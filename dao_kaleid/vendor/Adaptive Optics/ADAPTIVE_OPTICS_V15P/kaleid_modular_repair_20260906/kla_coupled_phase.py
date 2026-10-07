"""Restore coupled phase before local MTF refinement; reuse saved module states."""
import argparse
import copy
import json
import time
import numpy as np
from scipy.optimize import minimize
from restore_kla import HERE, KLAFullFOVEngine, PARAMS, save

OUT=HERE/'results/kla_phase';OUT.mkdir(parents=True,exist_ok=True)
SCALE=np.tile([.12,.12,.12,np.deg2rad(3.5/60),np.deg2rad(3.5/60)],8)


class PhaseControl:
    def __init__(self):
        self.engine=KLAFullFOVEngine();self.lenses=sum([list(self.engine.module_lenses[m]) for m in (2,3,4)],[])
        self.fields=[self.engine.field_points_y[i] for i in (0,4,8)];self.calls=0;self.exposures=0
        coord=np.linspace(-1,1,48);x,y=np.meshgrid(coord,coord);r=np.hypot(x,y);self.mask=(r>.1)&(r<.9)
        xx,yy=x[self.mask],y[self.mask]
        columns=[np.ones(len(xx)),xx,yy]
        for degree in range(2,7):columns.extend(xx**a*yy**(degree-a) for a in range(degree+1))
        self.basis=np.linalg.qr(np.stack(columns,axis=1))[0][:,3:]

    def decode(self,vector):return {l:{k:float(vector[i*5+j]) for j,k in enumerate(PARAMS)} for i,l in enumerate(self.lenses)}
    def vector(self,state):return np.array([state[l][k] for l in self.lenses for k in PARAMS])

    def phase(self,state,averages=4):
        readings=[];self.calls+=1;self.exposures+=averages
        for _ in range(averages):
            out=[]
            for f in self.fields:
                r=self.engine.get_field_opd_map(f,state,grid_size=48,add_noise=True)
                if r is None:raise RuntimeError('Missing pupil during phase measurement')
                out.extend(self.basis.T@r['opd_map'][self.mask])
            readings.append(out)
        return np.mean(readings,axis=0)

    def calibration(self):
        path=OUT/'calibration.npz'
        if path.exists():
            with np.load(path) as d:self.reference=d['reference'];self.jacobian=d['jacobian']
            return
        started=time.perf_counter();np.random.seed(202609061)
        self.reference=self.phase(self.decode(np.zeros(40)),16);cols=[]
        for i in range(40):
            probe=np.zeros(40);probe[i]=.15*SCALE[i]
            cols.append((self.phase(self.decode(probe),8)-self.phase(self.decode(-probe),8))/.3)
        self.jacobian=np.stack(cols,axis=1)
        np.savez_compressed(path,reference=self.reference,jacobian=self.jacobian)
        save(OUT/'calibration.json',{'settings':81,'three_field_exposure_stacks':656,
             'translation_probe_mm':.018,'tilt_probe_arcmin':.525,'new_training_states':0,
             'phase_description':'Three fields, orthonormal pupil polynomials through degree 6; piston and tilt removed',
             'singular_values':np.linalg.svd(self.jacobian,compute_uv=False).tolist(),
             'elapsed_seconds':time.perf_counter()-started})
        print('PHASE_CALIBRATION_COMPLETE',flush=True)

    def terminal(self,state,seed):
        rng=np.random.get_state();np.random.seed(seed*1000+90099)
        rows=[self.engine.get_field_opd_map(f,state,grid_size=64,add_noise=True) for f in self.engine.field_points_y]
        np.random.set_state(rng)
        if any(r is None for r in rows):raise RuntimeError('Missing terminal pupil')
        v=np.array([r['mtf_tangential'][np.argmin(abs(r['freqs']-2000))] for r in rows]);q=float(v.mean()-.3*v.std())
        return {'q2000':q,'paper_pass':bool(q>=.45),'field_mtf2000':v.tolist(),
                'center_mtf':float(v[4]),'minimum_edge_mtf':float(min(v[0],v[-1])),'above_reference':bool(q>.5962)}

    def case(self,seed,proposal,polish=True):
        path=OUT/f'{proposal}_{seed}.json'
        if path.exists():return
        source=HERE/f'results/kla/{proposal}_{seed}.json';record=json.loads(source.read_text())
        started=time.perf_counter();state=copy.deepcopy(record['acquired_state']);np.random.seed(seed*1000+61007)
        self.calls=0;self.exposures=0;trace=[];stages={};U,s,Vt=np.linalg.svd(self.jacobian,full_matrices=False)
        for step,damping in enumerate((.03,.01,.003,.001,.001,.001),1):
            measured=self.phase(state);residual=measured-self.reference
            direction=-Vt.T@((s/(s*s+(damping*s[0])**2))*(U.T@residual))
            direction=np.clip(direction,-.5,.5)*SCALE;origin=self.vector(state)
            candidates=[(float(residual@residual),0.,state)]
            for gain in (1.,.5,.25):
                candidate=self.decode(origin+gain*direction);v=self.phase(candidate)-self.reference
                candidates.append((float(v@v),gain,candidate))
            best=min(candidates,key=lambda r:r[0]);state=best[2]
            trace.append({'round':step,'damping':damping,'phase_cost_before':float(residual@residual),
                          'phase_cost_after':best[0],'gain':best[1]})
        stages['phase_restored']=self.terminal(state,seed);phase_state=copy.deepcopy(state);solvers=[];score_calls=0
        if polish:
            for kind,radius in (('broad',.03),('target',.01)):
                origin=self.vector(state);best=[-1.,np.zeros(40)]
                def objective(x):
                    nonlocal score_calls
                    candidate=self.decode(origin+x*SCALE);vals=[]
                    for f in self.fields:
                        r=self.engine.get_field_opd_map(f,candidate,grid_size=48,add_noise=True)
                        if r is None:return 0.
                        fs=np.linspace(500,4000,8) if kind=='broad' else [2000.]
                        vals.extend(r['mtf_tangential'][np.argmin(abs(r['freqs']-fr))] for fr in fs)
                    v=np.asarray(vals);score=float(v.mean() if kind=='broad' else v.mean()-.3*v.std());score_calls+=1
                    if score>best[0]:best[:]=[score,x.copy()]
                    return -score
                objective(np.zeros(40))
                fit=minimize(objective,np.zeros(40),method='Powell',bounds=[(-radius,radius)]*40,
                             options={'maxfev':600,'maxiter':5,'xtol':.0001,'ftol':.0001})
                state=self.decode(origin+best[1]*SCALE);stages['joint_'+kind]=self.terminal(state,seed)
                solvers.append({'phase':kind,'nit':int(fit.nit),'nfev':int(fit.nfev),'success':bool(fit.success),
                                'message':str(fit.message),'normalized_half_range':radius})
        result={'system':'KLA','seed':seed,'proposal':proposal,'source_modules':str(source),
          'stages':stages,'phase_state':phase_state,'terminal_state':state,'phase_rounds':trace,
          'phase_acquisitions':self.calls,'phase_exposure_stacks':self.exposures,
          'mtf_candidate_stacks':score_calls,'joint_solvers':solvers,'elapsed_seconds':time.perf_counter()-started,
          'training_states_added':0,'verification_noise_seed':seed*1000+90099}
        save(path,result);print('COUPLED_PHASE_COMPLETE',seed,proposal,stages,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=128);p.add_argument('--proposal',default='arch_a')
    p.add_argument('--phase-only',action='store_true');a=p.parse_args();control=PhaseControl();control.calibration();control.case(a.seed,a.proposal,not a.phase_only)
