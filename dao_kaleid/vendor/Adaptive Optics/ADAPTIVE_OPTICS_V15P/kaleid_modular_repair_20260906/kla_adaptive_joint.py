"""Complete measured joint correction; identical rules for every fixed case."""
import argparse
import copy
import json
import time
import numpy as np
from scipy.optimize import minimize
from kla_coupled_phase import PhaseControl, OUT as BASE, SCALE
from restore_kla import HERE, save
OUT=HERE/'results/kla_adaptive';OUT.mkdir(parents=True,exist_ok=True)


def run(seed,proposal):
    dest=OUT/f'{proposal}_{seed}.json'
    if dest.exists():return
    started=time.perf_counter();c=PhaseControl();c.calibration()
    path=BASE/f'{proposal}_{seed}.json'
    if not path.exists():c.case(seed,proposal,polish=False)
    base=json.loads(path.read_text());state=copy.deepcopy(base['phase_state']);local=[]
    local_calls=0;local_exposures=0
    # Refresh only when the measured three-field residual remains above 1.
    if base['phase_rounds'][-1]['phase_cost_after']>1.:
        np.random.seed(seed*100000+61108);c.calls=0;c.exposures=0
        prior=BASE/f'local_response_{proposal}_{seed}.json'
        if prior.exists():
            reused=json.loads(prior.read_text());local=reused['rows'];state=reused['terminal_state']
            local_calls=reused['acquisitions'];local_exposures=reused['exposure_stacks']
        else:
            for step in range(2):
                origin=c.vector(state);columns=[]
                for i in range(40):
                    probe=np.zeros(40);probe[i]=.15*SCALE[i]
                    columns.append((c.phase(c.decode(origin+probe),8)-c.phase(c.decode(origin-probe),8))/.3)
                J=np.stack(columns,axis=1);U,s,Vt=np.linalg.svd(J,full_matrices=False)
                residual=c.phase(state,16)-c.reference;candidates=[(float(residual@residual),0.,state,0.)]
                for damping in (.003,.001,.0003):
                    direction=-Vt.T@((s/(s*s+(damping*s[0])**2))*(U.T@residual))
                    direction=np.clip(direction,-.25,.25)*SCALE
                    for gain in (1.,.5,.25):
                        test=c.decode(origin+gain*direction);diff=c.phase(test,8)-c.reference
                        candidates.append((float(diff@diff),gain,test,damping))
                best=min(candidates,key=lambda r:r[0]);state=best[2]
                local.append({'step':step+1,'phase_cost':best[0],'gain':best[1],'damping':best[3],
                              'terminal':c.terminal(state,seed)})
                if best[0]<=1.:break
            local_calls=c.calls;local_exposures=c.exposures
            save(prior,{'source':str(path),'rows':local,'terminal_state':state,
                        'acquisitions':local_calls,'exposure_stacks':local_exposures})
    phase_state=copy.deepcopy(state);stages={'phase_restored':c.terminal(state,seed)};solvers=[];score_calls=0
    np.random.seed(seed*100000+61109)
    for kind,radius in (('broad',.03),('target',.01)):
        origin=c.vector(state);best=[-1.,np.zeros(40)]
        def objective(x):
            nonlocal score_calls
            candidate=c.decode(origin+x*SCALE);vals=[];score_calls+=1
            for f in c.fields:
                r=c.engine.get_field_opd_map(f,candidate,grid_size=48,add_noise=True)
                if r is None:return 0.
                freqs=np.linspace(500,4000,8) if kind=='broad' else [2000.]
                vals.extend(r['mtf_tangential'][np.argmin(abs(r['freqs']-fr))] for fr in freqs)
            v=np.asarray(vals);score=float(v.mean() if kind=='broad' else v.mean()-.3*v.std())
            if score>best[0]:best[:]=[score,x.copy()]
            return -score
        objective(np.zeros(40));fit=minimize(objective,np.zeros(40),method='Powell',bounds=[(-radius,radius)]*40,
                 options={'maxfev':600,'maxiter':5,'xtol':.0001,'ftol':.0001})
        state=c.decode(origin+best[1]*SCALE);stages['joint_'+kind]=c.terminal(state,seed)
        solvers.append({'phase':kind,'nit':int(fit.nit),'nfev':int(fit.nfev),'success':bool(fit.success),
                        'message':str(fit.message),'normalized_half_range':radius})
    rec={'system':'KLA','seed':seed,'proposal':proposal,'source_module_state':f'results/kla/{proposal}_{seed}.json',
         'source_fixed_phase':str(path),'stages':stages,'phase_rounds':base['phase_rounds'],'local_response_rounds':local,
         'phase_state':phase_state,'terminal_state':state,'phase_acquisitions':base['phase_acquisitions']+local_calls,
         'phase_exposure_stacks':base['phase_exposure_stacks']+local_exposures,'local_response_acquisitions':local_calls,
         'mtf_candidate_stacks':score_calls,'joint_solvers':solvers,'elapsed_seconds':time.perf_counter()-started,
         'protocol':'Six fixed-response rounds; measured phase cost > 1 triggers at most two local-response rounds; broad then target MTF polish.',
         'new_training_states':0}
    save(dest,rec);print('ADAPTIVE_JOINT_COMPLETE',proposal,seed,stages['joint_target'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--proposal',default='arch_a')
    a=p.parse_args();run(a.seed,a.proposal)
