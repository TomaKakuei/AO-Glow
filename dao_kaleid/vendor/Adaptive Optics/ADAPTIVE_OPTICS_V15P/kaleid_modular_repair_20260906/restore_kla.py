"""Three compact module proposals, followed by coupled broad/target refinement."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for directory in (ROOT.parent,ROOT/'canonical_three_phase_universal',ROOT/'minimal_ablation_completion',ROOT/'OL'):
    sys.path.insert(0,str(directory))
from models import build_universal_model
from kla_full_fov_engine import KLAFullFOVEngine
from run_poppy_dm_system_audit import KLA_CASE_DEFINITIONS,make_kla_perturbations
OUT=HERE/'results/kla';OUT.mkdir(parents=True,exist_ok=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
PARAMS=('dx','dy','dz','tx','ty')

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def load(module,proposal='arch_a'):
    if module==2:
        path=ROOT/'canonical_three_phase_universal/checkpoints'/f'kla_{proposal}.pt'
        if proposal=='arch_a_7k':path=ROOT/'canonical_three_phase_universal/checkpoints/kla_arch_a_7k_warmstart.pt'
    else:path=HERE/'checkpoints'/f'kla_module{module}_arch_a.pt'
    if module==2 and proposal in ('slor_mlp','sensitivity_svd'):
        from models_and_data import checkpoint_path,build_proposal
        path=checkpoint_path('kla',proposal);ckpt=torch.load(path,map_location='cpu',weights_only=False)
        net=build_proposal('kla',proposal,ckpt)
    else:
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        net=build_universal_model(proposal if module==2 and proposal in ('arch_b','arch_c') else 'arch_a',9,ckpt['group_dofs'])
        net.load_state_dict(ckpt['state_dict'])
    net.to(DEVICE).eval();return net,ckpt,path


def case(seed,proposal='arch_a',maxfev=600):
    path=OUT/f'{proposal}_{seed}.json'
    if path.exists():return json.loads(path.read_text())
    started=time.perf_counter();engine=KLAFullFOVEngine()
    definition=next(r for r in KLA_CASE_DEFINITIONS if r['seed']==seed)
    initial=make_kla_perturbations(engine,definition);state=copy.deepcopy(initial)
    np.random.seed(seed*1000+60906)
    fields=list(engine.field_points_y);fast_fields=[fields[i] for i in (0,4,8)]
    lenses=sum([list(engine.module_lenses[i]) for i in (2,3,4)],[])
    networks={m:load(m,proposal) for m in (2,3,4)}
    calls={'network_stacks':0,'candidate_stacks':0,'candidate_field_traces':0,'terminal_stacks':0}
    history=[];gain_records=[];stages={}
    def measure(s,selected_fields,grid=48):
        return [engine.get_field_opd_map(f,s,grid_size=grid,add_noise=True) for f in selected_fields]
    def metric(s,noise_seed):
        rng=np.random.get_state();np.random.seed(noise_seed)
        rows=measure(s,fields,64);np.random.set_state(rng);calls['terminal_stacks']+=1
        if any(r is None for r in rows):return {'q2000':-1.,'field_mtf2000':None,'paper_pass':False}
        mtf=np.asarray([r['mtf_tangential'][np.argmin(abs(r['freqs']-2000.))] for r in rows])
        return {'q2000':float(mtf.mean()-.3*mtf.std()),'field_mtf2000':mtf.tolist(),
                'center_mtf':float(mtf[4]),'minimum_edge_mtf':float(min(mtf[0],mtf[-1])),
                'paper_pass':bool(mtf.mean()-.3*mtf.std()>=.45),'above_reference':bool(mtf.mean()-.3*mtf.std()>.5962)}
    def score(s,kind):
        rows=measure(s,fast_fields,48);calls['candidate_stacks']+=1;calls['candidate_field_traces']+=3
        if any(r is None for r in rows):return 0.
        targets=np.linspace(500.,4000.,8) if kind=='broad' else [2000.]
        values=np.asarray([[r['mtf_tangential'][np.argmin(abs(r['freqs']-f))] for f in targets] for r in rows])
        return float(values.mean() if kind=='broad' else values.mean()-.3*values.std())
    stages['initial']=metric(state,seed*1000+90099)
    for round_index,weight in enumerate((1.,.5,.25)):
        for module in (2,3,4):
            rows=measure(state,fields,64);calls['network_stacks']+=1
            if any(r is None for r in rows):raise RuntimeError(f'Network pupil missing {seed} {module}')
            obs=np.asarray([r['opd_map'] for r in rows],dtype=np.float32)
            net,ckpt,_=networks[module]
            with torch.no_grad():pred=net(torch.from_numpy(obs).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            pred=weight*(pred*np.asarray(ckpt['target_std'])+np.asarray(ckpt['target_mean']))
            for index,lens in enumerate(engine.module_lenses[module]):
                direction=pred[index*5:(index+1)*5];origin=copy.deepcopy(state)
                def candidate(gain):
                    p=copy.deepcopy(origin);p[lens]={k:float(origin[lens][k]-gain*direction[j]) for j,k in enumerate(PARAMS)};return p
                before=calls['candidate_stacks'];zero=score(origin,'broad')
                fit=minimize_scalar(lambda g:-score(candidate(g),'broad'),bounds=(-.5,2.5),method='bounded',options={'maxiter':6})
                accepted=bool(-fit.fun>zero)
                if accepted:state=candidate(fit.x)
                gain_records.append({'round':round_index+1,'module':module,'lens':lens,
                     'gain':float(fit.x) if accepted else 0.,'nit':int(fit.nit),'nfev':int(fit.nfev),
                     'candidate_calls':calls['candidate_stacks']-before})
        stages[f'module_round{round_index+1}']=metric(state,seed*1000+90099)
    acquired=copy.deepcopy(state);solver=[]
    # Normalized coordinates give translations and tilts comparable search ranges.
    scales=np.tile([.12,.12,.12,(3.5/60)*(np.pi/180),(3.5/60)*(np.pi/180)],8)
    for phase,kind in (('joint_broad','broad'),('joint_target','target')):
        origin=copy.deepcopy(state);initial_vector=np.asarray([origin[l][k] for l in lenses for k in PARAMS])
        def decode(x):
            values=initial_vector+x*scales
            return {l:{k:float(values[i*5+j]) for j,k in enumerate(PARAMS)} for i,l in enumerate(lenses)}
        best=[score(origin,kind),np.zeros(40)];before=calls['candidate_stacks']
        def objective(x):
            value=score(decode(x),kind)
            history.append({'phase':phase,'score':value})
            if value>best[0]:best[:]=[value,x.copy()]
            return -value
        fit=minimize(objective,np.zeros(40),method='Powell',bounds=[(-1.,1.)]*40,
                     options={'maxfev':maxfev,'maxiter':5,'xtol':1e-3,'ftol':1e-4})
        state=decode(best[1]);stages[phase]=metric(state,seed*1000+90099)
        solver.append({'phase':phase,'method':'Powell','nit':int(fit.nit),'nfev':int(fit.nfev),
                       'candidate_calls':calls['candidate_stacks']-before,'success':bool(fit.success),'message':str(fit.message)})
    record={'system':'KLA','seed':seed,'case_definition':definition,'module2_proposal':proposal,'stages':stages,
            'initial_state':initial,'acquired_state':acquired,'terminal_state':state,'calls':calls,
            'module_gain_searches':gain_records,'joint_solvers':solver,'trace':history,'maxfev_per_joint_stage':maxfev,
            'network_parameters':{str(m):sum(p.numel() for p in net.parameters()) for m,(net,_,_) in networks.items()},
            'elapsed_seconds':time.perf_counter()-started}
    save(path,record);print('KLA_COMPLETE',seed,proposal,stages['joint_target'],calls,record['elapsed_seconds'],flush=True)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=156);parser.add_argument('--proposal',default='arch_a')
    parser.add_argument('--maxfev',type=int,default=600);args=parser.parse_args();case(args.seed,args.proposal,args.maxfev)
