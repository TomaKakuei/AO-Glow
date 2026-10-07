"""Compact front/rear proposals plus the original joint 40-coordinate solver."""
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
from scipy.optimize import minimize, OptimizeResult
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for directory in (ROOT,ROOT/'canonical_three_phase_universal',ROOT/'minimal_ablation_completion'):
    sys.path.insert(0,str(directory))
from models import build_universal_model
from eval_selected_trepan2p_ablation import (OpticalRuntime,TRAINING_GRID,initial_state_for_seed,
    vec_to_state_group,FRONT_ANCHORS,REAR_ANCHORS,ALL_ANCHORS,NOMINAL,SurfacePerturbation)
OUT=HERE/'results/trepan';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(3,34,36,96,100,142,136,188,187,246,272)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
RETRAIN=ROOT/'rigid_group_retrain_20260908'
if (RETRAIN/'status.json').exists() or (RETRAIN/'active_manifest.json').exists():
    OUT=RETRAIN/'results/control_trepan';OUT.mkdir(parents=True,exist_ok=True)

def state_json(s):return {str(k):dict(zip(SurfacePerturbation._fields,map(float,v))) for k,v in s.items()}

def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def load(group,front='arch_a'):
    # Once the corrected retraining pipeline has started, archived checkpoints
    # are intentionally unavailable: their labels came from the pre-fix tilt
    # geometry. The bridge becomes usable only after its atomic ready manifest.
    if (RETRAIN/'status.json').exists() or (RETRAIN/'active_manifest.json').exists():
        if str(RETRAIN) not in sys.path:sys.path.insert(0,str(RETRAIN))
        from control_bridge import load_trepan_branch
        return load_trepan_branch(group,front)
    path=(HERE/'checkpoints/trepan_rear_arch_a.pt' if group=='rear' else
          ROOT/'canonical_three_phase_universal/checkpoints'/f'trepan2p_{front}.pt')
    if front in ('slor_mlp','sensitivity_svd') and group=='front':
        from models_and_data import checkpoint_path,build_proposal
        path=checkpoint_path('trepan2p',front);ckpt=torch.load(path,map_location='cpu',weights_only=False)
        net=build_proposal('trepan2p',front,ckpt)
    else:
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        net=build_universal_model(front if group=='front' else 'arch_a',9,ckpt['group_dofs'])
        net.load_state_dict(ckpt['state_dict'])
    net.to(DEVICE).eval()
    return net,ckpt,path

def case(seed,front='arch_a',joint=True,*,use_nn=True,score_budget=None):
    suffix='' if use_nn and score_budget is None else f'_{"nn" if use_nn else "no_nn"}_budget{score_budget}'
    path=OUT/f'{front}_{seed}{suffix}.json'
    if path.exists():return json.loads(path.read_text())
    started=time.perf_counter();runtime=OpticalRuntime(TRAINING_GRID,'speckle')
    initial=initial_state_for_seed(seed);stages={};trace=[];observations=0;score_calls=0
    def metric(state):
        mean,fields=runtime.wide_wrms(state)
        return {'paper_wrms':float(mean),'center_wrms':float(fields[0]),'field_wrms':list(map(float,fields)),
                'maximum_field_wrms':float(max(fields)),'paper_pass':bool(mean<.07)}
    stages['initial']=metric(initial);modules={};unit_state=copy.deepcopy(initial);acquired=copy.deepcopy(initial)
    groups=(('front',FRONT_ANCHORS),('rear',REAR_ANCHORS)) if use_nn else ()
    for index,(group,anchors) in enumerate(groups):
        net,ckpt,source=load(group,front);current=np.zeros(20)
        isolated={a:initial[a] if a in anchors else NOMINAL for a in ALL_ANCHORS}
        counts=[]
        for round_index,target_gain in enumerate((1.,.7,.3)):
            obs=runtime.observe(isolated,noise_seed=seed*1000+(index*3+round_index+1)*100);observations+=1
            with torch.no_grad():pred=net(torch.from_numpy(obs).unsqueeze(0).float().to(DEVICE))[0].cpu().numpy()
            pred=pred*np.asarray(ckpt['target_std'])+np.asarray(ckpt['target_mean'])
            if round_index==0:
                unit_candidate=vec_to_state_group(pred,initial,anchors)
                for a in anchors:unit_state[a]=unit_candidate[a]
            before=score_calls
            def objective(value):
                nonlocal score_calls
                gain=float(np.asarray(value).ravel()[0]);candidate=vec_to_state_group(current+gain*pred,initial,anchors)
                isolated_candidate={a:candidate[a] if a in anchors else NOMINAL for a in ALL_ANCHORS}
                score=float(runtime.center_wrms(isolated_candidate));score_calls+=1
                trace.append({'phase':'module_gain','module':group,'round':round_index+1,'gain':gain,'center_wrms':score})
                return score
            grid=np.linspace(target_gain-.2,target_gain+.2,5);grid_scores=[objective([g]) for g in grid]
            fit=minimize(objective,[grid[int(np.argmin(grid_scores))]],method='BFGS',options={'eps':1e-3,'maxiter':5})
            candidates=[(float(fit.fun),float(fit.x[0])),(objective([target_gain]),target_gain),(objective([0.]),0.)]
            best_score,best_gain=min(candidates);current+=best_gain*pred
            full=vec_to_state_group(current,initial,anchors)
            isolated={a:full[a] if a in anchors else NOMINAL for a in ALL_ANCHORS}
            counts.append({'round':round_index+1,'nit':int(fit.nit),'nfev':int(fit.nfev),
                           'total_score_calls':score_calls-before,'selected_gain':best_gain,'score':best_score})
        for a in anchors:acquired[a]=isolated[a]
        modules[group]={'checkpoint':str(source),'parameters':sum(p.numel() for p in net.parameters()),'rounds':counts}
        save(OUT/f'{front}_{seed}_progress.json',{'status':'modules_in_progress','seed':seed,'completed_modules':modules,
             'acquired_state':state_json(acquired),'candidate_score_calls':score_calls,'trace':trace,
             'elapsed_seconds':time.perf_counter()-started})
        del net
    stages['unit_proposals']=metric(unit_state)
    fixed_state={a:SurfacePerturbation(*(np.asarray(initial[a])-.7*(np.asarray(initial[a])-np.asarray(unit_state[a])))) for a in ALL_ANCHORS}
    stages['fixed_07']=metric(fixed_state);stages['measured_modular']=metric(acquired)
    save(OUT/f'{front}_{seed}_progress.json',{'status':'joint_in_progress','seed':seed,'stages':stages,
         'acquired_state':state_json(acquired),'candidate_score_calls':score_calls,'trace':trace,
         'elapsed_seconds':time.perf_counter()-started})
    x0=np.concatenate([np.asarray(initial[a])-np.asarray(acquired[a]) for a in ALL_ANCHORS])
    def decode(x):return {a:SurfacePerturbation(*(np.asarray(initial[a])-x[i*5:(i+1)*5])) for i,a in enumerate(ALL_ANCHORS)}
    start_calls=score_calls
    best_joint={'x':x0.copy(),'score':float('inf')}
    class ScoreBudgetReached(RuntimeError):pass
    def objective_joint(x):
        nonlocal score_calls
        if score_budget is not None and score_calls>=score_budget:raise ScoreBudgetReached()
        score=float(runtime.center_wrms(decode(x)));score_calls+=1
        if score<best_joint['score']:best_joint.update(x=np.asarray(x).copy(),score=score)
        trace.append({'phase':'joint_40','center_wrms':score});return score
    try:
        fit=minimize(objective_joint,x0,method='L-BFGS-B',options={'maxiter':30,'ftol':1e-4,'eps':1e-2})
    except ScoreBudgetReached:
        fit=OptimizeResult(x=best_joint['x'],fun=best_joint['score'],nit=-1,nfev=score_calls-start_calls,
                           success=False,message='Shared measurement budget exhausted; best measured state retained')
    final=decode(fit.x);stages['joint_40']=metric(final)
    result={'system':'Trepan2p','seed':seed,'front_proposal':front,'stages':stages,'modules':modules,
            'terminal_state':state_json(final),'acquired_state':state_json(acquired),'initial_state':state_json(initial),
            'unit_state':state_json(unit_state),'fixed_state':state_json(fixed_state),
            'network_observations':observations,'candidate_score_calls':score_calls,'use_nn':use_nn,'score_budget':score_budget,
            'joint_solver':{'method':'L-BFGS-B','nit':int(fit.nit),'nfev':int(fit.nfev),
                            'score_calls':score_calls-start_calls,'success':bool(fit.success),'message':str(fit.message)},
            'stage_verifications':5,'stage_verification_fields':9,'trace':trace,
            'observation_scope':'front and rear separately observed with other module nominal, as original modular study',
            'elapsed_seconds':time.perf_counter()-started}
    save(path,result);print('TREPAN_COMPLETE',seed,front,stages['joint_40'],result['joint_solver'],flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=136);parser.add_argument('--front',default='arch_a')
    args=parser.parse_args()
    if str(RETRAIN) not in sys.path:sys.path.insert(0,str(RETRAIN))
    from fast_execution import execution
    with execution('trepan'):case(args.seed,args.front)
