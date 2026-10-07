"""Whole-objective controller ablations and a paired, bounded pupil DM."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.ndimage import map_coordinates
from nikon35 import HERE, ROOT, BIAS, NODES, AUDIT_NODES, TEST_SEEDS, Plant, grid, sample, save_json
from prepare_nikon35 import build, CKPTS, DEVICE

OUT = HERE / 'results/full_control80'
OUT.mkdir(parents=True,exist_ok=True)
RETRAIN = ROOT / 'rigid_group_retrain_20260908'
_corrected_pipeline_started = (RETRAIN/'status.json').exists() or (RETRAIN/'active_manifest.json').exists()
if _corrected_pipeline_started:
    OUT=RETRAIN/'results/control_nikon';OUT.mkdir(parents=True,exist_ok=True)
    if str(RETRAIN) not in sys.path:sys.path.insert(0,str(RETRAIN))
    from control_bridge import manifest as _corrected_manifest, nikon_calibration
    _corrected_manifest()  # refuse pre-fix checkpoints while retraining is pending
    JAC, CAPTURE_JAC = nikon_calibration()
else:
    JAC = np.load(HERE / 'results/center_calibration35.npz')['jacobian']
    CAPTURE_JAC = np.load(HERE / 'results/calibration35.npz')['jacobian']
MODELS = {}
CAP = 80
torch.set_num_threads(2)


def predictor(name):
    if _corrected_pipeline_started:
        from control_bridge import nikon_predict
        return lambda raw:nikon_predict(raw,architecture=name)
    if name not in MODELS:
        checkpoint = torch.load(CKPTS/f'nikon35_{name}.pt',map_location='cpu',weights_only=False)
        model = build(name).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict']); model.eval()
        MODELS[name] = (model, checkpoint)
    model, checkpoint = MODELS[name]
    def predict(raw):
        if isinstance(raw,dict):raw=raw['network']
        with torch.inference_mode():
            out = model(torch.from_numpy(grid(raw)).to(DEVICE))[0].cpu().numpy()
        return out*checkpoint['target_std']+checkpoint['target_mean']
    return predict


def score(raw):
    if isinstance(raw,dict):raw=raw['center']
    return float(np.sqrt(np.mean(raw**2))) if raw is not None else float('inf')


def inverse(jac, raw, fraction):
    if isinstance(raw,dict):raw=raw['center'] if jac.shape[0]==197 else raw['network']
    u,s,vt = np.linalg.svd(jac,full_matrices=False)
    return -(vt.T*(s/(s*s+(fraction*s[0])**2)))@u.T@raw.ravel()


def controller(plant, proposal, variant, damping, budget=CAP):
    """No hidden state or final exact optical scores enter control decisions."""
    zero=np.zeros(35); raw=plant.measure(zero)
    if raw is None:return zero, []
    q=zero.copy(); best=score(raw); accepted=[]
    # All variants share a final diagnostic; initial diagnosis is not a search step.
    max_readings=budget  # at most 39 searches plus the initial diagnosis
    if proposal is not None:
        pred=np.clip(proposal(raw),-1.5,1.5)
        if variant=='L1':
            q=-pred; plant.measure(q);return q, [{'phase':'unit proposal'}]
        if variant=='L2':
            q=-.7*pred;plant.measure(q);return q,[{'phase':'fixed 0.7 proposal'}]
        for gain in (.25,.5,.75,1.):
            candidate=-gain*pred
            measured=plant.measure(candidate)
            if score(measured)<best:
                q,raw,best=candidate,measured,score(measured)
        accepted.append({'phase':'measured proposal gain','score':best,'command':q.tolist()})
        if variant=='gain_only':return q,accepted
    if variant=='L3':
        # Fixed cyclic schedule, covering dx/dy/dz/tx/ty without hidden-group selection.
        for index in range(35):
            if plant.readings+2>max_readings:break
            for sign in (1.,-1.):
                candidate=q.copy();candidate[index]+=sign*.025
                candidate=np.clip(candidate,-1.5,1.5)
                measured=plant.measure(candidate)
                if score(measured)<best:
                    q,raw,best=candidate,measured,score(measured)
                    accepted.append({'phase':'fixed coordinate','coordinate':index,'score':best})
        return q,accepted
    # The full 35-column local calibration is acquired once at the fixed work point.
    # Accepted measured differences update this response during correction.
    jac=CAPTURE_JAC.copy()
    center_jac=JAC.copy()
    local_groups=[]
    for iteration in range(30):
        if plant.readings>=max_readings or best<=.04:break
        direction=inverse(jac,raw,damping)
        norm=float(np.max(np.abs(direction)))
        if norm>1.:direction/=norm
        old_q,old_raw=q.copy(),raw.copy()
        improved=False
        for gain in (1.,.5,.25):
            if plant.readings>=max_readings:break
            candidate=np.clip(old_q+gain*direction,-1.5,1.5)
            measured=plant.measure(candidate)
            if score(measured)<best:
                q,raw,best=candidate,measured,score(measured)
                improved=True
                # Accept the first decrease so a good proposal saves measurements.
                break
        if improved:
            dq=q-old_q; denominator=float(dq@dq)
            if denominator>1e-10:
                dr=(raw['network']-old_raw['network']).ravel()
                jac += np.outer(dr-jac@dq,dq)/denominator
                dc=raw['center']-old_raw['center']
                center_jac += np.outer(dc-center_jac@dq,dq)/denominator
            accepted.append({'phase':'residual refinement','iteration':iteration+1,
                             'score':best,'command':q.tolist()})
        else:
            # A small multi-field differential is not a center-WRMS stopping
            # criterion. Probe a complete five-axis body and refine the actual
            # center phase when the capture direction no longer helps.
            if plant.readings+11>max_readings:break
            optical_direction=inverse(center_jac,raw,damping)
            energies=[np.linalg.norm(optical_direction[g*5:(g+1)*5]) for g in range(7)]
            for g in local_groups:energies[g]=-1
            group=int(np.argmax(energies));local_groups.append(group)
            indices=np.arange(group*5,group*5+5)
            local=[];probe=.05
            for index in indices:
                dq=np.zeros(35);dq[index]=probe
                upper=np.clip(q+dq,-1.5,1.5);lower=np.clip(q-dq,-1.5,1.5)
                plus,minus=plant.measure(upper),plant.measure(lower)
                if plus is None or minus is None:local=[];break
                local.append((plus['center']-minus['center'])/(upper[index]-lower[index]))
            if not local:break
            local=np.column_stack(local)
            correction=inverse(local,raw,damping)
            scale=max(1.,float(np.max(np.abs(correction))))
            correction/=scale
            base_q=q.copy()
            for gain in (1.,.5,.25):
                if plant.readings>=max_readings:break
                candidate=base_q.copy();candidate[indices]+=gain*correction
                candidate=np.clip(candidate,-1.5,1.5)
                measured=plant.measure(candidate)
                if score(measured)<best:
                    q,raw,best=candidate,measured,score(measured)
                    accepted.append({'phase':'measured local center refinement','group':group+1,
                                     'score':best,'command':q.tolist()})
                    break
            jac=CAPTURE_JAC.copy();center_jac=JAC.copy()
    return q,accepted


def case(seed,name,variant,damping,budget=CAP):
    hidden,groups=sample(seed)
    plant=Plant(hidden,noise_seed=seed,score_center=True)
    started=time.perf_counter()
    try:
        initial=plant.terminal(np.zeros(35))[0]['absolute_pttd_rms_waves']
        q,history=controller(plant,predictor(name) if name else None,variant,damping,budget)
        endpoint=plant.terminal(q)[0]['absolute_pttd_rms_waves']
        error=None
    except Exception as exc:
        initial=endpoint=None; q=np.zeros(35);history=[];error=f'{type(exc).__name__}: {exc}'
    return {'system':'Nikon 100x','seed':seed,'proposal':name or 'calibrated SVD','variant':variant,
            'groups':groups,'dofs':35,'initial_metric':initial,'terminal_metric':endpoint,
            'success':int(endpoint is not None and endpoint<.07),
            'search_steps':max(0,plant.readings-1),'terminal_measurements':1,
            'moves_used':plant.readings,'all_optical_measurements':plant.readings+1,
            'offline_calibration_measurements':70,'command':q.tolist(),'error':error,
            'history':history,'readings':plant.records,'elapsed_seconds':time.perf_counter()-started}


def tune():
    (OUT/'development').mkdir(parents=True,exist_ok=True)
    path=OUT/'selected_control.json'
    if path.exists():return json.loads(path.read_text())['damping_fraction']
    # Six new development states, disjoint from training seeds and all 11 test seeds.
    seeds=tuple(range(2030000,2030006))
    alternatives=[]
    for damping in (1e-5,1e-4,1e-3,.01):
        rows=[]
        for seed in seeds:
            record=OUT/'development'/f'damping{damping:g}_seed{seed}.json'
            if record.exists():row=json.loads(record.read_text(encoding='utf-8'))
            else:
                row=case(seed,'arch_a','L4',damping)
                save_json(record,row)
            rows.append(row)
        values=[r['terminal_metric'] for r in rows]
        if any(v is None for v in values):mean=None
        else:mean=float(np.mean(values))
        choice={'damping_fraction':damping,'successes':sum(r['success'] for r in rows),
                'mean_wrms':mean,'mean_steps':float(np.mean([r['moves_used'] for r in rows]))}
        alternatives.append(choice)
        print('DEVELOPMENT '+json.dumps(choice),flush=True)
    selected=dict(min(alternatives,key=lambda r:(-r['successes'],r['mean_wrms'] if r['mean_wrms'] is not None else 1e9,r['mean_steps'])))
    selected.update({'development_seeds':seeds,'candidates':alternatives,'test_seeds':TEST_SEEDS,
                     'test_outcomes_used_for_selection':False,'budget':CAP,
                     'refinement':'full 35-coordinate measured response, damped inverse, secant updates'})
    save_json(path,selected)
    return selected['damping_fraction']


def dm_basis():
    import poppy
    dm=poppy.ContinuousDeformableMirror(dm_shape=(12,12),radius=.001,include_factor_of_two=True)
    wave=poppy.Wavefront(wavelength=587.5618e-9,npix=64,diam=.002)
    maps=[]
    for y in range(12):
        for x in range(12):
            dm.flatten();dm.set_actuator(x,y,1e-6)
            value=dm.get_opd(wave)
            if hasattr(value,'get'):value=value.get()
            maps.append(np.asarray(value,float)/1e-6)
    return np.asarray(maps)


def influences(basis,nodes):
    # POPPY pixel centres run from -1+1/N to 1-1/N pupil radii.
    index=(nodes+1)*basis.shape[1]/2-.5
    vals=np.column_stack([map_coordinates(item,[index[:,1],index[:,0]],order=1,mode='nearest') for item in basis])
    design=np.c_[np.ones(len(nodes)),nodes[:,0],nodes[:,1],np.sum(nodes**2,axis=1)]
    return vals-design@np.linalg.lstsq(design,vals,rcond=None)[0]


def dm_case(seed,basis):
    hidden,groups=sample(seed);plant=Plant(hidden,noise_seed=seed,score_center=True)
    raw=plant.measure(np.zeros(35))
    # Fit the same absolute center-WRMS objective used by DAO. This one
    # pupil command is then applied to center and every off-axis field.
    matrix=influences(basis,AUDIT_NODES)*3.5e-6/587.5618e-9
    target=-raw['center'].ravel()
    fit=lsq_linear(np.vstack([matrix,1e-6*np.eye(144)]),np.r_[target,np.zeros(144)],
                   bounds=(-1.,1.),tol=1e-12,max_iter=2000,lsmr_tol=None)
    fields=plant.terminal(np.zeros(35),fields=(0.,-.1,.1))
    dm_field=influences(basis,AUDIT_NODES)@fit.x*3.5e-6/587.5618e-9
    before=[float(item['absolute_pttd_rms_waves']) for item in fields]
    after=[float(np.sqrt(np.mean((item['residual_waves']+dm_field)**2))) for item in fields]
    return {'seed':seed,'groups':groups,'dofs':35,'initial_metric':before[0],
            'terminal_metric':after[0],'success':int(after[0]<.07),
            'initial_field_wrms':before,'terminal_field_wrms':after,
            'command_um':(fit.x*3.5).tolist(),'solver_success':bool(fit.success),
            'solver_iterations':int(fit.nit),'optimality':float(fit.optimality),
            'stroke_um':3.5,'fitted_fields_mm':[0.],'evaluated_fields_mm':[0.,-.1,.1],
            'metric':'absolute center PTTD WRMS, waves'}


def evaluate():
    (OUT/'cases').mkdir(parents=True,exist_ok=True)
    damping=tune()
    entries=[('arch_a',v) for v in ('L1','L2','gain_only','L3','L4')]
    entries += [(name,'L4') for name in ('arch_b','arch_c','slor')]
    entries += [('slor','L1'),(None,'L4')]
    rows=[]
    for name,variant in entries:
        for i,seed in enumerate(TEST_SEEDS,1):
            path=OUT/'cases'/f'{name or "svd"}_{variant}_{i:02d}.json'
            if path.exists():row=json.loads(path.read_text(encoding='utf-8'))
            else:
                row=case(seed,name,variant,damping);row['case_id']=f'Case_{i:02d}';save_json(path,row)
            rows.append(row)
        sub=rows[-11:]
        print(f'TEST {name}/{variant} success={sum(r["success"] for r in sub)}/11 mean={np.mean([r["terminal_metric"] for r in sub if r["terminal_metric"] is not None]):.6f}',flush=True)
    basis=dm_basis()
    for i,seed in enumerate(TEST_SEEDS,1):
        path=OUT/'cases'/f'dm_{i:02d}.json'
        if not path.exists():
            row=dm_case(seed,basis);row['case_id']=f'Case_{i:02d}';save_json(path,row)
    columns=['system','case_id','seed','proposal','variant','initial_metric','terminal_metric','success',
             'search_steps','terminal_measurements','moves_used','all_optical_measurements','offline_calibration_measurements','elapsed_seconds']
    with (OUT/'comparison35.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary=[]
    for name,variant in entries:
        sub=[r for r in rows if r['proposal']==(name or 'calibrated SVD') and r['variant']==variant]
        vals=[r['terminal_metric'] for r in sub]
        item={'proposal':name or 'calibrated SVD','variant':variant,'n':len(sub),
              'successes':sum(r['success'] for r in sub),'mean':None if any(v is None for v in vals) else float(np.mean(vals)),
              'median':None if any(v is None for v in vals) else float(np.median(vals)),
              'mean_steps':float(np.mean([r['moves_used'] for r in sub]))}
        summary.append(item)
    dm=[json.loads((OUT/'cases'/f'dm_{i:02d}.json').read_text()) for i in range(1,12)]
    summary.append({'proposal':'12x12 DM','variant':'shared pupil','n':11,'successes':sum(r['success'] for r in dm),
                    'mean':float(np.mean([r['terminal_metric'] for r in dm])),
                    'median':float(np.median([r['terminal_metric'] for r in dm])),
                    'initial_mean':float(np.mean([r['initial_metric'] for r in dm]))})
    save_json(OUT/'summary35.json',{'status':'complete','dofs':35,'groups':7,'test_seeds':TEST_SEEDS,
                                  'damping':damping,'results':summary})
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tune-only',action='store_true');args=parser.parse_args()
    if str(RETRAIN) not in sys.path:sys.path.insert(0,str(RETRAIN))
    from fast_execution import execution
    with execution('nikon'):
        if args.tune_only:tune()
        else:evaluate()
