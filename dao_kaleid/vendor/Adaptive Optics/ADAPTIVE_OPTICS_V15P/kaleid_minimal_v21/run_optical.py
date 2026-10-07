"""Fixed existing cases, complete modules; standalone Slor never enters Kaleid."""
from pathlib import Path
import argparse
import ast
import copy
import inspect
import json
import os
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OLD=ROOT/'kaleid_modular_repair_20260906'
sys.path.insert(0,str(OLD))
from fit_missing import checkpoint,build,save,DEVICE
OUT=HERE/'results'

def network(module,arch):
    path=checkpoint(module,arch)
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    return build(module,arch,ckpt),ckpt,path

def trepan(seed,arch,grid=32):
    import restore_trepan as original
    folder=OUT/f'trepan_grid{grid}';folder.mkdir(parents=True,exist_ok=True)
    dest=folder/f'{arch}_{seed}.json'
    if dest.exists():return
    runtime_class=original.OpticalRuntime
    instances=[]
    class Runtime(runtime_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.score_cache={};self.traces=0;self.invalid_candidates=0
            instances.append(self)
        def center_wrms(self,state):
            key=tuple(float(v) for a in original.ALL_ANCHORS for v in state[a])
            if key in self.score_cache:return self.score_cache[key]
            engine=self.center_engine
            engine.set_surface_perturbations(state,clear_others=True)
            try:
                _,_,opd,valid=engine._sample_wavefront(num_rays=grid,field_index=0,wavelength_nm=engine.wavelength_nm,focus=4.647)
                score=self.compute_true_wrms(opd,valid) if np.any(valid) else 999.
            except ValueError as error:
                if str(error)!='math domain error':raise
                # No real reference-sphere intersection: an invalid trial move,
                # with the same penalty as a missing candidate pupil.
                score=999.;self.invalid_candidates+=1
            self.traces+=1;self.score_cache[key]=score
            return score
    original.OUT=folder;original.OpticalRuntime=Runtime
    original.load=lambda group,front:network('trepan_'+group,front)
    torch.set_num_threads(1)
    row=original.case(seed,arch)
    row.update(algorithm='Kaleid',architecture=arch,control_pupil_grid=grid,terminal_pupil_grid=64,
               network_parameters=sum(m['parameters'] for m in row['modules'].values()),
               unique_candidate_traces=instances[0].traces,
               invalid_candidate_count=instances[0].invalid_candidates,
               source='Same modular gain and joint-40 algorithm; cached candidate scoring on stated pupil grid.')
    save(dest,row)

def trepan_slor(seed):
    import restore_trepan as t
    path=OUT/'slor_trepan'/f'slor_{seed}.json'
    if path.exists():return
    start=time.perf_counter();r=t.OpticalRuntime(t.TRAINING_GRID,'speckle');initial=t.initial_state_for_seed(seed)
    state=copy.deepcopy(initial);parameters=0;predictions={}
    for i,(group,anchors) in enumerate((('front',t.FRONT_ANCHORS),('rear',t.REAR_ANCHORS))):
        model,ckpt,source=network('trepan_'+group,'slor');parameters+=sum(p.numel() for p in model.parameters())
        isolated={a:initial[a] if a in anchors else t.NOMINAL for a in t.ALL_ANCHORS}
        obs=r.observe(isolated,noise_seed=seed*1000+(i*3+1)*100)
        with torch.inference_mode():pred=model(torch.from_numpy(obs).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
        pred=pred*np.asarray(ckpt['target_std'])+np.asarray(ckpt['target_mean'])
        corrected=t.vec_to_state_group(pred,initial,anchors)
        for a in anchors:state[a]=corrected[a]
        predictions[group]={'checkpoint':str(source),'prediction':pred.tolist()}
    mean,fields=r.wide_wrms(state)
    row={'system':'Trepan2p','method':'Slor standalone','seed':seed,'parameters':parameters,
         'terminal':{'paper_wrms':mean,'field_wrms':fields,'center_wrms':fields[0],
                     'maximum_field_wrms':max(fields),'paper_pass':bool(mean<.07)},
         'initial_state':t.state_json(initial),'terminal_state':t.state_json(state),'predictions':predictions,
         'network_calls':2,'network_observation_stacks':2,'applied_corrections':1,
         'gain_searches':0,'refinement_iterations':0,'terminal_pupil_grid':64,
         'elapsed_seconds':time.perf_counter()-start}
    save(path,row);print('SLOR_TREPAN_COMPLETE',seed,row['terminal'],flush=True)

def kla(seed,arch):
    import restore_kla as r
    import kla_coupled_phase as p
    import kla_adaptive_joint as j
    r.OUT=OUT/'kla_modules';r.OUT.mkdir(parents=True,exist_ok=True)
    r.load=lambda module,proposal:network(f'kla_module{module}',proposal)
    dest=r.OUT/f'{arch}_{seed}.json'
    if not dest.exists():
        # Execute the original module stage exactly, without its discarded broad
        # Powell experiment. The adopted joint solver runs only once below.
        tree=ast.parse(inspect.getsource(r.case))
        class ModulesOnly(ast.NodeTransformer):
            def visit_For(self,node):
                if isinstance(node.target,ast.Tuple) and [getattr(e,'id',None) for e in node.target.elts]==['phase','kind']:
                    node.iter=ast.Tuple(elts=[],ctx=ast.Load())
                return self.generic_visit(node)
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='print':return None
                return self.generic_visit(node)
        tree=ModulesOnly().visit(tree);ast.fix_missing_locations(tree)
        env=dict(r.__dict__);exec(compile(tree,'original_module_stage_without_discarded_powell','exec'),env)
        env['case'](seed,arch)
    p.HERE=HERE;p.OUT=OUT/'kla_phase';p.OUT.mkdir(parents=True,exist_ok=True)
    # The old PhaseControl expects results/kla/<proposal>. Keep a read-only
    # source mapping in its function globals by linking the same saved record.
    stagepath=OUT/'kla'/f'{arch}_{seed}.json';save(stagepath,json.loads(dest.read_text()))
    def calibration(self):
        with np.load(OLD/'results/kla_phase/calibration.npz') as d:self.reference=d['reference'];self.jacobian=d['jacobian']
    p.PhaseControl.calibration=calibration
    j.HERE=HERE;j.BASE=p.OUT;j.OUT=OUT/'kla_final';j.OUT.mkdir(parents=True,exist_ok=True)
    j.run(seed,arch)

def kla_slor(seed):
    import restore_kla as k
    from kla_coupled_phase import PhaseControl
    path=OUT/'slor_kla'/f'slor_{seed}.json'
    if path.exists():return
    start=time.perf_counter();c=PhaseControl();e=c.engine
    definition=next(d for d in k.KLA_CASE_DEFINITIONS if d['seed']==seed)
    initial=k.make_kla_perturbations(e,definition);state=copy.deepcopy(initial)
    np.random.seed(seed*1000+60906)
    measurements=[e.get_field_opd_map(f,initial,grid_size=64,add_noise=True) for f in e.field_points_y]
    if any(m is None for m in measurements):raise RuntimeError('Missing input pupil')
    obs=np.asarray([m['opd_map'] for m in measurements],np.float32)
    parameters=0;predictions={}
    for module in (2,3,4):
        model,ckpt,source=network(f'kla_module{module}','slor');parameters+=sum(p.numel() for p in model.parameters())
        with torch.inference_mode():pred=model(torch.from_numpy(obs).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
        pred=pred*np.asarray(ckpt['target_std'])+np.asarray(ckpt['target_mean'])
        for i,lens in enumerate(e.module_lenses[module]):
            state[lens]={key:float(initial[lens][key]-pred[i*5+d]) for d,key in enumerate(k.PARAMS)}
        predictions[str(module)]={'checkpoint':str(source),'prediction':pred.tolist()}
    row={'system':'KLA','method':'Slor standalone','seed':seed,'parameters':parameters,
         'initial_state':initial,'terminal_state':state,'terminal':c.terminal(state,seed),'predictions':predictions,
         'network_calls':3,'network_observation_stacks':1,'applied_corrections':1,'gain_searches':0,
         'refinement_iterations':0,'elapsed_seconds':time.perf_counter()-start}
    save(path,row);print('SLOR_KLA_COMPLETE',seed,row['terminal'],flush=True)

def grid_check():
    import restore_trepan as t
    old=json.loads((OLD/'results/trepan/arch_a_136.json').read_text())
    r=t.OpticalRuntime(t.TRAINING_GRID,'speckle');rows=[]
    for label in ('initial_state','unit_state','acquired_state','terminal_state'):
        if label not in old:continue
        state={int(a):t.SurfacePerturbation(**v) for a,v in old[label].items()}
        for grid in (16,32,64):
            start=time.perf_counter();e=r.center_engine;e.set_surface_perturbations(state,clear_others=True)
            _,_,opd,valid=e._sample_wavefront(num_rays=grid,field_index=0,wavelength_nm=e.wavelength_nm,focus=4.647)
            rows.append({'state':label,'grid':grid,'center_wrms':r.compute_true_wrms(opd,valid),
                         'seconds':time.perf_counter()-start,'valid_rays':int(valid.sum())})
    save(OUT/'candidate_grid_check.json',rows);print(json.dumps(rows,indent=2),flush=True)

def render_frozen(arch):
    import restore_trepan as t
    path=OUT/f'trepan_grid32/{arch}_136.json';record=json.loads(path.read_text())
    runtime=t.OpticalRuntime(t.TRAINING_GRID,'speckle');arrays={}
    for name,key in (('initial','initial_state'),('final','terminal_state')):
        state={int(a):t.SurfacePerturbation(**v) for a,v in record[key].items()}
        phases=[];masks=[]
        for index in (0,1):
            e=runtime.wide_engines[index];e.set_surface_perturbations(state,clear_others=True)
            _,_,opd,valid=e._sample_wavefront(num_rays=64,field_index=0,wavelength_nm=e.wavelength_nm,focus=4.647)
            yy,xx=np.mgrid[-1:1:64j,-1:1:64j];mask=valid&(np.hypot(xx,yy)<=1.)
            matrix=np.column_stack((xx[mask],yy[mask],(xx*xx+yy*yy)[mask],np.ones(mask.sum())))
            coeff=np.linalg.lstsq(matrix,opd[mask],rcond=None)[0]
            phase=np.zeros_like(opd);phase[mask]=(opd[mask]-matrix@coeff)/runtime.wavelength_system
            target=record['stages']['initial' if name=='initial' else 'joint_40']['field_wrms'][index]
            assert abs(np.sqrt(np.mean(phase[mask]**2))-target)<1e-7
            phases.append(phase);masks.append(mask)
        arrays[name+'_residual_waves']=np.asarray(phases);arrays[name+'_masks']=np.asarray(masks)
    np.savez_compressed(OUT/f'fig8_{arch}_136.npz',**arrays)
    print('FROZEN_FIG8_FIELDS_COMPLETE',arch,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('system',choices=['trepan','trepan_slor','kla','kla_slor','grid_check','render_frozen'])
    p.add_argument('--seed',type=int,default=136);p.add_argument('--arch',default='arch_a');p.add_argument('--grid',type=int,default=32)
    a=p.parse_args()
    if a.system=='grid_check':grid_check()
    elif a.system=='render_frozen':render_frozen(a.arch)
    elif a.system=='trepan':trepan(a.seed,a.arch,a.grid)
    elif a.system=='kla':kla(a.seed,a.arch)
    else:globals()[a.system](a.seed)
