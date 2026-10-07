"""Headless server entry: inspect, generate, infer, feedback, and exact legacy replay."""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from .paths import PACKAGE,DAO,registry,activate,legacy_bridge,sha

SYSTEMS=('mod3','trepan','kla','nikon')

def doctor(system=None,device=None):
    import importlib.metadata as md
    import torch
    r=registry();checks=[]
    systems=(system,) if system else SYSTEMS
    seen=set()
    for name in systems:
        for profile,entry in r['systems'][name].items():
            for branch,row in entry['checkpoints'].items():
                if row['path'] in seen:continue
                seen.add(row['path']);path=PACKAGE/row['path']
                if not path.is_file() or sha(path)!=row['sha256']:raise RuntimeError(f'Broken checkpoint {name}/{profile}/{branch}')
                checks.append({'system':name,'profile':profile,'branch':branch,'bytes':row['bytes']})
    versions={}
    for name in ('torch','numpy','scipy','rayoptics','opticalglass'):
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]='NOT_INSTALLED'
    out={'python':sys.version.split()[0],'packages':versions,'cuda_available':torch.cuda.is_available(),
         'device':device or ('cuda' if torch.cuda.is_available() else 'cpu'),
         'checkpoint_files_checked':len(checks),'weight_bytes':sum(v['bytes'] for v in checks),'profiles':{k:r['systems'][k] for k in systems}}
    if system:
        activate(system)
        from .predictors import Predictor
        if system in ('mod3','trepan'):branch='front'
        elif system=='kla':branch='module2'
        else:branch=None
        model=Predictor(system,branch,profile='latest',device=device)
        import numpy as np
        shape=(3,49) if system=='nikon' else ((9,64,64) if system=='kla' else (9,128,128))
        # A shape/interface check only; synthetic zeros are never a recovery claim.
        value=model(np.zeros(shape,np.float32))
        out['inference_shape_check']={'input':shape,'output':value.shape,'finite':bool(np.isfinite(value).all())}
        if system=='kla':
            from restore_kla import KLA_CASE_DEFINITIONS
            out['original_case_seeds']=[x['seed'] for x in KLA_CASE_DEFINITIONS]
    return out

def published_replay(system,seed,output,device=None):
    """Exact original controller paths. Trepan retains its historical exact score."""
    activate(system);legacy_bridge(device)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if system=='trepan':
        import run_trepan_limit20 as t
        t.OUT=output
        if device:
            import torch
            t.base.DEVICE=torch.device(device)
        return t.run(('arch_a','L4',seed))
    if system=='nikon':
        from .feedback import nikon
        return nikon(seed,profile='closed_loop',device=device)
    if system=='kla':
        folder=DAO/'kaleid_minimal_v21'
        sys.path.insert(0,str(folder))
        import run_optical as r
        # Original phase routines address HERE/results/kla. Preserve that
        # relative contract inside the caller's output directory.
        r.HERE=output/'kla_runtime'
        r.OUT=r.HERE/'results'
        r.OUT.mkdir(parents=True,exist_ok=True)
        if device:
            import torch
            r.DEVICE=torch.device(device)
        # These are the exact original module/phase/local-response/MTF stages.
        r.kla(seed,'arch_a')
        return json.loads((r.OUT/'kla_final'/f'arch_a_{seed}.json').read_text(encoding='utf-8'))
    if system=='mod3':
        from .feedback import mod3
        return mod3(seed,profile='closed_loop',device=device,score_grid=256)
    raise ValueError(system)

def parser():
    p=argparse.ArgumentParser(description='Portable DAO/Kaleid: noisy simulation, generation, inference and feedback')
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor');d.add_argument('--system',choices=SYSTEMS);d.add_argument('--device',choices=('cpu','cuda'))
    g=sub.add_parser('generate');g.add_argument('--system',choices=SYSTEMS,required=True);g.add_argument('--branch',required=True)
    g.add_argument('--count',type=int,required=True);g.add_argument('--output',type=Path,required=True)
    g.add_argument('--start-index',type=int,default=0);g.add_argument('--batch-size',type=int,default=25);g.add_argument('--device',choices=('cpu','cuda'))
    i=sub.add_parser('predict');i.add_argument('--system',choices=SYSTEMS,required=True);i.add_argument('--branch')
    i.add_argument('--profile',default='closed_loop');i.add_argument('--input',type=Path,required=True);i.add_argument('--key',default='observation')
    i.add_argument('--output',type=Path,required=True);i.add_argument('--device',choices=('cpu','cuda'))
    f=sub.add_parser('feedback');f.add_argument('--system',choices=SYSTEMS,required=True);f.add_argument('--profile',default='closed_loop')
    f.add_argument('--seed',type=int);f.add_argument('--budget',type=int,default=80);f.add_argument('--module-budget',type=int,default=20)
    f.add_argument('--score-grid',type=int,default=64);f.add_argument('--output',type=Path,required=True);f.add_argument('--device',choices=('cpu','cuda'))
    l=sub.add_parser('replay-published');l.add_argument('--system',choices=SYSTEMS,required=True);l.add_argument('--seed',type=int)
    l.add_argument('--output',type=Path,required=True);l.add_argument('--device',choices=('cpu','cuda'))
    s=sub.add_parser('serve');s.add_argument('--system',choices=SYSTEMS,required=True);s.add_argument('--profile',default='closed_loop')
    s.add_argument('--host',default='127.0.0.1');s.add_argument('--port',type=int,default=8000);s.add_argument('--device',choices=('cpu','cuda'))
    return p

def main(argv=None):
    a=parser().parse_args(argv);start=time.perf_counter()
    if a.command=='doctor':result=doctor(a.system,a.device)
    elif a.command=='generate':
        from .simulation import generate
        result=generate(a.system,a.branch,a.count,a.output,a.start_index,a.batch_size,a.device)
    elif a.command=='predict':
        import numpy as np
        from .predictors import Predictor
        from .simulation import atomic_json
        with np.load(a.input,allow_pickle=False) as z:x=z[a.key]
        model=Predictor(a.system,a.branch,a.profile,a.device)
        estimate=model(x);command=model.command(x)
        result={'system':a.system,'branch':a.branch,'profile':a.profile,'predicted_pose':estimate,'command':command,
            'units':'normalized 100um/5arcmin body coordinates' if a.system=='mod3' else ('normalized Nikon .02mm/.04degree coordinates' if a.system=='nikon' else 'checkpoint original physical units')}
        a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,result)
    elif a.command in ('feedback','replay-published'):
        defaults={'mod3':4101000,'trepan':136,'nikon':4,'kla':156};seed=a.seed if a.seed is not None else defaults[a.system]
        if a.command=='feedback':
            from . import feedback
            kwargs=dict(seed=seed,profile=a.profile,budget=a.budget,device=a.device)
            if a.system in ('mod3','trepan'):kwargs['module_budget']=a.module_budget
            if a.system=='mod3':kwargs['score_grid']=a.score_grid
            result=getattr(feedback,a.system)(**kwargs)
        else:result=published_replay(a.system,seed,a.output.parent/'replay_records',a.device)
        from .simulation import atomic_json
        result['elapsed_seconds']=time.perf_counter()-start
        a.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(a.output,result)
    elif a.command=='serve':
        from .service import serve
        return serve(a.system,a.profile,a.device,a.host,a.port)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
    return 0
