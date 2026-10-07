"""Verify portable inputs/weights and forbid reads from the original workspace.

Run python scripts/verify_dao_package.py --system mod3 (one process per system).
This is functional deployment validation, not a recovery benchmark.
"""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
from pathlib import Path
import argparse
import importlib.abc
import json
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('PySide6','PyQt6','PyQt5','qtconsole','ipywidgets','poppy'):
            raise ImportError(f'Headless verification forbids {fullname}')
sys.meta_path.insert(0,NoGui())

def no_original_workspace(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=str(Path(os.fsdecode(args[0])).absolute()).replace('\\','/').lower()
    original=REPO.parent
    for folder in ('Adaptive Optics','RAYOPTICS_RESTORE_20260929'):
        if path.startswith(str(original/folder).replace('\\','/').lower()+'/'):
            raise PermissionError('Portable check forbids a read from original workspace: '+path)
sys.addaudithook(no_original_workspace)

def main():
    p=argparse.ArgumentParser();p.add_argument('--system',choices=('mod3','nikon','trepan','kla'),required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--output',type=Path)
    p.add_argument('--sample-output',type=Path);a=p.parse_args()
    import numpy as np
    from dao_kaleid.paths import PACKAGE,registry
    from dao_kaleid.predictors import Predictor
    from dao_kaleid.cli import doctor
    result=doctor(a.system,a.device);start=time.perf_counter();rows=[];seen=set()
    profiles=registry()['systems'][a.system]
    fixture=None
    for profile,record in profiles.items():
        for branch,entry in record['checkpoints'].items():
            if entry['path'] in seen:continue
            seen.add(entry['path'])
            if a.system=='mod3':fixture=PACKAGE/'fixtures'/f'mod3_{branch}_validated.npz'
            elif a.system=='nikon':fixture=PACKAGE/'fixtures/nikon_latest_validated.npz'
            else:fixture=PACKAGE/'fixtures'/f'{a.system}_noisy.npz'
            with np.load(fixture) as z:obs=z['observation'];expected=z['expected_pose'] if 'expected_pose' in z else None;tol=float(z['tolerance']) if 'tolerance' in z else None
            model=Predictor(a.system,None if a.system=='nikon' else branch,profile,a.device)
            pose=model(obs);command=model.command(obs)
            assert np.isfinite(pose).all() and np.isfinite(command).all()
            error=None
            if profile=='latest' and expected is not None:
                error=float(np.max(abs(pose-expected)))
                assert error<=tol,(a.system,branch,error,tol)
            rows.append({'profile':profile,'branch':branch,'output_size':pose.size,'fixture_max_error':error})
    result={'status':'passed','system':a.system,'device':a.device,'original_workspace_reads_forbidden':True,
            'qt_notebook_dm_imports_forbidden':True,'checkpoints_loaded':len(rows),'checks':rows,
            'elapsed_seconds':time.perf_counter()-start}
    if a.sample_output:
        from dao_kaleid.simulation import generate
        branch={'mod3':'front','nikon':'G1','trepan':'front','kla':'module2'}[a.system]
        result['generated']=generate(a.system,branch,1,a.sample_output,batch_size=1,device=a.device)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
