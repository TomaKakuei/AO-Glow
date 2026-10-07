from pathlib import Path
import sys,os,json,time,hashlib,ctypes
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(HERE))
from camera import RelayCamera
import numpy as np
import torch
sys.path.insert(0,str(HERE))
DATA=HERE/'data';RESULTS=HERE/'results';CHECKPOINTS=HERE/'checkpoints'
for folder in (DATA,RESULTS,CHECKPOINTS):folder.mkdir(parents=True,exist_ok=True)
BRANCHES={'front':tuple(range(3)),'rear':tuple(range(3,7))}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def active_seconds():
    value=ctypes.c_ulonglong()
    if os.name=='nt' and ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):return value.value/1e7
    return time.monotonic()


def status(stage,**kwargs):
    p=HERE/'state.json';value=json.loads(p.read_text()) if p.exists() else {}
    value.update(kwargs);value.update(stage=stage,updated_unix=time.time());atomic_json(p,value)


def protocol():return json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
def protocol_sha():return sha(HERE/'protocol.json')


def gram_factor(matrix):
    val,vec=np.linalg.eigh(matrix.T@matrix)
    return (np.sqrt(np.maximum(val,0))[:,None]*vec.T).astype(np.float32)


def isolated_candidate(branch,index,attempt=0):
    from distribution import candidate
    q,meta=candidate(branch,400000+index,attempt)
    columns=np.array([g*5+d for g in BRANCHES[branch] for d in range(5)])
    other=np.setdiff1d(np.arange(35),columns)
    q[other]=0.
    meta.update(context=False,active_mask=np.any(q.reshape(7,5)!=0,axis=1).astype(np.uint8),
                deployment='isolated_module_with_nominal_reference_partner')
    assert np.all(q[other]==0.)
    return q,meta
