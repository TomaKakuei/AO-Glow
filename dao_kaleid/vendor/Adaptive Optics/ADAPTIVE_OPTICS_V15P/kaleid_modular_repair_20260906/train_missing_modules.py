"""Fit only missing modular Arch-A networks, using archived arrays only."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CANON=ROOT/'canonical_three_phase_universal'
sys.path.insert(0,str(CANON))
from models import build_universal_model
OUT=HERE/'checkpoints'
RESULTS=HERE/'results'
for folder in (OUT,RESULTS):folder.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_data(name):
    if name=='trepan_rear':
        paths=sorted((ROOT/'artifacts/rear_speckle_dataset').glob('batch_*.npz'))[:50]
        if len(paths)!=50:raise RuntimeError('Expected 50 existing rear batches')
        x=np.empty((5000,9,128,128),dtype=np.float32);y=np.empty((5000,20),dtype=np.float32)
        for index,path in enumerate(paths):
            with np.load(path) as d:
                x[index*100:(index+1)*100]=d['speckles'];y[index*100:(index+1)*100]=d['mechs']
        return x,y,paths
    path=ROOT.parent/'artifacts'/f'dataset_module{name[-1]}_5dof.npz'
    with np.load(path) as d:x=d['opd_maps'].astype(np.float32);y=d['labels'].astype(np.float32)
    assert len(x)==5000
    return x,y,[path]


def train(name):
    dest=OUT/f'{name}_arch_a.pt'
    if dest.exists():print('REUSE_COMPLETED',name,flush=True);return
    start=time.perf_counter();torch.manual_seed(20260907);np.random.seed(20260907)
    x,y,paths=load_data(name);split=int(.8*len(y));groups=[5]*(y.shape[1]//5)
    mean=y[:split].mean(axis=0);std=y[:split].std(axis=0);std[std<1e-6]=1.
    target=(y-mean)/std
    model=build_universal_model('arch_a',9,groups)
    source=CANON/'checkpoints'/('trepan2p_arch_a.pt' if name=='trepan_rear' else 'kla_arch_a.pt')
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    encoder={k:v for k,v in checkpoint['state_dict'].items() if not k.startswith('head.')}
    model.load_state_dict(encoder,strict=False);model.to(DEVICE)
    batch=8 if name=='trepan_rear' else 32
    train_dl=DataLoader(TensorDataset(torch.from_numpy(x[:split]),torch.from_numpy(target[:split])),batch_size=batch,shuffle=True)
    val_dl=DataLoader(TensorDataset(torch.from_numpy(x[split:]),torch.from_numpy(target[split:])),batch_size=batch)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,25);criterion=nn.HuberLoss()
    best=float('inf');history=[]
    for epoch in range(1,26):
        model.train();total=0.
        for inp,label in train_dl:
            inp,label=inp.to(DEVICE),label.to(DEVICE);opt.zero_grad(set_to_none=True)
            loss=criterion(model(inp),label);loss.backward();opt.step();total+=float(loss.detach())*len(label)
        model.eval();val=0.;squared=0.
        with torch.no_grad():
            for inp,label in val_dl:
                label=label.to(DEVICE);pred=model(inp.to(DEVICE));val+=float(criterion(pred,label))*len(label)
                squared+=float(((pred-label)**2).sum())
        val/=len(val_dl.dataset);scheduler.step()
        history.append({'epoch':epoch,'train_huber':total/split,'validation_huber':val,
                        'validation_normalized_rmse':float(np.sqrt(squared/(len(val_dl.dataset)*y.shape[1]))),
                        'elapsed_seconds':time.perf_counter()-start})
        if val<best:
            best=val
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'target_mean':mean,'target_std':std,'group_dofs':groups,'arch_name':'arch_a',
                        'source_encoder':str(source),'selected_epoch':epoch,'best_val_loss':best,
                        'input_preprocessing':'raw archived maps, as existing compact controller',
                        'states':len(y),'new_generated_states':0},OUT/f'{name}_best_progress.pt')
        save(RESULTS/f'training_{name}.json',{'status':'running','name':name,'history':history,'sources':[str(p) for p in paths],
             'batch_size':batch,'epochs':25,'split':split,'new_generated_states':0})
        print(f'EPOCH {name} {epoch}/25 val={val:.6f} elapsed={time.perf_counter()-start:.1f}',flush=True)
    (OUT/f'{name}_best_progress.pt').replace(dest)
    save(RESULTS/f'training_{name}.json',{'status':'complete','name':name,'history':history,'sources':[str(p) for p in paths],
         'batch_size':batch,'epochs':25,'split':split,'new_generated_states':0,'best_val_loss':best,
         'parameters':sum(p.numel() for p in model.parameters()),'elapsed_seconds':time.perf_counter()-start})
    print('TRAINING_COMPLETE',name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--module',choices=['all','trepan_rear','kla_module3','kla_module4'],default='all')
    name=parser.parse_args().module
    for item in (('kla_module3','kla_module4','trepan_rear') if name=='all' else (name,)):train(item)
    print('ALL_MISSING_MODULES_COMPLETE',flush=True)
