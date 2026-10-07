"""Only missing networks, using archived optical arrays. No optical generation."""
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OLD=ROOT/'kaleid_modular_repair_20260906'
CANON=ROOT/'canonical_three_phase_universal'
for directory in (CANON, ROOT/'minimal_ablation_completion', OLD):
    sys.path.insert(0,str(directory))
from models import build_universal_model
from models_and_data import SlorMLP
from train_missing_modules import load_data
torch.set_num_threads(2)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT=HERE/'checkpoints'
RESULTS=HERE/'results'

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def checkpoint(module,arch):
    if module in ('trepan_front','kla_module2'):
        system='trepan2p' if module=='trepan_front' else 'kla'
        if arch=='slor':
            return ROOT/'minimal_ablation_completion/slor_standalone/checkpoints'/f'{system}_slor_mlp_epoch500.pt'
        return CANON/'checkpoints'/f'{system}_{arch}.pt'
    if arch=='arch_a':return OLD/'checkpoints'/f'{module}_arch_a.pt'
    return OUT/f'{module}_{arch}.pt'

def build(module,arch,ckpt):
    n=len(ckpt['target_mean'])
    net=SlorMLP(9,n) if arch=='slor' else build_universal_model(arch,9,[5]*(n//5))
    net.load_state_dict(ckpt['state_dict'])
    return net.to(DEVICE).eval()

def fit(module,arch,x,y,sources):
    dest=checkpoint(module,arch)
    if dest.exists():return
    start=time.perf_counter()
    torch.manual_seed(20260907);np.random.seed(20260907)
    split=int(.8*len(y));groups=[5]*(y.shape[1]//5)
    ym=y[:split].mean(0);ys=y[:split].std(0);ys[ys<1e-6]=1.
    yn=(y-ym)/ys
    is_slor=arch=='slor'
    epochs=500 if is_slor else 25
    batch=250 if is_slor else (8 if module=='trepan_rear' else 32)
    if is_slor:
        features=np.concatenate([F.adaptive_avg_pool2d(torch.from_numpy(x[i:i+100]),(16,16)).flatten(1).numpy() for i in range(0,len(x),100)])
        model=SlorMLP(9,y.shape[1])
        xm=features[:split].mean(0);xs=features[:split].std(0);xs[xs<1e-6]=1.
        model.feature_mean.copy_(torch.from_numpy(xm));model.feature_std.copy_(torch.from_numpy(xs))
        data=features
    else:
        model=build_universal_model(arch,9,groups)
        source=CANON/'checkpoints'/f'{"trepan2p" if module=="trepan_rear" else "kla"}_{arch}.pt'
        old=torch.load(source,map_location='cpu',weights_only=False)
        model.load_state_dict({k:v for k,v in old['state_dict'].items() if not k.startswith('head.')},strict=False)
        data=x
    model.to(DEVICE)
    train=DataLoader(TensorDataset(torch.from_numpy(data[:split]),torch.from_numpy(yn[:split])),batch_size=batch,shuffle=True)
    val=DataLoader(TensorDataset(torch.from_numpy(data[split:]),torch.from_numpy(yn[split:])),batch_size=batch)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5 if is_slor else 1e-4,weight_decay=.01 if is_slor else 1e-4)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.1,patience=10) if is_slor else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,25))
    loss_fn=F.mse_loss if is_slor else F.huber_loss
    forward=model.forward_features if is_slor else model
    best=float('inf');history=[];weights=None;selected=0
    for epoch in range(1,epochs+1):
        model.train()
        for inputs,labels in train:
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(forward(inputs.to(DEVICE)),labels.to(DEVICE));loss.backward();optimizer.step()
        model.eval();total=0.;squared=0.
        with torch.inference_mode():
            for inputs,labels in val:
                labels=labels.to(DEVICE);pred=forward(inputs.to(DEVICE))
                total+=float(loss_fn(pred,labels))*len(labels);squared+=float(((pred-labels)**2).sum())
        value=total/(len(y)-split)
        scheduler.step(value) if is_slor else scheduler.step()
        history.append({'epoch':epoch,'validation_loss':value,'normalized_rmse':float(np.sqrt(squared/((len(y)-split)*y.shape[1])))})
        if is_slor or value<best:
            best=value;selected=epoch;weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    meta={'module':module,'architecture':arch,'epochs':epochs,'selected_epoch':selected,
          'selection':'final epoch' if is_slor else 'minimum validation Huber loss',
          'parameters':sum(p.numel() for p in model.parameters()),'train_states':split,'validation_states':len(y)-split,
          'new_training_states':0,'elapsed_seconds':time.perf_counter()-start,'history':history,
          'sources':[str(p) for p in sources],'training_seed':20260907,
          'optimizer':'AdamW','lr':1e-5 if is_slor else 1e-4,'batch':batch,'weight_decay':.01 if is_slor else 1e-4}
    dest.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':weights,'target_mean':ym,'target_std':ys,'group_dofs':groups,'metadata':meta},dest)
    save(RESULTS/f'fit_{module}_{arch}.json',meta)
    print('FIT_COMPLETE',module,arch,round(meta['elapsed_seconds'],1),flush=True)

def main():
    for module in ('kla_module3','kla_module4','trepan_rear'):
        missing=[a for a in ('arch_b','arch_c','slor') if not checkpoint(module,a).exists()]
        if not missing:continue
        x,y,sources=load_data(module)
        for arch in missing:fit(module,arch,x,y,sources)
        del x,y
    print('ALL_MISSING_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
