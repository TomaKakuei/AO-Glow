"""Reuse 3000 Nikon states; add coverage for L1 and all axial coordinates."""
from __future__ import annotations
import argparse
import copy
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

from nikon35 import HERE, ROOT, DATA_OLD, Plant, grid, sample, save_json
from models import build_universal_model
import sys
sys.path.insert(0, str(ROOT / 'minimal_ablation_completion'))
from models_and_data import SlorMLP

OUT = HERE / 'results'
CKPTS = HERE / 'checkpoints'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(2)


def dataset(count=1200):
    destination = OUT / 'training35.npz'
    if destination.exists():
        return
    started = time.perf_counter()
    plant = Plant(noise_seed=2026090601)
    raw, targets, seeds, ledger = [], [], [], []
    # Save chunks so interruption does not discard completed optical work.
    for start in range(0, count, 100):
        chunk = OUT / f'training_chunk_{start:04d}.npz'
        if chunk.exists():
            with np.load(chunk) as data:
                raw.extend(data['features']); targets.extend(data['targets']); seeds.extend(data['seeds'])
                ledger.extend(json.loads(str(data['ledger'])))
            continue
        batch_x, batch_y, batch_s, batch_l = [], [], [], []
        for index in range(start, min(count, start+100)):
            seed = 2027000 + index
            hidden, groups = sample(seed)
            # This plant has zero hidden state: a training label is the command.
            measured = plant.measure(hidden)
            batch_l.append({'seed': seed, 'valid': measured is not None, 'groups': groups})
            if measured is not None:
                batch_x.append(measured.ravel().astype(np.float32))
                batch_y.append(hidden.astype(np.float32)); batch_s.append(seed)
        np.savez_compressed(chunk, features=np.asarray(batch_x), targets=np.asarray(batch_y),
                            seeds=np.asarray(batch_s), ledger=json.dumps(batch_l))
        raw.extend(batch_x); targets.extend(batch_y); seeds.extend(batch_s); ledger.extend(batch_l)
        # The plant caches deterministic traces only during a small batch.
        plant._plant.cache.clear()
        print(f'DATA {min(count,start+100)}/{count} valid={len(raw)} elapsed={time.perf_counter()-started:.1f}s', flush=True)
    with np.load(DATA_OLD) as old:
        xo = old['features'].astype(np.float32)
        yo = np.pad(old['targets'].astype(np.float32), ((0,0),(5,0)))
    xn, yn = np.asarray(raw, np.float32), np.asarray(targets, np.float32)
    # Retain the existing first-80% / last-20% split within each source.
    old_split, new_split = int(len(xo)*.8), int(len(xn)*.8)
    train = np.r_[np.arange(old_split), len(xo)+np.arange(new_split)]
    validation = np.r_[np.arange(old_split, len(xo)), len(xo)+np.arange(new_split, len(xn))]
    x = grid(np.concatenate([xo, xn]))
    y = np.concatenate([yo, yn])
    np.savez_compressed(destination, maps=x, targets=y, train=train, validation=validation,
                        new_seeds=np.asarray(seeds), old_count=len(xo))
    save_json(OUT / 'training35.json', {'old_states': len(xo), 'new_attempts': count, 'new_valid_states': len(xn),
              'training_states': len(train), 'validation_states': len(validation), 'elapsed_seconds': time.perf_counter()-started,
              'coverage_std35': y.std(0).tolist(), 'new_generation_ledger': ledger})


def build(name):
    return SlorMLP(3, 35) if name.startswith('slor') else build_universal_model(name, 3, [5]*7)


def transfer(model, name, ym, ys):
    """Preserve the learned six-body response while adding the missing body."""
    old_path = ROOT / 'canonical_three_phase_universal/checkpoints' / f'nikon_{name}.pt'
    if name.startswith('slor'):
        old_path = ROOT / 'minimal_ablation_completion/checkpoints/nikon_slor_mlp.pt'
    old = torch.load(old_path, map_location='cpu', weights_only=False)
    state = model.state_dict()
    for key, value in old['state_dict'].items():
        if key.startswith('head.sub_heads.'):
            parts = key.split('.'); parts[2] = str(int(parts[2])+1)
            target = '.'.join(parts)
            if state[target].shape == value.shape:
                state[target] = value.clone()
        elif key in state and state[key].shape == value.shape:
            state[key] = value.clone()
    om, osd = np.asarray(old['target_mean']), np.asarray(old['target_std'])
    for g in range(6):
        scale = torch.as_tensor(osd[g*5:g*5+5]/ys[(g+1)*5:(g+2)*5], dtype=torch.float32)
        offset = torch.as_tensor((om[g*5:g*5+5]-ym[(g+1)*5:(g+2)*5])/ys[(g+1)*5:(g+2)*5], dtype=torch.float32)
        if name.startswith('slor') or name=='tolnet':
            sl = slice((g+1)*5,(g+2)*5); so = slice(g*5,g*5+5)
            prefix='output' if name.startswith('slor') else 'head.3'
            state[prefix+'.weight'][sl] = old['state_dict'][prefix+'.weight'][so] * scale[:,None]
            state[prefix+'.bias'][sl] = old['state_dict'][prefix+'.bias'][so] * scale + offset
        else:
            prefix=f'head.sub_heads.{g+1}.2'
            state[prefix+'.weight'] *= scale[:,None]
            state[prefix+'.bias'] = state[prefix+'.bias'] * scale + offset
    model.load_state_dict(state)


def train(name, epochs=35):
    CKPTS.mkdir(parents=True,exist_ok=True)
    path = CKPTS / f'nikon35_{name}.pt'
    if path.exists():
        print(f'REUSE {name}', flush=True); return
    with np.load(OUT / 'training35.npz') as data:
        x, y, ti, vi = data['maps'], data['targets'], data['train'], data['validation']
    ym, ys = y[ti].mean(0), y[ti].std(0)
    ys[ys < 1e-6] = 1
    yn = (y-ym)/ys
    torch.manual_seed(20260906)
    model = build(name)
    transfer(model, name, ym, ys)
    model.to(DEVICE)
    tx, ty = torch.from_numpy(x[ti]).to(DEVICE), torch.from_numpy(yn[ti]).to(DEVICE)
    vx, vy = torch.from_numpy(x[vi]).to(DEVICE), torch.from_numpy(yn[vi]).to(DEVICE)
    is_slor = name.startswith('slor')
    if is_slor:
        tx, vx = F.adaptive_avg_pool2d(tx, (16,16)).flatten(1), F.adaptive_avg_pool2d(vx,(16,16)).flatten(1)
        run = model.forward_features
    else:
        run = model
    batch = 250 if is_slor else 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5 if is_slor else 1e-4,
                                  weight_decay=.01 if is_slor else 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=10) if is_slor else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    def loss(a, b):
        return F.mse_loss(a,b) if is_slor else F.huber_loss(a,b)
    def validate():
        model.eval()
        with torch.inference_mode():
            pred = torch.cat([run(vx[i:i+batch]) for i in range(0,len(vx),batch)])
            return float(loss(pred,vy)), float(torch.mean((pred-vy)**2).sqrt())
    started = time.perf_counter()
    best, rmse = validate()
    weights = copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
    best_epoch = 0
    history = [{'epoch':0, 'validation_loss': best, 'normalized_pose_rmse':rmse}]
    for epoch in range(1, epochs+1):
        model.train()
        order=torch.randperm(len(tx),device=DEVICE)
        for start in range(0,len(order),batch):
            index=order[start:start+batch]
            optimizer.zero_grad(set_to_none=True)
            value=loss(run(tx[index]),ty[index]);value.backward();optimizer.step()
        val, rmse=validate()
        if is_slor:scheduler.step(val)
        else:scheduler.step()
        if val < best:
            best, best_epoch=val, epoch
            weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'epoch':epoch,'validation_loss':val,'normalized_pose_rmse':rmse})
        if epoch==1 or epoch%10==0 or epoch==epochs:
            print(f'TRAIN {name} {epoch}/{epochs} val={val:.6f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    metadata={'name':name,'epochs':epochs,'selected_epoch':best_epoch,'best_validation_loss':best,
              'parameters':sum(p.numel() for p in model.parameters()),'history':history,
              'train_states':len(ti),'validation_states':len(vi),'elapsed_seconds':time.perf_counter()-started,
              'initialization':'saved six-body model transferred to seven independent five-axis heads',
              'selection':'minimum held-out training-source validation loss; no optical test outcomes'}
    torch.save({'state_dict':weights,'target_mean':ym,'target_std':ys,'group_dofs':[5]*7,
                'name':name,'metadata':metadata},path)
    save_json(OUT / f'fit_{name}.json', metadata)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--new-states',type=int,default=1200)
    args=parser.parse_args()
    dataset(args.new_states)
    for name in ('arch_a','arch_b','arch_c','slor'):
        train(name, epochs=500 if name=='slor' else 35)
    print('TRAINING_COMPLETE', flush=True)
