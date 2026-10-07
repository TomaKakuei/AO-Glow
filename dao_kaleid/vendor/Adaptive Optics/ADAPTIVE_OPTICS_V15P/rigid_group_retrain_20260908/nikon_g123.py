"""Independent datasets for the patent's three optical groups."""
import json
import os
from pathlib import Path
import numpy as np
from common import HERE

GROUP_INDICES = {'G1':(0,1,2), 'G2':(3,4,5), 'G3':(6,)}
TARGET_SLICES = {'G1':slice(0,15), 'G2':slice(15,30), 'G3':slice(30,35)}


def config():
    return json.loads((HERE/'nikon_g123_protocol.json').read_text(encoding='utf-8'))


def sample(group, index, mode):
    cfg = config()
    seed = cfg['seed_bases'][group]+int(index)
    rng = np.random.RandomState(seed)
    q = np.zeros(35,dtype=np.float64)
    selected = list(GROUP_INDICES[group])
    if mode == 'joint_context':
        others = [b for b in range(7) if b not in selected]
        selected += list(rng.choice(others,size=1+index%len(others),replace=False))
    elif mode != 'isolated':
        raise ValueError(mode)
    for body in selected:
        q[5*body:5*body+5] = rng.uniform(-1.,1.,5)
    # All bodies of the target optical group receive independent five-axis
    # states; they do not share one rigid transform.
    return q,seed


def generate_batch(group,batch_index,batch_size,directory,mode):
    from nikon35 import Plant,grid
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'batch_{batch_index:04d}.npz'
    if path.exists():
        with np.load(path) as d:
            assert str(d['group']) == group and str(d['sampling_mode']) == mode
            assert d['targets'].shape == (batch_size,35)
            assert d['maps'].shape == (batch_size,3,32,32)
        return path
    maps=[];raws=[];targets=[];seeds=[]
    from sampling_backend import nikon_sampler
    sampler=nikon_sampler()
    for local in range(batch_size):
        q,seed=sample(group,batch_index*batch_size+local,mode)
        if sampler is None:
            plant=Plant(hidden=q,noise_seed=seed,noise_std=.015)
            raw=plant.measure(np.zeros(35))
        else:
            raw=sampler.measure(q,seed)
        if raw is None:
            raise RuntimeError(f'{group} trace failed, seed={seed}')
        maps.append(grid(raw)[0]);raws.append(np.asarray(raw,np.float32))
        targets.append(q.astype(np.float32));seeds.append(seed)
    temporary=path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,maps=np.stack(maps),raw=np.stack(raws),targets=np.stack(targets),
                            group_targets=np.stack(targets)[:,TARGET_SLICES[group]],
                            seeds=np.asarray(seeds,np.int64),group=group,sampling_mode=mode)
    os.replace(temporary,path)
    return path
