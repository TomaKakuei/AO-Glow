"""Noise-enabled optical plants and resumable streaming sample generation."""
import contextlib
import json
from pathlib import Path
import numpy as np
from .paths import activate,BASE,DAO,legacy_bridge

def mod3_camera(device=None):
    activate('mod3')
    from camera import RelayCamera
    camera=RelayCamera(device=device)
    original=camera.capture
    def capture(q,noise_seed,grid=2048,verify=True):
        # Same state, seed and threshold; bounded phase-fit refinement as in the
        # audited continuation. Never discard a hard state because of fit error.
        initial=camera.degree
        try:
            for degree in (24,28,32,36):
                camera.degree=degree
                try:return original(q,noise_seed,grid=grid,verify=verify)
                except RuntimeError as error:
                    if 'fit' not in str(error).lower() and 'continuous' not in str(error).lower():raise
                    if degree==36:raise
        finally:camera.degree=initial
    camera.capture=capture
    return camera

def execution(system):
    activate(system)
    if system in ('nikon','trepan'):
        from fast_execution import execution as accelerated
        return accelerated(system)
    return contextlib.nullcontext()

def generate(system,branch,count,output,start_index=0,batch_size=25,device=None):
    """Writes bounded batches, and continues only missing files on restart."""
    activate(system)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if count<1 or batch_size<1 or start_index<0:raise ValueError('Invalid sample range')
    camera=mod3_camera(device) if system=='mod3' else None
    if system=='nikon':
        from nikon_g123 import sample, TARGET_SLICES
        from nikon35 import Plant,grid
    elif system=='trepan':
        import fast_trepan_mainline
    elif system=='kla':
        from kla_full_fov_engine import KLAFullFOVEngine
        engine=KLAFullFOVEngine()
        module=int(str(branch).replace('module',''))
        active=engine.module_lenses[module]
    elif system=='mod3':
        from settings import isolated_candidate
    else:raise ValueError(system)
    manifest={'system':system,'branch':branch,'count':count,'start_index':start_index,'batch_size':batch_size,'noise_enabled':True,'records':[]}
    descriptor=output/'generation.json'
    if descriptor.exists():
        previous=json.loads(descriptor.read_text(encoding='utf-8'))
        for key in ('system','branch','count','start_index','batch_size'):
            if previous[key]!=manifest[key]:raise RuntimeError(f'Output already belongs to a different {key}')
    with execution(system):
        for offset in range(0,count,batch_size):
            n=min(batch_size,count-offset);index=start_index+offset
            path=output/f'samples_{index:08d}_{n:04d}.npz'
            if path.exists():
                with np.load(path) as saved:
                    if len(saved['sample_indices'])!=n:raise RuntimeError('Incomplete existing batch')
            elif system=='trepan':
                # Existing fast sampler already writes its own atomic batches.
                # Use a full-size namespace so changing n cannot reuse another file.
                folder=output/f'legacy_size{n}_start{index}'
                p=fast_trepan_mainline.generate(branch,index,n,folder)
                with np.load(p) as z:data={k:z[k] for k in z.files}
                frames=data['speckles']
                if np.all((frames>=0)&(frames<=65535)) and np.array_equal(frames,np.rint(frames)):
                    data['speckles']=frames.astype(np.uint16)
                data['sample_indices']=np.arange(index,index+n)
                atomic_npz(path,**data)
                # The delegated file is an intermediate, not a second dataset.
                intermediate=Path(p).resolve()
                intermediate.relative_to(output)
                intermediate.unlink()
                if not any(folder.iterdir()):folder.rmdir()
            else:
                rows=[]
                for i in range(index,index+n):
                    if system=='mod3':
                        for attempt in range(2000):
                            q,meta=isolated_candidate(branch,i,attempt)
                            if not camera.engine.geometry.path(q)['safe']:continue
                            try:image,render=camera.capture(q,(meta['seed']+97181)%2**32)
                            except ValueError:continue
                            break
                        else:raise RuntimeError('Cannot fill assigned amplitude stratum')
                        sl=slice(0,15) if branch=='front' else slice(15,35)
                        rows.append({'speckles':image,'targets_normalized':q,'group_targets_normalized':q[sl],'seeds':meta['seed'],'amplitude_bin':i%5})
                    elif system=='nikon':
                        q,seed=sample(branch,i,'joint_context')
                        plant=Plant(hidden=q,noise_seed=seed,noise_std=.015)
                        raw=plant.measure(np.zeros(35))
                        if raw is None:raise RuntimeError(f'Nikon trace failed at seed {seed}')
                        rows.append({'raw':raw,'maps':grid(raw)[0],'targets':q,'group_targets':q[TARGET_SLICES[branch]],'seeds':seed})
                    else:
                        seed=202610060+module*1000000+i
                        rng=np.random.default_rng(seed)
                        scales=np.array([.12,.12,.12,np.deg2rad(3.5/60),np.deg2rad(3.5/60)])
                        state={lens:dict(zip(('dx','dy','dz','tx','ty'),rng.uniform(-1,1,5)*scales if lens in active else np.zeros(5))) for lenses in engine.module_lenses.values() for lens in lenses}
                        old=np.random.get_state();np.random.seed(seed%2**32)
                        try:measured=[engine.get_field_opd_map(f,state,grid_size=64,add_noise=True) for f in engine.field_points_y]
                        finally:np.random.set_state(old)
                        if any(v is None for v in measured):raise RuntimeError(f'KLA missing pupil at seed {seed}; sample retained as failure, not silently reduced')
                        targets=np.array([[state[l][k] for k in ('dx','dy','dz','tx','ty')] for l in active]).ravel()
                        rows.append({'maps':np.array([v['opd_map'] for v in measured],np.float32),'targets':targets,'seeds':seed})
                data={key:np.array([row[key] for row in rows]) for key in rows[0]}
                if system=='nikon':data.update(group=branch,sampling_mode='joint_context')
                atomic_npz(path,sample_indices=np.arange(index,index+n),**data)
            from .paths import sha
            manifest['records'].append({'file':path.name,'sha256':sha(path),'count':n})
            atomic_json(descriptor,manifest)
            print(f'SAMPLED {system}/{branch} {offset+n}/{count}',flush=True)
    return manifest

def atomic_npz(path,**arrays):
    temp=path.with_suffix('.partial')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def atomic_json(path,value):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n',encoding='utf-8')
    temp.replace(path)
