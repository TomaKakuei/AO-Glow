"""Speckle-only sampling using the validated batch path and original state/noise seeds."""
import os
import time
from pathlib import Path
import numpy as np
from common import ROOT
from fast_raytrace import trepan_backend

_ENGINES = None


def generate(branch, batch_index, size, destination):
    global _ENGINES
    os.chdir(ROOT)
    import generate_speckle_dataset as g
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    path=destination/f'batch_{batch_index:04d}.npz'
    if path.exists():return str(path)
    # Match the original nominal camera reference exactly; reuse it across batches.
    if _ENGINES is None:
        _ENGINES=[]
        for x,y in g._pov_points():
            engine=g._build_model();g._set_field_point(engine,x,y)
            engine.get_wavefront_opd(focus=g.REFERENCE_FOCUS_MM)
            _ENGINES.append(engine)
    np.random.seed(920000+batch_index+(0 if branch=='front' else 100000))
    rng=np.random.default_rng(42+batch_index)
    anchors=sorted(_ENGINES[0].group_anchor_to_members)
    active=[68,70,72,74] if branch=='front' else [77,79,81,84]
    wavelength=_ENGINES[0].wavelength_nm*1e-6
    dx=_ENGINES[0].pupil_diameter_mm/64
    mask=np.load(g.MASK_PATH)
    images=[];targets=[];centers=[];wrms=[];attempts=0
    while len(images)<size:
        attempts+=1
        if attempts>size*3:raise RuntimeError(f'Too many invalid optical samples in {branch}/{batch_index}')
        state=g._sample_state(rng,anchors,branch)
        stack=[];center=None
        try:
            with trepan_backend():
                for (x,y),engine in zip(g._pov_points(),_ENGINES):
                    engine.set_surface_perturbations(state,clear_others=True)
                    opd=np.asarray(engine.get_wavefront_opd(focus=g.REFERENCE_FOCUS_MM),np.float64)
                    if not np.isfinite(opd).all() or not np.any(opd!=0):
                        raise ValueError('Invalid camera OPD')
                    if x==0 and y==0:center=opd.copy()
                    pupil=np.zeros((128,128),complex)
                    pupil[32:96,32:96]=(opd!=0)*np.exp(2j*np.pi/wavelength*opd)
                    sensor=g.fresnel_propagate(pupil*np.exp(1j*mask),dx=dx,z=2.,wavelength=wavelength)
                    stack.append(g.apply_minimal_scmos_noise(abs(sensor)**2))
        except Exception as exc:
            print(f'FAST_TREPAN_REJECT batch={batch_index} attempt={attempts}: {exc!r}',flush=True)
            continue
        images.append(np.stack(stack));centers.append(center.astype(np.float32))
        targets.append(np.array([[getattr(state[a],k) for k in ('dx_mm','dy_mm','dz_mm','tilt_x_deg','tilt_y_deg')]
                                 for a in active],np.float32).ravel())
        wrms.append(np.std(center[center!=0])/wavelength)
        print(f'FAST_TREPAN batch={batch_index} branch={branch} case={len(images)}/{size}',flush=True)
    temporary=path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,speckles=np.stack(images),mechs=np.stack(targets),opds=np.stack(centers),
            wrms=np.asarray(wrms,np.float32),backend='batch_current_rayoptics_path',
            observation='speckle_mainline',auxiliary_shwfs_generated=False,attempts=attempts)
    os.replace(temporary,path)
    print(f'FAST_TREPAN_SAVED {path}',flush=True)
    return str(path)
