"""Accelerated feedback and endpoint evaluation with the original reference-sphere metric."""
from contextlib import contextmanager
import numpy as np
from fast_raytrace import trepan_backend,nikon_backend,trace_batch


@contextmanager
def nikon_execution():
    from nikon35 import base
    absolute=base.absolute;corrected=base.corrected
    original=absolute.reference_sphere_rms
    original_trace=absolute.trace_explicit
    def evaluate(opm,effective_q,field_mm,nodes,*,focus_offset_mm=0.):
        corrected.apply_normalized_state(opm,np.asarray(effective_q,float))
        directions=[[0.,0.,1.]]
        for px,py in np.asarray(nodes,float):
            ux=corrected.SAMPLE_SIDE_NA*float(px)/corrected.SAMPLE_INDEX
            uy=corrected.SAMPLE_SIDE_NA*float(py)/corrected.SAMPLE_INDEX
            directions.append([ux,uy,np.sqrt(1-ux*ux-uy*uy)])
        points=np.tile([float(field_mm),0.,0.],(len(directions),1))
        packages=trace_batch(opm.seq_model,points,directions,corrected.WAVELENGTH_MM*1e6)
        if any(p is None for p in packages):
            return original(opm,effective_q,field_mm,nodes,focus_offset_mm=focus_offset_mm)
        iterator=iter(packages)
        # Keep the original chief-ray sphere, backtracking and PTTD projection.
        absolute.trace_explicit=lambda *_args,**_kwargs:next(iterator)
        try:
            return original(opm,effective_q,field_mm,nodes,focus_offset_mm=focus_offset_mm)
        finally:
            absolute.trace_explicit=original_trace
    absolute.reference_sphere_rms=evaluate
    try:
        with nikon_backend():yield
    finally:
        absolute.reference_sphere_rms=original
        absolute.trace_explicit=original_trace


def execution(system):
    return trepan_backend() if system=='trepan' else nikon_execution()
