"""New mod03 prescription adapter using the existing fast tracer and camera functions."""
from pathlib import Path
import ast
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bootstrap import setup, ARCHIVE, DAO
setup()
import numpy as np
import scipy.fft as sfft
from mod03_adapter import Mod03Adapter, TracePath
from mechanics import Geometry
import inspect
import fast_raytrace


def endpoint_kernel():
    # Keep the existing numerical body verbatim; replace only final RayPkg/list
    # construction, since sampling consumes exit points/directions/OPL/masks.
    tree=ast.parse(inspect.getsource(fast_raytrace.trace_batch))
    node=tree.body[0];node.name='trace_endpoints'
    assert isinstance(node.body[-1],ast.Return)
    node.body[-1]=ast.parse('return dict(p=segments[-2][0], d=segments[-2][1], opl=opl, valid=valid)').body[0]
    ast.fix_missing_locations(tree)
    namespace=dict(fast_raytrace.__dict__)
    exec(compile(tree,'existing_fast_raytrace_endpoint_output','exec'),namespace)
    return namespace['trace_endpoints']


TRACE_ENDPOINTS=endpoint_kernel()


def legacy_camera_functions():
    # Load these exact pure functions without importing the old dataset module,
    # whose module-level code changes environment variables and creates assets.
    source=DAO/'generate_speckle_dataset.py'
    names={'fresnel_propagate','apply_minimal_scmos_noise'}
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(selected)!=2:raise RuntimeError('Legacy camera functions not found')
    namespace={'np':np,'sfft':sfft}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),namespace)
    return namespace['fresnel_propagate'],namespace['apply_minimal_scmos_noise']


class SamplingOptics(Mod03Adapter):
    def __init__(self):
        self.geometry=Geometry(ARCHIVE/'US07199938-1-mod03.zmx')
        super().__init__(grid_size=64)
        self.propagate,self.camera_noise=legacy_camera_functions()
        self.diffuser=np.load(DAO/'artifacts/diffuser_mask.npy')
        if self.diffuser.shape!=(128,128):raise ValueError('Legacy diffuser shape changed')
        self.diffuser_factor=np.exp(1j*self.diffuser)

    def poses(self,q):
        return self.geometry.poses(q)

    def _trace(self,q,field,pupils=None,scalar=False):
        if scalar:return super()._trace(q,field,pupils=pupils,scalar=True)
        points,directions=self.launch(field,pupils)
        result=TRACE_ENDPOINTS(TracePath(self.path(q)),points,directions,self.wave,
                              check_apertures=True,first_surf=0,last_surf=22,
                              intersect_obj=False,pt_inside_fuzz=1e-8)
        if not isinstance(result,dict):raise RuntimeError('Endpoint kernel requires spherical mod03 surfaces')
        p,d=result['p'],result['d'];valid=result['valid']
        opl=(result['opl']+np.sum(points*directions,axis=1))/self.n_image
        return dict(p=p[1:],d=d[1:],opl=opl[1:],valid=valid[1:],
                    chief_valid=bool(valid[0]),chief_point=p[0],chief_opl=opl[0])

    def sample(self,q,noise_seed):
        observation=self.observe(q)
        survival=observation['ray_survival_fraction']
        if min(survival)<.2:raise ValueError('Less than 20% of pupil survives in a field')
        state=np.random.get_state()
        np.random.seed(int(noise_seed))
        try:
            images=[]
            for phase,valid in zip(observation['opd_waves'],observation['valid']):
                pupil=np.zeros((128,128),complex)
                pupil[32:96,32:96]=valid*np.exp(2j*np.pi*phase)
                field=self.propagate(pupil*self.diffuser_factor,dx=9./64,z=2.,wavelength=self.wave*1e-6)
                image=self.camera_noise(abs(field)**2)
                # The legacy camera already rounds ADUs; uint16 is lossless.
                images.append(image.astype(np.uint16))
        finally:
            np.random.set_state(state)
        return dict(speckles=np.stack(images),center_opd_waves=observation['opd_waves'][0].astype(np.float32),
                    valid=observation['valid'],wrms_waves=observation['wrms_waves'],
                    ray_survival=survival,centroid_mm=observation['centroid_mm'])
