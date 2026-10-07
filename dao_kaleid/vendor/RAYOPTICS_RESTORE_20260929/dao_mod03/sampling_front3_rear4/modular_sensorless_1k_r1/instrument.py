"""Simulator-side instrument; unknown poses never enter the controller module."""
from settings import *
from control import pool,MeasuredFeatures


class Instrument:
    def __init__(self,camera,hidden,seed,budget,scope):
        self.__camera=camera;self.__hidden=np.array(hidden,float).copy()
        self.__command=np.zeros(35);self.__seed=int(seed)
        self.budget=int(budget);self.measurements=0;self.invalid=0;self.scope=scope
        if scope=='front':assert np.all(self.__hidden[15:]==0.)
        if scope=='rear':assert np.all(self.__hidden[:15]==0.)

    @property
    def remaining(self):return self.budget-self.measurements

    def allowed(self,command):
        q=np.asarray(command,float)
        if q.shape!=(35,) or not np.isfinite(q).all():return False
        if self.scope=='front' and np.any(q[15:]!=0.):return False
        if self.scope=='rear' and np.any(q[:15]!=0.):return False
        state=self.__hidden+q
        return bool(abs(state).max()<=1. and self.__camera.engine.geometry.path(state,start=self.__hidden+self.__command)['safe'])

    def read(self,command):
        if self.remaining<=0:raise RuntimeError('Instrument exposure budget exceeded')
        if not self.allowed(command):self.invalid+=1;return None
        self.measurements+=1
        image,_=self.__camera.capture(self.__hidden+command,
                                     self.__seed+100003*self.measurements,
                                     grid=protocol()['camera']['production_grid'])
        self.__command=np.asarray(command).copy()
        return image

    def commit(self,command):
        if not self.allowed(command):raise RuntimeError('Collision/travel/scope interlock')
        self.__command=np.asarray(command).copy()


def measured_calibration(camera):
    path=RESULTS/'measured_response.npz'
    if path.exists():return load_features()
    with np.load(RESULTS/'calibration.npz') as z:
        reference=pool(z['reference'])
        variance=pool(z['noise_std'].astype(float)**2)/4.*(1+1/32)
    sigma=np.sqrt(variance);columns_a=[];columns_b=[];steps=[];exposures=32
    for axis in range(35):
        for step in [.04,.02,.01,.005]:
            dq=np.zeros(35);dq[axis]=step
            if camera.engine.geometry.path(dq)['safe'] and camera.engine.geometry.path(-dq)['safe']:break
        else:raise RuntimeError('No safe calibration probe for axis')
        plus=[];minus=[]
        for repeat in range(4):
            images,_=camera.capture(dq,912000+axis*20+repeat,grid=protocol()['camera']['production_grid']);plus.append(pool(images))
            images,_=camera.capture(-dq,922000+axis*20+repeat,grid=protocol()['camera']['production_grid']);minus.append(pool(images))
        plus=np.stack(plus);minus=np.stack(minus)
        columns_a.append((plus[:2].mean(0)-minus[:2].mean(0))/(2*step)/sigma)
        columns_b.append((plus[2:].mean(0)-minus[2:].mean(0))/(2*step)/sigma)
        steps.append(step);exposures+=8
        print('MEASURED_CALIBRATION_AXIS',axis+1,'/35',flush=True)
    a=np.column_stack(columns_a);b=np.column_stack(columns_b)
    cross=(a.T@b+b.T@a)/2.;values,vectors=np.linalg.eigh(cross)
    order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    keep=values>max(values[0]*1e-4,0.)
    count=min(12,int(keep.sum()))
    if count<4:raise RuntimeError('Fewer than four independently reproducible measured modes; do not pad the control basis with noise')
    directions=vectors[:,:count]
    average=(a+b)/2.
    basis=np.linalg.qr(average@directions,mode='reduced')[0]
    jac=basis.T@average
    np.savez_compressed(path,reference=reference,sigma=sigma,basis=basis,
                        jacobian=jac,directions=directions,steps=steps,
                        independent_half_cross_eigenvalues=values)
    atomic_json(RESULTS/'measured_response.json',dict(noisy_exposure_sets=exposures,
        known_command_states=71,unknown_state_phase_used=False,shwfs_used=False,
        feature_rank=count,independent_half_cross_eigenvalues=values.tolist(),sha256=sha(path)))
    return load_features()


def load_features():
    with np.load(RESULTS/'measured_response.npz') as z:
        return MeasuredFeatures(z['reference'],z['sigma'],z['basis'],z['jacobian'],z['directions'])
