"""Offline endpoint scoring; never imported by the observation-only controller."""
from settings import *
from analyze_low_order_wfe import decompose


def score(camera,q,size=256):
    y,x=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    disk=x*x+y*y<=1.;xy=np.c_[x[disk],y[disk]]
    maps=[];masks=[];rms=[];support=[]
    for field in range(9):
        values,valid=camera.phase_at(q,field,xy);support.append(float(valid.mean()))
        mask=np.zeros((size,size),bool);mask[disk]=valid
        phase=np.zeros((size,size));phase[mask]=values[valid]
        maps.append(phase);masks.append(mask);rms.append(float(np.sqrt(np.mean(values[valid]**2))))
    result=decompose(dict(opd_waves=np.array(maps),valid=np.array(masks),wrms_waves=np.array(rms)))
    result['pupil_support_fraction']=support
    result['minimum_pupil_support_fraction']=min(support)
    return result
