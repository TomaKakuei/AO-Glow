"""Observation-only modular Kaleid controller.

This module receives noisy images, measured calibration and command interlocks.
It has no optical model, unknown pose, exact phase, WFE or endpoint scores.
"""
import numpy as np


def pool(images):
    return np.asarray(images,float).reshape(9,64,2,64,2).mean(axis=(2,4)).ravel()


class MeasuredFeatures:
    def __init__(self,reference,sigma,basis,jacobian,directions):
        self.reference=np.asarray(reference);self.sigma=np.asarray(sigma)
        self.basis=np.asarray(basis);self.jacobian=np.asarray(jacobian)
        self.directions=np.asarray(directions)

    def vector(self,image):return self.basis.T@((pool(image)-self.reference)/self.sigma)


def pair(rig,command,features):
    images=[rig.read(command),rig.read(command)]
    if any(x is None for x in images):return None
    vectors=np.stack([features.vector(x) for x in images]);mean=vectors.mean(0)
    # Cross product of independent noisy readings estimates the squared
    # response without the positive self-noise bias of a single squared norm.
    score=float(vectors[0]@vectors[1])/len(mean)
    variance=max(.25,float(np.mean((vectors[0]-vectors[1])**2))/2)
    return dict(images=images,vectors=vectors,mean=mean,score=score,variance=variance)


def compare(rig,current,trial,features,confidence=1.645):
    if rig.remaining<4 or not rig.allowed(trial):return None
    # Refresh both positions every comparison; never compare a new reading to
    # the smallest noisy score seen on an earlier visit.
    before=pair(rig,current,features);after=pair(rig,trial,features)
    if before is None or after is None:return None
    k=len(before['mean']);v1=before['variance'];v2=after['variance']
    stderr=np.sqrt(2*v1*np.dot(before['mean'],before['mean'])+
                   2*v2*np.dot(after['mean'],after['mean'])+k*(v1*v1+v2*v2))/k
    improvement=before['score']-after['score']
    accepted=bool(improvement>confidence*stderr)
    rig.commit(trial if accepted else current)
    return dict(accepted=accepted,before=before,after=after,
                improvement=float(improvement),standard_error=float(stderr),
                threshold=float(confidence*stderr))


def module_align(rig,model,branch,features,budget):
    start=rig.measurements;current=np.zeros(35);image=rig.read(current);history=[]
    sl=slice(0,15) if branch=='front' else slice(15,35)
    gain=1.
    while rig.measurements-start+4<=budget and rig.remaining>=4:
        proposal=model.predict(branch,image)
        trial=current.copy();trial[sl]=np.clip(current[sl]-gain*proposal,-1,1)
        if not rig.allowed(trial):
            gain*=.5
            if gain<.0625:break
            continue
        result=compare(rig,current,trial,features)
        if result is None:break
        accepted=result['accepted']
        if accepted:current=trial;image=result['after']['images'][-1];gain=min(1.,gain*1.5)
        else:image=result['before']['images'][-1];gain*=.5
        history.append(dict(gain=gain,accepted=accepted,improvement=result['improvement'],
                            standard_error=result['standard_error'],measurements=rig.measurements))
        if gain<.0625:break
    other=np.r_[15:35] if branch=='front' else np.r_[0:15]
    assert np.all(current[other]==0)
    return dict(command=current,history=history,measurements=rig.measurements-start)


def joint_refine(rig,features,budget):
    # The assembled residual modules are the starting plant. Joint commands
    # are increments from that assembly, and no module network is run here.
    start=rig.measurements;current=np.zeros(35);history=[]
    directions=features.directions
    jac=features.jacobian@directions
    radius=.08;failures=0
    while rig.measurements-start+4<=budget and rig.remaining>=4:
        reading=rig.read(current)
        if reading is None:raise RuntimeError('Current assembled pose invalid')
        residual=features.vector(reading)
        gram=jac.T@jac;scale=max(float(np.linalg.eigvalsh(gram)[-1]),1e-12)
        step=np.linalg.solve(gram+.001*scale*np.eye(len(gram)),-jac.T@residual)
        move=directions@step;move*=min(1.,radius/max(abs(move).max(),1e-12))
        trial=np.clip(current+move,-1,1)
        result=compare(rig,current,trial,features)
        if result is None:
            radius*=.5
            if radius<.0005:break
            continue
        accepted=result['accepted']
        if accepted:
            dq=np.linalg.lstsq(directions,trial-current,rcond=None)[0]
            dy=result['after']['mean']-result['before']['mean']
            if dq@dq>1e-10:jac+=np.outer(dy-jac@dq,dq)/(dq@dq)
            current=trial;failures=0;radius=min(.15,radius*1.25)
        else:failures+=1;radius*=.5
        history.append(dict(stage='joint_proposal',accepted=accepted,measurements=rig.measurements,
                            improvement=result['improvement'],standard_error=result['standard_error'],radius=radius))
        if failures>=2 and rig.measurements-start+8<=budget and rig.remaining>=8:
            # Refresh only two informative actuator directions from independent
            # noisy central probes, rather than spending 36 reads on one
            # coordinate gradient. Directions come from measured calibration.
            selected=np.argsort(abs(jac.T@residual))[-2:]
            for axis in selected:
                step_size=max(.015,radius)
                delta=directions[:,axis]*step_size/max(abs(directions[:,axis]).max(),1e-12)
                a=current+delta;b=current-delta
                if not rig.allowed(a) or not rig.allowed(b):continue
                plus=pair(rig,a,features);minus=pair(rig,b,features)
                rig.commit(current)
                if plus is None or minus is None:continue
                amount=step_size/max(abs(directions[:,axis]).max(),1e-12)
                jac[:,axis]=(plus['mean']-minus['mean'])/(2*amount)
            history.append(dict(stage='measured_direction_refresh',directions=selected.tolist(),measurements=rig.measurements))
            failures=0;radius=max(radius,.025)
        if radius<.0005:break
    return dict(command=current,history=history,measurements=rig.measurements-start)
