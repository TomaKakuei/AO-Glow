"""Registered coverage schedule: heterogeneous active bodies, axes and magnitudes."""
import numpy as np

BRANCHES={'front':(0,1,2),'rear':(3,4,5,6)}
BIN_EDGES=((.0005,.01),(.01,.05),(.05,.2),(.2,.6),(.6,1.))
FAMILIES=('independent','single_axis','translation_dominant','tilt_dominant','one_dominant','mixed_scales','edge_probe','weak_background')


def candidate(branch,index,attempt=0):
    # Each of 5 amplitude strata has exactly 1000 accepted records per branch.
    # Failed optical/collision candidates retry the same stratum/family, not a
    # smaller amplitude. No common factor rescales all bodies after rejection.
    branch_number=0 if branch=='front' else 1
    seed=930000000+branch_number*100000000+int(index)*10000+int(attempt)
    rng=np.random.default_rng(seed)
    group=np.array(BRANCHES[branch]);other=np.array(BRANCHES['rear' if branch=='front' else 'front'])
    amplitude_bin=index%5;family=(index//5)%8
    count=1+(index//40)%len(group)
    active=list(rng.choice(group,size=count,replace=False))
    # Mixed context is explicitly represented; full-assembly inference does not
    # get access to an oracle that resets the unobserved half of the objective.
    context=(index//120)%5<2
    if context:
        active+=list(rng.choice(other,size=int(rng.integers(1,len(other)+1)),replace=False))
    q=np.zeros((7,5));lo,hi=BIN_EDGES[amplitude_bin]
    dominant=int(rng.choice(group))
    if dominant not in active:
        dominant=int(rng.choice([i for i in active if i in group]))
    peak=rng.uniform(lo,hi)
    for gi in active:
        translation_scale=np.exp(rng.uniform(np.log(max(lo/5,1e-5)),np.log(hi)))
        tilt_scale=np.exp(rng.uniform(np.log(max(lo/5,1e-5)),np.log(hi)))
        values=rng.uniform(-1,1,5)*np.r_[[translation_scale]*3,[tilt_scale]*2]
        if family==1:
            selected=int(rng.integers(0,5));values[:]=0;values[selected]=rng.choice([-1,1])*rng.uniform(lo,hi)
        elif family==2:values[3:]*=.1
        elif family==3:values[:3]*=.1
        elif family==4 and gi!=dominant:values*=rng.uniform(.01,.15)
        elif family==5:values*=10**rng.uniform(-2,0,5)
        elif family==7:values*=rng.uniform(.02,.2)
        q[gi]=values
    axis=int(rng.integers(0,5))
    if family==2:axis=int(rng.integers(0,3))
    if family==3:axis=int(rng.integers(3,5))
    # At least one target coordinate reaches its assigned stratum. Boundary
    # samples put only one selected coordinate near the upper edge.
    if family==6:peak=rng.uniform(max(lo,hi*.9),hi)
    if family==1:q[dominant]=0
    q[dominant,axis]=rng.choice([-1,1])*peak
    return q.ravel(),dict(seed=seed,amplitude_bin=amplitude_bin,family=FAMILIES[family],
                         active_mask=np.any(q!=0,axis=1).astype(np.uint8),
                         context=bool(context),dominant_group=dominant,dominant_axis=axis)
