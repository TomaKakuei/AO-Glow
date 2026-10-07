"""Conservative clearance certificates for mod03's seven separate lens bodies.

The prescription supplies optical caps (DIAM) and mechanical radii (MEMA).
Unknown peripheral shoulders are modeled explicitly as flat annuli at each
cap edge, closed by a cylinder at MEMA. No housing geometry is inferred.
Most neighboring bodies are separated by global Z support bounds. The last
two interlocking menisci require a curved-surface bound, not vertex gaps.
All accepted motions also certify the whole linear command path from start
to finish using adaptive interval bounds. An unresolved pose is rejected.
"""
import json
from pathlib import Path
import numpy as np

GROUPS = ((3,5),(6,8),(9,10),(11,14),(15,18),(19,20),(21,22))
SCALES = np.array([.1,.1,.1,5/60,5/60])  # mm and degrees
MARGIN_MM = .001  # 1 um clearance in the explicit body-envelope model


def read_rows(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf-16').splitlines():
        p=line.split()
        if not p: continue
        if p[0]=='SURF': rows.append({'surface':int(p[1])})
        elif rows and p[0] in ('CURV','DISZ','DIAM','MEMA'):
            rows[-1][p[0].lower()]=float(p[1])
    return rows


def rotation(tx_deg,ty_deg):
    tx,ty=np.deg2rad([tx_deg,ty_deg]);cx,sx=np.cos(tx),np.sin(tx);cy,sy=np.cos(ty),np.sin(ty)
    return np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])


def sag(cv,r):
    return cv*r*r/(1+np.sqrt(1-(cv*r)**2)) if cv else np.zeros_like(r)


class Geometry:
    def __init__(self,zmx):
        self.rows=read_rows(zmx)
        self.origins=np.zeros((25,3))
        self.origins[2:,2]=np.cumsum([r['disz'] for r in self.rows[1:24]])
        self.pivots=np.array([(self.origins[a]+self.origins[b])/2 for a,b in GROUPS])
        self.outer=np.array([max(self.rows[i]['mema'] for i in range(a,b+1)) for a,b in GROUPS])
        self.vertex_radius=[]
        self.face_specs=[]
        for gi,(a,b) in enumerate(GROUPS):
            pair=[]
            extent=0.
            for i in (a,b):
                r=self.rows[i]
                cap=r['diam'];cv=r['curv'];outer=self.outer[gi]
                if abs(cv)*cap>=1 or outer<cap-1e-9: raise ValueError('Invalid cap or mechanical radius')
                pair.append((cv,cap,outer))
                axial=max(abs(self.origins[i,2]-self.pivots[gi,2]),
                          abs(self.origins[i,2]-self.pivots[gi,2]+float(sag(cv,cap))))
                extent=max(extent,np.hypot(outer,axial))
            self.face_specs.append(pair);self.vertex_radius.append(extent)
        self.vertex_radius=np.array(self.vertex_radius)
        self.nominal_bounds=self.clearance_bounds(np.zeros(35))
        if min(self.nominal_bounds)<MARGIN_MM: raise ValueError(f'Nominal body envelopes overlap: {self.nominal_bounds}')

    def poses(self,q):
        q=np.asarray(q,float)
        if q.shape!=(35,) or not np.isfinite(q).all() or max(abs(q))>1+1e-12:
            raise ValueError('Expected 35 normalized coordinates within +/-100 um and +/-5 arcmin')
        state=q.reshape(7,5)*SCALES
        origins=self.origins.copy();rot=np.broadcast_to(np.eye(3),(25,3,3)).copy()
        for gi,((a,b),v) in enumerate(zip(GROUPS,state)):
            r=rotation(*v[3:]);pivot=self.pivots[gi]
            origins[a:b+1]=(self.origins[a:b+1]-pivot)@r.T+pivot+v[:3]
            rot[a:b+1]=r
        return origins,rot

    @staticmethod
    def face_z_support(origin,rot,cv,cap,outer):
        h=float(np.hypot(rot[2,0],rot[2,1]));a=float(rot[2,2])
        radii=[0.,cap]
        if cv:
            stationary=h/abs(cv)/np.sqrt(a*a+h*h)
            if stationary<cap:radii.append(stationary)
        radii=np.asarray(radii)
        values=np.r_[a*sag(cv,radii)+h*radii,a*sag(cv,radii)-h*radii,
                     a*sag(cv,cap)+h*outer,a*sag(cv,cap)-h*outer]
        return origin[2]+np.array([values.min(),values.max()])

    def clearance_bounds(self,q):
        origins,rot=self.poses(q)
        front=[];back=[]
        for gi,(a,b) in enumerate(GROUPS):
            front.append(self.face_z_support(origins[a],rot[a],*self.face_specs[gi][0])[0])
            back.append(self.face_z_support(origins[b],rot[b],*self.face_specs[gi][1])[1])
        bounds=np.array(front[1:])-np.array(back[:-1])
        # S20/S21 are nested positive-curvature surfaces. Over their common
        # lower-hemisphere graphs, the unrestricted minimum vertical separation
        # is dc_z + sqrt((R_left-R_right)^2 - ||dc_xy||^2). Restricting to caps
        # cannot make the minimum smaller. Use it only while right body's
        # projection is wholly inside the left optical cap (no shoulder crossing).
        a,b=20,21
        rl=1/self.rows[a]['curv'];rr=1/self.rows[b]['curv']
        cl=origins[a]+rot[a]@np.array([0.,0.,rl]);cr=origins[b]+rot[b]@np.array([0.,0.,rr])
        dc=cr-cl;dr=rl-rr
        # Bound projected right-body radius and coordinates in the left frame.
        gi=6;left=5
        pivot=self.pivots[gi]+np.asarray(q).reshape(7,5)[gi,:3]*SCALES[:3]
        local_center=rot[a].T@(pivot-origins[a])
        relative=rot[a].T@rot[b]
        height=self.vertex_radius[gi]
        projection_bound=self.outer[gi]+height*np.linalg.norm(relative[:2,2])
        domain_margin=self.rows[a]['diam']-np.linalg.norm(local_center[:2])-projection_bound
        if dr<=np.linalg.norm(dc[:2]):
            raise ValueError('Nested-sphere bound outside registered geometric domain')
        bounds[-1]=dc[2]+np.sqrt(dr*dr-dc[:2]@dc[:2])
        # The extra constraint certifies validity of the curved bound itself.
        # Keeping it in the adaptive path proof avoids a discontinuous switch
        # between curved and planar bounds near the cap boundary.
        nonadjacent=[front[j]-back[i] for i in range(7) for j in range(i+2,7)]
        return np.r_[bounds,domain_margin,nonadjacent]

    def state(self,q,margin=MARGIN_MM):
        bounds=self.clearance_bounds(q)
        return bool(np.all(bounds>=margin)),bounds

    def path(self,end,start=None,margin=MARGIN_MM,max_depth=12):
        """Certify the whole interpolation; no shared rescaling or pose clamping."""
        end=np.asarray(end,float);start=np.zeros(35) if start is None else np.asarray(start,float)
        self.poses(end);self.poses(start)
        delta=(end-start).reshape(7,5)*SCALES
        # Lipschitz constants for support-Z bounds, including all rotations.
        speed=abs(delta[:,2])+self.vertex_radius*np.sum(abs(np.deg2rad(delta[:,3:])),axis=1)
        lips=speed[:-1]+speed[1:]
        # Safe bound for nested-sphere clearance over the registered box.
        # The centers' speed uses the pivot-to-sphere-center distance.
        center_speeds=[]
        for gi,surf in ((5,20),(6,21)):
            radius=1/self.rows[surf]['curv']
            distance=abs(self.origins[surf,2]+radius-self.pivots[gi,2])
            w=np.sum(abs(np.deg2rad(delta[gi,3:])))
            center_speeds.append((abs(delta[gi,2])+distance*w,np.linalg.norm(delta[gi,:2])+distance*w))
        dr=1/self.rows[20]['curv']-1/self.rows[21]['curv']
        dc_bound=.35  # > maximum center lateral difference in this +/- .1 mm, +/-5' box
        curved_lips=sum(v[0] for v in center_speeds)+dc_bound/np.sqrt(dr*dr-dc_bound*dc_bound)*sum(v[1] for v in center_speeds)
        lips[-1]=max(lips[-1],curved_lips)
        angular=np.sum(abs(np.deg2rad(delta[:,3:])),axis=1)
        left_vertex_distance=abs(self.origins[20,2]-self.pivots[5,2])
        domain_lips=(np.linalg.norm(delta[6,:3]-delta[5,:3])+
                     (left_vertex_distance+3.)*angular[5]+
                     self.vertex_radius[6]*(angular[5]+angular[6]))
        lips=np.r_[lips,domain_lips,[speed[i]+speed[j] for i in range(7) for j in range(i+2,7)]]
        cache={};evaluations=0;worst=float('inf')
        def at(t):
            nonlocal evaluations,worst
            if t not in cache:
                cache[t]=self.clearance_bounds(start+t*(end-start));evaluations+=1
                worst=min(worst,float(min(cache[t])))
            return cache[t]
        pending=[(0.,1.,0)]
        while pending:
            lo,hi,depth=pending.pop();mid=(lo+hi)/2
            l,m,r=at(lo),at(mid),at(hi)
            if min(l.min(),m.min(),r.min())<margin:
                return dict(safe=False,reason='clearance_bound_below_margin',lower_mm=worst,evaluations=evaluations)
            certificate=np.minimum(np.minimum(l,m),r)-lips*(hi-lo)/4
            if np.all(certificate>=margin):continue
            if depth>=max_depth:
                return dict(safe=False,reason='unresolved_path_interval',lower_mm=worst,evaluations=evaluations)
            pending.extend(((lo,mid,depth+1),(mid,hi,depth+1)))
        return dict(safe=True,reason='certified_body_envelopes',lower_mm=worst,evaluations=evaluations)

    def describe(self):
        return dict(groups=GROUPS,scales_mm_mm_mm_deg_deg=SCALES.tolist(),clearance_margin_mm=MARGIN_MM,
                    mechanical_radii_mm=self.outer.tolist(),nominal_pair_clearance_lower_bounds_mm=self.nominal_bounds.tolist(),
                    constraint_names=['G1_G2','G2_G3','G3_G4','G4_G5','G5_G6','G6_G7','nested_cap_domain']+
                        [f'G{i+1}_G{j+1}' for i in range(7) for j in range(i+2,7)],
                    cap_extension='flat annular shoulder from DIAM to MEMA; cylindrical outer wall',
                    scope='prescription-derived closed lens-body envelopes, no barrel or holder geometry',
                    collision_policy='conservative sufficient clearance; unresolved states rejected, never silently clamped',
                    motion_path='all t in straight interpolation of translations and Euler commands')
