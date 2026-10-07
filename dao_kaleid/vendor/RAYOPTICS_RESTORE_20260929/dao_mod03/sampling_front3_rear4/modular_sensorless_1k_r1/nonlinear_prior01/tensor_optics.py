"""Differentiable known-prescription ray model for offline training losses.

It is not a sensor and is never an unknown-state inference input. The
controller still receives noisy intensity images only. This research kernel
must match the existing ray tracer before any training use is permitted.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from optics import SamplingOptics
from mechanics import GROUPS, SCALES
import numpy as np
import torch
from torch import nn


class TensorOptics(nn.Module):
    def __init__(self, engine=None, pupil_grid=32, device='cpu'):
        super().__init__()
        engine = engine or SamplingOptics()
        self.pupil_grid = pupil_grid
        y,x = np.mgrid[-1:1:complex(pupil_grid),-1:1:complex(pupil_grid)]
        disk = x*x+y*y<=1
        xy = np.c_[x[disk],y[disk]]
        pupils = np.vstack(([0.,0.],xy))
        launches = [engine.launch(field,pupils) for field in engine.fields]
        def buffer(name, values):
            self.register_buffer(name,torch.as_tensor(values,dtype=torch.float64,device=device))
        buffer('xy',xy)
        buffer('launch_points',np.stack([a for a,b in launches]))
        buffer('launch_directions',np.stack([b for a,b in launches]))
        buffer('origins',engine.geometry.origins)
        buffer('pivots',engine.geometry.pivots)
        buffer('scales',SCALES)
        buffer('reference_centers',np.array([r['center'] for r in engine.reference]))
        buffer('reference_exits',np.array([r['E'] for r in engine.reference]))
        buffer('reference_opls',np.array([r['chief_opl'] for r in engine.reference]))
        buffer('low_design',np.c_[np.ones(len(xy)),xy,2*np.sum(xy*xy,axis=1)-1])
        self.wavelength_mm = float(engine.wave)*1e-6
        self.n_image = float(engine.n_image)
        self.path_data = []
        for row in engine.path(np.zeros(35)):
            interface = row[0]
            if type(interface.profile).__name__!='Spherical' or hasattr(interface,'phase_element'):
                raise ValueError('Only the audited spherical prescription is supported')
            apertures = []
            for aperture in interface.clear_apertures:
                kind = type(aperture).__name__
                if kind=='Circular':
                    apertures.append((kind,aperture.x_offset,aperture.y_offset,aperture.radius))
                elif kind=='Rectangular':
                    apertures.append((kind,aperture.x_offset,aperture.y_offset,aperture.x_half_width,aperture.y_half_width))
                else:
                    raise ValueError('Unsupported aperture geometry')
            if not apertures:
                apertures=[('Circular',0.,0.,interface.max_aperture)]
            self.path_data.append(dict(cv=float(interface.profile.cv),index=None if row[3] is None else float(row[3]),
                                       z_dir=None if row[4] is None else float(row[4]),mode=interface.interact_mode,apertures=apertures))
        assert len(self.path_data)==24

    def poses(self,q):
        if q.ndim!=2 or q.shape[1]!=35 or q.dtype!=torch.float64:
            raise ValueError('Expected batch x35 float64 normalized coordinates')
        state = q.reshape(-1,7,5)*self.scales
        count = len(q)
        origins = self.origins.expand(count,-1,-1).clone()
        rotations = torch.eye(3,dtype=q.dtype,device=q.device).expand(count,25,3,3).clone()
        zero,one = torch.zeros_like(q[:,0]),torch.ones_like(q[:,0])
        for group,(a,b) in enumerate(GROUPS):
            tx,ty = torch.deg2rad(state[:,group,3]),torch.deg2rad(state[:,group,4])
            cx,sx,cy,sy = torch.cos(tx),torch.sin(tx),torch.cos(ty),torch.sin(ty)
            rotation = torch.stack([cy,sy*sx,sy*cx,zero,cx,-sx,-sy,cy*sx,cy*cx],dim=-1).reshape(count,3,3)
            shifted = self.origins[a:b+1]-self.pivots[group]
            origins[:,a:b+1] = torch.einsum('bij,sj->bsi',rotation,shifted)+self.pivots[group]+state[:,group,None,:3]
            rotations[:,a:b+1] = rotation[:,None]
        return origins,rotations

    def trace(self,q):
        origins,rotations = self.poses(q)
        p = self.launch_points.expand(len(q),-1,-1,-1)
        d = self.launch_directions.expand_as(p)
        valid = torch.ones(p.shape[:-1],dtype=torch.bool,device=q.device)
        opl = torch.zeros(p.shape[:-1],dtype=q.dtype,device=q.device)
        exit_p=exit_d=None
        for surf in range(1,len(self.path_data)):
            before,after = self.path_data[surf-1],self.path_data[surf]
            # Path index0 is prescription surface1. Preserve the reference
            # tracer's local transforms and perpendicular-plane intersection.
            ra,rb = rotations[:,surf],rotations[:,surf+1]
            rotation = rb.transpose(-1,-2)@ra
            translation = torch.einsum('bij,bj->bi',ra.transpose(-1,-2),origins[:,surf+1]-origins[:,surf])
            bp = torch.einsum('bij,bfrj->bfri',rotation,p-translation[:,None,None])
            bd = torch.einsum('bij,bfrj->bfri',rotation,d)
            perpendicular = -(bp*bd).sum(-1)
            pp = bp+perpendicular[...,None]*bd
            cv = after['cv']
            c = cv*(pp*pp).sum(-1)-2*pp[...,2]
            beta = cv*(bd*pp).sum(-1)-bd[...,2]
            disc = beta*beta-cv*c
            valid = valid & (disc>0)
            distance = c/(before['z_dir']*torch.sqrt(torch.clamp(disc,min=1e-30))-beta)
            hit = pp+distance[...,None]*bd
            normal = -cv*hit+torch.tensor([0.,0.,1.],dtype=q.dtype,device=q.device)
            normal = normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True)
            if surf-1<22:
                opl = opl+before['index']*(distance+perpendicular)
            if surf<=22:
                for aperture in after['apertures']:
                    ax,ay = hit[...,0]-aperture[1],hit[...,1]-aperture[2]
                    if aperture[0]=='Circular':
                        inside = torch.sqrt(ax*ax+ay*ay)<=aperture[3]+1e-8
                    else:
                        inside = (abs(ax)<=aperture[3]+1e-8)&(abs(ay)<=aperture[4]+1e-8)
                    valid = valid & inside
            cosine = (bd*normal).sum(-1)/torch.linalg.vector_norm(normal,dim=-1)
            if after['mode']=='transmit':
                ni,no = before['index'],after['index']
                refraction = no*no-ni*ni*(1-cosine*cosine)
                valid = valid & (refraction>=0)
                ncos = torch.copysign(torch.sqrt(torch.clamp(refraction,min=1e-30)),cosine)
                outgoing = (ni*bd+(ncos-ni*cosine)[...,None]*normal)/no
            elif after['mode']=='reflect':
                outgoing = bd-2*cosine[...,None]*normal
            else:
                outgoing = bd
            valid = valid & torch.isfinite(hit).all(-1) & torch.isfinite(outgoing).all(-1)
            p,d = hit,outgoing
            if surf==22:
                exit_p,exit_d = p,d
        valid = valid & torch.isfinite(opl)
        opl = (opl+(self.launch_points*self.launch_directions).sum(-1))/self.n_image
        return exit_p,exit_d,opl,valid

    def phase(self,q):
        p,d,opl,valid = self.trace(q)
        chief_valid = valid[:,:,0]
        p,d,opl,valid = p[:,:,1:],d[:,:,1:],opl[:,:,1:],valid[:,:,1:]
        centers = self.reference_centers[None,:,None,:]
        offset = p-centers
        radius2 = ((self.reference_centers-self.reference_exits)**2).sum(-1)[None,:,None]
        beta = (offset*d).sum(-1)
        c = (offset*offset).sum(-1)-radius2
        disc = beta*beta-c
        valid = valid & (disc>0)
        root = torch.sqrt(torch.clamp(disc,min=1e-30))
        denominator = -beta+root
        safe_denominator = torch.where(abs(denominator)>1e-12,denominator,torch.ones_like(denominator))
        small = torch.where(abs(denominator)>1e-12,c/safe_denominator,-beta-root)
        large = -beta+root
        distance = torch.where(abs(small)<abs(large),small,large)
        phase = (opl+distance-self.reference_opls[None,:,None])*self.n_image/self.wavelength_mm
        mean = torch.where(valid,phase,0).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        phase = torch.where(valid,phase-mean,0)
        return phase,valid,chief_valid

    def high_order(self,q):
        phase,valid,chief_valid = self.phase(q)
        design = self.low_design
        weight = valid.to(phase.dtype)
        gram = torch.einsum('nr,bfn,ns->bfrs',design,weight,design)
        rhs = torch.einsum('nr,bfn->bfr',design,phase)
        coefficients = torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
        residual = torch.where(valid,phase-torch.einsum('nr,bfr->bfn',design,coefficients),0)
        variance = (residual*residual).sum(-1)/weight.sum(-1)
        return variance,residual,valid,chief_valid
