"""A pupil-relay camera with continuous OPD and integrated sensor pixels.

The optical entrance pupil remains 9 mm. A 2/9 magnification pupil relay
maps it to the legacy 2 mm camera pupil, followed by 2 mm propagation.
Only noisy sensor frames are observations. Internal irradiance is used for
numerical integration/convergence, never as controller feedback.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from optics import SamplingOptics
from wavefront_check import opd
import numpy as np
import torch
import scipy.ndimage as ndi
from scipy.interpolate import CubicSpline
from collections import OrderedDict


class RelayCamera:
    entrance_diameter_mm=9.
    camera_diameter_mm=2.
    propagation_mm=2.
    sensor_size=128

    def __init__(self,device=None,degree=24,boundary_angles=512):
        self.engine=SamplingOptics()
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.degree=degree;self.boundary_angles=boundary_angles
        y,x=np.mgrid[-1:1:64j,-1:1:64j]
        base=np.c_[x[self.engine.pupil_mask],y[self.engine.pupil_mask]]
        # Cartesian grids sparsely cover the circular rim. Add direct traced
        # annuli so a high-order fit is interpolated there, not extrapolated.
        angle=np.arange(384)*2*np.pi/384
        rim=np.concatenate([np.c_[r*np.cos(angle),r*np.sin(angle)] for r in [.96,.99,.9999]])
        self.base_count=len(base)
        self.xy=np.vstack([base,rim])
        self.terms=[(i,j) for i in range(degree+1) for j in range(degree+1-i)]
        self.design=self.basis(self.xy)
        self.pinv=np.linalg.pinv(self.design,rcond=1e-12)
        self.cache={};self.aperture_cache={};self.intensity_cache=OrderedDict()

    def basis(self,xy):
        vx=np.polynomial.chebyshev.chebvander(xy[:,0],self.degree)
        vy=np.polynomial.chebyshev.chebvander(xy[:,1],self.degree)
        return np.column_stack([vx[:,i]*vy[:,j] for i,j in self.terms])

    def phase_at(self,q,field_index,xy):
        engine=self.engine;field=engine.fields[field_index]
        raw=engine._trace(q,field,np.vstack(([0.,0.],xy)))
        if not raw['chief_valid']:raise ValueError('Chief ray missing')
        ref=engine.reference[field_index]
        data=engine._wave_data(raw,ref['E'],ref['chief_opl'])
        values=opd(data,ref['center'])*engine.n_image/(engine.wave*1e-6)
        values-=values.mean()
        phase=np.full(len(xy),np.nan);phase[raw['valid']]=values
        return phase,raw['valid']

    def pupil_boundary(self,q,field_index,count=None):
        # The intersection of these weakly tilted circular-aperture pupils is
        # star-shaped about the valid chief ray. Verify radial monotonicity.
        count=count or self.boundary_angles
        angle=np.arange(count)*2*np.pi/count
        unit=np.c_[np.cos(angle),np.sin(angle)]
        radii=np.array([.125,.25,.5,.75,.9,1.])
        xy=(unit[:,None,:]*radii[None,:,None]).reshape(-1,2)
        _,valid=self.phase_at(q,field_index,xy)
        valid=valid.reshape(count,-1)
        if np.any(np.diff(valid.astype(int),axis=1)>0) or not valid[:,0].all():
            raise ValueError('Pupil boundary is not in the certified star-shaped domain')
        if valid[:,-1].all():return np.ones(count)
        lo=np.max(np.where(valid,radii[None,:],0.),axis=1)
        hi=np.min(np.where(~valid,radii[None,:],1.),axis=1)
        for _ in range(13):
            mid=(lo+hi)/2
            _,keep=self.phase_at(q,field_index,unit*mid[:,None])
            lo=np.where(keep,mid,lo);hi=np.where(keep,hi,mid)
        return (lo+hi)/2

    def wave_representation(self,q,verify=False):
        out=[]
        rng=np.random.default_rng(179076001)
        radius=np.r_[np.sqrt(rng.uniform(0,.995**2,600)),np.full(128,.998)]
        angle=np.r_[rng.uniform(0,2*np.pi,600),np.arange(128)*2*np.pi/128+.001]
        probe=np.c_[radius*np.cos(angle),radius*np.sin(angle)]
        for fi in range(len(self.engine.fields)):
            values,keep=self.phase_at(q,fi,self.xy)
            survival=float(np.mean(keep[:self.base_count]))
            if survival<.2:raise ValueError('Less than 20% of pupil survives in a field')
            coef=self.pinv@values if keep.all() else np.linalg.lstsq(self.design[keep],values[keep],rcond=1e-12)[0]
            fit=self.design[keep]@coef-values[keep]
            linear=np.linalg.lstsq(np.c_[np.ones(keep.sum()),self.xy[keep]],values[keep],rcond=None)[0]
            matrix=np.zeros((self.degree+1,self.degree+1))
            for c,(i,j) in zip(coef,self.terms):matrix[i,j]=c
            boundary=self.pupil_boundary(q,fi)
            row=dict(coefficients=matrix,boundary=boundary,carrier=linear[1:],fit_rms_waves=float(np.sqrt(np.mean(fit**2))),
                     fit_max_waves=float(abs(fit).max()))
            if verify:
                truth,ok=self.phase_at(q,fi,probe);error=self.basis(probe[ok])@coef-truth[ok];error-=error.mean()
                row.update(independent_fit_rms_waves=float(np.sqrt(np.mean(error**2))),
                           independent_fit_max_waves=float(abs(error).max()))
            row['pupil_survival_fraction']=survival
            out.append(row)
        return out

    def resources(self,n):
        if n in self.cache:return self.cache[n]
        # Ray samples include endpoints; field integration uses pixel centers.
        coord=(np.arange(n)+.5)*2/n-1
        v=np.polynomial.chebyshev.chebvander(coord,self.degree)
        xx,yy=np.meshgrid(coord,coord)
        radii=np.hypot(xx,yy);angle=np.mod(np.arctan2(yy,xx),2*np.pi)
        big=2*n
        # Fourier interpolation preserves the physical diffuser footprint.
        from scipy.signal import resample
        centered=np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(self.engine.diffuser),
                              shift=(-32./n,-32./n))).real
        mask=resample(resample(centered,big,axis=0),big,axis=1).real
        dx=self.camera_diameter_mm/n
        freq=np.fft.fftfreq(big,dx);fx,fy=np.meshgrid(freq,freq)
        wavelength=self.engine.wave*1e-6
        term=1-(wavelength*fx)**2-(wavelength*fy)**2
        # Complex root attenuates evanescent components instead of aliasing them.
        transfer=np.exp(2j*np.pi/wavelength*self.propagation_mm*(np.sqrt(term.astype(complex))-1))
        row=dict(v=torch.tensor(v,dtype=torch.float64,device=self.device),radii=radii,angle=angle,
                 coords=coord,
                 fx=torch.tensor(fx,dtype=torch.float32,device=self.device),
                 fy=torch.tensor(fy,dtype=torch.float32,device=self.device),
                 diffuser=torch.tensor(np.exp(1j*mask),dtype=torch.complex64,device=self.device),
                 transfer=torch.tensor(transfer,dtype=torch.complex64,device=self.device))
        self.cache[n]=row
        return row

    def coverage_array(self,resource,boundary,n):
        angles=np.arange(len(boundary)+1)*2*np.pi/len(boundary)
        spline=CubicSpline(angles,np.r_[boundary,boundary[0]],bc_type='periodic')
        radial=np.minimum(spline(resource['angle']),1.)
        distance=radial-resource['radii'];coverage=(distance>0).astype(float)
        iy,ix=np.where(abs(distance)<np.sqrt(2)/n)
        x=resource['coords'][ix];y=resource['coords'][iy]
        counts=np.zeros(len(x))
        for sy in (np.arange(8)+.5)/8-.5:
            for sx in (np.arange(8)+.5)/8-.5:
                xx=x+sx*2/n;yy=y+sy*2/n
                rr=np.hypot(xx,yy);theta=np.mod(np.arctan2(yy,xx),2*np.pi)
                counts+=(rr<=np.minimum(spline(theta),1.))
        coverage[iy,ix]=counts/64.
        return coverage

    def aperture(self,resource,boundary,n):
        key=(n,boundary.tobytes())
        if key in self.aperture_cache:return self.aperture_cache[key]
        coverage=self.coverage_array(resource,boundary,n)
        value=torch.tensor(np.sqrt(coverage),dtype=torch.float64,device=self.device)
        if np.all(boundary==1.):self.aperture_cache[key]=value
        return value

    def integrate_field(self,representation,n,carrier_shift=True):
        resource=self.resources(n);v=resource['v']
        coef=torch.tensor(representation['coefficients'],dtype=torch.float64,device=self.device)
        cx,cy=representation['carrier']
        if carrier_shift:coef[1,0]-=cx;coef[0,1]-=cy
        phase=v@coef.T@v.T
        amplitude=self.aperture(resource,representation['boundary'],n)
        pupil=torch.zeros((2*n,2*n),dtype=torch.complex64,device=self.device)
        angle=torch.remainder(2*np.pi*phase,2*np.pi).to(torch.float32)
        pupil[n//2:3*n//2,n//2:3*n//2]=torch.polar(amplitude.to(torch.float32),angle)
        transfer=resource['transfer']
        if carrier_shift:
            # Modulation theorem: preserve the exact carrier in H(f+f0).
            # The omitted output carrier has unit modulus, so intensity is
            # unchanged. No physical tilt is removed from the camera image.
            wavelength=self.engine.wave*1e-6
            fx=resource['fx']+2*cx/self.camera_diameter_mm
            fy=resource['fy']+2*cy/self.camera_diameter_mm
            r2=(wavelength*fx)**2+(wavelength*fy)**2
            root=torch.sqrt(torch.clamp(1-r2,min=0.))
            kz=2*np.pi/wavelength*self.propagation_mm
            # Rationalized sqrt(1-r2)-1 avoids subtractive cancellation.
            phase_h=torch.where(r2<=1.,-kz*r2/(root+1.),-kz)
            decay=torch.exp(-kz*torch.sqrt(torch.clamp(r2-1,min=0.)))
            transfer=torch.polar(decay,phase_h)
        field=torch.fft.ifft2(torch.fft.fft2(pupil*resource['diffuser'])*transfer)
        factor=2*n//self.sensor_size
        intensity=field.abs().square().reshape(self.sensor_size,factor,self.sensor_size,factor).mean(dim=(1,3))
        return intensity.cpu().numpy()

    def integrated_irradiance(self,representation,n=2048):
        return self.integrate_field(representation,n)

    def reference_irradiance(self,representation,n=4096,carrier_shift=False):
        """Bounded-memory CPU reference, with optional full physical carrier.

        Independent implementation used only to validate the production
        renderer. Chunk the transfer function instead of allocating several
        8192-square double-complex work arrays on the small laptop GPU.
        """
        from scipy import fft as sf
        from scipy.signal import resample
        coord=(np.arange(n)+.5)*2/n-1
        v=np.polynomial.chebyshev.chebvander(coord,self.degree)
        xx,yy=np.meshgrid(coord,coord)
        geometry=dict(coords=coord,radii=np.hypot(xx,yy),angle=np.mod(np.arctan2(yy,xx),2*np.pi))
        del xx,yy
        coverage=self.coverage_array(geometry,representation['boundary'],n)
        del geometry
        coef=representation['coefficients'].copy();cx,cy=representation['carrier']
        if carrier_shift:coef[1,0]-=cx;coef[0,1]-=cy
        phase=v@coef.T@v.T
        big=2*n;field=np.zeros((big,big),np.complex64)
        field[n//2:3*n//2,n//2:3*n//2]=np.sqrt(coverage)*np.exp(2j*np.pi*phase)
        del coverage,phase
        centered=np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(self.engine.diffuser),
                              shift=(-32./n,-32./n))).real.astype(np.float32)
        mask=resample(resample(centered,big,axis=0),big,axis=1).real
        field*=np.exp(np.complex64(1j)*mask)
        del mask
        spectrum=sf.fft2(field,overwrite_x=True,workers=4)
        del field
        wavelength=self.engine.wave*1e-6;freq=np.fft.fftfreq(big,self.camera_diameter_mm/n)
        offsetx=2*cx/self.camera_diameter_mm if carrier_shift else 0.
        offsety=2*cy/self.camera_diameter_mm if carrier_shift else 0.
        for start in range(0,big,32):
            fx=freq[None,:]+offsetx;fy=freq[start:start+32,None]+offsety
            root=np.sqrt((1-(wavelength*fx)**2-(wavelength*fy)**2).astype(complex))
            h=np.exp(2j*np.pi/wavelength*self.propagation_mm*(root-1)).astype(np.complex64)
            spectrum[start:start+32]*=h
        field=sf.ifft2(spectrum,overwrite_x=True,workers=4)
        factor=big//self.sensor_size
        intensity=(field.real**2+field.imag**2).reshape(128,factor,128,factor).mean(axis=(1,3))
        return intensity

    @staticmethod
    def expected_adu(intensity):
        return 108.+16000.*intensity/max(float(intensity.max()),1e-30)

    @staticmethod
    def noise_sigma(expected):
        signal=np.maximum(expected-108.,0.)
        return np.sqrt(4*signal+16*(1.2**2+.5**2)+(.005*signal)**2+1/12)

    def capture(self,q,seed,grid=2048,verify=True):
        key=(grid,tuple(np.asarray(q,float)))
        if key in self.intensity_cache and (not verify or all('independent_fit_rms_waves' in r for r in self.intensity_cache[key][1])):
            intensities,representations=self.intensity_cache[key]
            self.intensity_cache.move_to_end(key)
        else:
            representations=self.wave_representation(q,verify=verify)
            if max(r['fit_rms_waves'] for r in representations)>.0003:
                raise RuntimeError('Continuous OPD fit exceeds 0.0003 waves; stop instead of returning an inaccurate sensor frame')
            if verify and max(r['independent_fit_rms_waves'] for r in representations)>.0003:
                raise RuntimeError('Independent continuous OPD fit exceeds 0.0003 waves')
            intensities=[self.integrated_irradiance(r,grid) for r in representations]
            self.intensity_cache[key]=(intensities,representations)
            if len(self.intensity_cache)>16:self.intensity_cache.popitem(last=False)
        state=np.random.get_state();np.random.seed(seed)
        try:
            images=[self.engine.camera_noise(intensity).astype(np.uint16) for intensity in intensities]
        finally:np.random.set_state(state)
        return np.stack(images),representations
