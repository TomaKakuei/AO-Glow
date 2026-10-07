"""Observation-only predictors. Optical truth is never a network input."""
import numpy as np
import torch
from .architectures import CoordinatedSpeckle, CoordinatedNikon, ModularNetwork
from ._canonical import build_universal_model
from .paths import activate, checkpoint

def device_name(device=None):
    return device or ('cuda' if torch.cuda.is_available() else 'cpu')

def payload(path):
    # Only load hash-checked checkpoints shipped by this package.
    return torch.load(path,map_location='cpu',weights_only=False)

class Predictor:
    def __init__(self,system,branch=None,profile='closed_loop',device=None):
        self.system=system;self.branch=branch;self.profile=profile
        self.device=device_name(device)
        activate(system)
        if system=='mod3':
            if branch not in ('front','rear'):raise ValueError('Mod3 needs front or rear')
            cp=payload(checkpoint(system,profile,branch))
            if profile=='latest':
                self.model=CoordinatedSpeckle(branch,cp['calibration'])
                self.model.load_state_dict(cp['model'])
            else:
                self.model=ModularNetwork(3 if branch=='front' else 4)
                self.model.load_state_dict(cp['state_dict'])
            self.cp=cp;self.model.to(self.device).eval()
        elif system=='nikon':
            import nikon35
            self.grid=nikon35.grid
            cps=[payload(checkpoint(system,'closed_loop',g)) for g in ('G1','G2','G3')]
            if profile=='latest':
                cp=payload(checkpoint(system,profile,'joint'))
                self.model=CoordinatedNikon(cps,cp['raw_mean'],cp['raw_std'],nikon35.NODES)
                self.model.load_state_dict(cp['state_dict']);self.model.to(self.device).eval()
            else:
                self.models=[self._ordinary(p) for p in cps];self.cps=cps
        elif system in ('trepan','kla'):
            if branch is None:raise ValueError(f'{system} needs a branch/module')
            cp=payload(checkpoint(system,profile,branch))
            self.cp=cp;self.model=self._ordinary(cp)
        else:raise ValueError(system)

    def _ordinary(self,cp):
        # Older KLA module checkpoints omit in_fields, but all use nine maps.
        net=build_universal_model('arch_a',cp.get('in_fields',9),cp.get('group_dofs',[5]*(len(cp['target_mean'])//5)))
        net.load_state_dict(cp['state_dict'])
        return net.to(self.device).eval()

    def __call__(self,observation):
        if isinstance(observation,dict):observation=observation['network']
        x=np.asarray(observation,dtype=np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite observation')
        if self.system=='nikon':
            if x.shape not in ((3,49),(147,)):raise ValueError('Expected noisy Nikon 3x49 nodes')
            x=x.reshape(1,3,49)
            image=torch.as_tensor(self.grid(x),device=self.device)
            with torch.inference_mode():
                if self.profile=='latest':
                    return self.model.physical_prediction(image,torch.as_tensor(x,device=self.device))[0].cpu().numpy()
                return np.concatenate([net(image)[0].cpu().numpy()*cp['target_std']+cp['target_mean'] for net,cp in zip(self.models,self.cps)])
        if x.shape!=(9,128,128) and self.system!='kla':raise ValueError('Expected noisy 9x128x128 speckle ADU')
        if self.system=='kla' and x.shape!=(9,64,64):raise ValueError('Expected original noisy KLA 9x64x64 input')
        image=torch.as_tensor(x[None],device=self.device)
        with torch.inference_mode():
            if self.system=='mod3':
                if self.profile=='latest':return self.model.physical_prediction(image)[0].cpu().numpy()
                cp=self.cp
                image=torch.asinh((image-torch.as_tensor(cp['reference'],device=self.device))/torch.as_tensor(cp['noise_std'],device=self.device)/3.)
            return self.model(image)[0].cpu().numpy()*np.asarray(self.cp['target_std'])+np.asarray(self.cp['target_mean'])

    def predict(self,branch,observation):
        if branch!=self.branch:raise ValueError('Predictor branch mismatch')
        return self(observation)

    def command(self,observation):
        predicted=self(observation)
        if self.system in ('mod3','nikon'):
            return -np.clip(predicted,-(1. if self.system=='mod3' else 1.5),1. if self.system=='mod3' else 1.5)
        return -predicted
