"""Kaleid's separate group heads, with explicit measured-image position cues."""
from settings import *
from models import ArchA_CanonicalResNet,GroupFactorizedHead
from torch import nn


class ModularNetwork(ArchA_CanonicalResNet):
    def __init__(self,groups):
        super().__init__(9,[5]*groups)
        # Global spatial averaging alone largely suppresses a translated
        # speckle image. Fixed sensor-coordinate moments retain that observable
        # displacement without adding any phase or mechanical-state input.
        y,x=torch.meshgrid((torch.arange(128)+.5)/64-1,(torch.arange(128)+.5)/64-1,indexing='ij')
        basis=torch.stack([torch.ones_like(x),x,y,x*y,x*x,y*y,
                           torch.sin(torch.pi*x),torch.sin(torch.pi*y)])
        self.register_buffer('sensor_moments',basis)
        self.head=GroupFactorizedHead(256+9*len(basis),[5]*groups)

    def forward(self,x):
        feat=self.stage3(self.stage2(self.stage1(self.stem(x))))
        global_features=self.pool(feat).flatten(1)
        moments=torch.einsum('bfxy,kxy->bfk',x,self.sensor_moments)/16384.
        return self.head(torch.cat([global_features,moments.flatten(1)],dim=1))


def build(branch):return ModularNetwork(len(BRANCHES[branch]))
