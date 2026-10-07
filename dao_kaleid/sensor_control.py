"""Acceptance from independent noisy vectors; no optical model import."""
import numpy as np

def compare_vectors(before,after):
    def pair(v):
        a,b=np.asarray(v,dtype=float)
        mean=(a+b)/2;noise=(a-b)/np.sqrt(2)
        variance=2*float(mean@noise)**2+float(noise@noise)**2
        return float(a@b),variance
    b,bv=pair(before);a,av=pair(after)
    stderr=float(np.sqrt(max(bv+av,0)))
    improvement=b-a
    return {'before':b,'after':a,'improvement':improvement,'stderr':stderr,'accepted':bool(improvement>1.645*stderr)}
