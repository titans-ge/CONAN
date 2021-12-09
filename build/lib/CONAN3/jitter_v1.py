import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt    

def jitter(rarr, earr, indlist, nphot, nRV, njumpRV):
    
    jitters = np.array([])
    
    for j in range(nRV):
        res=rarr[indlist[nphot+j][0]]
        err=earr[indlist[nphot+j][0]]
        nfree = len(res)-njumpRV[j]
        jit=1
        jit, dump = scipy.optimize.leastsq(redchisqmin, jit, args=(nfree, res, err))
        jit = np.abs(jit)
        jitters = np.append(jitters,jit)
            
    return jitters


def redchisqmin(jit,nfree, res, err):
    
    redchisq = np.sum(res**2/(err**2 + jit**2)) / nfree
    
    return 1-redchisq
