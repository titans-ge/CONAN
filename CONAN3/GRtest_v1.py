# this routine calculates the GR statistics for a set of ndim MCMC chains


import sys
import numpy as np

def grtest_emcee(chains_out):
    # note: chains_out has the dimensions (nchains,nsteps,ndim)
    nc,ns,nd = np.shape(chains_out)
    mcs = np.mean(chains_out, axis=1)      # mcs should be an (nchains, ndim) array
    mc = np.mean(mcs, axis=0)              # mc should be an (ndim) array
    BV = ns/(nc-1.) * np.sum((mcs-mc)**2, axis=0)  # BV should be an (ndim) array
    mvs = np.var(chains_out, axis=1)       # mvs should be an (nchains, ndim) array
    WV = np.mean(mvs, axis=0)              # WV should be an (ndim) array
    VV = (ns-1.)/ns * WV + (nc + 1.)/(ns*nc) * BV # VV should be an (ndim) array
    GR = VV/WV                             # GR should be an (ndim) array
    
    return GR
