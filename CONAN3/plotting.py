from .utils import bin_data, phase_fold
from ._classes import __default_backend__
import numpy as np
import matplotlib, os
from matplotlib.ticker import FormatStrFormatter    
import matplotlib.pyplot as plt    


def mcmc_plots(yval,tarr,farr,earr, nphot, nRV, indlist, filters,names,RVnames,prefix,RVunit,T0,period,Dur):
    
    matplotlib.use('Agg')

    if np.iterable(T0): npl = len(T0)
    else: npl = 1; T0 = [T0]
    
    phase_phasefold = {}    # dictionary containing phase of each planet across several datasets
    flux_phasefold  = {}
    model_phasefold = {}

    for n in range(npl):
        phase_phasefold[n] = np.empty(0) #array containing phase of each data point
        flux_phasefold[n]  = np.empty(0) #array containig baseline corrected flux of each data point
        model_phasefold[n] = np.empty(0) #array containg baseline corrected model of each data point

    for j in range(nphot):
        
        infile=prefix.split("/")[0] + "/" + names[j][:-4]+'_lcout.dat'
        tt, ft, et, mt, bfunc, mm, fco = np.loadtxt(infile, usecols=(0,1,2,3,4,5,6), unpack = True)  # reading in the lightcurve data
