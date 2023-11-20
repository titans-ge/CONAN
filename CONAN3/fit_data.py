from logging import exception
from telnetlib import STATUS
import numpy as np
from types import SimpleNamespace
import os
from multiprocessing import Pool
import pickle
import emcee
import time

from occultquad import *
from occultnl import *
from .basecoeff_v14_LM import *
from .model_GP_v3 import *
from .logprob_multi_sin_v4 import *
from .plots_v12 import *
from .corfac_v8 import *
from .groupweights_v2 import *
from .ecc_om_par import *
from .outputs_v6_GP import *
from .jitter_v1 import *
from .GRtest_v1 import *
from george.modeling import Model
from george import kernels
import corner
from .gpnew import *
import celerite
from celerite import terms
from .celeritenew import GPnew as clnGPnew


from ._classes import _raise, mcmc_setup, __default_backend__, load_chains
import matplotlib
matplotlib.use(__default_backend__)

import multiprocessing as mp 
mp.set_start_method('fork')
__all__ = ["fit_data"]

def fit_data(lc, rv=None, mcmc=None, statistic = "median",
verbose=False, debug=False, save_burnin_chains=True, **kwargs):
    """
    function to fit the data using the light-curve object lc, rv_object rv and mcmc setup object mcmc.

    Parameters
    ----------
    statistic : str;
        statistic to run on posteriors to obtain model parameters and create model output file ".._out_full.dat".
        must be one of ["median", "max", "bestfit"], default is "median". 
        "max" and "median" calculate the maximum and median of each parameter posterior respectively while "bestfit" \
            is the parameter combination that gives the maximum joint posterior probability.

    Returns:
    --------
    result : object containing labeled mcmc chains
        Object that contains methods to plot the chains, corner, and histogram of parameters.
        e.g result.plot_chains(), result.plot_burnin_chains(), result.plot_corner, result.plot_posterior("T_0")

    **kwargs: other parameters sent to emcee.EnsembleSampler.run_mcmc() function
    """
    print('CONAN3 launched!!!\n') 
    #begin loading data from the 3 objects and calling the methods
    assert statistic in ["median", "max", "bestfit"], 'statistic can only be either median, max or bestfit'

#============lc_data=========================
    #from load_lightcurves()
    fpath      = lc._fpath
    names      = lc._names
    filters    = lc._filters
    lamdas     = lc._lamdas
    bases      = lc._bases
    groups     = lc._groups
    useGPphot  = lc._useGPphot

    nphot      = len(names)             # the number of photometry input files
    njumpphot  = np.zeros(nphot)
    filnames   = np.array(list(sorted(set(filters),key=filters.index))) 
    ulamdas    = np.array(list(sorted(set(lamdas),key=lamdas.index))) 
    grnames    = np.array(list(sorted(set(groups))))
    
    nfilt      = len(filnames)
    ngroup     = len(grnames)
    useSpline  = lc._spline

 #============GP Setup=============================
    #from load_lightcurves.add_GP()
    GPchoices = ["time", "xshift", "yshift", "air", "fwhm", "sky", "eti"]
    ndimGP = len(GPchoices)

    GPphotlc = np.zeros((nphot,ndimGP))
    GPphotvars = np.zeros((nphot,ndimGP))
    GPphotkerns = np.zeros((nphot,ndimGP),dtype=object)
    GPphotWN = np.zeros((nphot,1),dtype=object)
    GPphotWNstartppm = 50      # start at 50 ppm 
    GPphotWNstart = -50*np.ones((nphot,1)) # set WN very low in case WN is not used
    GPphotWNstep = np.zeros((nphot,1))
    GPphotWNprior = np.zeros((nphot,1))
    GPphotWNpriorwid = np.zeros((nphot,1))
    GPphotWNlimup = np.zeros((nphot,1))
    GPphotWNlimlo = np.zeros((nphot,1))
    GPphotpars1 = np.zeros((nphot,ndimGP))
    GPphotstep1 = np.zeros((nphot,ndimGP))
    GPphotprior1 = np.zeros((nphot,ndimGP))
    GPphotpriorwid1 = np.zeros((nphot,ndimGP))
    GPphotlim1up = np.zeros((nphot,ndimGP))
    GPphotlim1lo = np.zeros((nphot,ndimGP))
    GPphotpars2 = np.zeros((nphot,ndimGP))
    GPphotstep2 = np.zeros((nphot,ndimGP))
    GPphotprior2 = np.zeros((nphot,ndimGP))
    GPphotpriorwid2 = np.zeros((nphot,ndimGP))
    GPphotlim2up = np.zeros((nphot,ndimGP))
    GPphotlim2lo = np.zeros((nphot,ndimGP))
    GPncomponent = np.zeros((nphot,ndimGP))           # number of components in the kernel
    GPjumping = np.zeros((nphot,ndimGP),dtype=bool)
    GPall = np.zeros((nphot,ndimGP),dtype=bool) #Joint hyperparameters

    DA_gp = lc._GP_dict         #load input gp parameters from dict

    for j, nm in enumerate(DA_gp["lc_list"]):

        k = np.where(np.array(GPchoices) == DA_gp["pars"][j])

        if nm == 'all':
            i = np.where(np.array(names) == nm)
            GPjumping[0,k]=True
            GPphotkerns[0,k]=DA_gp["kernels"][j]
            GPphotpars1[0,k]=float(DA_gp["scale"][j])
            GPphotstep1[0,k]=float(DA_gp["s_step"][j])
            GPphotprior1[0,k]=float(DA_gp["s_pri"][j])
            GPphotpriorwid1[0,k]=float(DA_gp["s_pri_wid"][j])
            GPphotlim1up[0,k]=float(DA_gp["s_up"][j])
            GPphotlim1lo[0,k]=float(DA_gp["s_lo"][j])
            GPphotpars2[0,k]=float(DA_gp["metric"][j])
            GPphotstep2[0,k]=float(DA_gp["m_step"][j])
            GPphotprior2[0,k]=float(DA_gp["m_pri"][j])
            GPphotpriorwid2[0,k]=float(DA_gp["m_pri_wid"][j])
            GPphotlim2up[0,k]=float(DA_gp["m_up"][j])
            GPphotlim2lo[0,k]=float(DA_gp["m_lo"][j])
            GPall[0,k]=True

            if DA_gp["WN"][j] == 'y' and useGPphot[i]=='ce':
                GPphotWNstart[0] = -8.   
                GPphotWNstep[0] = 0.1 
                GPphotWNprior[0] = 0.
                GPphotWNpriorwid[0] = 0.
                GPphotWNlimup[0] = -5
                GPphotWNlimlo[0] = -12
                GPphotWN[0] = 'all'

            elif DA_gp["WN"][j] == 'y' and useGPphot[i]=='y':
                GPphotWNstart[:] = np.log((GPphotWNstartppm/1e6)**2) # in absolute
                GPphotWNstep[0] = 0.1
                GPphotWNprior[0] = 0.0
                GPphotWNpriorwid[0] = 0.
                GPphotWNlimup[0] = -3
                GPphotWNlimlo[0] = -25.0
                GPphotWN[0] = 'all'
            elif DA_gp["WN"][j] == 'n':
                GPphotWN[:] = 'n'
            else:
                raise ValueError('For at least one GP an invalid White-Noise option input was provided. Set it to either y or n.')

        else: 
            i = np.where(np.array(names) == nm)
            GPjumping[i,k]=True
            GPphotkerns[i,k]=DA_gp["kernels"][j]
            GPphotpars1[i,k]=float(DA_gp["scale"][j])
            GPphotstep1[i,k]=float(DA_gp["s_step"][j])
            GPphotprior1[i,k]=float(DA_gp["s_pri"][j])
            GPphotpriorwid1[i,k]=float(DA_gp["s_pri_wid"][j])
            GPphotlim1up[i,k]=float(DA_gp["s_up"][j])
            GPphotlim1lo[i,k]=float(DA_gp["s_lo"][j])
            GPphotpars2[i,k]=float(DA_gp["metric"][j])
            GPphotstep2[i,k]=float(DA_gp["m_step"][j])
            GPphotprior2[i,k]=float(DA_gp["m_pri"][j])
            GPphotpriorwid2[i,k]=float(DA_gp["m_pri_wid"][j])
            GPphotlim2up[i,k]=float(DA_gp["m_up"][j])
            GPphotlim2lo[i,k]=float(DA_gp["m_lo"][j])
            GPall[i,k]=False

            if DA_gp["WN"][j] == 'y' and useGPphot[i[0][0]] == 'ce':
                GPphotWNstart[i[0][0]] = -8.
                GPphotWNstep[i[0][0]] = 0.1
                GPphotWNprior[i[0][0]] = 0.
                GPphotWNpriorwid[i[0][0]] = 0.
                GPphotWNlimup[i[0][0]] = -5
                GPphotWNlimlo[i[0][0]] = -12
                GPphotWN[i[0][0]] = DA_gp["WN"][j]
            elif DA_gp["WN"][j] == 'y' and useGPphot[i[0][0]] == 'y':
                GPphotWNstart[i] = np.log((GPphotWNstartppm/1e6)**2) # in absolute
                GPphotWNstep[i] = 0.1
                GPphotWNprior[i] = 0.0
                GPphotWNpriorwid[i] = 0.
                GPphotWNlimup[i] = -3
                GPphotWNlimlo[i] = -21.0
                GPphotWN[i] = 'y'
            elif DA_gp["WN"][j] == 'n':
                GPphotWN[:] = 'n'
            else:
                raise ValueError('For at least one GP an invalid White-Noise option input was provided. Set it to either y or n.')





#============rv_data========================== 
    # from load_rvs
    if rv is not None and rv._names == []: rv = None   
    RVnames  = [] if rv is None else rv._names
    RVbases  = [] if rv is None else rv._RVbases
    gammas   = [] if rv is None else rv._gammas
    gamsteps = [] if rv is None else rv._gamsteps
    gampri   = [] if rv is None else rv._gampri
    gamprilo = [] if rv is None else rv._gamprilo
    gamprihi = [] if rv is None else rv._gamprihi
    sinPs    = [] if rv is None else rv._sinPs
    rv_fpath = [] if rv is None else rv._fpath
    if rv is None: #remove k as a free parameter
        lc._config_par["K"].to_fit = "n"
        lc._config_par["K"].step_size = 0   
    
    nRV=len(RVnames)             # the number of RV input files
    njumpRV=np.zeros(nRV)
    
    gamma_in = np.zeros((nRV,7)) # set up array to contain the gamma values
    extinpars = []

 
#============transit and RV jump parameters===============
    #from load_lightcurves.setup_transit_rv()

    config_par = lc._config_par
    
   #rprs
    rprs0 = config_par['RpRs'].start_value
    rprs_in = [ config_par['RpRs'].start_value,config_par['RpRs'].step_size, \
                config_par['RpRs'].bounds_lo,config_par['RpRs'].bounds_hi, \
                config_par['RpRs'].prior_mean,config_par['RpRs'].prior_width_lo,\
                config_par['RpRs'].prior_width_hi ]  
    rprs_ext = [config_par['RpRs'].prior_mean,config_par['RpRs'].prior_width_lo,\
                config_par['RpRs'].prior_width_hi]    
    if config_par['RpRs'].step_size != 0.: njumpphot=njumpphot+1
    if (config_par['RpRs'].to_fit == 'n' and config_par['RpRs'].prior == 'p'):
        extinpars.append('RpRs')

    erprs0=np.copy(0)
    
   #imp par
    b_in = [config_par['Impact_para'].start_value,config_par['Impact_para'].step_size,\
            config_par['Impact_para'].bounds_lo,config_par['Impact_para'].bounds_hi,\
            config_par['Impact_para'].prior_mean,config_par['Impact_para'].prior_width_lo,\
            config_par['Impact_para'].prior_width_hi]  
    b_ext = [config_par['Impact_para'].prior_mean,config_par['Impact_para'].prior_width_lo,\
            config_par['Impact_para'].prior_width_hi]     
    if config_par['Impact_para'].step_size != 0.: njumpphot=njumpphot+1
    if (config_par['Impact_para'].to_fit == 'n' and config_par['Impact_para'].prior == 'p'):
        extinpars.append('b')

   #duration
    dur_in = [config_par['Duration'].start_value,config_par['Duration'].step_size,\
            config_par['Duration'].bounds_lo,config_par['Duration'].bounds_hi,\
            config_par['Duration'].prior_mean,config_par['Duration'].prior_width_lo,\
            config_par['Duration'].prior_width_hi]  
    dur_ext = [config_par['Duration'].prior_mean,config_par['Duration'].prior_width_lo,\
                config_par['Duration'].prior_width_hi]     
    if config_par['Duration'].step_size != 0.: njumpphot=njumpphot+1
    if (config_par['Duration'].to_fit == 'n' and config_par['Duration'].prior == 'p'):
        extinpars.append('dur_[d]')
        
   #T0
    T0_in = [config_par['T_0'].start_value,config_par['T_0'].step_size,config_par['T_0'].bounds_lo,\
            config_par['T_0'].bounds_hi,config_par['T_0'].prior_mean,config_par['T_0'].prior_width_lo,\
            config_par['T_0'].prior_width_hi]  
    T0_ext = [config_par['T_0'].prior_mean,config_par['T_0'].prior_width_lo,config_par['T_0'].prior_width_hi]    
    if config_par['T_0'].step_size != 0. != 0.: 
        njumpphot=njumpphot+1
        njumpRV=njumpRV+1

    if (config_par['T_0'].to_fit == 'n' and config_par['T_0'].prior == 'p'):
        extinpars.append('T_0')
        
    
   #per
    per_in = [config_par['Period'].start_value,config_par['Period'].step_size,config_par['Period'].bounds_lo,\
            config_par['Period'].bounds_hi,config_par['Period'].prior_mean,config_par['Period'].prior_width_lo,\
            config_par['Period'].prior_width_hi]  
    per_ext = [config_par['Period'].prior_mean,config_par['Period'].prior_width_lo,config_par['Period'].prior_width_hi]    
    if config_par['Period'].step_size != 0. != 0.: 
        njumpphot=njumpphot+1
        njumpRV=njumpRV+1

    if (config_par['Period'].to_fit == 'n' and config_par['Period'].prior == 'p'):
        extinpars.append('Period_[d]')

   #ecc
    ecc_in = [config_par['Eccentricity'].start_value,config_par['Eccentricity'].step_size,\
        config_par['Eccentricity'].bounds_lo,config_par['Eccentricity'].bounds_hi,\
        config_par['Eccentricity'].prior_mean,config_par['Eccentricity'].prior_width_lo,\
        config_par['Eccentricity'].prior_width_hi]
    eccpri= config_par['Eccentricity'].prior
    if config_par['Eccentricity'].step_size != 0.: 
        njumpRV=njumpRV+1
    if (config_par['Eccentricity'].to_fit == 'n' and config_par['Eccentricity'].prior == 'p'):
        _raise(ValueError, 'cant externally input eccentricity at this time!')

   #omega
    opri= config_par['omega'].prior
    omega_in = [config_par['omega'].start_value,config_par['omega'].step_size,config_par['omega'].bounds_lo,\
                config_par['omega'].bounds_hi,config_par['omega'].prior_mean,config_par['omega'].prior_width_lo,\
                config_par['omega'].prior_width_hi]
    omega_in = np.multiply(omega_in,np.pi)/180.
    if config_par['omega'].step_size != 0.: 
        njumpRV=njumpRV+1
    if (config_par['omega'].to_fit == 'n' and config_par['omega'].prior == 'p'):
        _raise(ValueError, 'cant externally input eccentricity at this time!')
        
   #K
    Kpri= config_par['K'].prior
    K_in = [config_par['K'].start_value,config_par['K'].step_size,config_par['K'].bounds_lo,\
            config_par['K'].bounds_hi,config_par['K'].prior_mean,config_par['K'].prior_width_lo,\
            config_par['K'].prior_width_hi]   
    K_in=np.divide(K_in,1000.)  # convert to km/s
    
    K_ext = [config_par['K'].prior_mean,config_par['K'].prior_width_lo,config_par['K'].prior_width_hi] 
    K_ext = np.divide(K_ext,1000.)  # convert to km/s
    if config_par['K'].step_size != 0.: njumpRV=njumpRV+1

    if (config_par['K'].to_fit == 'n' and config_par['K'].prior == 'p'):
        extinpars.append('K')
        

    for i in range(nRV):
        gamma_in[i,:]=[float(gammas[i]),float(gamsteps[i]),-1000,1000,float(gampri[i]),float(gamprilo[i]),float(gamprihi[i])]
        if (gamma_in[i,1] != 0.) :
            njumpRV[i]=njumpRV[i]+1


   # adapt the eccentricity and omega jump parameters sqrt(e)*sin(o), sqrt(e)*cos(o)

    if ((eccpri == 'y' and opri == 'n') or (eccpri == 'n' and opri == 'y')):
        _raise(ValueError,'priors on eccentricity and omega: either both on or both off')
        
    eos_in, eoc_in = ecc_om_par(ecc_in, omega_in)
    
    
#============ddfs ==========================
    #from load_lighcurves.transit_depth_variations()
    drprs_op = lc._ddfs.drprs_op
    divwhite = lc._ddfs.divwhite
    ddfYN    = lc._ddfs.ddfYN
                
    grprs  = lc._ddfs.depth_per_group   # the group rprs values
    egrprs = lc._ddfs.depth_err_per_group  # the uncertainties of the group rprs values
    dwfiles = [f"dw_00{n}.dat" for n in grnames]   

    dwCNMarr=np.array([])      # initializing array with all the dwCNM values
    dwCNMind=[]                # initializing array with the indices of each group's dwCNM values
    dwind=np.array([])
    if (divwhite=='y'):           # do we do a divide-white?    
        for i in range(ngroup):   # read fixed dwCNMs for each group
            tdwCNM, dwCNM = np.loadtxt(lc._fpath + dwfiles[i], usecols=(0,1), unpack = True)
            dwCNMarr      = np.concatenate((dwCNMarr,dwCNM), axis=0)
            dwind         = np.concatenate((dwind,np.zeros(len(dwCNM),dtype=int)+i), axis=0)
            indices       = np.where(dwind==i)
            dwCNMind.append(indices)        

   
#============occultation setup=============
    #from load_lightcurves.setup_occultation()

    DA_occ = lc._occ_dict
    nocc = len(DA_occ["filters_occ"])
    occ_in = np.zeros((nocc,7))

    for i, f in enumerate(DA_occ["filters_occ"]):
        j = np.where(filnames == f )
        k = np.where(np.array(lc._filters)== f)

        occ_in[j,:] = [DA_occ["start_value"][i], DA_occ["step_size"][i], DA_occ["bounds_lo"][i], DA_occ["bounds_hi"][i],
                        DA_occ["prior_mean"][i], DA_occ["prior_width_lo"][i], DA_occ["prior_width_hi"][i] ]           

        if occ_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1
    

#============limb darkening===============
    #from load_lightcurves.limb_darkening()
    DA_ld = lc._ld_dict

    c1_in=np.zeros((nfilt,7))
    c2_in=np.zeros((nfilt,7))
    c3_in=np.zeros((nfilt,7))
    c4_in=np.zeros((nfilt,7)) 

    for i in range(nfilt):
        j=np.where(filnames == filnames[i])              # make sure the sequence in this array is the same as in the "filnames" array
        k=np.where(np.array(lc._filters) == filnames[i])

        c1_in[j,:] = [DA_ld["c1"][i], DA_ld["step1"][i],DA_ld["bound_lo1"][i],DA_ld["bound_hi1"][i],DA_ld["c1"][i],DA_ld["sig_lo1"][i],DA_ld["sig_hi1"][i]]
        c1_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step1"][i] == 0.) else c1_in[j,5])
        c1_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step1"][i] == 0.) else c1_in[j,6])
        if c1_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1


        c2_in[j,:] = [DA_ld["c2"][i], DA_ld["step2"][i],DA_ld["bound_lo2"][i],DA_ld["bound_hi2"][i],DA_ld["c2"][i],DA_ld["sig_lo2"][i],DA_ld["sig_hi2"][i]]  # the limits are -3 and 3 => very safe
        c2_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step2"][i] == 0.) else c2_in[j,5])
        c2_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step2"][i] == 0.) else c2_in[j,6])
        if c2_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1

        c3_in[j,:] = [DA_ld["c3"][i], DA_ld["step3"][i],DA_ld["bound_lo3"][i],DA_ld["bound_hi3"][i],DA_ld["c3"][i],DA_ld["sig_lo3"][i],DA_ld["sig_hi3"][i]]  # the limits are -3 and 3 => very safe
        c3_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step3"][i] == 0.) else c3_in[j,5])
        c3_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step3"][i] == 0.) else c3_in[j,6])
        if c3_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1

        c4_in[j,:] = [DA_ld["c4"][i], DA_ld["step4"][i],DA_ld["bound_lo4"][i],DA_ld["bound_hi4"][i],DA_ld["c4"][i],DA_ld["sig_lo4"][i],DA_ld["sig_hi4"][i]]  # the limits are -3 and 3 => very safe
        c4_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step4"][i] == 0.) else c4_in[j,5])
        c4_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step4"][i] == 0.) else c4_in[j,6])
        if c4_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1

        
        if (c3_in[j,0] == 0. and c4_in[j,0]==0 and c3_in[j,1] == 0. and c4_in[j,1] == 0.):
            if verbose: print('Limb-darkening law: quadratic')
            v1=2.*c1_in[j,0]+c2_in[j,0] #transform c1 and c2
            v2=c1_in[j,0]-c2_in[j,0]
            # ev1=np.sqrt(4.*c1_in[j,1]**2+c2_in[j,1]**2)  #transform steps (not needed)
            # ev2=np.sqrt(c1_in[j,1]**2+c2_in[j,1]**2)
            lov1=np.sqrt(4.*c1_in[j,5]**2+c2_in[j,5]**2) if c1_in[j,5] else 0 #transform sig_los
            lov2=np.sqrt(c1_in[j,5]**2+c2_in[j,5]**2)    if c2_in[j,5] else 0
            hiv1=np.sqrt(4.*c1_in[j,6]**2+c2_in[j,6]**2) if c1_in[j,6] else 0 #transform sig_his
            hiv2=np.sqrt(c1_in[j,6]**2+c2_in[j,6]**2)    if c2_in[j,6] else 0
            lo_lim1 = 2.*c1_in[j,2]+c2_in[j,2]  if c1_in[j,2] else 0   #transform bound_lo
            lo_lim2 = c1_in[j,2]-c2_in[j,3]     if c2_in[j,3] else 0
            hi_lim1 = 2.*c1_in[j,3]+c2_in[j,3]  if c1_in[j,3] else 0
            hi_lim2 = c1_in[j,3]-c2_in[j,2]     if c2_in[j,2] else 0
            if debug:
                print("\nDEBUG: In fit_data.py")
                print(f"LD:\nuniform: c1 = ({lo_lim1},{v1},{hi_lim1})/{c1_in[j,1]}, c2 = ({lo_lim2},{v2},{hi_lim2})/{c2_in[j,1]}")
                print(f"normal: c1=({v1},-{lov1}+{hiv1})/{c1_in[j,1]}, c2=({v2},-{lov2}+{hiv2})/{c2_in[j,1]}")
            #replace inputs LDs with transformations
            c1_in[j,0]=np.copy(v1)  #replace c1 and c2
            c2_in[j,0]=np.copy(v2)
            c1_in[j,4]=np.copy(v1)  #replace prior mean
            c2_in[j,4]=np.copy(v2)
            # c1_in[j,1]=np.copy(ev1) #replace steps (not needed)
            # c2_in[j,1]=np.copy(ev2)
            c1_in[j,2]=np.copy(lo_lim1)  #replace bound_lo
            c2_in[j,2]=np.copy(lo_lim2)
            c1_in[j,3]=np.copy(hi_lim1)  #replace bound_hi
            c2_in[j,3]=np.copy(hi_lim2)            
            if (DA_ld["priors"][i] == 'y'):    # replace prior on LDs
                c1_in[j,5]=np.copy(lov1)
                c1_in[j,6]=np.copy(hiv1)
                c2_in[j,5]=np.copy(lov2)
                c2_in[j,6]=np.copy(hiv2)


#============contamination factors=======================
    #from load_lightcurves.contamination
    DA_cont = lc._contfact_dict
    cont=np.zeros((nfilt,2))

    for i in range(nfilt):
        j = np.where(filnames == filnames[i])               # make sure the sequence in this array is the same as in the "filnames" array
        if verbose: print(j)
        cont[j,:]= [DA_cont["cont_ratio"][i], DA_cont["err"][i]]


#========= stellar properties==========================
    #from setup_fit.stellar_parameters()
    DA_stlr = lc._stellar_dict
    
    Rs_in  = DA_stlr["R_st"][0]
    sRs_lo = DA_stlr["R_st"][1]
    sRs_hi = DA_stlr["R_st"][2]

    Ms_in  = DA_stlr["M_st"][0]
    sMs_lo = DA_stlr["M_st"][1]
    sMs_hi = DA_stlr["M_st"][2]

    howstellar = DA_stlr["par_input"]


#=============mcmc setup===============
    #from setup_fit.mcmc()
    if mcmc is None: mcmc = mcmc_setup()
    DA_mc =  mcmc._mcmc_dict

    nsamples    = int(DA_mc['n_steps']*DA_mc['n_chains'])   # total number of integrations
    nchains     = int(DA_mc['n_chains'])  #  number of chains
    ppchain     = int(nsamples/nchains)  # number of points per chain
    nproc       = int(DA_mc['n_cpus'])   #  number of processes
    burnin      = int(DA_mc['n_burn'])    # Length of bun-in
    walk        = DA_mc['sampler']            # Differential Evolution?          
    grtest      = True if DA_mc['GR_test'] == 'y' else False  # GRtest done?
    plots       = True if DA_mc['make_plots'] == 'y' else False  # Make plots done
    leastsq     = True if DA_mc['leastsq'] == 'y' else False  # Do least-square?
    savefile    = DA_mc['savefile']   # Filename of save file
    savemodel   = DA_mc['savemodel']   # Filename of model save file
    adaptBL     = DA_mc['adapt_base_stepsize']   # Adapt baseline coefficent
    paraCNM     = DA_mc['remove_param_for_CNM']   # remove parametric model for CNM computation
    baseLSQ     = DA_mc['leastsq_for_basepar']   # do a leas-square minimization for the baseline (not jump parameters)
    lm          = True if DA_mc['lssq_use_Lev_Marq'] =='y' else False  # use Levenberg-Marquardt algorithm for minimizer?
    cf_apply    = DA_mc['apply_CFs']  # which CF to apply
    jit_apply   = DA_mc['apply_jitter'] # apply jitter



#********************************************************************
#============Start computations as in original CONANGP===============
#********************************************************************
############################ GPs for RVs setup #############################################

    ### BUG: this should be read in! And should contain RV inputs ###
    # all of these should be lists with nphot bzw nRV items

    useGPrv=['n']*nRV

    GPrvpars1=np.array([0.])
    GPrvpars2=np.array([0.])
    GPrvstep1=np.array([0.001])
    GPrvstep2=np.array([0.001])
    GPrvWN=['y']    #fitWN
    GPrvwnstartms = np.array([10])
    GPrvwnstart = np.log((GPrvwnstartms/1e3)**2)  # in km/s
    GPrvWNstep = np.array([0.1])

    #============================= SETUP ARRAYS =======================================
    print('Setting up photometry arrays ...')
    if useSpline.use: print('Setting up Spline fitting ...')                                
    tarr=np.array([]) # initializing array with all timestamps
    farr=np.array([]) # initializing array with all flux values
    earr=np.array([]) # initializing array with all error values
    xarr=np.array([]) # initializing array with all x_shift values
    yarr=np.array([]) # initializing array with all y_shift values
    aarr=np.array([]) # initializing array with all airmass values
    warr=np.array([]) # initializing array with all fwhm values
    sarr=np.array([]) # initializing array with all sky values
    lind=np.array([]) # initializing array with the lightcurve indices
    barr=np.array([]) # initializing array with all bisector values
    carr=np.array([]) # initializing array with all contrast values

    indlist = []    # the list of the array indices
    bvars    = []   # a list that will contain lists of [0, 1] for each of the baseline parameters, for each of the LCs. 0 means it's fixed. 1 means it's variable
    bvarsRV    = []   # a list that will contain lists of [0, 1] for each of the baseline parameters, for each of the RV curves. 0 means it's fixed. 1 means it's variable


    if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the value specified at the jump parameters, and fix it.
        rprs_in=[rprs_in[0],0,0,1,0,0,0]
        nddf=nfilt
    else:
        nddf=0

    # set up the parameters
    params   = np.array([T0_in[0], rprs_in[0], b_in[0], dur_in[0], per_in[0], eos_in[0], eoc_in[0], K_in[0]])  # initial guess params
    stepsize = np.array([T0_in[1], rprs_in[1], b_in[1], dur_in[1], per_in[1], eos_in[1], eoc_in[1], K_in[1]])  # stepsizes
    pmin     = np.array([T0_in[2], rprs_in[2], b_in[2], dur_in[2], per_in[2], eos_in[2], eoc_in[2], K_in[2]])  # Boundaries (min)
    pmax     = np.array([T0_in[3], rprs_in[3], b_in[3], dur_in[3], per_in[3], eos_in[3], eoc_in[3], K_in[3]])  # Boundaries (max)
    prior    = np.array([T0_in[4], rprs_in[4], b_in[4], dur_in[4], per_in[4], eos_in[4], eoc_in[4], K_in[4]])  # Prior centers
    priorlow = np.array([T0_in[5], rprs_in[5], b_in[5], dur_in[5], per_in[5], eos_in[5], eoc_in[5], K_in[5]])  # Prior sigma low side
    priorup  = np.array([T0_in[6], rprs_in[6], b_in[6], dur_in[6], per_in[6], eos_in[6], eoc_in[6], K_in[6]])  # Prior sigma high side
    pnames   = np.array(['T_0', 'RpRs', 'b', 'dur_[d]', 'Period_[d]', 'esin(w)', 'ecos(w)', 'K']) # Parameter names

    extcens = np.array([T0_ext[0], rprs_ext[0], b_ext[0], dur_ext[0], per_ext[0], 0., 0., K_ext[0]])
    extup = np.array([T0_ext[1], rprs_ext[1], b_ext[1], dur_ext[1], per_ext[1], 0., 0., K_ext[1]])
    extlow = np.array([T0_ext[2], rprs_ext[2], b_ext[2], dur_ext[2], per_ext[2], 0., 0., K_ext[2]])


    if (divwhite=='y'):           # do we do a divide-white? If yes, then fix all the transit shape parameters
        stepsize[0:6] = 0
        prior[0:6] = 0

    if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the specified value, and fix it.
        drprs_in=np.zeros((nfilt,7))
        njumpphot=njumpphot+1   # each LC has another jump pm

        for i in range(nfilt):  # and make an array with the drprs inputs |  drprs_op=[0.,float(adump[3]),float(adump[4]),float(adump[5])]  # the dRpRs options
            drprs_in[i,:]=drprs_op
            params=np.concatenate((params, [drprs_in[i,0]]))     # add them to the parameter arrays    
            stepsize=np.concatenate((stepsize, [drprs_in[i,1]]))
            pmin=np.concatenate((pmin, [drprs_in[i,2]]))
            pmax=np.concatenate((pmax, [drprs_in[i,3]]))
            prior=np.concatenate((prior, [drprs_in[i,4]]))
            priorlow=np.concatenate((priorlow, [drprs_in[i,5]]))
            priorup=np.concatenate((priorup, [drprs_in[i,6]]))
            pnames=np.concatenate((pnames, [filnames[i]+'_dRpRs']))
            

    for i in range(nfilt):  # add the occultation depths
        params=np.concatenate((params, [occ_in[i,0]]))
        stepsize=np.concatenate((stepsize, [occ_in[i,1]]))
        pmin=np.concatenate((pmin, [occ_in[i,2]]))
        pmax=np.concatenate((pmax, [occ_in[i,3]]))
        prior=np.concatenate((prior, [occ_in[i,4]]))
        priorlow=np.concatenate((priorlow, [occ_in[i,5]]))
        priorup=np.concatenate((priorup, [occ_in[i,6]]))
        pnames=np.concatenate((pnames, [filnames[i]+'_DFocc']))

    for i in range(nfilt):  # add the LD coefficients for the filters to the parameters
        params=np.concatenate((params, [c1_in[i,0], c2_in[i,0], c3_in[i,0], c4_in[i,0]]))
        stepsize=np.concatenate((stepsize, [c1_in[i,1], c2_in[i,1], c3_in[i,1], c4_in[i,1]]))
        pmin=np.concatenate((pmin, [c1_in[i,2], c2_in[i,2], c3_in[i,2], c4_in[i,2]]))
        pmax=np.concatenate((pmax, [c1_in[i,3], c2_in[i,3], c3_in[i,3], c4_in[i,3]]))
        prior=np.concatenate((prior, [c1_in[i,4], c2_in[i,4], c3_in[i,4], c4_in[i,4]]))
        priorlow=np.concatenate((priorlow, [c1_in[i,5], c2_in[i,5], c3_in[i,5], c4_in[i,5]]))
        priorup=np.concatenate((priorup, [c1_in[i,6], c2_in[i,6], c3_in[i,6], c4_in[i,6]]))
        pnames=np.concatenate((pnames, [filnames[i]+'_c1',filnames[i]+'_c2',filnames[i]+'_c3',filnames[i]+'_c4']))

    for i in range(nRV):
        params=np.concatenate((params,[gamma_in[i,0]]), axis=0)
        stepsize=np.concatenate((stepsize,[gamma_in[i,1]]), axis=0)
        pmin=np.concatenate((pmin,[gamma_in[i,2]]), axis=0)
        pmax=np.concatenate((pmax,[gamma_in[i,3]]), axis=0)
        prior=np.concatenate((prior,[gamma_in[i,4]]), axis=0)
        priorlow=np.concatenate((priorlow,[gamma_in[i,5]]), axis=0)
        priorup=np.concatenate((priorup,[gamma_in[i,6]]), axis=0)
        pnames=np.concatenate((pnames,[RVnames[i]+'_gamma']), axis=0)
        
        if (jit_apply=='y'):
            # print('does jitter work?')
            # print(nothing)
            params=np.concatenate((params,[0.01]), axis=0)
            stepsize=np.concatenate((stepsize,[0.001]), axis=0)
            pmin=np.concatenate((pmin,[0.]), axis=0)
            pmax=np.concatenate((pmax,[100]), axis=0)
            prior=np.concatenate((prior,[0.]), axis=0)
            priorlow=np.concatenate((priorlow,[0.]), axis=0)
            priorup=np.concatenate((priorup,[0.]), axis=0)
            pnames=np.concatenate((pnames,[RVnames[i]+'_jitter']), axis=0)
        else:
            params=np.concatenate((params,[0.]), axis=0)
            stepsize=np.concatenate((stepsize,[0.]), axis=0)
            pmin=np.concatenate((pmin,[0.]), axis=0)
            pmax=np.concatenate((pmax,[0]), axis=0)
            prior=np.concatenate((prior,[0.]), axis=0)
            priorlow=np.concatenate((priorlow,[0.]), axis=0)
            priorup=np.concatenate((priorup,[0.]), axis=0)
            pnames=np.concatenate((pnames,[RVnames[i]+'_jitter']), axis=0)         
        
    nbc_tot = np.copy(0)  # total number of baseline coefficients let to vary (leastsq OR jumping)

    #################################### GP setup #########################################
    print('Setting up photometry GPs ...')
    GPobjects = []
    GPparams = []
    GPstepsizes = []
    GPindex = []  # this array contains the lightcurve index of the lc it applies to
    GPprior = []
    GPpriwid = []
    GPlimup = []
    GPlimlo =[]
    GPnames = []
    GPcombined = []
    pargps = []

    for i in range(nphot):
        t, flux, err, xshift, yshift, airm, fwhm, sky, eti = np.loadtxt(fpath+names[i], usecols=(0,1,2,3,4,5,6,7,8), unpack = True)  # reading in the data
        if (divwhite=='y'): # if the divide - white is activated, divide the lcs by the white noise model before proceeding
            dwCNM = np.copy(dwCNMarr[dwCNMind[groups[i]-1]])
            flux=np.copy(flux/dwCNM)
        
        sky=sky-np.mean(sky)
        tarr=np.concatenate((tarr,t), axis=0)
        farr=np.concatenate((farr,flux), axis=0)
        earr=np.concatenate((earr,err), axis=0)
        xarr=np.concatenate((xarr,xshift), axis=0)
        yarr=np.concatenate((yarr,yshift), axis=0)
        aarr=np.concatenate((aarr,airm), axis=0)
        warr=np.concatenate((warr,fwhm), axis=0)
        sarr=np.concatenate((sarr,sky), axis=0)
        barr=np.concatenate((barr,np.zeros(len(t),dtype=int)), axis=0)   # bisector array: filled with 0s
        carr=np.concatenate((carr,np.zeros(len(t),dtype=int)), axis=0)   # contrast array: filled with 0s
        lind=np.concatenate((lind,np.zeros(len(t),dtype=int)+i), axis=0)
        indices=np.where(lind==i)
        if verbose: print(lind, indices)
        indlist.append(indices)
        
        pargp = np.vstack((t, xshift, yshift, airm, fwhm, sky, eti)).T  # the matrix with all the possible inputs to the GPs

        if (useGPphot[i]=='n'):
            pargps.append([])   #Akin: to keep the indexes of the lists pargps and GPobjects index correct, append empty list if no gp
            GPobjects.append([])
            A_in,B_in,C1_in,C2_in,D_in,E_in,G_in,H_in,nbc = basecoeff(bases[i])  # the baseline coefficients for this lightcurve; each is a 2D array
            nbc_tot = nbc_tot+nbc # add up the number of jumping baseline coeff
            njumpphot[i]=njumpphot[i]+nbc   # each LC has another jump pm

            # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
            if (baseLSQ == "y"):
                abvar=np.concatenate(([A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]]))
                abind=np.where(abvar!=0.)
                bvars.append(abind)
                A_in[1,:]=B_in[1,:]=C1_in[1,:]=C2_in[1,:]=D_in[1,:]=E_in[1,:]=G_in[1,:]=H_in[1,:]=0                             # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

            # append these to the respective mcmc input arrays
            params=np.concatenate((params,A_in[0,:],B_in[0,:],C1_in[0,:],C2_in[0,:],D_in[0,:],E_in[0,:],G_in[0,:],H_in[0,:]))
            stepsize=np.concatenate((stepsize,A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]))
            pmin=np.concatenate((pmin,A_in[2,:],B_in[2,:],C1_in[2,:],C2_in[2,:],D_in[2,:],E_in[2,:],G_in[2,:],H_in[2,:]))
            pmax=np.concatenate((pmax,A_in[3,:],B_in[3,:],C1_in[3,:],C2_in[3,:],D_in[3,:],E_in[3,:],G_in[3,:],H_in[3,:]))
            prior=np.concatenate((prior, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            priorlow=np.concatenate((priorlow, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            priorup=np.concatenate((priorup, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            pnames=np.concatenate((pnames, [names[i]+'_A0', names[i]+'_A1',names[i]+'_A2',names[i]+'_A3',names[i]+'_A4',names[i]+'_B1',names[i]+'_B2',names[i]+'_C11', names[i]+'_C12',names[i]+'_C21', names[i]+'_C22',names[i]+'_D1',names[i]+'_D2',names[i]+'_E1',names[i]+'_E2',names[i]+'_G1',names[i]+'_G2',names[i]+'_G3',names[i]+'_H1',names[i]+'_H2']))
            # note currently we have the following parameters in these arrays:
            #   [T0,RpRs,b,dur,per,eos, eoc, ddf_1, ..., ddf_n, occ_1, ... , occ_n, c1_f1,c2_f1,c3_f1,c4_f1, c1_f2, .... , c4fn, A0_lc1,A1_lc1,A2_lc0,A3_lc0,A4_lc0, B0_lc1,B1_lc1,C0_lc1,C1_lc1,C2_lc1,C3_lc1,C4_lc1,D0_lc1,D1_lc1,E0_lc1,E1_lc1,G0_lc1,G1_lc1,H0_lc1,H1_lc1,H2_lc1,A0_lc2, ...]
            #    0  1    2  3   4  5   6    |  7 - 4+nddf     |          [7+nddf -- 6+nddf+4*n_filt]       |                           7+nddf+4*n_filt -- 7+nddf+4*n_filt + 15                                                |    7+nddf+4*n_filt + 16
            #    p a r a m e t e r s   |  d d F s        | Limb Darkening                             | const.  time  (5)     |      AM (2)  |     coordinate shifts   (5)      |     FWHM  (2) |   sky  (2)   | SIN (3)  | CNM (2) |
            #    each lightcurve has 21 baseline jump parameters, starting with index  8+nddf+4*n_filt+nRV

        elif (useGPphot[i]=='y'):
            # first, also allocate spots in the params array for the BL coefficients, but set them all to 0/1 and the stepsize to 0
            A_in,B_in,C1_in,C2_in,D_in,E_in,G_in,H_in,nbc = basecoeff(bases[i])  # the baseline coefficients for this lightcurve; each is a 2D array
            nbc_tot = nbc_tot+nbc # add up the number of jumping baseline coeff
            
            # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
            if (baseLSQ == "y"):
                abvar=np.concatenate(([A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]]))
                abind=np.where(abvar!=0.)
                bvars.append(abind)
                A_in[1,:]=B_in[1,:]=C1_in[1,:]=C2_in[1,:]=D_in[1,:]=E_in[1,:]=G_in[1,:]=H_in[1,:]=0                             # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

            params=np.concatenate((params,A_in[0,:],B_in[0,:],C1_in[0,:],C2_in[0,:],D_in[0,:],E_in[0,:],G_in[0,:],H_in[0,:]))
            stepsize=np.concatenate((stepsize,A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]))
            pmin=np.concatenate((pmin,A_in[2,:],B_in[2,:],C1_in[2,:],C2_in[2,:],D_in[2,:],E_in[2,:],G_in[2,:],H_in[2,:]))
            pmax=np.concatenate((pmax,A_in[3,:],B_in[3,:],C1_in[3,:],C2_in[3,:],D_in[3,:],E_in[3,:],G_in[3,:],H_in[3,:]))
            prior=np.concatenate((prior, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            priorlow=np.concatenate((priorlow, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            priorup=np.concatenate((priorup, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            pnames=np.concatenate((pnames, [names[i]+'_A0', names[i]+'_A1',names[i]+'_A2',names[i]+'_A3',names[i]+'_A4',names[i]+'_B1',names[i]+'_B2',names[i]+'_C11', names[i]+'_C12',names[i]+'_C21', names[i]+'_C22',names[i]+'_D1',names[i]+'_D2',names[i]+'_E1',names[i]+'_E2',names[i]+'_G1',names[i]+'_G2',names[i]+'_G3',names[i]+'_H1',names[i]+'_H2']))

            # define the index in the set of filters that this LC has:
            k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
            k = k[0].item()
            # ddf of this LC:
            if (ddfYN=='y'):           
                ddfhere=params[8+k]
            else:
                ddfhere=0.
            
            # define the correct LDCs c1, c2:
            c1here=params[8+nddf+nocc+k*4]
            c2here=params[8+nddf+nocc+k*4+1]
            
            mean_model=Transit_Model(T0=params[0],RpRs=params[1],b=params[2],dur=params[3],per=params[4],eos=params[5],eoc=params[6],ddf=ddfhere,occ=params[8+nddf+k],c1=c1here,c2=c2here)
        
            if GPphotWNstart[i] == GPphotWNstart[0] and GPphotWNstep[i]==0.0:
                d_combined = 1.0
            elif GPphotWN[0] == 'all':
                d_combined = 1.0
            else:
                d_combined = 0.0


            GPparams=np.concatenate((GPparams,GPphotWNstart[i]), axis=0)   # start gppars with the white noise
            GPstepsizes=np.concatenate((GPstepsizes,GPphotWNstep[i]),axis=0)
            GPindex=np.concatenate((GPindex,[i]),axis=0)
            GPprior=np.concatenate((GPprior,GPphotWNprior[i]),axis=0)
            GPpriwid=np.concatenate((GPpriwid,GPphotWNpriorwid[i]),axis=0)
            GPlimup=np.concatenate((GPlimup,GPphotWNlimup[i]),axis=0)
            GPlimlo=np.concatenate((GPlimlo,GPphotWNlimlo[i]),axis=0)
            GPnames=np.concatenate((GPnames,['GPphotWN_lc'+str(i+1)]),axis=0)
            GPcombined=np.concatenate((GPcombined,[d_combined]),axis=0)

            iii=0

            for ii in range(ndimGP):
                if GPall[0,ii] == True:
                    j = 0
                else:
                    j = i

                if GPjumping[j,ii]==True:
                    if iii>0:
                        k2 = k3

                    if (GPphotkerns[j,ii]=='sqexp'):
                        k1 = GPphotpars1[j,ii] * kernels.ExpSquaredKernel(GPphotpars2[j,ii], ndim=ndimGP, axes=ii)  
                    elif (GPphotkerns[j,ii]=='mat32'):
                        k1 = GPphotpars1[j,ii] * kernels.Matern32Kernel(GPphotpars2[j,ii], ndim=ndimGP, axes=ii)  
                    else:
                        _raise(ValueError, 'kernel not recognized! Must be either "sqexp" or "mat32" ')
                        
                    if iii==0:
                        k3 = k1
                    else:
                        k3 = k2 + k1
                        
                    GPparams=np.concatenate((GPparams,(np.log(GPphotpars1[j,ii]),np.log(GPphotpars2[j,ii]))), axis=0)           
                    if GPall[0,ii] == True and i == 0:         
                        GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,ii],GPphotstep2[j,ii])),axis=0)
                    elif GPall[0,ii] == True and i != 0:
                        GPstepsizes=np.concatenate((GPstepsizes,(0.,0.)),axis=0)
                    else:
                        GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,ii],GPphotstep2[j,ii])),axis=0)
                    GPindex=np.concatenate((GPindex,(np.zeros(2)+i)),axis=0)
                    GPprior=np.concatenate((GPprior,(GPphotprior1[j,ii],GPphotprior2[j,ii])),axis=0)
                    GPpriwid=np.concatenate((GPpriwid,(GPphotpriorwid1[j,ii],GPphotpriorwid2[j,ii])),axis=0)
                    GPlimup=np.concatenate((GPlimup,(GPphotlim1up[j,ii],GPphotlim2up[j,ii])),axis=0)
                    GPlimlo=np.concatenate((GPlimlo,(GPphotlim1lo[j,ii],GPphotlim2lo[j,ii])),axis=0)
                    GPnames=np.concatenate((GPnames,(['GPphotscale_lc'+str(i+1)+'dim'+str(ii),"GPphotmetric_lc"+str(i+1)+'dim'+str(ii)])),axis=0)
                    GPcombined=np.concatenate((GPcombined,(GPall[j,ii],GPall[j,ii])),axis=0)

                    iii=iii+1
                    
            gp = GPnew(k3, mean=mean_model,white_noise=GPphotWNstart[i],fit_white_noise=True)
            
  
            # freeze the parameters that are not jumping!
            # indices of the GP mean model parameters in the params model
            pindices = [0,1,2,3,4,5,6,8+k,8+nddf+k,8+nddf+nocc+k*4,8+nddf+nocc+k*4+1]
            
            GPparnames=gp.get_parameter_names(include_frozen=True)
            frz = []
            for ii in range(len(pindices)):
                if (stepsize[pindices[ii]]==0.):
                    gp.freeze_parameter(GPparnames[ii])
                    frz.append(ii)
            if debug: 
                print('\nDEBUG: In fit_data.py') 
                print(f'GP frozen parameters:{np.array(GPparnames)[frz]}')
                print(f'Model+GP parameters to fit: {gp.get_parameter_names()}')
                print(f'with values: {gp.get_parameter_vector()}')

            gp.compute(pargp, err)
            GPobjects.append(gp)
            pargps.append(pargp) 
            
    # ================ MODIFICATIONS: adding celerite ===============
        elif (useGPphot[i]=='ce'):
            pargp = np.copy(t)
            # first, also allocate spots in the params array for the BL coefficients, but set them all to 0/1 and the stepsize to 0
            A_in,B_in,C1_in,C2_in,D_in,E_in,G_in,H_in,nbc = basecoeff(bases[i])  # the baseline coefficients for this lightcurve; each is a 2D array
            nbc_tot = nbc_tot+nbc # add up the number of jumping baseline coeff
            # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
            if (baseLSQ == "y"):
                abvar=np.concatenate(([A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]]))
                abind=np.where(abvar!=0.)
                bvars.append(abind)
                A_in[1,:]=B_in[1,:]=C1_in[1,:]=C2_in[1,:]=D_in[1,:]=E_in[1,:]=G_in[1,:]=H_in[1,:]=0                             # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

            params=np.concatenate((params,A_in[0,:],B_in[0,:],C1_in[0,:],C2_in[0,:],D_in[0,:],E_in[0,:],G_in[0,:],H_in[0,:]))
            stepsize=np.concatenate((stepsize,A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]))
            pmin=np.concatenate((pmin,A_in[2,:],B_in[2,:],C1_in[2,:],C2_in[2,:],D_in[2,:],E_in[2,:],G_in[2,:],H_in[2,:]))
            pmax=np.concatenate((pmax,A_in[3,:],B_in[3,:],C1_in[3,:],C2_in[3,:],D_in[3,:],E_in[3,:],G_in[3,:],H_in[3,:]))
            prior=np.concatenate((prior, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            priorlow=np.concatenate((priorlow, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            priorup=np.concatenate((priorup, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
            pnames=np.concatenate((pnames, [names[i]+'_A0', names[i]+'_A1',names[i]+'_A2',names[i]+'_A3',names[i]+'_A4',names[i]+'_B1',names[i]+'_B2',names[i]+'_C11', names[i]+'_C12',names[i]+'_C21', names[i]+'_C22',names[i]+'_D1',names[i]+'_D2',names[i]+'_E1',names[i]+'_E2',names[i]+'_G1',names[i]+'_G2',names[i]+'_G3',names[i]+'_H1',names[i]+'_H2']))

            # define the index in the set of filters that this LC has:
            k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
            k = k[0].item()
            # ddf of this LC:
            if (ddfYN=='y'):           
                ddfhere=params[8+k]
            else:
                ddfhere=0.
            
            # define the correct LDCs c1, c2:
            c1here=params[8+nddf+nocc+k*4]
            c2here=params[8+nddf+nocc+k*4+1]
            
            mean_model=Transit_Model(T0=params[0],RpRs=params[1],b=params[2],dur=params[3],per=params[4],eos=params[5],eoc=params[6],ddf=ddfhere,occ=params[8+nddf+k],c1=c1here,c2=c2here)
        
            if GPphotWNstart[i] == GPphotWNstart[0] and GPphotWNstep[i]==0.0:
                d_combined = 1.0
            elif GPphotWN[0] == 'all':
                d_combined = 1.0
            else:
                d_combined = 0.0

            # define a Matern 3/2 kernel
            #celerite.terms.Matern32Term(*args, **kwargs)

            c_sigma=np.copy(np.log(GPphotpars1[i][0]))
            c_rho =np.copy(np.log(GPphotpars2[i][0]))
            # #for celerite
            # Q  = 1/np.sqrt(2)
            # w0 = 2*np.pi/(np.exp(c_rho))
            # S0 = np.exp(c_sigma)**2/(w0*Q)
            c_eps=0.001
            if debug: print(f"DEBUG: In fit_data.py - kernel = {GPphotkerns[i,0]} ")
            
            if GPphotWNstep[i]>1e-12:  #if the white nois is jumping

                GPparams=np.concatenate((GPparams,GPphotWNstart[i]), axis=0)   
                GPstepsizes=np.concatenate((GPstepsizes,GPphotWNstep[i]),axis=0)
                GPindex=np.concatenate((GPindex,[i]),axis=0)
                GPprior=np.concatenate((GPprior,GPphotWNprior[i]),axis=0)
                GPpriwid=np.concatenate((GPpriwid,GPphotWNpriorwid[i]),axis=0)
                GPlimup=np.concatenate((GPlimup,GPphotWNlimup[i]),axis=0)
                GPlimlo=np.concatenate((GPlimlo,GPphotWNlimlo[i]),axis=0)
                GPnames=np.concatenate((GPnames,['CEphotWN_lc'+str(i+1)]),axis=0)
                GPcombined=np.concatenate((GPcombined,[d_combined]),axis=0)

                c_WN=np.copy(GPphotWNstart[i])
                bounds_w = dict(log_sigma=(GPphotWNlimlo[i],GPphotWNlimup[i]))
                k1 = terms.JitterTerm(log_sigma=c_WN, bounds=bounds_w)
                if GPphotkerns[i,0] == "mat32":
                    bounds = dict(log_sigma=(GPphotlim1lo[i][0], GPphotlim1up[i][0]), log_rho=(GPphotlim2lo[i][0], GPphotlim2up[i][0]))
                    if debug: print(f"celerite gp bounds = {bounds}, starting: {c_sigma},{c_rho}")
                    k2 = terms.Matern32Term(log_sigma=c_sigma, log_rho=c_rho, bounds=bounds)
                # elif GPphotkerns[i,0] == "sho":
                #     k2 = terms.SHOTerm(log_S0=np.log(S0), log_omega0=np.log(w0), log_Q = np.log(Q))
                #     k2.freeze_parameter("log_Q")
                # else: _raise(ValueError, f'Celerite kernel not recognized! Must be either "sho" or "mat32" but {GPphotkerns[i,0]} given')
                #sho not working, problem with number of parameters (3 expected 2)
                k3=k1 + k2
                NparGP=3

            else:
                bounds = dict(log_sigma=(GPphotlim1lo[i][0], GPphotlim1up[i][0]), log_rho=(GPphotlim2lo[i][0], GPphotlim2up[i][0]))
                if debug: print(f"celerite gp bounds = {bounds}, starting: {c_sigma},{c_rho}")
                if GPphotkerns[i,0] == "mat32": k3 = terms.Matern32Term(log_sigma=c_sigma, log_rho=c_rho, bounds=bounds)
                # elif GPphotkerns[i,0] == "sho": 
                #     k3 = terms.SHOTerm(log_S0=np.log(S0), log_omega0=np.log(w0), log_Q = np.log(Q))
                #     k3.freeze_parameter("log_Q")
                # else: _raise(ValueError, f'Celerite kernel not recognized! Must be either "sho" or "mat32" but {GPphotkerns[i,0]} given')
                NparGP=2


            ii=0 
            if GPall[0,ii] == True:
                j = 0
            else:
                j=i           
            GPparams=np.concatenate((GPparams,[c_sigma,c_rho]), axis=0)             
            if GPall[0,ii] == True and i == 0:         
                GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,ii],GPphotstep2[j,ii])),axis=0)
            elif GPall[0,ii] == True and i != 0:
                GPstepsizes=np.concatenate((GPstepsizes,(0.,0.)),axis=0)
            else:
                GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,ii],GPphotstep2[j,ii])),axis=0)
            GPindex=np.concatenate((GPindex,(np.zeros(2)+i)),axis=0)
            GPprior=np.concatenate((GPprior,(GPphotprior1[j,ii],GPphotprior2[j,ii])),axis=0)
            GPpriwid=np.concatenate((GPpriwid,(GPphotpriorwid1[j,ii],GPphotpriorwid2[j,ii])),axis=0)
            GPlimup=np.concatenate((GPlimup,(GPphotlim1up[j,ii],GPphotlim2up[j,ii])),axis=0)
            GPlimlo=np.concatenate((GPlimlo,(GPphotlim1lo[j,ii],GPphotlim2lo[j,ii])),axis=0)
            GPnames=np.concatenate((GPnames,(['CEphotscale_lc'+str(i+1)+'dim'+str(ii),"CEphotmetric_lc"+str(i+1)+'dim'+str(ii)])),axis=0)
            GPcombined=np.concatenate((GPcombined,(GPall[j,ii],GPall[j,ii])),axis=0)
        

            gp = clnGPnew(k3, mean=mean_model,fit_mean=True)#,log_white_noise=GPphotWNstart[i],fit_white_noise=True)

  
            # freeze the parameters that are not jumping!
            # indices of the GP mean model parameters in the params model
            pindices = [0,1,2,3,4,5,6,8+k,8+nddf+k,8+nddf+nocc+k*4,8+nddf+nocc+k*4+1]

            GPparnames=gp.get_parameter_names(include_frozen=True)
            if debug: print(f"pindices= {pindices}\nGPparnames={GPparnames}")

            frz=[]
            for ii in range(len(pindices)):
                if (stepsize[pindices[ii]]==0.):
                    _ = gp.freeze_parameter(GPparnames[ii+NparGP])
                    # print(f"gppars from main: {GPparnames[ii+NparGP]}")
                    frz.append(ii+NparGP)
            if debug:
                print('\nDEBUG: In fit_data.py') 
                print(f'frz={frz}')
                print(f'GP frozen parameters:{np.array(GPparnames)[frz]}')
                print(f'GP+Model parameters to fit: {gp.get_parameter_names()}')
                print(f'with values: {gp.get_parameter_vector()}')

            gp.compute(pargp, err)
            GPobjects.append(gp)
            pargps.append(pargp)  
        # time.sleep(3000)

    if rv is not None: print('Setting up RV arrays ...')

    for i in range(nRV):
        t, rv, err, bis, fwhm, contrast = np.loadtxt(rv_fpath+RVnames[i], usecols=(0,1,2,3,4,5), unpack = True)  # reading in the data
        
        tarr = np.concatenate((tarr,t), axis=0)
        farr = np.concatenate((farr,rv), axis=0)    # ! add the RVs to the "flux" array !
        earr = np.concatenate((earr,err), axis=0)   # ! add the RV errors to the "earr" array !
        xarr=np.concatenate((xarr,np.zeros(len(t),dtype=int)), axis=0)  # xshift array: filled with 0s
        yarr=np.concatenate((yarr,np.zeros(len(t),dtype=int)), axis=0)  # yshift array: filled with 0s
        aarr=np.concatenate((aarr,np.zeros(len(t),dtype=int)), axis=0)  # airmass array: filled with 0s
        warr=np.concatenate((warr,fwhm), axis=0)
        sarr=np.concatenate((sarr,np.zeros(len(t),dtype=int)), axis=0)  # sky array: filled with 0s
        barr=np.concatenate((barr,bis), axis=0)  # bisector array
        carr=np.concatenate((carr,contrast), axis=0)  # contrast array
        lind=np.concatenate((lind,np.zeros(len(t),dtype=int)+i+nphot), axis=0)   # indices
        indices=np.where(lind==i+nphot)
        indlist.append(indices)
        Pin = sinPs[i]

        
        if (useGPrv[i]=='n'):
            W_in,V_in,U_in,S_in,P_in,nbcRV = basecoeffRV(RVbases[i],Pin)  # the baseline coefficients for this lightcurve; each is a 2D array
            nbc_tot = nbc_tot+nbcRV # add up the number of jumping baseline coeff
            abvar=np.concatenate(([W_in[1,:],V_in[1,:],U_in[1,:],S_in[1,:],P_in[1,:]]))
            abind=np.where(abvar!=0.)
            njumpRV[i] = njumpRV[i]+len(abind)
        
            if (baseLSQ == "y"):
                bvarsRV.append(abind)
                W_in[1,:]=V_in[1,:]=U_in[1,:]=S_in[1,:]=P_in[1,:]=0        # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters
            # append these to the respective mcmc input arrays
            params=np.concatenate((params,W_in[0,:],V_in[0,:],U_in[0,:],S_in[0,:],P_in[0,:]))
            stepsize=np.concatenate((stepsize,W_in[1,:],V_in[1,:],U_in[1,:],S_in[1,:],P_in[1,:]))
            pmin=np.concatenate((pmin,W_in[2,:],V_in[2,:],U_in[2,:],S_in[2,:],P_in[2,:]))
            pmax=np.concatenate((pmax,W_in[3,:],V_in[3,:],U_in[3,:],S_in[3,:],P_in[3,:]))
            prior=np.concatenate((prior, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
            priorlow=np.concatenate((priorlow, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
            priorup=np.concatenate((priorup, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
            pnames=np.concatenate((pnames, [RVnames[i]+'_W1',RVnames[i]+'_W2',RVnames[i]+'_V1',RVnames[i]+'_V2',RVnames[i]+'_U1', RVnames[i]+'_U2',RVnames[i]+'_S1',RVnames[i]+'_S2',RVnames[i]+'_P1',RVnames[i]+'_P2',RVnames[i]+'_P3',RVnames[i]+'_P4']))
            # note currently we have the following parameters in these arrays:
            #   [T0,RpRs,b,dur,per,eos, eoc,K,ddf_1, ..., ddf_n, c1_f1,c2_f1,c3_f1,c4_f1, c1_f2, .... , c4fn, A0_lc1,A1_lc1,A2_lc0,A3_lc0,A4_lc0, B0_lc1,B1_lc1,C0_lc1,C1_lc1,C2_lc1,C3_lc1,C4_lc1,D0_lc1,D1_lc1,E0_lc1,E1_lc1,G0_lc1,G1_lc1,H0_lc1,H1_lc1,H2_lc1,A0_lc2, ...]
            #    0  1    2  3   4  5   6    |  7 - 4+nddf     |          [7+nddf -- 6+nddf+4*n_filt]       |                           7+nddf+4*n_filt -- 7+nddf+4*n_filt + 15                                                |    7+nddf+4*n_filt + 16
            #    p a r a m e t e r s   |  d d F s        | Limb Darkening                             | const.  time  (5)     |      AM (2)  |     coordinate shifts   (5)      |     FWHM  (2) |   sky  (2)   | SIN (3)  | CNM (2) | time_rv (2) | bisector (2) | fwhm_rv (2) | contrast (2)   | sinus (3)
            # each rv curve has 8 baseline jump parameters, starting with index  8+nddf+4*n_filt+nRV + nphot* 21


    # calculate the weights for the lightcurves to be used for the CNM calculation later: do this in a function!
    #ewarr=grweights(earr,indlist,grnames,groups,ngroup)

    for i in range(len(params)):
        if verbose: print(pnames[i], params[i], stepsize[i], pmin[i], pmax[i], priorup[i], priorlow[i])
        
    inmcmc='n'

    LCjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific lc

    #ATTENTION: pass to the lnprob function the individual subscript (of variable p) that are its jump parameters for each LC
    # which indices of p0 are referring to lc n

    for i in range(nphot):
        
        temp=np.ndarray([0])
        
        lcstep1 = np.where(stepsize[0:7]!=0.)  # the common transit jump parameters
        lcstep = lcstep1[0]
        
        if (len(lcstep) > 0): 
            temp=np.copy(lcstep)
        
        # define the index in the set of filters that this LC has:
        k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
        k = k[0].item()

        if (ddfYN=='y'):    
            if temp.shape:    
                temp=np.concatenate((np.asarray(temp),np.asarray([8+k])),axis=0)
            else:
                temp=np.asarray([8+k])

        occind=8+nddf+k 
        if (stepsize[occind]!=0.):
            temp=np.concatenate((np.asarray(temp),[occind]),axis=0)
        
        if verbose: print(temp)

        c1ind=8+nddf+nocc+k*4
        if (stepsize[c1ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[c1ind]),axis=0)
        
        c2ind=8+nddf+nocc+k*4+1
        if (stepsize[c2ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[c2ind]),axis=0)
    
        bfstart= 8+nddf+nocc+nfilt*4 + nRV*2  # the first index in the param array that refers to a baseline function    
        blind = np.asarray(list(range(bfstart+i*20,bfstart+i*20+20)))  # the indices for the coefficients for the base function    
        if verbose: print(bfstart, blind, nocc, nfilt)

        #BUG: this here is not set to the correct indices
        lcstep1 = np.where(stepsize[blind]!=0.)
        
        if (len(lcstep1) > 0): 
            lcstep = lcstep1[0]
            temp=np.concatenate((np.asarray(temp),blind[lcstep]),axis=0)

        #and also add the GPparams
        gind = np.where(GPindex==i)
        gindl = list(gind[0]+len(params))
        gind = gind[0]+len(params)

        if gindl:
            temp = np.concatenate((temp,gind),axis=0)
        
        LCjump.append(temp)
    # print(f"LCjump:{LCjump}")
    RVjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific RV dataset

    for i in range(nRV):
        
        temp=np.ndarray([])
        
        rvstep1 = np.where(stepsize[0:8]!=0.)  # the common RV jump parameters: transit + K
        rvstep = rvstep1[0]
        
        if (len(rvstep) > 0): 
            temp=np.copy(rvstep)

        # identify the gamma index of this RV
        gammaind = 8+nddf+nocc+nfilt*4+i*2
        
        if (stepsize[gammaind]!=0.):           
            temp=np.concatenate((temp,[gammaind]),axis=0)

        bfstart= 8+nddf+nocc+nfilt*4 + nRV*2 + nphot*20  # the first index in the param array that refers to an RV baseline function    
        blind = list(range(bfstart+i*8,bfstart+i*8+8))  # the indices for the coefficients for the base function    

        rvstep = np.where(stepsize[blind]!=0.)
        if rvstep[0]: 
            # temp.append(rvstep)
            temp=np.concatenate((temp,rvstep[0]),axis=0)
        
        RVjump.append(temp)
    # print(f"RVjump:{RVjump}")
    # =============================== CALCULATION ==========================================


    pnames_all = np.concatenate((pnames,GPnames))
    initial = np.concatenate((params,GPparams))
    steps = np.concatenate((stepsize,GPstepsizes))
    priors = np.concatenate((prior,GPprior))
    priwid = (priorup+priorlow)/2.
    priorwids = np.concatenate((priwid,GPpriwid))
    lim_low = np.concatenate((pmin,GPlimlo))
    lim_up = np.concatenate((pmax,GPlimup))
    ndim = np.count_nonzero(steps)
    jumping=np.where(steps!=0.)
    jumping_noGP = np.where(stepsize!=0.)
    jumping_GP = np.where(GPstepsizes!=0.)

    pindices = []
    for i in range(nphot):
        fullist=list(jumping[0])
        lclist=list(LCjump[i])
        both = list(set(fullist).intersection(lclist))  # attention: set() makes it unordered. we'll need to reorder it
        both.sort()
        indices_A = [fullist.index(x) for x in both]
        pindices.append(indices_A)

    for i in range(nRV):
        fullist=list(jumping[0])
        rvlist=list(RVjump[i])
        both = list(set(fullist).intersection(rvlist))  # attention: set() makes it unordered. we'll need to reorder it
        both.sort()
        indices_A = [fullist.index(x) for x in both]
        pindices.append(indices_A)

    ewarr=np.nan#grweights(earr,indlist,grnames,groups,ngroup,nphot)


    ############################## Initial guess ##################################
    print('\nPlotting initial guess\n---------------------------')

    inmcmc = 'n'
    indparams = [tarr,farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,
                nocc,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, 
                cont,names,RVnames,earr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,
                pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,
                jumping_noGP,GPphotWN,jit_apply,jumping_GP,GPstepsizes,GPcombined,useSpline]
    
    debug_t1 = time.time()
    mval, merr,dump1,dump2 = logprob_multi(initial[jumping],*indparams,make_out_file=True,verbose=True,debug=debug)
    if debug: print(f'finished logprob_multi, took {(time.time() - debug_t1)} secs')
    if not os.path.exists("init"): os.mkdir("init")    #folder to put initial plots    
    debug_t2 = time.time()
    mcmc_plots(mval,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr,lind, nphot, nRV, indlist, filters, names, RVnames, 'init/init_',initial,T0_in[0],per_in[0])
    if debug: print(f'finished mcmc_plots, took {(time.time() - debug_t2)} secs')


    ########################### MCMC run ###########################################
    print('\nRunning MCMC')

    inmcmc = 'y'
    indparams = [tarr,farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,
                nocc,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV,
                cont,names,RVnames,earr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,
                pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,
                jumping_noGP,GPphotWN,jit_apply,jumping_GP,GPstepsizes,GPcombined,useSpline]

    print('No of dimensions: ', ndim)
    print('No of chains: ', nchains)
    print('fitting parameters: ', pnames_all[jumping])


    ijnames = np.where(steps != 0.)
    jnames = pnames_all[[ijnames][0]]  # jnames are the names of the jump parameters

    # put starting points for all walkers, i.e. chains
    p0 = np.random.rand(ndim * nchains).reshape((nchains, ndim))*np.asarray(steps[jumping])*2 + (np.asarray(initial[jumping])-np.asarray(steps[jumping]))
    assert np.all([np.isfinite(logprob_multi(p0[i],*indparams)) for i in range(nchains)]),f'Range of start values of a(some) jump parameter(s) are outside the prior distribution'

    if walk == "demc": moves = emcee.moves.DEMove()
    elif walk == "snooker": moves = emcee.moves.DESnookerMove()
    else: moves = emcee.moves.StretchMove()
    sampler = emcee.EnsembleSampler(nchains, ndim, logprob_multi, args=(indparams),pool=Pool(nproc), moves=moves)

    print("\nRunning first burn-in...")
    p0, lp, _ = sampler.run_mcmc(p0, 20, progress=True, **kwargs)

    print("Running second burn-in...")
    p0 = p0[np.argmax(lp)] + steps[jumping] * np.random.randn(nchains, ndim) # this can create problems!
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(p0, burnin, progress=True)
    if save_burnin_chains:
        burnin_chains = sampler.chain

        #save burn-in chains to file
        burnin_chains_dict =  {}
        for ch in range(burnin_chains.shape[2]):
            burnin_chains_dict[jnames[ch]] = burnin_chains[:,:,ch]
        pickle.dump(burnin_chains_dict,open("burnin_chains_dict.pkl","wb"))  
        print("burn-in chain written to disk")
        matplotlib.use('Agg')
        burn_result = load_chains()
        try:
            fig = burn_result.plot_burnin_chains()
            fig.savefig("burnin_chains.png", bbox_inches="tight")
            print("Burn-in chains plot saved as: burnin_chains.png")
        except: 
            print(f"full burn-in chains not plotted (number of parameters ({ndim}) exceeds 20. use result.plot_burnin_chains()")
            print(f"saving burn-in chain plot for the first 20 parameters")
            pl_pars = list(burn_result._par_names)[:20]
            fig = burn_result.plot_burnin_chains(pl_pars)
            fig.savefig("burnin_chains.png", bbox_inches="tight")  

        matplotlib.use(__default_backend__)
    sampler.reset()

    print("\nRunning production...")
    pos, prob, state = sampler.run_mcmc(pos, ppchain,skip_initial_state_check=True, progress=True )
    bp = pos[np.argmax(prob)]

    posterior = sampler.flatchain
    chains = sampler.chain

    print(("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))))
    GRvals = grtest_emcee(chains)

    gr_print(jnames,GRvals)

    nijnames = np.where(steps == 0.)
    njnames = pnames_all[[nijnames][0]]  # njnames are the names of the fixed parameters

    exti = np.intersect1d(pnames_all,extinpars, return_indices=True)
    exti[1].sort()
    extins=np.copy(exti[1])

    #save chains to file
    chains_dict =  {}
    for ch in range(chains.shape[2]):
        chains_dict[jnames[ch]] = chains[:,:,ch]
    pickle.dump(chains_dict,open("chains_dict.pkl","wb"))
    print("Production chain written to disk as `chains_dict.pkl`. Run `result=CONAN3.load_chains()` to load it.")  

    dim=posterior.shape
    # calculate PDFs of the stellar parameters given 
    Rs_PDF = get_PDF_Gauss(Rs_in,sRs_lo,sRs_hi,dim)
    Ms_PDF = get_PDF_Gauss(Ms_in,sMs_lo,sMs_hi,dim)
    extind_PDF = np.zeros((dim[0],len(extins)))
    for i in range(len(extinpars)):
        ind = extins[i]
        par_PDF = get_PDF_Gauss(extcens[ind],extup[ind],extlow[ind],dim)
        extind_PDF[:,i] = par_PDF

    # newparams == initial here are the parameter values as they come in. they get used only for the fixed values
    bpfull = np.copy(initial)
    bpfull[[ijnames][0]] = bp

    medvals,maxvals =mcmc_outputs(posterior,jnames, ijnames, njnames, nijnames, bpfull, ulamdas, Rs_in, Ms_in, Rs_PDF, Ms_PDF, nfilt, filnames, howstellar, extinpars, extins, extind_PDF)

    npar=len(jnames)
    if (baseLSQ == "y"):
        # print("TODO: validate if GPs and leastsquare_for_basepar works together")
        npar = npar + nbc_tot   # add the baseline coefficients if they are done by leastsq

    medp=np.copy(initial)
    medp[[ijnames][0]]=medvals
    maxp=initial
    maxp[[ijnames][0]]=maxvals

    #============================== PLOTTING ===========================================
    print('Plotting output figures')

    inmcmc='n'
    indparams = [tarr,farr,xarr,yarr,warr,aarr,sarr,barr,carr, nphot, nRV, indlist, filters, nfilt,
         filnames,nddf,nocc,rprs0,erprs0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, 
              baseLSQ, bvars, bvarsRV, cont,names,RVnames,earr,divwhite,dwCNMarr,dwCNMind,params,
                  useGPphot,useGPrv,GPobjects,GPparams,GPindex,pindices,jumping,pnames,LCjump, 
                      priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps, 
                          jumping_noGP,GPphotWN,jumping_GP,jit_apply,GPstepsizes,GPcombined,useSpline]
    
    #AKIN: save config parameters indparams and summary_stats and as a hidden files. 
    #can be used to run logprob_multi() to generate out_full.dat files for median posterior, max posterior and best fit values
    pickle.dump(indparams, open(".par_config.pkl","wb"))
    stat_vals = dict(med = medp[jumping], max=maxp[jumping], bf = bpfull[jumping])
    pickle.dump(stat_vals, open(".stat_vals.pkl","wb"))
    
    #median
    mval, merr,T0_post,p_post = logprob_multi(medp[jumping],*indparams,make_out_file=(statistic=="median"), verbose=True,)
    mcmc_plots(mval,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr,lind, nphot, nRV, indlist, filters, names, RVnames, 'med_',medp,T0_post,p_post)

    #max_posterior
    mval2, merr2, T0_post, p_post = logprob_multi(maxp[jumping],*indparams,make_out_file=(statistic=="max"),verbose=False)
    mcmc_plots(mval2,tarr,farr,earr,xarr,yarr,warr,aarr,sarr,barr,carr,lind, nphot, nRV, indlist, filters, names, RVnames, 'max_',maxp, T0_post,p_post)


    maxresiduals = farr - mval2 if statistic != "median" else farr - mval  #Akin allow statistics to be based on median of posterior
    chisq = np.sum(maxresiduals**2/earr**2)

    ndat = len(tarr)
    bic=get_BIC_emcee(npar,ndat,chisq)
    aic=get_AIC_emcee(npar,ndat,chisq)


    rarr=farr-mval2  if statistic != "median" else farr - mval  # the full residuals
    bw, br, brt, cf, cfn = corfac(rarr, tarr, earr, indlist, nphot, njumpphot) # get the beta_w, beta_r and CF and the factor to get redchi2=1

    outfile='CF.dat'

    of=open(outfile,'w')
    for i in range(nphot):   #adapt the error values
    # print(earr[indlist[i][0]])
        of.write('%8.3f %8.3f %8.3f %8.3f %10.6f \n' % (bw[i], br[i], brt[i],cf[i],cfn[i]))
        if (cf_apply == 'cf'):
            earr[indlist[i][0]] = np.multiply(earr[indlist[i][0]],cf[i])
            if verbose: print((earr[indlist[i][0]]))        
        if (cf_apply == 'rchisq'):
            earr[indlist[i][0]] = np.sqrt((earr[indlist[i][0]])**2 + (cfn[i])**2)
            if verbose: print((earr[indlist[i][0]]))

    of.close()

    T0_out = maxp[np.where(pnames_all=='T_0')]  if statistic != "median" else medp[np.where(pnames_all=='T_0')]
    # T0_out_err = sig1[np.where(jnames=='T_0')]

    Period_out = maxp[np.where(pnames_all=='Period_[d]')] if statistic != "median" else medp[np.where(pnames_all=='T_0')]
    # Period_out_err = sig1[np.where(jnames=='Period_[d]')]

    dur_out = maxp[np.where(pnames_all=='dur_[d]')] if statistic != "median" else medp[np.where(pnames_all=='T_0')]
    # dur_out_err = sig1[np.where(jnames=='dur_[d]')]

    
        # ==== corner plot ================

    #### corner plot only if few parameters
    # if len(jnames) <=12:
    #     c=corner.corner(sampler.flatchain, labels=jnames)
    #     c.savefig("corner_GP.png", bbox_inches='tight')
    #     plt.close()
        
    #     ndim, nwalkers = sampler.chain.shape[2], sampler.chain.shape[0]
    #     fig = plt.figure(figsize=(12,15))
    #     for j in range(ndim):
    #         plt.subplot(ndim,1,j+1)
    #         # plt.ylabel(jnames[j])
    #         plt.autoscale(enable=True, axis='x', tight=True)
    #         for i in range(0,nwalkers-1): 
    #             plt.plot(sampler.chain[i,:, j],"k",alpha=0.05, label=f"{jnames[j]}")
    #             if i==0: plt.legend(loc='upper left')
    #         plt.subplots_adjust(hspace=0.01,bottom=0.1, top=0.9)
    #     fig.savefig("chains.png", bbox_inches="tight")

    result = load_chains()

    result.T0 = T0_out
    # result.T0_err = T0_out_err
    result.P = Period_out
    # result.P_err = Period_out_err
    result.dur = dur_out
    # result.dur_err = dur_out_err

    matplotlib.use('Agg')
    try:
        fig = result.plot_chains()
        fig.savefig("chains.png", bbox_inches="tight")
    except:
        print(f"\nfull chains not plotted (number of parameters ({ndim}) exceeds 20. use result.plot_burnin_chains()")
        print(f"saving chain plot for the first 20 parameters")
        pl_pars = list(result._par_names)[:20]
        fig = result.plot_chains(pl_pars)
        fig.savefig("chains.png", bbox_inches="tight")  

    try:
        fig = result.plot_corner()
        fig.savefig("corner.png", bbox_inches="tight")
    except: 
        print(f"\ncorner not plotted (number of parameters ({ndim}) exceeds 14. use result.plot_corner(force_plot=True)")
        print("saving corner plot for the first 14 parameters")
        pl_pars = list(result._par_names)[:14]
        fig = result.plot_corner(pl_pars,force_plot=True)
        fig.savefig("corner.png", bbox_inches="tight")

    matplotlib.use(__default_backend__)

    return result