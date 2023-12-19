from logging import exception
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

import george
from george import kernels, GP
import corner
# from .gpnew import GPnew as GP

import celerite
from celerite import terms
from celerite import GP as cGP
# from .celeritenew import GPnew as clnGPnew
# from .gpnew import *
from copy import deepcopy


from ._classes import _raise, mcmc_setup, __default_backend__, load_result
import matplotlib
matplotlib.use(__default_backend__)

import multiprocessing as mp 
mp.set_start_method('fork')
__all__ = ["fit_data"]

def fit_data(lc, rv=None, mcmc=None, statistic = "median", out_folder="output", progress=True,rerun_result=False,
verbose=False, debug=False, save_burnin_chains=True, **kwargs):
    """
    function to fit the data using the light-curve object lc, rv_object rv and mcmc setup object mcmc.

    Parameters
    ----------
    lc : lightcurve object;
        object containing lightcurve data and setup parameters. 
        see CONAN3.load_lightcurves() for more details.

    rv : rv object
        object containing radial velocity data and setup parameters. 
        see CONAN3.load_rvs() for more details.

    mcmc : mcmc_setup object;
        object containing mcmc setup parameters. 
        see CONAN3.mcmc_setup() for more details.

    statistic : str;
        statistic to run on posteriors to obtain model parameters and create model output file ".._lcout.dat".
        must be one of ["median", "max", "bestfit"], default is "median". 
        "max" and "median" calculate the maximum and median of each parameter posterior respectively while "bestfit" \
            is the parameter combination that gives the maximum joint posterior probability.

    progress : bool;
        if True, show MCMC progress bar, default is True.

    out_folder : str;
        path to output folder, default is "output".

    rerun_result : bool;
        if True, rerun CONAN with previous fit result in order to regenerate plots and files. Default is False.

    verbose : bool;
        if True, print out additional information, default is False.

    debug : bool;
        if True, print out additional debugging information, default is False.

    save_burnin_chains : bool;
        if True, save burn-in chains to file, default is True.

    **kwargs : dict;
        other parameters sent to emcee.EnsembleSampler.run_mcmc() function

    Returns:
    --------
    result : object containing labeled mcmc chains
        Object that contains methods to plot the chains, corner, and histogram of parameters.
        e.g result.plot_chains(), result.plot_burnin_chains(), result.plot_corner, result.plot_posterior("T_0")

    """

    if not os.path.exists(out_folder):
        print(f"Creating output folder...{out_folder}")
        os.mkdir(out_folder)

    if os.path.exists(f'{out_folder}/chains_dict.pkl'):
        if not rerun_result:
            print(f'Fit result already exists in this folder: {out_folder}.\n Loading results...')
            result = load_result(out_folder)
            return result
        else:
            print(f'Fit result already exists in this folder: {out_folder}.\nRerunning with results to generate plots and files...\n')

    print('CONAN fit launched!!!\n') 

        
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

    nphot      = len(names)                                                 # the number of photometry input files
    njumpphot  = np.zeros(nphot)                                            # the number of jumping parameters for each photometry input file
    filnames   = np.array(list(sorted(set(filters),key=filters.index)))     # the unique filter names
    ulamdas    = np.array(list(sorted(set(lamdas),key=lamdas.index)))       # the unique wavelengths
    grnames    = np.array(list(sorted(set(groups))))                        # the unique group names
    
    nfilt      = len(filnames)                                              # the number of unique filters
    ngroup     = len(grnames)                                               # the number of unique groups
    useSpline_lc  = lc._lcspline                                            # use spline to interpolate the light curve
    s_samp        = lc._ss

 #============GP Setup=============================
    #from load_lightcurves.add_GP()
    GPchoices          = ["col0", "col3", "col4", "col5", "col6", "col7", "col8"]# ["time", "xshift", "yshift", "air", "fwhm", "sky", "eti"] #
    ndimGP             = len(GPchoices)    # the number of GP dimensions

    GPphotkerns        = np.zeros((nphot, ndimGP), dtype=object)
    #white noise
    GPphotWN           = np.zeros((nphot, 1), dtype=object)
    GPphotWNstartppm   = 50      # start at 50 ppm 
    GPphotWNstart      = -50 * np.ones((nphot, 1)) # set WN very low in case WN is not used
    GPphotWNstep       = np.zeros((nphot, 1))
    GPphotWNprior      = np.zeros((nphot, 1))
    GPphotWNpriorwid   = np.zeros((nphot, 1))
    GPphotWNlimup      = np.zeros((nphot, 1))
    GPphotWNlimlo      = np.zeros((nphot, 1))
    
    #assumes GP with 2 parameters
    GPphotpars1        = np.zeros((nphot, ndimGP))   # the first GP parameter
    GPphotstep1        = np.zeros((nphot, ndimGP))          # the step size of the first GP parameter
    GPphotprior1       = np.zeros((nphot, ndimGP))          # the prior mean of the first GP parameter
    GPphotpriorwid1    = np.zeros((nphot, ndimGP))          # the prior width of the first GP parameter
    GPphotlim1up       = np.zeros((nphot, ndimGP))          # the upper limit of the first GP parameter
    GPphotlim1lo       = np.zeros((nphot, ndimGP))          # the lower limit of the first GP parameter

    GPphotpars2        = np.zeros((nphot, ndimGP))   # the second GP parameter
    GPphotstep2        = np.zeros((nphot, ndimGP))        # the step size of the second GP parameter
    GPphotprior2       = np.zeros((nphot, ndimGP))       # the prior mean of the second GP parameter 
    GPphotpriorwid2    = np.zeros((nphot, ndimGP))     # the prior width of the second GP parameter
    GPphotlim2up       = np.zeros((nphot, ndimGP))      # the upper limit of the second GP parameter
    GPphotlim2lo       = np.zeros((nphot, ndimGP))      # the lower limit of the second GP parameter

    GPncomponent       = np.zeros((nphot, ndimGP))               # number of components in the kernel
    GPjumping          = np.zeros((nphot, ndimGP), dtype=bool)   # which parameters are jumping
    GPall              = np.zeros((nphot, ndimGP), dtype=bool)   # Joint hyperparameters

    DA_gp = lc._GP_dict         #load input gp parameters from dict

    for j, nm in enumerate(DA_gp["lc_list"]):

        k = np.where(np.array(GPchoices) == DA_gp["pars"][j])   # get index of the GP parameter name in the GPchoices array

        if nm == 'all':    # if the GP is applied to all LCs
            i = np.where(np.array(names) == nm)
            GPjumping[0, k]        = True
            GPphotkerns[0, k]      = DA_gp["kernels"][j]
            GPphotpars1[0, k]      = float(DA_gp["scale"][j])
            GPphotstep1[0, k]      = float(DA_gp["s_step"][j])
            GPphotprior1[0, k]     = float(DA_gp["s_pri"][j])
            GPphotpriorwid1[0, k]  = float(DA_gp["s_pri_wid"][j])
            GPphotlim1up[0, k]     = float(DA_gp["s_up"][j])
            GPphotlim1lo[0, k]     = float(DA_gp["s_lo"][j])
            GPphotpars2[0, k]      = float(DA_gp["metric"][j])
            GPphotstep2[0, k]      = float(DA_gp["m_step"][j])
            GPphotprior2[0, k]     = float(DA_gp["m_pri"][j])
            GPphotpriorwid2[0, k]  = float(DA_gp["m_pri_wid"][j])
            GPphotlim2up[0, k]     = float(DA_gp["m_up"][j])
            GPphotlim2lo[0, k]     = float(DA_gp["m_lo"][j])
            GPall[0, k]            = True

            if DA_gp["WN"][j] == 'y' and useGPphot[i]=='ce':
                GPphotWNstart[0]    = -8.   
                GPphotWNstep[0]     = 0.1 
                GPphotWNprior[0]    = 0.
                GPphotWNpriorwid[0] = 0.
                GPphotWNlimup[0]    = -5
                GPphotWNlimlo[0]    = -12
                GPphotWN[0]         = 'all'

            elif DA_gp["WN"][j] == 'y' and useGPphot[i]=='y':
                GPphotWNstart[:]    = np.log((GPphotWNstartppm/1e6)**2) # in absolute
                GPphotWNstep[0]     = 0.1
                GPphotWNprior[0]    = 0.0
                GPphotWNpriorwid[0] = 0.
                GPphotWNlimup[0]    = -3
                GPphotWNlimlo[0]    = -25.0
                GPphotWN[0]         = 'all'
            elif DA_gp["WN"][j] == 'n':
                GPphotWN[:] = 'n'
            else:
                raise ValueError('For at least one GP an invalid White-Noise option input was provided. Set it to either y or n.')

        else: 
            i = np.where(np.array(names) == nm)                 # get index of the LC name in the names array
            GPjumping[i,k]       = True                         #lcname nm has GP jumping in GPparameter k of the GPchoices
            GPphotkerns[i,k]     = DA_gp["kernels"][j]
            GPphotpars1[i,k]     = float(DA_gp["scale"][j])
            GPphotstep1[i,k]     = float(DA_gp["s_step"][j])
            GPphotprior1[i,k]    = float(DA_gp["s_pri"][j])
            GPphotpriorwid1[i,k] = float(DA_gp["s_pri_wid"][j])
            GPphotlim1up[i,k]    = float(DA_gp["s_up"][j])
            GPphotlim1lo[i,k]    = float(DA_gp["s_lo"][j])

            GPphotpars2[i,k]     = float(DA_gp["metric"][j])
            GPphotstep2[i,k]     = float(DA_gp["m_step"][j])
            GPphotprior2[i,k]    = float(DA_gp["m_pri"][j])
            GPphotpriorwid2[i,k] = float(DA_gp["m_pri_wid"][j])
            GPphotlim2up[i,k]    = float(DA_gp["m_up"][j])
            GPphotlim2lo[i,k]    = float(DA_gp["m_lo"][j])
            GPall[i,k]=False

            if DA_gp["WN"][j] == 'y' and useGPphot[i[0][0]] == 'ce':      # if WN is used and celerite is used for the GP
                GPphotWNstart[i[0][0]]    = -8.
                GPphotWNstep[i[0][0]]     = 0.1
                GPphotWNprior[i[0][0]]    = 0.
                GPphotWNpriorwid[i[0][0]] = 0.
                GPphotWNlimup[i[0][0]]    = -5
                GPphotWNlimlo[i[0][0]]    = -12
                GPphotWN[i[0][0]]         = DA_gp["WN"][j]
            elif DA_gp["WN"][j] == 'y' and useGPphot[i[0][0]] == 'y':    # if WN is used and george is used for the GP
                GPphotWNstart[i]    = np.log((GPphotWNstartppm/1e6)**2) # in absolute
                GPphotWNstep[i]     = 0.1
                GPphotWNprior[i]    = 0.0
                GPphotWNpriorwid[i] = 0.
                GPphotWNlimup[i]    = -3
                GPphotWNlimlo[i]    = -21.0
                GPphotWN[i]         = 'y'
            elif DA_gp["WN"][j] == 'n':
                GPphotWN[:] = 'n'
            else:
                raise ValueError('For at least one GP an invalid White-Noise option input was provided. Set it to either y or n.')


#============rv_data========================== 
    # from load_rvs
    if rv is not None and rv._names == []: rv = None   
    RVnames  = [] if rv is None else rv._names
    RVbases  = [] if rv is None else rv._RVbases
    sinPs    = [] if rv is None else rv._sinPs
    rv_fpath = [] if rv is None else rv._fpath
    nRV      = len(RVnames)             # the number of RV input files
    njumpRV  = np.zeros(nRV)        # the number of jumping parameters for each RV input file
    useSpline_rv  = [] if rv is None else rv._rvspline                                                 # use spline to interpolate the light curve

    extinpars= []               # set up array to contain the names of the externally input parameters
    
    for i in range(nRV):
        if (float(rv._gamsteps[i]) != 0.) :
            njumpRV[i]=njumpRV[i]+1

 
#============transit and RV jump parameters===============
    #from load_lightcurves.setup_transit_rv()

    CP  = deepcopy(lc._config_par)    #load input transit and RV parameters from dict
    npl = lc._nplanet

    # # rho_star (same for all planets)
    rho_star = CP[f"pl{1}"]["rho_star"]
    if rho_star.step_size != 0.: njumpphot=njumpphot+1
    if (rho_star.to_fit == 'n' and rho_star.prior == 'p'):   
        extinpars.append('rho_star')
        
    for n in range(1,npl+1): 
        if rv is None: #remove k as a free parameter
            CP[f"pl{n}"]['K'].to_fit = "n"
            CP[f"pl{n}"]['K'].step_size = 0

        # # rprs   
        if CP[f"pl{n}"]['RpRs'].step_size != 0.: njumpphot=njumpphot+1                 # if step_size is 0, then rprs is not a jumping parameter
        if (CP[f"pl{n}"]['RpRs'].to_fit == 'n' and CP[f"pl{n}"]['RpRs'].prior == 'p'):   # if to_fit is 'n' and prior is 'p', then rprs is an externally input parameter
            extinpars.append('RpRs')
        
        # # imp par     
        if CP[f"pl{n}"]['Impact_para'].step_size != 0.: njumpphot=njumpphot+1
        if (CP[f"pl{n}"]['Impact_para'].to_fit == 'n' and CP[f"pl{n}"]['Impact_para'].prior == 'p'):
            extinpars.append('Impact_para')


        # # T0   
        if CP[f"pl{n}"]['T_0'].step_size != 0.: 
            njumpphot=njumpphot+1
            njumpRV=njumpRV+1
        if (CP[f"pl{n}"]['T_0'].to_fit == 'n' and CP[f"pl{n}"]['T_0'].prior == 'p'):
            extinpars.append('T_0')
            
        
        # # per    
        if CP[f"pl{n}"]['Period'].step_size != 0.: 
            njumpphot=njumpphot+1
            njumpRV=njumpRV+1
        if (CP[f"pl{n}"]['Period'].to_fit == 'n' and CP[f"pl{n}"]['Period'].prior == 'p'):
            extinpars.append('Period')

        # #ecc
        if CP[f"pl{n}"]['Eccentricity'].step_size != 0.: njumpRV=njumpRV+1
        if (CP[f"pl{n}"]['Eccentricity'].to_fit == 'n' and CP[f"pl{n}"]['Eccentricity'].prior == 'p'):
            _raise(ValueError, 'cant externally input eccentricity at this time!')

        # #omega
        for key,val in CP[f"pl{n}"]["omega"].__dict__.items():   #convert to radians
            if isinstance(val, (float,int)): CP[f"pl{n}"]["omega"].__dict__[key] *= np.pi/180
        if CP[f"pl{n}"]['omega'].step_size != 0.: njumpRV=njumpRV+1
        if (CP[f"pl{n}"]['omega'].to_fit == 'n' and CP[f"pl{n}"]['omega'].prior == 'p'):
            _raise(ValueError, 'cant externally input eccentricity at this time!')
            
        # #K
        for key,val in CP[f"pl{n}"]["K"].__dict__.items(): #convert K to km/s
            if isinstance(val, (float,int)): CP[f"pl{n}"]["K"].__dict__[key] /= 1000 
        if CP[f"pl{n}"]['K'].step_size != 0.: njumpRV=njumpRV+1
        if (CP[f"pl{n}"]['K'].to_fit == 'n' and CP[f"pl{n}"]['K'].prior == 'p'):
            extinpars.append('K')

        # adapt the eccentricity and omega jump parameters sqrt(e)*sin(o), sqrt(e)*cos(o)
        if ((CP[f"pl{n}"]['Eccentricity'].prior == 'y' and CP[f"pl{n}"]['omega'].prior == 'n') or (CP[f"pl{n}"]['Eccentricity'].prior == 'n' and CP[f"pl{n}"]['omega'].prior == 'y')):
            _raise(ValueError,'priors on eccentricity and omega: either both on or both off')
            
        CP[f"pl{n}"]["sesin(w)"], CP[f"pl{n}"]["secos(w)"] = ecc_om_par(CP[f"pl{n}"]["Eccentricity"], CP[f"pl{n}"]["omega"])

        #now remove rho_star, Eccentricty and omega from the dictionary 
        _ = [CP[f"pl{n}"].pop(key) for key in ["rho_star","Eccentricity", "omega"]]

        
    
#============ddfs ==========================
    #from load_lighcurves.transit_depth_variations()
    drprs_op = lc._ddfs.drprs_op    # drprs options --> [0., step, bounds_lo, bounds_hi, 0., width_lo, width_hi]
    divwhite = lc._ddfs.divwhite    # do we do a divide-white?
    ddfYN    = lc._ddfs.ddfYN       # do we do a depth-dependent fit?
                
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

   
#============phasecurve setup=============
    #from load_lightcurves.setup_phasecurve()

    DA_occ = lc._PC_dict["D_occ"]
    DA_Apc = lc._PC_dict["A_pc"]
    DA_off = lc._PC_dict["ph_off"]

    nocc     = len(filnames) #len(DA_occ["filters_occ"])
    occ_in   = np.zeros((nocc,7))
    Apc_in   = np.zeros((nocc,7))
    phoff_in = np.zeros((nocc,7))

    for i, f in enumerate(filnames):
        k = np.where(np.array(lc._filters)== f)     #  get indices where the filter name is the same as the one in the input file

        occ_in[i,:] = [DA_occ[f].start_value, DA_occ[f].step_size, DA_occ[f].bounds_lo, DA_occ[f].bounds_hi,
                        DA_occ[f].prior_mean, DA_occ[f].prior_width_lo, DA_occ[f].prior_width_hi ]           
        if DA_occ[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1


        Apc_in[i,:] = [DA_Apc[f].start_value, DA_Apc[f].step_size, DA_Apc[f].bounds_lo, DA_Apc[f].bounds_hi,
                        DA_Apc[f].prior_mean, DA_Apc[f].prior_width_lo, DA_Apc[f].prior_width_hi ]           
        if DA_Apc[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1
    

        phoff_in[i,:] = [DA_off[f].start_value, DA_off[f].step_size, DA_off[f].bounds_lo, DA_off[f].bounds_hi,
                        DA_off[f].prior_mean, DA_off[f].prior_width_lo, DA_off[f].prior_width_hi ]           
        if DA_off[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1
    
    


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
        c1_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step1"][i] == 0.) else c1_in[j,5])   #sig_lo
        c1_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step1"][i] == 0.) else c1_in[j,6])   #sig_hi
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
            hi_lim1 = 2.*c1_in[j,3]+c2_in[j,3]  if c1_in[j,3] else 0   #transform bound_hi
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
    if np.any([spl.use for spl in useSpline_lc]): print('Setting up Spline fitting for LCS ...')                            
    t_arr      = np.array([])  # initializing array with all timestamps (col0)
    f_arr      = np.array([])  # initializing array with all flux values(col1)
    e_arr      = np.array([])  # initializing array with all error values(col2)
    col3_arr   = np.array([])  # initializing array with all col4 values (prev xarr)
    col4_arr   = np.array([])  # initializing array with all col4 values (prev yarr)
    col5_arr   = np.array([])  # initializing array with all col5 values (prev aarr)
    col6_arr   = np.array([])  # initializing array with all col6 values (prev warr)
    col7_arr   = np.array([])  # initializing array with all col7 values (prev sarr)

    lind       = np.array([])  # initializing array with the lightcurve indices
    bis_arr    = np.array([])  # initializing array with all bisector values
    contr_arr  = np.array([])  # initializing array with all contrast values

    indlist    = []   # the list of the array indices
    bvars      = []   # a list that will contain lists of [0, 1] for each of the baseline parameters, for each of the LCs. 0 means it's fixed. 1 means it's variable
    bvarsRV    = []   # a list that will contain lists of [0, 1] for each of the baseline parameters, for each of the RV curves. 0 means it's fixed. 1 means it's variable


    if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the value specified at the jump parameters, and fix it. 
        # the RpRs options [start,step,bound_lo,bound_hi,pri_mean,pri_widthlo,pri_widthhi] 
        for key in ["step_size","bounds_lo","bounds_hi","prior_mean","prior_width_lo","prior_width_hi"]:
            for n in range(1,npl+1):
                CP[f"pl{n}"]["RpRs"].__dict__[key]=0 if key!="bounds_hi" else 1
        nddf = nfilt
    else:
        nddf = 0

    pnames   = ['T_0', 'RpRs', 'Impact_para', 'Period', 'sesin(w)', 'secos(w)', 'K'] # Parameter names
    # reorder dictionary CP as above
    for n in range(1,npl+1): 
        CP[f"pl{n}"] = {key:CP[f"pl{n}"][key] for key in pnames}
    # set up the parameters. loop thorugh the planet number (n) and the parameter name (key) for that planet
    params   = np.concatenate(([rho_star.start_value],   [CP[f"pl{n}"][key].start_value    for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # initial guess params
    stepsize = np.concatenate(([rho_star.step_size],     [CP[f"pl{n}"][key].step_size      for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # stepsizes
    pmin     = np.concatenate(([rho_star.bounds_lo],     [CP[f"pl{n}"][key].bounds_lo      for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Boundaries (min)
    pmax     = np.concatenate(([rho_star.bounds_hi],     [CP[f"pl{n}"][key].bounds_hi      for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Boundaries (max)
    prior    = np.concatenate(([rho_star.prior_mean],    [CP[f"pl{n}"][key].prior_mean     for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Prior centers
    priorlow = np.concatenate(([rho_star.prior_width_lo],[CP[f"pl{n}"][key].prior_width_lo for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Prior sigma low side
    priorup  = np.concatenate(([rho_star.prior_width_hi],[CP[f"pl{n}"][key].prior_width_hi for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Prior sigma high side
    pnames   = np.concatenate((["rho_star"],             [nm+(f"_{n}" if npl>1 else "")    for n in range(1,npl+1)  for nm in pnames]))

    extcens  = np.concatenate(([rho_star.prior_mean],     [CP[f"pl{n}"][key].prior_mean     for n in range(1,npl+1)  for key in CP[f"pl1"].keys()])) # External parameter prior mean
    extlow   = np.concatenate(([rho_star.prior_width_lo], [CP[f"pl{n}"][key].prior_width_lo for n in range(1,npl+1)  for key in CP[f"pl1"].keys()])) # External parameter prior sig lo
    extup    = np.concatenate(([rho_star.prior_width_hi], [CP[f"pl{n}"][key].prior_width_hi for n in range(1,npl+1)  for key in CP[f"pl1"].keys()])) # External parameter prior sig hi
    #set 5th(secin(w)) and 6th(secos(w)) element of each planet parameter array to 0
    for n in range(npl):
        extcens[5+7*n : 7+7*n] = 0
        extlow[5+7*n  : 7+7*n] = 0
        extup[5+7*n   : 7+7*n] = 0


    if (divwhite=='y'):           # do we do a divide-white? If yes, then fix all the transit shape parameters (index 0 + index 1-6)
        #fixing index 0, rho_star 
        stepsize[0] = 0
        prior[0]    = 0
        for n in range(npl):
            stepsize[1+7*n : 7+7*n] = 0     #index 1-6
            prior[1+7*n : 7+7*n]    = 0

    if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the specified value, and fix it.
        drprs_in  = np.zeros((nfilt,7))
        njumpphot = njumpphot+1   # each LC has another jump pm

        for i in range(nfilt):  # and make an array with the drprs inputs  # the dRpRs options
            drprs_in[i,:] = drprs_op     #[0., step, bounds_lo, bounds_hi, 0., width_lo, width_hi]
            params        = np.concatenate((params,   [drprs_in[i,0]]))     # add them to the parameter arrays    
            stepsize      = np.concatenate((stepsize, [drprs_in[i,1]]))
            pmin          = np.concatenate((pmin,     [drprs_in[i,2]]))
            pmax          = np.concatenate((pmax,     [drprs_in[i,3]]))
            prior         = np.concatenate((prior,    [drprs_in[i,4]]))
            priorlow      = np.concatenate((priorlow, [drprs_in[i,5]]))
            priorup       = np.concatenate((priorup,  [drprs_in[i,6]]))
            pnames        = np.concatenate((pnames,   [filnames[i]+'_dRpRs']))
            

    for i in range(nfilt):  # add the occultation depths
        params     = np.concatenate((params,   [occ_in[i,0]]))
        stepsize   = np.concatenate((stepsize, [occ_in[i,1]]))
        pmin       = np.concatenate((pmin,     [occ_in[i,2]]))
        pmax       = np.concatenate((pmax,     [occ_in[i,3]]))
        prior      = np.concatenate((prior,    [occ_in[i,4]]))
        priorlow   = np.concatenate((priorlow, [occ_in[i,5]]))
        priorup    = np.concatenate((priorup,  [occ_in[i,6]]))
        pnames     = np.concatenate((pnames,   [filnames[i]+'_DFocc']))

    for i in range(nfilt):  # add the pc amplitudes
        params     = np.concatenate((params,   [Apc_in[i,0]]))
        stepsize   = np.concatenate((stepsize, [Apc_in[i,1]]))
        pmin       = np.concatenate((pmin,     [Apc_in[i,2]]))
        pmax       = np.concatenate((pmax,     [Apc_in[i,3]]))
        prior      = np.concatenate((prior,    [Apc_in[i,4]]))
        priorlow   = np.concatenate((priorlow, [Apc_in[i,5]]))
        priorup    = np.concatenate((priorup,  [Apc_in[i,6]]))
        pnames     = np.concatenate((pnames,   [filnames[i]+'_Apc']))

    for i in range(nfilt):  # add the phase offsets
        params     = np.concatenate((params,   [phoff_in[i,0]]))
        stepsize   = np.concatenate((stepsize, [phoff_in[i,1]]))
        pmin       = np.concatenate((pmin,     [phoff_in[i,2]]))
        pmax       = np.concatenate((pmax,     [phoff_in[i,3]]))
        prior      = np.concatenate((prior,    [phoff_in[i,4]]))
        priorlow   = np.concatenate((priorlow, [phoff_in[i,5]]))
        priorup    = np.concatenate((priorup,  [phoff_in[i,6]]))
        pnames     = np.concatenate((pnames,   [filnames[i]+'_ph_off']))


    for i in range(nfilt):  # add the LD coefficients for the filters to the parameters
        params     = np.concatenate((params,   [c1_in[i,0], c2_in[i,0], c3_in[i,0], c4_in[i,0]]))
        stepsize   = np.concatenate((stepsize, [c1_in[i,1], c2_in[i,1], c3_in[i,1], c4_in[i,1]]))
        pmin       = np.concatenate((pmin,     [c1_in[i,2], c2_in[i,2], c3_in[i,2], c4_in[i,2]]))
        pmax       = np.concatenate((pmax,     [c1_in[i,3], c2_in[i,3], c3_in[i,3], c4_in[i,3]]))
        prior      = np.concatenate((prior,    [c1_in[i,4], c2_in[i,4], c3_in[i,4], c4_in[i,4]]))
        priorlow   = np.concatenate((priorlow, [c1_in[i,5], c2_in[i,5], c3_in[i,5], c4_in[i,5]]))
        priorup    = np.concatenate((priorup,  [c1_in[i,6], c2_in[i,6], c3_in[i,6], c4_in[i,6]]))
        pnames     = np.concatenate((pnames,   [filnames[i]+'_c1',filnames[i]+'_c2',filnames[i]+'_c3',filnames[i]+'_c4']))

    for i in range(nRV):
        params      = np.concatenate((params,  [rv._gammas[i]]),   axis=0)
        stepsize    = np.concatenate((stepsize,[rv._gamsteps[i]]), axis=0)
        pmin        = np.concatenate((pmin,    [-1000]), axis=0)
        pmax        = np.concatenate((pmax,    [1000]),  axis=0)
        prior       = np.concatenate((prior,   [rv._gampri[i]]),   axis=0)
        priorlow    = np.concatenate((priorlow,[rv._gamprilo[i]]), axis=0)
        priorup     = np.concatenate((priorup, [rv._gamprihi[i]]), axis=0)
        # pnames      = np.concatenate((pnames,  [RVnames[i]+'_gamma']), axis=0)
        pnames      = np.concatenate((pnames,  [f"rv{i+1}_gamma"]), axis=0)

        
        if (jit_apply=='y'):
            # print('does jitter work?')
            # print(nothing)
            params      = np.concatenate((params,  [0.01]), axis=0)
            stepsize    = np.concatenate((stepsize,[0.001]), axis=0)
            pmin        = np.concatenate((pmin,    [0.]), axis=0)
            pmax        = np.concatenate((pmax,    [100]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            # pnames      = np.concatenate((pnames,  [RVnames[i]+'_jitter']), axis=0)
            pnames      = np.concatenate((pnames,  [f"rv{i+1}_jitter"]), axis=0)

        else:
            params      = np.concatenate((params,  [0.]), axis=0)
            stepsize    = np.concatenate((stepsize,[0.]), axis=0)
            pmin        = np.concatenate((pmin,    [0.]), axis=0)
            pmax        = np.concatenate((pmax,    [0]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            # pnames      = np.concatenate((pnames,  [RVnames[i]+'_jitter']), axis=0)     
            pnames      = np.concatenate((pnames,  [f"rv{i+1}_jitter"]), axis=0)
    
        
    nbc_tot = np.copy(0)  # total number of baseline coefficients let to vary (leastsq OR jumping)

    #################################### GP setup #########################################
    print('Setting up photometry GPs ...')
    GPobjects   = []
    GPparams    = []
    GPstepsizes = []
    GPindex     = []  # this array contains the lightcurve index of the lc it applies to
    GPprior     = []
    GPpriwid    = []
    GPlimup     = []
    GPlimlo     = []
    GPnames     = []
    GPcombined  = []
    pargps      = []  #list to hold independent variables of GP for each lc

    for i in range(nphot):
        t, flux, err, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in = np.loadtxt(fpath+names[i], usecols=(0,1,2,3,4,5,6,7,8), unpack = True)  # reading in the data
        if (divwhite=='y'): # if the divide - white is activated, divide the lcs by the white noise model before proceeding
            dwCNM = np.copy(dwCNMarr[dwCNMind[groups[i]-1]])
            flux=np.copy(flux/dwCNM)
                
        col7_in   = col7_in - np.mean(col7_in)
        t_arr     = np.concatenate((t_arr,    t),       axis=0)
        f_arr     = np.concatenate((f_arr,    flux),    axis=0)
        e_arr     = np.concatenate((e_arr,    err),     axis=0)
        col3_arr  = np.concatenate((col3_arr, col3_in), axis=0)
        col4_arr  = np.concatenate((col4_arr, col4_in), axis=0)
        col5_arr  = np.concatenate((col5_arr, col5_in), axis=0)
        col6_arr  = np.concatenate((col6_arr, col6_in), axis=0)
        col7_arr  = np.concatenate((col7_arr, col7_in), axis=0)  #TODO: we can have column 8 right?
        
        bis_arr   = np.concatenate((bis_arr, np.zeros(len(t), dtype=int)),   axis=0)   # bisector array: filled with 0s
        contr_arr = np.concatenate((contr_arr, np.zeros(len(t), dtype=int)), axis=0)   # contrast array: filled with 0s
        lind      = np.concatenate((lind, np.zeros(len(t), dtype=int) + i),  axis=0)   # lightcurve index array: filled with i
        indices   = np.where(lind == i)
        if verbose: print(lind, indices)
        indlist.append(indices)
        
        #baseline parameters
        # first, also allocate spots in the params array for the BL coefficients, but set them all to 0/1 and the stepsize to 0
        A_in,B_in,C1_in,C2_in,D_in,E_in,G_in,H_in,nbc = basecoeff(bases[i],useSpline_lc[i])  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot      = nbc_tot+nbc # add up the number of jumping baseline coeff
        njumpphot[i] = njumpphot[i]+nbc   # each LC has another jump pm

        # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
        if (baseLSQ == "y"):
            abvar=np.concatenate(([A_in[1,:],B_in[1,:],C1_in[1,:],C2_in[1,:],D_in[1,:],E_in[1,:],G_in[1,:],H_in[1,:]]))
            abind=np.where(abvar!=0.)
            bvars.append(abind)
            A_in[1,:]=B_in[1,:]=C1_in[1,:]=C2_in[1,:]=D_in[1,:]=E_in[1,:]=G_in[1,:]=H_in[1,:]=0                             # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

        # append these to the respective mcmc input arrays
        params    = np.concatenate((params,   A_in[0,:], B_in[0,:], C1_in[0,:], C2_in[0,:], D_in[0,:], E_in[0,:], G_in[0,:], H_in[0,:]))
        stepsize  = np.concatenate((stepsize, A_in[1,:], B_in[1,:], C1_in[1,:], C2_in[1,:], D_in[1,:], E_in[1,:], G_in[1,:], H_in[1,:]))
        pmin      = np.concatenate((pmin,     A_in[2,:], B_in[2,:], C1_in[2,:], C2_in[2,:], D_in[2,:], E_in[2,:], G_in[2,:], H_in[2,:]))
        pmax      = np.concatenate((pmax,     A_in[3,:], B_in[3,:], C1_in[3,:], C2_in[3,:], D_in[3,:], E_in[3,:], G_in[3,:], H_in[3,:]))
        prior     = np.concatenate((prior,    np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        priorlow  = np.concatenate((priorlow, np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        priorup   = np.concatenate((priorup,  np.zeros(len(A_in[0,:])+len(B_in[0,:])+len(C1_in[0,:])+len(C2_in[0,:])+len(D_in[0,:])+len(E_in[0,:])+len(G_in[0,:])+len(H_in[0,:]))))
        # pnames    = np.concatenate((pnames, [names[i]+'_A0', names[i]+'_A1', names[i]+'_A2', names[i]+'_A3', names[i]+'_A4', names[i]+'_B1', names[i]+'_B2', names[i]+'_C11', names[i]+'_C12', names[i]+'_C21', names[i]+'_C22', names[i]+'_D1', names[i]+'_D2', names[i]+'_E1', names[i]+'_E2', names[i]+'_G1', names[i]+'_G2', names[i]+'_G3', names[i]+'_H1', names[i]+'_H2']))
        pnames   = np.concatenate((pnames, [f"lc{i+1}_off",f"lc{i+1}_A0",f"lc{i+1}_B0",f"lc{i+1}_C0",f"lc{i+1}_D0",
                                            f"lc{i+1}_A3",f"lc{i+1}_B3",
                                            f"lc{i+1}_A4",f"lc{i+1}_B4",
                                            f"lc{i+1}_A5",f"lc{i+1}_B5",
                                            f"lc{i+1}_A6",f"lc{i+1}_B6",
                                            f"lc{i+1}_A7",f"lc{i+1}_B7",
                                            f"lc{i+1}_sin_amp",f"lc{i+1}_sin_per",f"lc{i+1}_sin_off",
                                            f"lc{i+1}_ACNM",f"lc{i+1}_BCNM"
                                            ]))        
        # note currently we have the following parameters in these arrays:
        #   [rho_star,                                   (1)
        #   [T0,RpRs,b,per,eos, eoc,K,                   (7)*npl
        #   ddf_1, ..., ddf_n,                           (nddf)
        #   (occ_1,Apc_1,phoff_1),...,occ_n,Apc_n,phoff_n(3*nocc)
        #   c1_f1,c2_f1,c3_f1,c4_f1, c1_f2, .... , c4fn, (4*n_filt)
        #   Rv_gamma, RV_jit                              (2*nRVs)         
        #   A0_lc1,A1_lc1,A2_lc1,A3_lc1,A4_lc1,           (5)
        #   B0_lc1,B1_lc1,                                (2)
        #   C0_lc1,C1_lc1,C2_lc1,C3_lc1,C4_lc1,           (4)
        #   D0_lc1,D1_lc1,                                (2)
        #   E0_lc1,E1_lc1,                                (2)
        #   G0_lc1,G1_lc1,                                (3)
        #   H0_lc1,H1_lc1,H2_lc1,                         (2)
        #   A0_lc2, ...]
        #    = 1+7*npl+nddf+nocc*3+4*n_filt+2*nRV
        #    each lightcurve has 20 possible baseline jump parameters, starting with index  1+7*npl+nddf+nocc*3+4*n_filt+2*nRV

        pargp_all = np.vstack((t, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in)).T  # the matrix with all the possible inputs to the GPs

        if (useGPphot[i]=='n'):
            pargps.append([])   #to keep the indices of the lists, pargps and GPobjects, correct, append empty list if no gp
            GPobjects.append([])

        elif (useGPphot[i]=='y'):
            # define the index in the set of filters that this LC has:
            # k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
            # k = k[0].item()
        
            if GPphotWNstart[i] == GPphotWNstart[0] and GPphotWNstep[i]==0.0:
                d_combined = 1.0
            elif GPphotWN[0] == 'all':
                d_combined = 1.0
            else:
                d_combined = 0.0

            #add white noise setup tp GP arrays
            GPparams    = np.concatenate((GPparams,GPphotWNstart[i]), axis=0)   # start gppars with the white noise
            GPstepsizes = np.concatenate((GPstepsizes,GPphotWNstep[i]),axis=0)
            GPindex     = np.concatenate((GPindex,[i]),axis=0)
            GPprior     = np.concatenate((GPprior,GPphotWNprior[i]),axis=0)
            GPpriwid    = np.concatenate((GPpriwid,GPphotWNpriorwid[i]),axis=0)
            GPlimup     = np.concatenate((GPlimup,GPphotWNlimup[i]),axis=0)
            GPlimlo     = np.concatenate((GPlimlo,GPphotWNlimlo[i]),axis=0)
            GPnames     = np.concatenate((GPnames,['GPphotWN_lc'+str(i+1)]),axis=0)
            GPcombined  = np.concatenate((GPcombined,[d_combined]),axis=0)

            nGP=0   # counter for the number of GPs that are added for this
            dimGP_thislc = sum(GPjumping[i])
            pargp = pargp_all[:,GPjumping[i]]  # the matrix with the inputs to the GPs for this lightcurve
            for gpdim in range(ndimGP):   # loop through the dimensions of the GP
                if GPall[0,gpdim] == True:
                    j = 0
                else:
                    j = i

                if GPjumping[j,gpdim]==True:
                    if nGP>0:   # if this is not the first GP, then add the new kernel to the previous one
                        k2 = kern

                    if (GPphotkerns[j,gpdim]=='sqexp'):
                        k1 = GPphotpars1[j,gpdim] * kernels.ExpSquaredKernel(GPphotpars2[j,gpdim], ndim=dimGP_thislc, axes=nGP)  
                    elif (GPphotkerns[j,gpdim]=='mat32'):
                        k1 = GPphotpars1[j,gpdim] * kernels.Matern32Kernel(GPphotpars2[j,gpdim], ndim=dimGP_thislc, axes=nGP)  
                    else:
                        _raise(ValueError, 'kernel not recognized! Must be either "sqexp" or "mat32" ')
                        
                    if nGP==0:  # if this is the first GP, then set the kernel to the first one
                        kern = k1
                    else:
                        kern = k2 + k1
                        
                    GPparams=np.concatenate((GPparams,(np.log(GPphotpars1[j,gpdim]),np.log(GPphotpars2[j,gpdim]))), axis=0)           
                    if GPall[0,gpdim] == True and i == 0:         
                        GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,gpdim],GPphotstep2[j,gpdim])),axis=0)
                    elif GPall[0,gpdim] == True and i != 0:
                        GPstepsizes=np.concatenate((GPstepsizes,(0.,0.)),axis=0)
                    else:
                        GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,gpdim],GPphotstep2[j,gpdim])),axis=0)
                    
                    GPindex    = np.concatenate((GPindex,(np.zeros(2)+i)),axis=0)
                    GPprior    = np.concatenate((GPprior,(GPphotprior1[j,gpdim],GPphotprior2[j,gpdim])),axis=0)
                    GPpriwid   = np.concatenate((GPpriwid,(GPphotpriorwid1[j,gpdim],GPphotpriorwid2[j,gpdim])),axis=0)
                    GPlimup    = np.concatenate((GPlimup,(GPphotlim1up[j,gpdim],GPphotlim2up[j,gpdim])),axis=0)
                    GPlimlo    = np.concatenate((GPlimlo,(GPphotlim1lo[j,gpdim],GPphotlim2lo[j,gpdim])),axis=0)
                    GPnames    = np.concatenate((GPnames,(['GPphotscale_lc'+str(i+1)+'dim'+str(gpdim),"GPphotmetric_lc"+str(i+1)+'dim'+str(gpdim)])),axis=0)
                    GPcombined = np.concatenate((GPcombined,(GPall[j,gpdim],GPall[j,gpdim])),axis=0)

                    nGP=nGP+1
                    
            gp = GP(kern, mean=1,white_noise=GPphotWNstart[i],fit_white_noise=True)

            gp.compute(pargp, err)
            GPobjects.append(gp)
            pargps.append(pargp) 
            
    # ================ MODIFICATIONS: adding celerite ===============
        elif (useGPphot[i]=='ce'):
            pargp = np.copy(t)   #

            # define the index in the set of filters that this LC has:
            # k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
            # k = k[0].item()

            if GPphotWNstart[i] == GPphotWNstart[0] and GPphotWNstep[i]==0.0:
                d_combined = 1.0
            elif GPphotWN[0] == 'all':
                d_combined = 1.0
            else:
                d_combined = 0.0

            # define a Matern 3/2 kernel
            #celerite.terms.Matern32Term(*args, **kwargs)

            c_sigma = np.copy(np.log(GPphotpars1[i][0]))
            c_rho   = np.copy(np.log(GPphotpars2[i][0]))
            #for celerite
            Q  = 1/np.sqrt(2)
            w0 = 2*np.pi/(np.exp(c_rho))
            S0 = np.exp(c_sigma)**2/(w0*Q)   # https://celerite2.readthedocs.io/en/latest/api/python/#model-building
            c_eps=0.001
            if debug: print(f"DEBUG: In fit_data.py - kernel = {GPphotkerns[i,0]} ")
            
            if GPphotWNstep[i]>1e-12:  #if the white noise is jumping

                GPparams    = np.concatenate((GPparams,GPphotWNstart[i]), axis=0)   
                GPstepsizes = np.concatenate((GPstepsizes,GPphotWNstep[i]),axis=0)
                GPindex     = np.concatenate((GPindex,[i]),axis=0)
                GPprior     = np.concatenate((GPprior,GPphotWNprior[i]),axis=0)
                GPpriwid    = np.concatenate((GPpriwid,GPphotWNpriorwid[i]),axis=0)
                GPlimup     = np.concatenate((GPlimup,GPphotWNlimup[i]),axis=0)
                GPlimlo     = np.concatenate((GPlimlo,GPphotWNlimlo[i]),axis=0)
                GPnames     = np.concatenate((GPnames,['CEphotWN_lc'+str(i+1)]),axis=0)
                GPcombined  = np.concatenate((GPcombined,[d_combined]),axis=0)

                c_WN=np.copy(GPphotWNstart[i])
                bounds_w = dict(log_sigma=(GPphotWNlimlo[i],GPphotWNlimup[i]))
                k1 = terms.JitterTerm(log_sigma=c_WN, bounds=bounds_w)
                if GPphotkerns[i,0] == "mat32":
                    bounds = dict(log_sigma=(GPphotlim1lo[i][0], GPphotlim1up[i][0]), log_rho=(GPphotlim2lo[i][0], GPphotlim2up[i][0]))
                    if debug: print(f"celerite gp bounds = {bounds}, starting: {c_sigma},{c_rho}")
                    k2 = terms.Matern32Term(log_sigma=c_sigma, log_rho=c_rho, bounds=bounds)
                elif GPphotkerns[i,0] == "sho":
                    k2 = terms.SHOTerm(log_S0=np.log(S0), log_omega0=np.log(w0), log_Q = np.log(Q))
                    k2.freeze_parameter("log_Q")
                else: _raise(ValueError, f'Celerite kernel not recognized! Must be either "sho" or "mat32" but {GPphotkerns[i,0]} given')
                kern=k1 + k2
                NparGP=3

            else:
                bounds = dict(log_sigma=(GPphotlim1lo[i][0], GPphotlim1up[i][0]), log_rho=(GPphotlim2lo[i][0], GPphotlim2up[i][0]))
                if debug: print(f"celerite gp bounds = {bounds}, starting: {c_sigma},{c_rho}")
                if GPphotkerns[i,0] == "mat32": 
                    kern = terms.Matern32Term(log_sigma=c_sigma, log_rho=c_rho, bounds=bounds)
                elif GPphotkerns[i,0] == "sho": 
                    k3 = terms.SHOTerm(log_S0=np.log(S0), log_omega0=np.log(w0), log_Q = np.log(Q))
                    k3.freeze_parameter("log_Q")
                else: _raise(ValueError, f'Celerite kernel not recognized! Must be either "sho" or "mat32" but {GPphotkerns[i,0]} given')
                NparGP=2

            gpdim=0    #only dim0 (time) used for now
            if GPall[0,gpdim] == True:
                j = 0
            else:
                j=i

            GPparams=np.concatenate((GPparams,[c_sigma,c_rho]), axis=0)             
            if GPall[0,gpdim] == True and i == 0:         
                GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,gpdim],GPphotstep2[j,gpdim])),axis=0)
            elif GPall[0,gpdim] == True and i != 0:
                GPstepsizes=np.concatenate((GPstepsizes,(0.,0.)),axis=0)
            else:
                GPstepsizes=np.concatenate((GPstepsizes,(GPphotstep1[j,gpdim],GPphotstep2[j,gpdim])),axis=0)

            GPindex     = np.concatenate((GPindex,(np.zeros(2)+i)),axis=0)
            GPprior     = np.concatenate((GPprior,(GPphotprior1[j,gpdim],GPphotprior2[j,gpdim])),axis=0)
            GPpriwid    = np.concatenate((GPpriwid,(GPphotpriorwid1[j,gpdim],GPphotpriorwid2[j,gpdim])),axis=0)
            GPlimup     = np.concatenate((GPlimup,(GPphotlim1up[j,gpdim],GPphotlim2up[j,gpdim])),axis=0)
            GPlimlo     = np.concatenate((GPlimlo,(GPphotlim1lo[j,gpdim],GPphotlim2lo[j,gpdim])),axis=0)
            GPnames     = np.concatenate((GPnames,(['CEphotscale_lc'+str(i+1)+'dim'+str(gpdim),"CEphotmetric_lc"+str(i+1)+'dim'+str(gpdim)])),axis=0)
            GPcombined  = np.concatenate((GPcombined,(GPall[j,gpdim],GPall[j,gpdim])),axis=0)
        

            # gp = clnGPnew(kern, mean=mean_model,fit_mean=True)#,log_white_noise=GPphotWNstart[i],fit_white_noise=True)
            gp = cGP(kern, mean=1)

            gp.compute(pargp, err)
            GPobjects.append(gp)
            pargps.append(pargp)  

    if rv is not None: print('Setting up RV arrays ...')

    for i in range(nRV):
        t, rv, err, bis, fwhm, contrast = np.loadtxt(rv_fpath+RVnames[i], usecols=(0,1,2,3,4,5), unpack = True)  # reading in the data
        
        t_arr    = np.concatenate((t_arr,t), axis=0)
        f_arr    = np.concatenate((f_arr,rv), axis=0)    # ! add the RVs to the "flux" array !
        e_arr    = np.concatenate((e_arr,err), axis=0)   # ! add the RV errors to the "earr" array !
        col3_arr = np.concatenate((col3_arr,np.zeros(len(t),dtype=int)), axis=0)  # xshift array: filled with 0s
        col4_arr = np.concatenate((col4_arr,np.zeros(len(t),dtype=int)), axis=0)  # yshift array: filled with 0s
        col5_arr = np.concatenate((col5_arr,np.zeros(len(t),dtype=int)), axis=0)  # airmass array: filled with 0s
        col6_arr = np.concatenate((col6_arr,fwhm), axis=0)
        col7_arr = np.concatenate((col7_arr,np.zeros(len(t),dtype=int)), axis=0)  # sky array: filled with 0s

        bis_arr   = np.concatenate((bis_arr,bis), axis=0)  # bisector array
        contr_arr = np.concatenate((contr_arr,contrast), axis=0)  # contrast array
        lind      = np.concatenate((lind,np.zeros(len(t),dtype=int)+i+nphot), axis=0)   # indices
        indices   = np.where(lind==i+nphot)
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
            params    = np.concatenate((params,W_in[0,:],V_in[0,:],U_in[0,:],S_in[0,:],P_in[0,:]))
            stepsize  = np.concatenate((stepsize,W_in[1,:],V_in[1,:],U_in[1,:],S_in[1,:],P_in[1,:]))
            pmin      = np.concatenate((pmin,W_in[2,:],V_in[2,:],U_in[2,:],S_in[2,:],P_in[2,:]))
            pmax      = np.concatenate((pmax,W_in[3,:],V_in[3,:],U_in[3,:],S_in[3,:],P_in[3,:]))
            prior     = np.concatenate((prior, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
            priorlow  = np.concatenate((priorlow, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
            priorup   = np.concatenate((priorup, np.zeros(len(W_in[0,:])+len(V_in[0,:])+len(U_in[0,:])+len(S_in[0,:])+len(P_in[0,:]))))
            # pnames    = np.concatenate((pnames, [RVnames[i]+'_W1',RVnames[i]+'_W2',
            #                                      RVnames[i]+'_V1',RVnames[i]+'_V2',
            #                                      RVnames[i]+'_U1', RVnames[i]+'_U2',
            #                                      RVnames[i]+'_S1',RVnames[i]+'_S2',
            #                                      RVnames[i]+'_P1',RVnames[i]+'_P2',RVnames[i]+'_P3',RVnames[i]+'_P4']))
            pnames   = np.concatenate((pnames, [f"rv{i+1}_A0",f"rv{i+1}_B0",
                                                f"rv{i+1}_A3",f"rv{i+1}_B3",
                                                f"rv{i+1}_A4",f"rv{i+1}_B4",
                                                f"rv{i+1}_A5",f"rv{i+1}_B5",
                                                f"rv{i+1}_sin_amp",f"rv{i+1}_sin_per",f"rv{i+1}_sin_off",f"rv{i+1}_sin_off2"]))            

    # calculate the weights for the lightcurves to be used for the CNM calculation later: do this in a function!
    #ewarr=grweights(earr,indlist,grnames,groups,ngroup)

    for i in range(len(params)):
        if verbose: print(pnames[i], params[i], stepsize[i], pmin[i], pmax[i], priorup[i], priorlow[i])
        
    inmcmc='n'

    LCjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific lc

    #ATTENTION: pass to the lnprob function the individual subscript (of variable p) that are its jump parameters for each LC
    # which indices of p0 are referring to lc n

    for i in range(nphot):
        
        temp=np.ndarray([0])  # the indices of the parameters that jump for this LC
        
        tr_ind = np.array([0])   #rho_star index
        tr_ind = np.append(tr_ind, np.concatenate([np.arange(1,7)+7*n for n in range(npl)])) # add index of the 6 other transit jump parameters for all planets (no K)
        lcstep = tr_ind[np.where(stepsize[tr_ind]!=0.)[0]]
        
        if (len(lcstep) > 0): 
            temp=np.copy(lcstep)
        
        # define the index in the set of filters that this LC has:
        k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
        k = k[0].item()
        
        #transit depth variation
        if (ddfYN=='y'):    
            if temp.shape:    
                temp=np.concatenate((np.asarray(temp),np.asarray([1+7*npl+k])),axis=0)
            else:
                temp=np.asarray([1+7*npl+k])
        
        #occultation
        occind=1+7*npl+nddf+k                   # the index of the occultation parameter for this LC
        if (stepsize[occind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[occind]),axis=0)

        #pc
        Apc_ind=1+7*npl+nddf+nocc+k                   # the index of the first pc amplitude for this LC
        if (stepsize[Apc_ind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[Apc_ind]),axis=0)
        
        #phpff
        phoff_ind=1+7*npl+nddf+nocc*2+k                   # the index of the first pc amplitude for this LC
        if (stepsize[phoff_ind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[phoff_ind]),axis=0)
             
        #limb darkening
        c1ind=1+7*npl+nddf+nocc*3+k*4
        if (stepsize[c1ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[c1ind]),axis=0)
        
        c2ind=1+7*npl+nddf+nocc*3+k*4+1
        if (stepsize[c2ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[c2ind]),axis=0)
    
        #baseline
        bfstart= 1+7*npl+nddf+nocc*3+nfilt*4 + nRV*2  # the first index in the param array that refers to a baseline function    
        blind = np.asarray(list(range(bfstart+i*20,bfstart+i*20+20)))  # the indices for the coefficients for the base function  #TODO why 20, not 21  
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
        
        rv_ind  = np.array([0])     #rho_star
        rv_ind  = np.append(rv_ind, np.concatenate([np.arange(1,8)+7*n for n in range(npl)])) # the common RV jump parameters: transit + K for all planets
        rvstep1 = np.where(stepsize[rv_ind]!=0.)  
        rvstep = rvstep1[0]
        
        if (len(rvstep) > 0): 
            temp=np.copy(rvstep)

        # identify the gamma index of this RV (note: each gamma comes with a jitter, so 2 indices needed per rvdata)
        gammaind = 1+7*npl+nddf+nocc*3+nfilt*4+i*2
        
        if (stepsize[gammaind]!=0.):           
            temp=np.concatenate((temp,[gammaind]),axis=0)

        bfstart= 1+7*npl+nddf+nocc*3+nfilt*4 + nRV*2 + nphot*20  # the first index in the param array that refers to an RV baseline function    
        blind = np.asarray(list(range(bfstart+i*8,bfstart+i*8+8)))  # the indices for the coefficients for the base function    

        rvstep1 = np.where(stepsize[blind]!=0.)
        if len(rvstep1)>0:
            rvstep = rvstep1[0] 
            temp=np.concatenate((np.asarray(temp),blind[rvstep]),axis=0)
        
        RVjump.append(temp)
    # print(f"RVjump:{RVjump}")
    # =============================== CALCULATION ==========================================

    pnames_all   = np.concatenate((pnames, GPnames))
    initial      = np.concatenate((params, GPparams))
    steps        = np.concatenate((stepsize, GPstepsizes))
    priors       = np.concatenate((prior, GPprior))
    priwid       = (priorup + priorlow) / 2.
    priorwids    = np.concatenate((priwid, GPpriwid))
    lim_low      = np.concatenate((pmin, GPlimlo))
    lim_up       = np.concatenate((pmax, GPlimup))
    ndim         = np.count_nonzero(steps)
    jumping      = np.where(steps != 0.)
    jumping_noGP = np.where(stepsize != 0.)
    jumping_GP   = np.where(GPstepsizes != 0.)


    pindices = []       #holds the indices of the jumping parameters for each lc/rv in the list contained only of jumping parameters--> pnames_all[jumping]
    for i in range(nphot):
        fullist = list(jumping[0])   # the indices of all the jumping parameters
        lclist  = list(LCjump[i])    # the indices of the jumping parameters for this LC
        both    = list(set(fullist).intersection(lclist))  # attention: set() makes it unordered. we'll need to reorder it
        both.sort()
        indices_A = [fullist.index(x) for x in both]
        pindices.append(indices_A)

    for i in range(nRV):
        fullist = list(jumping[0])
        rvlist  = list(RVjump[i])
        both    = list(set(fullist).intersection(rvlist))  # attention: set() makes it unordered. we'll need to reorder it
        both.sort()
        indices_A = [fullist.index(x) for x in both]
        pindices.append(indices_A)

    ewarr=np.nan#grweights(earr,indlist,grnames,groups,ngroup,nphot)

    ############################## Initial guess ##################################
    print('\nPlotting initial guess\n---------------------------')

    inmcmc = 'n'
    indparams = [t_arr,f_arr,col3_arr,col4_arr,col6_arr,col5_arr,col7_arr,bis_arr,contr_arr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,
                nocc,0,0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, 
                cont,names,RVnames,e_arr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,
                pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,
                jumping_noGP,GPphotWN,jit_apply,jumping_GP,GPstepsizes,GPcombined,npl,useSpline_lc,useSpline_rv,s_samp]
    
    debug_t1 = time.time()
    mval, merr,T0_init,per_init,Dur_init = logprob_multi(initial[jumping],*indparams,make_out_file=True,verbose=True,debug=debug,out_folder=out_folder)
    if debug: print(f'finished logprob_multi, took {(time.time() - debug_t1)} secs')
    if not os.path.exists(out_folder+"/init"): os.mkdir(out_folder+"/init")    #folder to put initial plots    
    debug_t2 = time.time()
    mcmc_plots(mval,t_arr,f_arr,e_arr, nphot, nRV, indlist, filters, names, RVnames, out_folder+'/init/init_',initial,T0_init,per_init,Dur_init)
    if debug: print(f'finished mcmc_plots, took {(time.time() - debug_t2)} secs')


    ########################### MCMC run ###########################################
    print('\n============Running MCMC======================')

    inmcmc = 'y'
    indparams = [t_arr,f_arr,col3_arr,col4_arr,col6_arr,col5_arr,col7_arr,bis_arr,contr_arr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,
                nocc,0,0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV,
                cont,names,RVnames,e_arr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,
                pindices,jumping,pnames,LCjump,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,
                jumping_noGP,GPphotWN,jit_apply,jumping_GP,GPstepsizes,GPcombined,npl,useSpline_lc,useSpline_rv,s_samp]

    print('No of dimensions: ', ndim)
    if nchains < 2*ndim:
        print('WARNING: Number of chains is less than twice the number of dimensions. Increasing number of chains to 2*ndim')
        nchains = 2*ndim
    print('No of chains: ', nchains)
    print('fitting parameters: ', pnames_all[jumping])
    if debug: 
        starting = {k:v for k,v in zip(pnames_all[jumping],initial[jumping])}
        print(f'initial: {starting}')


    ijnames = np.where(steps != 0.)    #indices of the jumping parameters
    jnames = pnames_all[[ijnames][0]]  # jnames are the names of the jump parameters

    # put starting points for all walkers, i.e. chains
    p0 = np.random.rand(ndim * nchains).reshape((nchains, ndim))*np.asarray(steps[jumping])*2 + (np.asarray(initial[jumping])-np.asarray(steps[jumping]))
    assert np.all([np.isfinite(logprob_multi(p0[i],*indparams)) for i in range(nchains)]),f'Range of start values of a(some) jump parameter(s) are outside the prior distribution'

    if walk == "demc": moves = emcee.moves.DEMove()
    elif walk == "snooker": moves = emcee.moves.DESnookerMove()
    else: moves = emcee.moves.StretchMove()
    sampler = emcee.EnsembleSampler(nchains, ndim, logprob_multi, args=(indparams),pool=Pool(nproc), moves=moves)

    if not os.path.exists(f'{out_folder}/chains_dict.pkl'):

        print("\nRunning first burn-in...")
        p0, lp, _ = sampler.run_mcmc(p0, 20, progress=progress, **kwargs)

        print("Running second burn-in...")
        p0 = p0[np.argmax(lp)] + steps[jumping] * np.random.randn(nchains, ndim) # this can create problems!
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, burnin, progress=progress, **kwargs)
        if save_burnin_chains:
            burnin_chains = sampler.chain

            #save burn-in chains to file
            burnin_chains_dict =  {}
            for ch in range(burnin_chains.shape[2]):
                burnin_chains_dict[jnames[ch]] = burnin_chains[:,:,ch]
            pickle.dump(burnin_chains_dict,open(out_folder+"/"+"burnin_chains_dict.pkl","wb"))  
            print("burn-in chain written to disk")
            matplotlib.use('Agg')
            burn_result = load_result(out_folder)
            try:
                fig = burn_result.plot_burnin_chains()
                fig.savefig(out_folder+"/"+"burnin_chains.png", bbox_inches="tight")
                print(f"Burn-in chains plot saved as: {out_folder}/burnin_chains.png")
            except: 
                print(f"full burn-in chains not plotted (number of parameters ({ndim}) exceeds 20. use result.plot_burnin_chains()")
                print(f"saving burn-in chain plot for the first 20 parameters")
                pl_pars = list(burn_result._par_names)[:20]
                fig = burn_result.plot_burnin_chains(pl_pars)
                fig.savefig(out_folder+"/"+"burnin_chains.png", bbox_inches="tight") 
            #TODO save more than one plot if there are more than 20 parameters

            matplotlib.use(__default_backend__)
        sampler.reset()

        print("\nRunning production...")
        pos, prob, state = sampler.run_mcmc(pos, ppchain,skip_initial_state_check=True, progress=progress, **kwargs )
        bp = pos[np.argmax(prob)]

        posterior = sampler.flatchain
        chains    = sampler.chain
        print(("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))))

    else:
        print("\nSkipping burn-in and production. Loading chains from disk")
        result    = load_result(out_folder)
        posterior = result.flat_posterior
        chains    = np.stack([v for k,v in result._chains.items()],axis=2)
        try: bp   = result.params_max
        except: bp = np.median(posterior,axis=0)



    GRvals = grtest_emcee(chains)
    gr_print(jnames,GRvals,out_folder)

    nijnames = np.where(steps == 0.)     #indices of the fixed parameters
    njnames = pnames_all[[nijnames][0]]  # njnames are the names of the fixed parameters

    exti = np.intersect1d(pnames_all,extinpars, return_indices=True)
    exti[1].sort()
    extins=np.copy(exti[1])

    #save chains to file
    chains_dict =  {}
    for ch in range(chains.shape[2]):
        chains_dict[jnames[ch]] = chains[:,:,ch]
    pickle.dump(chains_dict,open(out_folder+"/"+"chains_dict.pkl","wb"))
    print(f"Production chain written to disk as {out_folder}/chains_dict.pkl. Run `result=CONAN3.load_result()` to load it.\n")  
    print("==================MCMC Finished==================\n\n")

    # ==== chain and corner plot ================
    result = load_result(out_folder)

    matplotlib.use('Agg')
    #chain plot
    try:
        fig = result.plot_chains()
        fig.savefig(out_folder+"/chains.png", bbox_inches="tight")
    except:
        if ndim<=30:
            nplotpars = int(np.ceil(ndim/2))
            nplot     = 2
        else:
            nplotpars = 14
            nplot = int(np.ceil(ndim/14))

        for i in range(nplot):
            fit_pars = list(result._par_names)[i*nplotpars:(i+1)*nplotpars]
            fig = result.plot_chains(fit_pars)
            fig.savefig(out_folder+f"/chains_{i}.png", bbox_inches="tight") 
        print(f"saved {nplot} chain plots as {out_folder}/chains_*.png")
    #corner plot
    try:
        fig = result.plot_corner()
        fig.savefig(out_folder+"/corner.png", bbox_inches="tight")
    except: 
        if ndim<=30:
            nplotpars = int(ndim/2)
            nplot     = 2
        else:
            nplotpars = 14
            nplot = int(np.ceil(ndim/14))
        for i in range(nplot):
            fit_pars = list(result._par_names)[i*nplotpars:(i+1)*nplotpars]
            fig = result.plot_corner(fit_pars)
            fig.savefig(out_folder+f"/corner_{i}.png", bbox_inches="tight")
        print(f"saved {nplot} corner plots as {out_folder}/corner_*.png")

    matplotlib.use(__default_backend__)

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

    medvals,maxvals =mcmc_outputs(posterior,jnames, ijnames, njnames, nijnames, bpfull, ulamdas, Rs_in, Ms_in, Rs_PDF, Ms_PDF, 
                                  nfilt, filnames, howstellar, extinpars, extins, extind_PDF,npl,out_folder)

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
    indparams = [t_arr,f_arr,col3_arr,col4_arr,col6_arr,col5_arr,col7_arr,bis_arr,contr_arr, nphot, nRV, indlist, filters, nfilt,
         filnames,nddf,nocc,0,0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, 
              baseLSQ, bvars, bvarsRV, cont,names,RVnames,e_arr,divwhite,dwCNMarr,dwCNMind,params,
                  useGPphot,useGPrv,GPobjects,GPparams,GPindex,pindices,jumping,pnames,LCjump, 
                      priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps, 
                          jumping_noGP,GPphotWN,jumping_GP,jit_apply,GPstepsizes,GPcombined,npl,useSpline_lc,useSpline_rv,s_samp]
    
    #AKIN: save config parameters indparams and summary_stats and as a hidden files. 
    #can be used to run logprob_multi() to generate out_full.dat files for median posterior, max posterior and best fit values
    pickle.dump(indparams, open(out_folder+"/.par_config.pkl","wb"))
    stat_vals = dict(med = medp[jumping], max=maxp[jumping], bf = bpfull[jumping])
    pickle.dump(stat_vals, open(out_folder+"/.stat_vals.pkl","wb"))

    #median
    mval, merr,T0_post,p_post,Dur_post = logprob_multi(medp[jumping],*indparams,make_out_file=(statistic=="median"), verbose=True,out_folder=out_folder)
    mcmc_plots(mval,t_arr,f_arr,e_arr, nphot, nRV, indlist, filters, 
               names, RVnames, out_folder+'/med_',medp,T0_post,p_post,Dur_post)

    #max_posterior
    mval2, merr2, T0_post, p_post, Dur_post = logprob_multi(maxp[jumping],*indparams,make_out_file=(statistic=="max"),verbose=False)
    mcmc_plots(mval2,t_arr,f_arr,e_arr, nphot, nRV, indlist, filters, 
               names, RVnames, out_folder+'/max_',maxp, T0_post,p_post,Dur_post)


    maxresiduals = f_arr - mval2 if statistic != "median" else f_arr - mval  #Akin allow statistics to be based on median of posterior
    chisq = np.sum(maxresiduals**2/e_arr**2)

    ndat = len(t_arr)
    bic=get_BIC_emcee(npar,ndat,chisq,out_folder)
    aic=get_AIC_emcee(npar,ndat,chisq,out_folder)

    rarr=f_arr-mval2  if statistic != "median" else f_arr - mval  # the full residuals
    bw, br, brt, cf, cfn = corfac(rarr, t_arr, e_arr, indlist, nphot, njumpphot) # get the beta_w, beta_r and CF and the factor to get redchi2=1
    of=open(out_folder+"/CF.dat",'w')
    for i in range(nphot):   #adapt the error values
    # print(earr[indlist[i][0]])
        of.write('%8.3f %8.3f %8.3f %8.3f %10.6f \n' % (bw[i], br[i], brt[i],cf[i],cfn[i]))
        if (cf_apply == 'cf'):
            e_arr[indlist[i][0]] = np.multiply(e_arr[indlist[i][0]],cf[i])
            if verbose: print((e_arr[indlist[i][0]]))        
        if (cf_apply == 'rchisq'):
            e_arr[indlist[i][0]] = np.sqrt((e_arr[indlist[i][0]])**2 + (cfn[i])**2)
            if verbose: print((e_arr[indlist[i][0]]))

    of.close()

    result = load_result(out_folder)
    # result.T0   = maxp[np.where(pnames_all=='T_0')]      if statistic != "median" else medp[np.where(pnames_all=='T_0')]
    # result.P    = maxp[np.where(pnames_all=='Period')]   if statistic != "median" else medp[np.where(pnames_all=='Period')]
    # result.dur  = maxp[np.where(pnames_all=='Duration')] if statistic != "median" else medp[np.where(pnames_all=='Duration')]
    # result.RpRs = maxp[np.where(pnames_all=='RpRs')]     if statistic != "median" else medp[np.where(pnames_all=='RpRs')]
    
    return result