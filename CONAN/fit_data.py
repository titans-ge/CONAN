import numpy as np
from types import SimpleNamespace
import os
from multiprocessing import Pool
import dill as pickle
import emcee, dynesty
from dynesty.utils import resample_equal
import time

from .basecoeff_setup import *
from .models import *
from .logprob_multi import logprob_multi
from .plotting import *
from .funcs import corfac, grtest_emcee
from .utils import ecc_om_par
from .outputs import *

import george
from george import GP
import celerite
from celerite import GP as cGP
import spleaf
from .geepee import gp_params_convert, celerite_kernels, george_kernels,spleaf_kernels


from copy import deepcopy
from .conf import create_configfile
from scipy.stats import norm, uniform, lognorm, loguniform,truncnorm


from ._classes import _raise, fit_setup, __default_backend__, load_result, load_lightcurves, load_rvs, _text_format
import matplotlib
matplotlib.use(__default_backend__)

import multiprocessing as mp 
# mp.set_start_method('fork')
__all__ = ["run_fit"]

def prior_transform(u,prior_dst,prior_names):
    """  
    function to transform the unit cube,u, to the prior space.
    
    Parameters:
    -----------
    u : array_like;
        unit cube, array of values between 0 and 1.
    prior_dst : list;
        list of scipy.stats prior distributions for each parameter.
    prior_names : list;
        list of parameter names.
        
    Returns:
    --------
    x : array_like;
        array of values in the prior space.
    """
    x = np.array(u)  # copy u 
    for i, pr in enumerate(prior_dst):
        x[i] = pr.ppf(u[i]) 
        x[i] = x[i]
    return x 

def run_fit(lc_obj=None, rv_obj=None, fit_obj=None, statistic = "median", out_folder="output", init_only=False,progress=True,
            rerun_result=False, get_parameter_names=False, shared_params={}, debug=False, save_burnin_chains=True, resume_sampling=False,
            dyn_kwargs=dict(sample='rwalk',bound='multi'), run_kwargs=dict(),verbose=True ):
    """
    function to fit the data using the light-curve object lc_obj, rv_object rv_obj, and fit_setup 
    object fit_obj.

    Parameters:
    -----------
    lc_obj : lightcurve object;
        object containing lightcurve data and setup parameters. 
        see CONAN.load_lightcurves() for more details.
    rv_obj : rv object
        object containing radial velocity data and setup parameters. 
        see CONAN.load_rvs() for more details.
    fit_obj : fit_setup object;
        object containing fit setup parameters. 
        see CONAN.fit_setup() for more details.
    statistic : str;
        statistic to run on posteriors to obtain model parameters and create model output file 
        "_\*out.dat". must be one of ["median", "max", "bestfit"], default is "median". "max" and 
        "median" calculate the maximum and median of each parameter posterior respectively while 
        "bestfit" is the parameter combination that gives the maximum joint posterior probability.
    progress : bool;
        if True, show MCMC progress bar, default is True.
    out_folder : str;
        path to output folder, default is "output".
    init_only : bool;
        generate only initial models, prior display, and \*out.dat files in the output folder. 
        useful for diagnosing starting points or generating full lc and rv models
    rerun_result : bool;
        if True, rerun CONAN with previous fit result in order to regenerate plots and files. 
        This also allows to create files compatibile with latest CONAN version. Default is False.
    shared_params: dict, optional
        dict specifying parameters that shared a value. Defualt is empty dict {}. 
        use `CONAN.get_parameter_names(lc_obj, rv_obj)` to see all parameter names.
    resume_sampling : bool;
        resume sampling from last saved position 
    verbose : bool;
        if True, print out additional information, default is False.
    debug : bool;
        if True, print out additional debugging information, default is False.
    save_burnin_chains : bool;
        if True, save burn-in chains to file, default is True.
    dyn_kwargs : dict;
        other parameters sent to the dynesty.NestedSampler() or dynesty.DynamicNestedSampler() 
        function. e.g dyn_kwargs=dict(sample='rwalk',bounds='multi')
    run_kwargs : dict;
        other parameters sent to emcee's run_mcmc() function or dynesty's run_nested() function.
        e.g., for emcee: run_kwargs=dict(thin_by=1, tune=True, skip_initial_state_check=False)
        e.g., for dynesty dynamic sampling: run_kwargs=dict(maxiter_init=10000, maxiter_batch=1000,n_effective=30000)
        e.g., for static sampling: run_kwargs=dict( nlive_batch=50, maxbatch=5,maxiter=10000, maxcall=50000, logl_max=12344, n_effective=30000)
    
    Returns:
    --------
    result : object containing labeled mcmc chains
        Object that contains methods to plot the chains, corner, and histogram of parameters.
        e.g ``result.plot_chains()``, ``result.plot_burnin_chains()``, ``result.plot_corner``, 
        ``result.plot_posterior("T_0")``
    """
    if not get_parameter_names:
        if not os.path.exists(out_folder):
            if verbose: print(f"Creating output folder...{out_folder}")
            os.makedirs(out_folder)
            create_configfile(lc_obj, rv_obj, fit_obj, f"{out_folder}/config_save.dat")  #create config file


        if os.path.exists(f'{out_folder}/chains_dict.pkl'):
            if rerun_result or resume_sampling:
                print(f'Fit result already exists in this folder: {out_folder}')
                if rerun_result:
                    print('Rerunning CONAN with saved posterior chains to regenerate plots and files...\n')
                    create_configfile(lc_obj, rv_obj, fit_obj, f"{out_folder}/config_save.dat")  #create config file
                if resume_sampling:
                    print('Resuming sampling from last saved position...\n')
            else:
                print(f'Fit result already exists in this folder: {out_folder}.\n Loading results...')
                result = load_result(out_folder)
                return result
        
    if verbose: print('\nCONAN fit launched!!!\n') 

    
    #begin loading data from the 3 objects and calling the methods
    assert statistic in ["median", "max", "bestfit"], 'statistic can only be either median, max or bestfit'

#============lc_obj=========================
    #from load_lightcurves()
    if lc_obj is None and rv_obj is not None: lc_obj=rv_obj._lcobj
    fpath      = lc_obj._fpath
    LCnames    = lc_obj._names
    filters    = lc_obj._filters
    wl         = lc_obj._wl
    bases      = lc_obj._bases
    bases_init = lc_obj._bases_init
    groups     = lc_obj._groups
    useGPphot  = lc_obj._useGPphot
    jitt_start = np.log(lc_obj._jitt_estimate)
    jitt_start = [-25 if jt<-25 else jt for jt in jitt_start] #set lowest start jitt to -25

    nphot      = len(LCnames)                                                 # the number of photometry input files
    njumpphot  = np.zeros(nphot)                                            # the number of jumping parameters for each photometry input file
    filnames   = np.array(list(sorted(set(filters),key=filters.index)))     # the unique filter names
    uwl        = np.array(list(sorted(set(wl),key=wl.index)))       # the unique wavelengths
    grnames    = np.array(list(sorted(set(groups))))                        # the unique group names
    nfilt      = len(filnames)                                              # the number of unique filters
    ngroup     = len(grnames)                                               # the number of unique groups

    useSpline_lc  = lc_obj._lcspline                                            # use spline to interpolate the light curve
    input_lcs     = lc_obj._input_lc
    s_samp        = lc_obj._ss
    custom_LCfunc = lc_obj._custom_LCfunc     # custom light curve function
    ncustom       = custom_LCfunc.npars 
    fit_offset    = lc_obj._fit_offset if isinstance(lc_obj._fit_offset,list) else [lc_obj._fit_offset]*nphot
    lc_Qsho       = lc_obj._Qsho

#============rv_obj========================== 
    # from load_rvs() 
    if rv_obj is None: rv_obj = load_rvs(verbose=False) 
    RVnames  = rv_obj._names
    RVbases  = rv_obj._RVbases
    rv_fpath = rv_obj._fpath
    rv_dict  = rv_obj._rvdict
    sinPs    = rv_dict["sinPs"]
    nRV      = len(RVnames)             # the number of RV input files
    njumpRV  = np.zeros(nRV)        # the number of jumping parameters for each RV input file
    useGPrv  = rv_obj._useGPrv 

    rvjitt_start = rv_obj._jitt_estimate 
    useSpline_rv = rv_obj._rvspline   
    input_rvs    = rv_obj._input_rv  
    RVbases_init = rv_obj._RVbases_init
    RVunit       = rv_obj._RVunit
    rv_Qsho      = rv_obj._Qsho


    custom_RVfunc = rv_obj._custom_RVfunc     # custom RV function
    ncustomRV     = custom_RVfunc.npars 

    extinpars= []  # set up array to contain the names of the externally input parameters (parameters always needed to generate a transit/RV model even if not a jumping parameter)

#============transit and RV jump parameters===============
    #from load_lightcurves.planet_parameters()

    CP  = deepcopy(lc_obj._planet_pars)    #load input transit and RV parameters from dict
    npl = lc_obj._nplanet

    if "rho_star" in CP[f"pl{1}"].keys():         #rho_star or Duration stored in rhoSt_Dur
        rho_dur = 'rho'
        # # rho_star (same for all planets)
        rhoSt_Dur = CP[f"pl{1}"]["rho_star"]
        if nphot > 0:    # if there is photometry, then define rho_star: can be a jumping parameter
            if rhoSt_Dur.step_size != 0.: njumpphot=njumpphot+1
            if (rhoSt_Dur.to_fit == 'n' and rhoSt_Dur.prior == 'p'):   
                extinpars.append('rho_star')
        else:
            rhoSt_Dur.to_fit == 'n'
            rhoSt_Dur.step_size = 0
    else:
        rho_dur = 'dur'
        # # Duration. only for single planet systems 
        rhoSt_Dur = CP[f"pl{1}"]["Duration"]    
        if rhoSt_Dur.step_size != 0.: njumpphot=njumpphot+1
        if (rhoSt_Dur.to_fit == 'n' and rhoSt_Dur.prior == 'p'):
            extinpars.append('Duration')

        
    for n in range(1,npl+1): 
        if nRV == 0: #remove k as a free parameter
            CP[f"pl{n}"]['K'].to_fit = "n"
            CP[f"pl{n}"]['K'].step_size = 0
        if nphot == 0: #remove rprs and impact_para as free parameters
            CP[f"pl{n}"]['RpRs'].to_fit = "n"
            CP[f"pl{n}"]['RpRs'].step_size = 0
            CP[f"pl{n}"]['Impact_para'].to_fit = "n"
            CP[f"pl{n}"]['Impact_para'].step_size = 0
        assert CP[f"pl{n}"]['Period'].start_value!=0,f"Period for planet {n} cannot be zero. Make sure to set planet parameters using the lc_obj.planet_parameters() function."

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

        # #K 
        if CP[f"pl{n}"]['K'].step_size != 0.: njumpRV=njumpRV+1
        if (CP[f"pl{n}"]['K'].to_fit == 'n' and CP[f"pl{n}"]['K'].prior == 'p'):
            extinpars.append('K')

        # #ecc
        if "Eccentricity" in CP[f"pl{n}"].keys():
            if CP[f"pl{n}"]['Eccentricity'].step_size != 0.: njumpRV=njumpRV+1
            if (CP[f"pl{n}"]['Eccentricity'].to_fit == 'n' and CP[f"pl{n}"]['Eccentricity'].prior == 'p'):
                _raise(ValueError, 'cant externally input eccentricity at this time!')

            # #omega
            for key,val in CP[f"pl{n}"]["omega"].__dict__.items():   #convert to radians
                if isinstance(val, (float,int)): CP[f"pl{n}"]["omega"].__dict__[key] *= np.pi/180
            if CP[f"pl{n}"]['omega'].step_size != 0.: njumpRV=njumpRV+1
            if (CP[f"pl{n}"]['omega'].to_fit == 'n' and CP[f"pl{n}"]['omega'].prior == 'p'):
                _raise(ValueError, 'cant externally input eccentricity at this time!')
        
            # adapt the eccentricity and omega jump parameters sqrt(e)*sin(o), sqrt(e)*cos(o)
            if ((CP[f"pl{n}"]['Eccentricity'].prior == 'y' and CP[f"pl{n}"]['omega'].prior == 'n') or (CP[f"pl{n}"]['Eccentricity'].prior == 'n' and CP[f"pl{n}"]['omega'].prior == 'y')):
                _raise(ValueError,'priors on eccentricity and omega: either both on or both off')
                
            CP[f"pl{n}"]["sesin(w)"], CP[f"pl{n}"]["secos(w)"] = ecc_om_par(CP[f"pl{n}"]["Eccentricity"], CP[f"pl{n}"]["omega"])

            #now remove Eccentricty and omega from the dictionary 
            _ = [CP[f"pl{n}"].pop(key) for key in ["Eccentricity", "omega"]]
        else:
            #sesinw
            if CP[f"pl{n}"]["sesin(w)"].step_size != 0.: njumpRV=njumpRV+1
            if (CP[f"pl{n}"]["sesin(w)"].to_fit == 'n' and CP[f"pl{n}"]["sesin(w)"].prior == 'p'):
                _raise(ValueError, 'cant externally input sesinw at this time!')

            #secosw
            if CP[f"pl{n}"]["secos(w)"].step_size != 0.: njumpRV=njumpRV+1
            if (CP[f"pl{n}"]["secos(w)"].to_fit == 'n' and CP[f"pl{n}"]["secos(w)"].prior == 'p'):
                _raise(ValueError, 'cant externally input secosw at this time!')

        _ = CP[f"pl{n}"].pop("rho_star" if "rho_star" in CP[f"pl{n}"] else "Duration")  #remove rho_star or Duration from the dictionary

    
#============ddfs ==========================
    #from load_lighcurves.transit_depth_variations()
    divwhite = lc_obj._ddfs.divwhite    # do we do a divide-white?
    ddfYN    = lc_obj._ddfs.ddfYN       # do we do a depth-dependent fit?
                
    grprs  = lc_obj._ddfs.depth_per_group   # the group rprs values
    egrprs = lc_obj._ddfs.depth_err_per_group  # the uncertainties of the group rprs values
    dwfiles = [f"dw_00{n}.dat" for n in grnames]   

    dwCNMarr=np.array([])      # initializing array with all the dwCNM values
    dwCNMind=[]                # initializing array with the indices of each group's dwCNM values
    dwind=np.array([])
    if (divwhite=='y'):           # do we do a divide-white?    
        for i in range(ngroup):   # read fixed dwCNMs for each group
            tdwCNM, dwCNM = np.loadtxt(lc_obj._fpath + dwfiles[i], usecols=(0,1), unpack = True)
            dwCNMarr      = np.concatenate((dwCNMarr,dwCNM), axis=0)
            dwind         = np.concatenate((dwind,np.zeros(len(dwCNM),dtype=int)+i), axis=0)
            indices       = np.where(dwind==i)
            dwCNMind.append(indices)        

#=============sampling setup===============
    #from setup_fit()
    if fit_obj is None: fit_obj = fit_setup(verbose=verbose)
    DA_mc =  fit_obj._fit_dict

    nsteps           = int(DA_mc['n_steps'])   # number of steps
    nchains          = int(DA_mc['n_chains'])  #  number of chains
    nproc            = int(DA_mc['n_cpus'])   #  number of processes
    burnin           = int(DA_mc['n_burn'])    # Length of bun-in
    emcee_move       = DA_mc['emcee_move']            # Differential Evolution? 
    fit_sampler      = DA_mc['sampler']               # Which sampler to use?   
    NS_type          = DA_mc["nested_sampling"]     #static or dynamic sampling
    nlive            = DA_mc["n_live"]  
    force_nlive      = DA_mc["force_nlive"]
    dlogz            = DA_mc["dyn_dlogz"]    
    jit_apply        = DA_mc['apply_RVjitter']       # apply rvjitter
    jit_LCapply      = DA_mc['apply_LCjitter']     # apply lcjitter
    LCjitter_loglims = DA_mc['LCjitter_loglims']   # log of the LC jitter limits
    RVjitter_lims    = DA_mc['RVjitter_lims']      # RV jitter limits
    LCbase_lims      = DA_mc['LCbasecoeff_lims']   # bounds of the LC baseline coefficients
    RVbase_lims      = DA_mc['RVbasecoeff_lims']   # bounds of the RV baseline coefficients
    paraCNM          = DA_mc['remove_param_for_CNM']   # remove parametric model for CNM computation
    baseLSQ          = DA_mc['leastsq_for_basepar']   # do a leas-square minimization for the baseline (not jump parameters)
    cf_apply         = DA_mc['apply_CFs']  # which CF to apply
    # fit_offset       = DA_mc['fit_LCoffset']  # fit offset for LC

    #limits on parametric baseline pars: auto or user-defined
    #LC
    lcbases_lims = [dict(off = [input_lcs[LCnames[i]]["col1"].min(),input_lcs[LCnames[i]]["col1"].max()], #min,max flux
                        amp=[0,1], freq=[1,333], phi=[0,1], ACNM=[0,1e8], BCNM=[0,1e8], C0=[-1,1], D0=[-1,1]) 
                        for i in range(nphot)]

    col_pars = [f"{L}{c}" for c in [0,3,4,5,6,7,8] for L in ["A","B"]] + ["C0","D0"]    #decorr_params
    if isinstance(LCbase_lims,list):   # set all lims to user-defined LCbase_lims
        for i in range(nphot):
            for k in col_pars: lcbases_lims[i][k] = LCbase_lims
                
    else:                              #auto determine best lims from data. range(flux)/range(column array)
        for i in range(nphot):
            off_step = 0.1*np.ptp(lcbases_lims[i]["off"])
            if bases_init[i]["off"]>=lcbases_lims[i]["off"][1] - off_step: lcbases_lims[i]["off"][1] += 2*off_step
            if bases_init[i]["off"]<=lcbases_lims[i]["off"][0] + off_step: lcbases_lims[i]["off"][0] -= 2*off_step
            
            fl_arr = input_lcs[LCnames[i]]["col1"]
            for c in [0,3,4,5,6,7,8]:      #col numbers
                arr = input_lcs[LCnames[i]][f"col{c}"]
                arr = arr-np.median(arr) if c==0 else arr    #if time array, subtract median
                if np.ptp(arr) > 0:      #if variation in this column
                    alim = np.ptp(fl_arr)/np.ptp(arr**1)    #max flux change occuring across the range of the decorr_column
                    alim = max(alim,1)
                    lcbases_lims[i][f"A{c}"] = [-alim,alim]

                    blim = np.ptp(fl_arr)/np.ptp(arr**2)
                    blim = max(blim,1)
                    lcbases_lims[i][f"B{c}"] = [-blim,blim]
                else:     #set to [-1,1]
                    lcbases_lims[i][f"A{c}"] = lcbases_lims[i][f"B{c}"] = [-1,1]

    #RV
    rvbases_lims = [dict(amp=[0,1], freq=[1,100], phi=[0,1], phi2=[0,1]) for i in range(nRV)]
    col_pars = [f"{L}{c}" for c in [0,3,4,5] for L in ["A","B"]]    #decorr_params
    if isinstance(RVbase_lims,list):   # set all lims to user-defined RVbase_lims
        for i in range(nRV):
            for k in col_pars: rvbases_lims[i][k] = RVbase_lims
    else:                              #auto determine best lims from data. rms/1% span of the decorr_column
        for i in range(nRV):
            rv_arr = input_rvs[RVnames[i]]["col1"]
            for c in [0,3,4,5]:
                arr = input_rvs[RVnames[i]][f"col{c}"]
                arr = arr-np.median(arr) if c==0 else arr
                if np.ptp(arr) > 0:
                    alim = np.ptp(rv_arr)/np.ptp(arr**1)
                    alim = max(alim,5)
                    rvbases_lims[i][f"A{c}"] = [-alim,alim]

                    blim = np.ptp(rv_arr)/np.ptp(arr**2)
                    blim = max(blim,5)
                    rvbases_lims[i][f"B{c}"] = [-blim,blim]
                else:
                    rvbases_lims[i][f"A{c}"] = rvbases_lims[i][f"B{c}"] = [-1,1]

    #LC jitter lims
    if isinstance(LCjitter_loglims,list):   # set all lims to user-defined LCjitter_loglims
        lcjitter_loglims = [LCjitter_loglims]*nphot
    else:               #auto determine best jiiter lims for each lcfile as [-15, log(10*mean(fluxerr))]
        lcjitter_loglims = [ [-15, np.log(input_lcs[LCnames[i]]["col2"].mean()*10)] for i in range(nphot)]
    #rv jitter lims
    if isinstance(RVjitter_lims,list):   # set all lims to user-defined RVjitter_lims
        rvjitter_lims = [RVjitter_lims]*nRV
    else:               #auto determine best jiiter lims for each rvfile as [0, 10*mean(RVerr)]
        rvjitter_lims = [ [0,input_rvs[RVnames[i]]["col2"].mean()*10] for i in range(nRV)]

#========= stellar properties==========================
    #from setup_fit.stellar_parameters()
    DA_stlr = fit_obj._stellar_dict
    
    Rs_in  = DA_stlr["R_st"][0]
    sRs_lo = sRs_hi = DA_stlr["R_st"][1]
    Ms_in  = DA_stlr["M_st"][0]
    sMs_lo = sMs_hi = DA_stlr["M_st"][1]
    LTT    = SimpleNamespace(flag="y",Rstar=Rs_in) if DA_mc["LTT_corr"]=="y" else SimpleNamespace(flag="n",Rstar=None)

    howstellar = DA_stlr["par_input"]

    #********************************************************************
    #============Start computations as in original CONANGP===============
    #********************************************************************
        
    #============================= SETUP ARRAYS =======================================
    if LCnames != []: 
        if verbose: print('Setting up photometry arrays ...')
    if np.any([spl.use for spl in useSpline_lc.values()]): 
        if verbose: print('Setting up Spline fitting for LCs ...')  

    t_arr      = np.array([])  # initializing array with all timestamps (col0)
    f_arr      = np.array([])  # initializing array with all flux values(col1)
    e_arr      = np.array([])  # initializing array with all error values(col2)
    lind       = np.array([])  # initializing array with the lightcurve indices
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
    params   = np.concatenate(([rhoSt_Dur.start_value],   [CP[f"pl{n}"][key].start_value    for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # initial guess params
    stepsize = np.concatenate(([rhoSt_Dur.step_size],     [CP[f"pl{n}"][key].step_size      for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # stepsizes
    pmin     = np.concatenate(([rhoSt_Dur.bounds_lo],     [CP[f"pl{n}"][key].bounds_lo      for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Boundaries (min)
    pmax     = np.concatenate(([rhoSt_Dur.bounds_hi],     [CP[f"pl{n}"][key].bounds_hi      for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Boundaries (max)
    prior    = np.concatenate(([rhoSt_Dur.prior_mean],    [CP[f"pl{n}"][key].prior_mean     for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Prior centers
    priorlow = np.concatenate(([rhoSt_Dur.prior_width_lo],[CP[f"pl{n}"][key].prior_width_lo for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Prior sigma low side
    priorup  = np.concatenate(([rhoSt_Dur.prior_width_hi],[CP[f"pl{n}"][key].prior_width_hi for n in range(1,npl+1)  for key in CP[f"pl1"].keys()]))  # Prior sigma high side
    pnames   = np.concatenate((["rho_star" if rho_dur=='rho' else "Duration"],  [nm+(f"_{n}" if npl>1 else "")      for n in range(1,npl+1)  for nm in pnames]))

    extcens  = np.concatenate(([rhoSt_Dur.prior_mean],     [CP[f"pl{n}"][key].prior_mean     for n in range(1,npl+1)  for key in CP[f"pl1"].keys()])) # External parameter prior mean
    extlow   = np.concatenate(([rhoSt_Dur.prior_width_lo], [CP[f"pl{n}"][key].prior_width_lo for n in range(1,npl+1)  for key in CP[f"pl1"].keys()])) # External parameter prior sig lo
    extup    = np.concatenate(([rhoSt_Dur.prior_width_hi], [CP[f"pl{n}"][key].prior_width_hi for n in range(1,npl+1)  for key in CP[f"pl1"].keys()])) # External parameter prior sig hi
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

    ttv_conf = lc_obj._ttvs.conf   #ttv configuration
    ttvYN    = lc_obj._ttvs.to_fit
    if ttvYN == "y":
        if verbose: print("Setting up TTV arrays ...")
        nttv = len(lc_obj._ttvs.fit_t0s)    
        for i in range(nttv):
            params    = np.concatenate((params,   [lc_obj._ttvs.prior[i].start_value]))
            stepsize  = np.concatenate((stepsize, [lc_obj._ttvs.prior[i].step_size]))
            pmin      = np.concatenate((pmin,     [lc_obj._ttvs.prior[i].bounds_lo]))
            pmax      = np.concatenate((pmax,     [lc_obj._ttvs.prior[i].bounds_hi]))
            prior     = np.concatenate((prior,    [lc_obj._ttvs.prior[i].prior_mean]))
            priorlow  = np.concatenate((priorlow, [lc_obj._ttvs.prior[i].prior_width_lo]))
            priorup   = np.concatenate((priorup,  [lc_obj._ttvs.prior[i].prior_width_hi]))
            pnames    = np.concatenate((pnames,   [lc_obj._ttvs.fit_labels[i]]))
            njumpphot = njumpphot+1
    else:
        nttv = 0

    if ddfYN == 'y':   # if ddFs are fit: set the Rp/Rs to the specified value, and fix it.
        njumpphot = njumpphot+1   # each LC has another jump param (i.e. dRpRs for each filter/lc)

        drprs = lc_obj._ddfs.drprs
        for i,f in enumerate(filnames):  # and make an array with the drprs inputs  # the dRpRs options
            params        = np.concatenate((params,   [drprs.start_value]))     # add them to the parameter arrays    
            stepsize      = np.concatenate((stepsize, [drprs.step_size]))
            pmin          = np.concatenate((pmin,     [drprs.bounds_lo]))
            pmax          = np.concatenate((pmax,     [drprs.bounds_hi]))
            prior         = np.concatenate((prior,    [drprs.prior_mean]))
            priorlow      = np.concatenate((priorlow, [drprs.prior_width_lo]))
            priorup       = np.concatenate((priorup,  [drprs.prior_width_hi]))
            pnames        = np.concatenate((pnames,   [f+'_dRpRs']))
            
    nocc = len(filnames)
    DA_occ  = lc_obj._PC_dict["D_occ"]
    for i,f in enumerate(filnames):  # add the occultation depths
        params     = np.concatenate((params,   [DA_occ[f].start_value]))
        stepsize   = np.concatenate((stepsize, [DA_occ[f].step_size]))
        pmin       = np.concatenate((pmin,     [DA_occ[f].bounds_lo]))
        pmax       = np.concatenate((pmax,     [DA_occ[f].bounds_hi]))
        prior      = np.concatenate((prior,    [DA_occ[f].prior_mean]))
        priorlow   = np.concatenate((priorlow, [DA_occ[f].prior_width_lo]))
        priorup    = np.concatenate((priorup,  [DA_occ[f].prior_width_hi]))
        pnames     = np.concatenate((pnames,   [f+'_DFocc']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_occ[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1

    DA_Fn = lc_obj._PC_dict["Fn"]
    for i,f in enumerate(filnames):  # add the nightside flux
        params     = np.concatenate((params,   [DA_Fn[f].start_value if DA_Fn[f].start_value!=None else 0]))
        stepsize   = np.concatenate((stepsize, [DA_Fn[f].step_size]))
        pmin       = np.concatenate((pmin,     [DA_Fn[f].bounds_lo]))
        pmax       = np.concatenate((pmax,     [DA_Fn[f].bounds_hi]))
        prior      = np.concatenate((prior,    [DA_Fn[f].prior_mean]))
        priorlow   = np.concatenate((priorlow, [DA_Fn[f].prior_width_lo]))
        priorup    = np.concatenate((priorup,  [DA_Fn[f].prior_width_hi]))
        pnames     = np.concatenate((pnames,   [f+'_Fn']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_Fn[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1

    DA_off  = lc_obj._PC_dict["ph_off"]
    for i,f in enumerate(filnames):  # add the phase offsets
        params     = np.concatenate((params,   [DA_off[f].start_value if DA_off[f].start_value!=None else 0]))
        stepsize   = np.concatenate((stepsize, [DA_off[f].step_size]))
        pmin       = np.concatenate((pmin,     [DA_off[f].bounds_lo]))
        pmax       = np.concatenate((pmax,     [DA_off[f].bounds_hi]))
        prior      = np.concatenate((prior,    [DA_off[f].prior_mean]))
        priorlow   = np.concatenate((priorlow, [DA_off[f].prior_width_lo]))
        priorup    = np.concatenate((priorup,  [DA_off[f].prior_width_hi]))
        pnames     = np.concatenate((pnames,   [f+'_ph_off']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_off[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1
    
    DA_Aev  = lc_obj._PC_dict["A_ev"]
    for i,f in enumerate(filnames):  # add the Aev amplitudes
        params     = np.concatenate((params,   [DA_Aev[f].start_value]))
        stepsize   = np.concatenate((stepsize, [DA_Aev[f].step_size]))
        pmin       = np.concatenate((pmin,     [DA_Aev[f].bounds_lo]))
        pmax       = np.concatenate((pmax,     [DA_Aev[f].bounds_hi]))
        prior      = np.concatenate((prior,    [DA_Aev[f].prior_mean]))
        priorlow   = np.concatenate((priorlow, [DA_Aev[f].prior_width_lo]))
        priorup    = np.concatenate((priorup,  [DA_Aev[f].prior_width_hi]))
        pnames     = np.concatenate((pnames,   [f+'_Aev']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_Aev[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1

    DA_Adb  = lc_obj._PC_dict["A_db"]
    for i,f in enumerate(filnames):  # add the Adb amplitudes
        params     = np.concatenate((params,   [DA_Adb[f].start_value]))
        stepsize   = np.concatenate((stepsize, [DA_Adb[f].step_size]))
        pmin       = np.concatenate((pmin,     [DA_Adb[f].bounds_lo]))
        pmax       = np.concatenate((pmax,     [DA_Adb[f].bounds_hi]))
        prior      = np.concatenate((prior,    [DA_Adb[f].prior_mean]))
        priorlow   = np.concatenate((priorlow, [DA_Adb[f].prior_width_lo]))
        priorup    = np.concatenate((priorup,  [DA_Adb[f].prior_width_hi ]))
        pnames     = np.concatenate((pnames,   [f+'_Adb']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_Adb[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1  

    DA_cont  = lc_obj._contfact_dict
    for i,f in enumerate(filnames):  # add the contamination factors
        params     = np.concatenate((params,   [DA_cont[f].start_value]))
        stepsize   = np.concatenate((stepsize, [DA_cont[f].step_size]))
        pmin       = np.concatenate((pmin,     [DA_cont[f].bounds_lo]))
        pmax       = np.concatenate((pmax,     [DA_cont[f].bounds_hi]))
        prior      = np.concatenate((prior,    [DA_cont[f].prior_mean]))
        priorlow   = np.concatenate((priorlow, [DA_cont[f].prior_width_lo]))
        priorup    = np.concatenate((priorup,  [DA_cont[f].prior_width_hi ]))
        pnames     = np.concatenate((pnames,   [f+'_cont']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_cont[f].step_size != 0.: njumpphot[k]=njumpphot[k]+1          

    DA_ld = lc_obj._ld_dict     #TODO modify ld_dict to use simplenamespace for consistency
    for i,f in enumerate(filnames):  # # add the LD coefficients for the filters to the parameters
        params     = np.concatenate((params,   [DA_ld["q1"][i],        DA_ld["q2"][i]]))
        stepsize   = np.concatenate((stepsize, [DA_ld["step1"][i],     DA_ld["step2"][i]]))
        pmin       = np.concatenate((pmin,     [DA_ld["bound_lo1"][i], DA_ld["bound_lo2"][i]]))
        pmax       = np.concatenate((pmax,     [DA_ld["bound_hi1"][i], DA_ld["bound_hi2"][i]]))
        prior      = np.concatenate((prior,    [DA_ld["q1"][i],        DA_ld["q2"][i]]))
        priorlow   = np.concatenate((priorlow, [DA_ld["sig_lo1"][i],   DA_ld["sig_lo2"][i]]))
        priorup    = np.concatenate((priorup,  [DA_ld["sig_hi1"][i],   DA_ld["sig_hi2"][i]]))
        pnames     = np.concatenate((pnames,   [f+'_q1',f+'_q2']))
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file
        if DA_ld["step1"][i] != 0.: njumpphot[k]=njumpphot[k]+1
        if DA_ld["step2"][i] != 0.: njumpphot[k]=njumpphot[k]+1

    for i in range(nphot):    #add jitter
        if (jit_LCapply[i]=='y'):
            params      = np.concatenate((params,  [jitt_start[i]]), axis=0)    
            stepsize    = np.concatenate((stepsize,[0.1]), axis=0)
            #ensure jitt start values are within the limits else set new limits around the start value
            if lcjitter_loglims[i][0] >= jitt_start[i]: lcjitter_loglims[i][0] = jitt_start[i] - 5
            if lcjitter_loglims[i][1] <= jitt_start[i]: lcjitter_loglims[i][1] = jitt_start[i] + 5

            pmin        = np.concatenate((pmin,    [lcjitter_loglims[i][0]]), axis=0)
            pmax        = np.concatenate((pmax,    [lcjitter_loglims[i][1]]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            pnames      = np.concatenate((pnames,  [f"lc{i+1}_logjitter"]), axis=0)

        else:
            params      = np.concatenate((params,  [-50]), axis=0)
            stepsize    = np.concatenate((stepsize,[0]), axis=0)
            pmin        = np.concatenate((pmin,    [0]), axis=0)
            pmax        = np.concatenate((pmax,    [0]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            pnames      = np.concatenate((pnames,  [f"lc{i+1}_logjitter"]), axis=0)
    
    if custom_LCfunc.func != None: 
        if verbose: print('Setting up parameters for custom LC function ...')
        params     = np.concatenate((params,   [custom_LCfunc.par_dict[key].start_value    for key in custom_LCfunc.par_dict.keys()]))
        stepsize   = np.concatenate((stepsize, [custom_LCfunc.par_dict[key].step_size      for key in custom_LCfunc.par_dict.keys()]))
        pmin       = np.concatenate((pmin,     [custom_LCfunc.par_dict[key].bounds_lo      for key in custom_LCfunc.par_dict.keys()]))
        pmax       = np.concatenate((pmax,     [custom_LCfunc.par_dict[key].bounds_hi      for key in custom_LCfunc.par_dict.keys()]))
        prior      = np.concatenate((prior,    [custom_LCfunc.par_dict[key].prior_mean     for key in custom_LCfunc.par_dict.keys()]))
        priorlow   = np.concatenate((priorlow, [custom_LCfunc.par_dict[key].prior_width_lo for key in custom_LCfunc.par_dict.keys()]))
        priorup    = np.concatenate((priorup,  [custom_LCfunc.par_dict[key].prior_width_hi for key in custom_LCfunc.par_dict.keys()]))
        pnames     = np.concatenate((pnames,   list(custom_LCfunc.par_dict.keys())) )
        njumpphot  = njumpphot + custom_LCfunc.nfree

    sine_dict = lc_obj._sine_dict
    sine_conf = SimpleNamespace(flag  = any([v.trig!= None for v in sine_dict.values()]),   # if any sinusoidal component is present
                                fit   = [v.fit     for v in sine_dict.values()][0],         #type of sine fit ["all","slct","same","filt"]
                                pars  = {k:[]      for k in sine_dict},                     #list to hold start values for each sine component
                                npars = {k:v.npars for k,v in sine_dict.items()},           #number of parameters for each sine component
                                index = {k:[]      for k in sine_dict},                     #index of sine component in the parameter array
                                nfree = {k:v.nfree for k,v in sine_dict.items()},           #number of free parameters for each sine component           
                                trig  = {k:v.trig  for k,v in sine_dict.items()},           #trig function for each sine component
                                n     = {k:v.n     for k,v in sine_dict.items()},           #number of harmonics for each sine component
                                x     = {k:v.par   for k,v in sine_dict.items()},           #column name for each sine component
                                )

    nsin = 0
    # sin_init={"lc1.dat": {"sin1_Amp": 0.01, "sin1_P": 1, "sin1_x0": 0}}
    if sine_conf.flag:
        if sine_conf.fit == "same":
            if verbose: print('Setting up same sinusoidal model for all LCs...')
            sinus      = sine_dict["same"]
            col_alias  = sinus.par.replace("ol","").capitalize()   #column name alias e.g. col5 -> C5
            trig_funcs = ["sin","cos"] if sinus.trig=="sincos" else ["sin"] if sinus.trig==None else [sinus.trig]
            for p in ["Amp","P","x0"]:
                for trig in trig_funcs:
                    for n in range(1,sinus.n+1):    #number of harmonics
                        params     = np.concatenate((params,   [sinus.__dict__[p].start_value]),    axis=0)
                        stepsize   = np.concatenate((stepsize, [sinus.__dict__[p].step_size]),      axis=0)
                        pmin       = np.concatenate((pmin,     [sinus.__dict__[p].bounds_lo]),      axis=0)
                        pmax       = np.concatenate((pmax,     [sinus.__dict__[p].bounds_hi]),      axis=0)
                        prior      = np.concatenate((prior,    [sinus.__dict__[p].prior_mean]),     axis=0)
                        priorlow   = np.concatenate((priorlow, [sinus.__dict__[p].prior_width_hi]), axis=0)
                        priorup    = np.concatenate((priorup,  [sinus.__dict__[p].prior_width_hi]), axis=0)
                        pnames     = np.concatenate((pnames,   [f'{trig}_{f"{n}*" if n>1 else ""}{col_alias}_{p}_same']))  #e.g Same_sin(2*C5)_Amp
                        # if sinus.__dict__[p].step_size>0: njumpphot  = njumpphot + 1
                        sine_conf.pars["same"].append(sinus.__dict__[p].start_value)  #append the start values to the sine_conf.pars list
                        if p!="Amp": break # only continue the n loop for Amp and not for P and x0 which remain same
                    if p!="Amp": break # only continue the trig loop for Amp and not for P and x0 which remain same for sin and cos
            njumpphot += sinus.nfree
            nsin      += sinus.npars
            sine_conf.index["same"] = np.arange(sinus.npars)

        if sine_conf.fit == "filt":  
            if verbose: print('Setting up filter dependent sinusoidal model')
            st_ind = 0
            for filt in filnames:
                sinus      = sine_dict[filt]
                col_alias  = sinus.par.replace("ol","").capitalize()   #column name alias e.g. col5 -> C5
                trig_funcs = ["sin","cos"] if sinus.trig=="sincos" else ["sin"] if sinus.trig==None else [sinus.trig]
                for p in ["Amp","P","x0"]:
                    for trig in trig_funcs:
                        for n in range(1,sinus.n+1):    #number of harmonics
                            params     = np.concatenate((params,   [sinus.__dict__[p].start_value]),    axis=0)
                            stepsize   = np.concatenate((stepsize, [sinus.__dict__[p].step_size]),      axis=0)
                            pmin       = np.concatenate((pmin,     [sinus.__dict__[p].bounds_lo]),      axis=0)
                            pmax       = np.concatenate((pmax,     [sinus.__dict__[p].bounds_hi]),      axis=0)
                            prior      = np.concatenate((prior,    [sinus.__dict__[p].prior_mean]),     axis=0)
                            priorlow   = np.concatenate((priorlow, [sinus.__dict__[p].prior_width_hi]), axis=0)
                            priorup    = np.concatenate((priorup,  [sinus.__dict__[p].prior_width_hi]), axis=0)
                            pnames     = np.concatenate((pnames,   [f'{filt}_{trig}_{f"{n}*" if n>1 else ""}{col_alias}_{p}']))  #e.g V_sin(2*C5)_Amp
                            sine_conf.pars[filt].append(sinus.__dict__[p].start_value)  #append the start values to the sine_conf.pars list
                            if p!="Amp": break # only continue the n loop for Amp and not for P and x0 which remain same
                        if p!="Amp": break # only continue the trig loop for Amp and not for P and x0 which remain same for sin and cos
                
                k = np.where(np.array(lc_obj._filters)== filt)     #  get indices where the filter name is the same as the one in the input file
                njumpphot[k]=njumpphot[k]+sinus.nfree
                nsin      += sinus.npars
                sine_conf.index[filt] = np.arange(st_ind,st_ind+sinus.npars)
                st_ind += sinus.npars

        if sine_conf.fit in ["slct","all"]:
            if verbose: print("Setting up different sinusoids for the LCs")
            st_ind = 0
            for nm in LCnames:
                sinus      = sine_dict[nm]
                col_alias  = sinus.par.replace("ol","").capitalize()   #column name alias e.g. col5 -> C5
                lc_alias   = f"lc{LCnames.index(nm)+1}"                #lc alias e.g. lc1
                trig_funcs = ["sin","cos"] if sinus.trig=="sincos" else ["sin"] if sinus.trig==None else [sinus.trig] 
                for p in ["Amp","P","x0"]:
                    for trig in trig_funcs:
                        for n in range(1,sinus.n+1):    #number of harmonics
                            params     = np.concatenate((params,   [sinus.__dict__[p].start_value]),    axis=0)
                            stepsize   = np.concatenate((stepsize, [sinus.__dict__[p].step_size]),      axis=0)
                            pmin       = np.concatenate((pmin,     [sinus.__dict__[p].bounds_lo]),      axis=0)
                            pmax       = np.concatenate((pmax,     [sinus.__dict__[p].bounds_hi]),      axis=0)
                            prior      = np.concatenate((prior,    [sinus.__dict__[p].prior_mean]),     axis=0)
                            priorlow   = np.concatenate((priorlow, [sinus.__dict__[p].prior_width_hi]), axis=0)
                            priorup    = np.concatenate((priorup,  [sinus.__dict__[p].prior_width_hi]), axis=0)
                            pnames     = np.concatenate((pnames,   [f'{lc_alias}_{trig}_{f"{n}*" if n>1 else ""}{col_alias}_{p}']))  #e.g lc1_sin(2*C5)_Amp
                            sine_conf.pars[nm].append(sinus.__dict__[p].start_value)  #append the start values to the sine_conf.pars_dict
                            if p!="Amp": break # only continue the n loop for Amp and not for P and x0 which remain same
                        if p!="Amp": break # only continue the trig loop for Amp and not for P and x0 which remain same for sin and cos
                njumpphot[LCnames.index(nm)] += sinus.nfree
                nsin      += sinus.npars
                sine_conf.index[nm] = np.arange(st_ind,st_ind+sinus.npars)
                st_ind += sinus.npars           


    for i in range(nRV):
        params      = np.concatenate((params,  [rv_dict["gamma"][i].start_value]),    axis=0)
        stepsize    = np.concatenate((stepsize,[rv_dict["gamma"][i].step_size]),      axis=0)
        pmin        = np.concatenate((pmin,    [rv_dict["gamma"][i].bounds_lo]),      axis=0)
        pmax        = np.concatenate((pmax,    [rv_dict["gamma"][i].bounds_hi]),      axis=0)
        prior       = np.concatenate((prior,   [rv_dict["gamma"][i].prior_mean]),     axis=0)
        priorlow    = np.concatenate((priorlow,[rv_dict["gamma"][i].prior_width_lo]), axis=0)
        priorup     = np.concatenate((priorup, [rv_dict["gamma"][i].prior_width_hi]), axis=0)
        pnames      = np.concatenate((pnames,  [f"rv{i+1}_gamma"]), axis=0)
        if rv_dict["gamma"][i].step_size!=0: njumpRV[i]=njumpRV[i]+1

        if (jit_apply[i]=='y'):
            params      = np.concatenate((params,  [rvjitt_start[i]]), axis=0)
            stepsize    = np.concatenate((stepsize,[0.0001]), axis=0)
            #ensure jitt start values are within the limits else set new limits around the start value
            if rvjitter_lims[i][0] >= rvjitt_start[i]: rvjitter_lims[i][0] = 0
            if rvjitter_lims[i][1] <= rvjitt_start[i]: rvjitter_lims[i][1] = rvjitt_start[i] * 5

            pmin        = np.concatenate((pmin,    [rvjitter_lims[i][0]]), axis=0)
            pmax        = np.concatenate((pmax,    [rvjitter_lims[i][1]]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            pnames      = np.concatenate((pnames,  [f"rv{i+1}_jitter"]), axis=0)

        else:
            params      = np.concatenate((params,  [0.]), axis=0)
            stepsize    = np.concatenate((stepsize,[0.]), axis=0)
            pmin        = np.concatenate((pmin,    [0.]), axis=0)
            pmax        = np.concatenate((pmax,    [0]),  axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            pnames      = np.concatenate((pnames,  [f"rv{i+1}_jitter"]), axis=0)
    
    if custom_RVfunc.func != None: 
        if verbose: print('Setting up parameters for custom RV function ...')
        params     = np.concatenate((params,   [custom_RVfunc.par_dict[key].start_value    for key in custom_RVfunc.par_dict.keys()]))
        stepsize   = np.concatenate((stepsize, [custom_RVfunc.par_dict[key].step_size      for key in custom_RVfunc.par_dict.keys()]))
        pmin       = np.concatenate((pmin,     [custom_RVfunc.par_dict[key].bounds_lo      for key in custom_RVfunc.par_dict.keys()]))
        pmax       = np.concatenate((pmax,     [custom_RVfunc.par_dict[key].bounds_hi      for key in custom_RVfunc.par_dict.keys()]))
        prior      = np.concatenate((prior,    [custom_RVfunc.par_dict[key].prior_mean     for key in custom_RVfunc.par_dict.keys()]))
        priorlow   = np.concatenate((priorlow, [custom_RVfunc.par_dict[key].prior_width_lo for key in custom_RVfunc.par_dict.keys()]))
        priorup    = np.concatenate((priorup,  [custom_RVfunc.par_dict[key].prior_width_hi for key in custom_RVfunc.par_dict.keys()]))
        pnames     = np.concatenate((pnames,   list(custom_RVfunc.par_dict.keys())) )
        njumpRV    = njumpRV + custom_RVfunc.nfree        


    #################################### GP setup #########################################
    if lc_obj._GP_dict != {}: 
        if verbose: print('Setting up photometry GPs ...')
    
    GPobjects   = []  # list to hold the GP objects for each lc
    GPparams    = []  # list to hold the GP parameters for each lc
    GPstepsizes = []  # list to hold the GP step sizes for each lc
    GPindex     = []  # this array contains the lightcurve index of the lc it applies to
    GPprior     = []  # list to hold the GP priors for each lc
    GPpriwid    = []  # list to hold the GP prior widths for each lc
    GPlimup     = []  # list to hold the GP upper limits for each lc
    GPlimlo     = []  # list to hold the GP lower limits for each lc
    GPnames     = []  # list to hold the GP names for each lc
    pargps      = []  # list to hold independent variables of GP for each lc
    gp_colnames  = []  # list to hold column name of independent variable
    gpkerns     = []  # list to hold the kernel for each lc

    GPdict   = lc_obj._GP_dict
    sameLCgp = lc_obj._sameLCgp

    # =================PHOTOMETRY =========================================
    nbc_tot = np.copy(0)  # total number of baseline coefficients let to vary (leastsq OR jumping)

    for i in range(nphot):
        # t, flux, err, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in = np.loadtxt(fpath+names[i], usecols=(0,1,2,3,4,5,6,7,8), unpack = True)  # reading in the data
        thisLCdata = input_lcs[LCnames[i]]
        t, flux, err, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in = thisLCdata.values()
        
        if (divwhite=='y'): # if the divide - white is activated, divide the lcs by the white noise model before proceeding
            dwCNM = np.copy(dwCNMarr[dwCNMind[groups[i]-1]])
            flux=np.copy(flux/dwCNM)
                
        t_arr     = np.concatenate((t_arr, t),    axis=0)
        f_arr     = np.concatenate((f_arr, flux), axis=0)
        e_arr     = np.concatenate((e_arr, err),  axis=0)
        lind      = np.concatenate((lind,  np.zeros(len(t), dtype=int) + i),  axis=0)   # lightcurve index array: filled with i
        indices   = np.where(lind == i)
        indlist.append(indices)
        
        #baseline parameters
        # first, also allocate spots in the params array for the BL coefficients, but set them all to 0/1 and the stepsize to 0
        offset, dcol0, dcol3, dcol4, dcol5, dcol6, dcol7, dcol8, dsin, dCNM, nbc = basecoeff(bases[i],useSpline_lc[LCnames[i]],bases_init[i],lcbases_lims[i],fit_offset[i])  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot      = nbc_tot+nbc # add up the number of jumping baseline coeff
        njumpphot[i] = njumpphot[i]+nbc   # each LC has another jump pm

        # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
        if (baseLSQ == "y"):
            abvar=np.concatenate(([offset[1,:],dcol0[1,:],dcol3[1,:],dcol4[1,:],dcol5[1,:],dcol6[1,:],dcol7[1,:],dcol8[1,:],dsin[1,:],dCNM[1,:]]))
            abind=np.where(abvar!=0.)
            bvars.append(abind)
            offset[1,:]=dcol0[1,:]=dcol3[1,:]=dcol4[1,:]=dcol5[1,:]=dcol6[1,:]=dcol7[1,:]=dcol8[1,:]=dsin[1,:]=dCNM[1,:]=0      # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

        # append these to the respective mcmc input arrays
        params    = np.concatenate((params,   offset[0,:],dcol0[0,:],dcol3[0,:],dcol4[0,:],dcol5[0,:],dcol6[0,:],dcol7[0,:],dcol8[0,:],dsin[0,:],dCNM[0,:]))
        stepsize  = np.concatenate((stepsize, offset[1,:],dcol0[1,:],dcol3[1,:],dcol4[1,:],dcol5[1,:],dcol6[1,:],dcol7[1,:],dcol8[1,:],dsin[1,:],dCNM[1,:]))
        pmin      = np.concatenate((pmin,     offset[2,:],dcol0[2,:],dcol3[2,:],dcol4[2,:],dcol5[2,:],dcol6[2,:],dcol7[2,:],dcol8[2,:],dsin[2,:],dCNM[2,:]))
        pmax      = np.concatenate((pmax,     offset[3,:],dcol0[3,:],dcol3[3,:],dcol4[3,:],dcol5[3,:],dcol6[3,:],dcol7[3,:],dcol8[3,:],dsin[3,:],dCNM[3,:]))
        prior     = np.concatenate((prior,    np.zeros(22)))
        priorlow  = np.concatenate((priorlow, np.zeros(22)))
        priorup   = np.concatenate((priorup,  np.zeros(22)))
        pnames    = np.concatenate((pnames, [f"lc{i+1}_off",f"lc{i+1}_A0",f"lc{i+1}_B0",f"lc{i+1}_C0",f"lc{i+1}_D0",
                                            f"lc{i+1}_A3",f"lc{i+1}_B3",
                                            f"lc{i+1}_A4",f"lc{i+1}_B4",
                                            f"lc{i+1}_A5",f"lc{i+1}_B5",
                                            f"lc{i+1}_A6",f"lc{i+1}_B6",
                                            f"lc{i+1}_A7",f"lc{i+1}_B7",
                                            f"lc{i+1}_A8",f"lc{i+1}_B8",
                                            f"lc{i+1}_sin_amp",f"lc{i+1}_sin_freq",f"lc{i+1}_sin_phi",
                                            f"lc{i+1}_ACNM",f"lc{i+1}_BCNM"
                                            ]))        
        # note currently we have the following parameters in these arrays:
        #   [rho_star,                                   (1)
        #   [T0,RpRs,b,per,sesinw, secosw,K,                   (7)*npl
        #   ttv,...                                      (nttv)
        #   ddf_1, ..., ddf_n,                           (nddf)
        #   (occ_1,Fn_1,phoff_1,Aev_1,A_db_1,cont_1),...,occ_n,Fn_n,phoff_n,Aev_n,Adb_n,cont_n(6*nocc)
        #   q1_f1,q2_f1, q1_f2, .... , q2fn,            (2*n_filt)
        #   LC_jit                                       (nphot)
        #   Rv_gamma, RV_jit                              (2*nRVs)         
        #   baseline                                       22, ...]
        #    = 1+7*npl+nttv+nddf+nocc*6+2*n_filt+nphot+ncustom +2*nRV + 22*nphot
        #    each lightcurve has 22 possible baseline jump parameters, starting with index  1+7*npl+nttv+nddf+nocc*6+2*n_filt+nphot+ncustom +2*nRV

        # pargp_all = np.vstack((t, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in)).T  # the matrix with all the possible inputs to the GPs

        if (useGPphot[i]=='n'):
            GPobjects.append([])
            pargps.append([])
            gpkerns.append([])
            gp_colnames.append([])

        elif useGPphot[i] in ['ge','ce','sp']:     #George or Celerite GP
            gp_conv  = gp_params_convert()   #class containing functions to convert gp amplitude and lengthscale to the required values for the different kernels 
            thisLCgp = GPdict[LCnames[i]]        #the GP dictionary for this LC
            gpcols   = [thisLCgp[f"amplitude{n}"].user_data.col for n in range(thisLCgp["ngp"])]     #data column names to use for the GP
            gpkerns.append([thisLCgp[f"amplitude{n}"].user_data.kernel for n in range(thisLCgp["ngp"])]) #GP kernels of this lc

            for n in range(thisLCgp["ngp"]):                      #loop through the number of GPs for this LC
                gpkern = thisLCgp[f"amplitude{n}"].user_data.kernel   #kernel to use for this GP
                gpcol  = thisLCgp[f"amplitude{n}"].user_data.col   #column of the data to use for this GP

                #TODO: FOR SAMEGP, dont append parameters that will not be used (with name None)
                #use if sameLCgp.flag and i != sameLCgp.first_index: continue
                #    else append the parameters

                GPparams    = np.concatenate((GPparams,    [thisLCgp[f"amplitude{n}"].start_value, thisLCgp[f"lengthscale{n}"].start_value]), axis=0)
                GPstepsizes = np.concatenate((GPstepsizes, [thisLCgp[f"amplitude{n}"].step_size, thisLCgp[f"lengthscale{n}"].step_size]), axis=0)
                GPindex     = np.concatenate((GPindex,     (np.zeros(2)+i)), axis=0)
                GPprior     = np.concatenate((GPprior,     [thisLCgp[f"amplitude{n}"].prior_mean, thisLCgp[f"lengthscale{n}"].prior_mean]), axis=0)
                GPpriwid    = np.concatenate((GPpriwid,    [thisLCgp[f"amplitude{n}"].prior_width_lo, thisLCgp[f"lengthscale{n}"].prior_width_lo]), axis=0)
                GPlimup     = np.concatenate((GPlimup,     [thisLCgp[f"amplitude{n}"].bounds_hi, thisLCgp[f"lengthscale{n}"].bounds_hi]), axis=0)
                GPlimlo     = np.concatenate((GPlimlo,     [thisLCgp[f"amplitude{n}"].bounds_lo, thisLCgp[f"lengthscale{n}"].bounds_lo]), axis=0)
                if sameLCgp.flag:
                    if i == sameLCgp.first_index:
                        GPnames = np.concatenate((GPnames,     [f"GPlcSame_Amp{n+1}_{gpcol}",f"GPlcSame_len{n+1}_{gpcol}"]), axis=0)
                    else:
                        GPnames = np.concatenate((GPnames,     ['GPlcSame_Amp_None', 'GPlcSame_len_None']),axis=0)
                elif sameLCgp.filtflag:
                    if i == sameLCgp.first_index[lc_obj._filters[i]]:
                        GPnames = np.concatenate((GPnames,     [f"GPlcFilt_{lc_obj._filters[i]}_Amp{n+1}_{gpcol}",f"GPlcFilt_{lc_obj._filters[i]}_len{n+1}_{gpcol}"]), axis=0)
                    else:
                        GPnames = np.concatenate((GPnames,     ['GPlcFilt_Amp_None', 'GPlcFilt_len_None']),axis=0)
                else:
                    GPnames = np.concatenate((GPnames,     [f"GPlc{i+1}_Amp{n+1}_{gpcol}",f"GPlc{i+1}_len{n+1}_{gpcol}"]), axis=0)

                if useGPphot[i]=="ge":  #George GP
                    assert gpkern in george_kernels.keys(), f"Invalid kernel '{gpkern}' for George GP, must be one of {list(george_kernels.keys())}"
                    ndim_gp  = len(set(gpcols))       #number of different columns used for the GP
                    axes_gp  = [gpcols.index(gpcol)]  #axes of the GP (0 or 1)

                    if n==0: 
                        kern = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels="ge_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value,
                                                                                                    thisLCgp[f"lengthscale{n}"].start_value])
                        kern.set_parameter_vector([gppar1, gppar2])  
                        gp_x = thisLCdata[gpcol]  # the x values for the GP
                        col_name = gpcol

                    if n==1:                       # if this is the second GP, then add/mult the new kernel to the previous one
                        kern2 = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        gppar1, gppar2 =  gp_conv.get_values(kernels="ge_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value,
                                                                                                    thisLCgp[f"lengthscale{n}"].start_value])
                        kern2.set_parameter_vector([gppar1, gppar2])
                        if thisLCgp["op"]=="+": kern += kern2
                        if thisLCgp["op"]=="*": kern *= kern2
                        
                        if ndim_gp >1: 
                            gp_x = np.vstack((gp_x, thisLCdata[gpcol])).T  #2D array with the x values for the GP
                            col_name = (col_name, gpcol)

                    gp = GP(kern, mean=0)
                    col_nm = col_name
            
                if useGPphot[i]=="ce":   #Celerite GP
                    assert gpkern in celerite_kernels.keys(), f"Invalid kernel '{gpkern}' for Celerite GP, must be one of {list(celerite_kernels.keys())}"
                    if n==0: 
                        if gpkern=="sho":
                            kern  = celerite_kernels[gpkern](log_S0 =-10, log_Q=np.log(lc_Qsho), log_omega0=1)  #dummy initialization
                            kern.freeze_parameter("log_Q")   #freeze Q
                        else:
                            kern  = celerite_kernels[gpkern](-10, 1)   #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels="ce_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value, thisLCgp[f"lengthscale{n}"].start_value],
                                                                fixed_arg = lc_Qsho)
                        kern.set_parameter_vector([gppar1, gppar2])
                    if n==1:
                        if gpkern=="sho": 
                            kern2 = celerite_kernels[gpkern](log_S0 =-10, log_Q=np.log(lc_Qsho), log_omega0=1) 
                            kern2.freeze_parameter("log_Q")
                        else:
                            kern2 = celerite_kernels[gpkern](-10, 1)
                        #starting values of next kernel
                        gppar1, gppar2 =  gp_conv.get_values(kernels="ce_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value, thisLCgp[f"lengthscale{n}"].start_value], 
                                                                fixed_arg = lc_Qsho)
                        kern2.set_parameter_vector([gppar1, gppar2])
                        
                        if thisLCgp["op"]=="+": kern += kern2
                        if thisLCgp["op"]=="*": kern *= kern2
                    
                    gp_x = thisLCdata[gpcol] # the x values for the GP,
                    gp   = cGP(kern, mean=0, fit_mean = False)
                    col_nm = gpcol

                if useGPphot[i]=="sp":   #spleaf GP
                    assert gpkern in spleaf_kernels.keys(), f"Invalid kernel '{gpkern}' for Spleaf GP, must be one of {list(spleaf_kernels.keys())}"
                    if n==0: 
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels="sp_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value, thisLCgp[f"lengthscale{n}"].start_value])
                        kern = spleaf_kernels[gpkern](gppar1, gppar2) if gpkern!="sho" else spleaf_kernels[gpkern](gppar1, gppar2, lc_Qsho)
                    if n==1:
                        #starting values of next kernel
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value, thisLCgp[f"lengthscale{n}"].start_value])
                        kern2 = spleaf_kernels[gpkern](gppar1, gppar2) if gpkern!="sho" else spleaf_kernels[gpkern](gppar1, gppar2, lc_Qsho)

                        
                        # if thisLCgp["op"]=="+": kern += kern2
                        # if thisLCgp["op"]=="*": kern *= kern2
                    
                    gp_x   = thisLCdata[gpcol] # the x values for the GP,
                    gp     = {"k1":kern} if thisLCgp["ngp"]==1 else {"k1":kern, "k2":kern2} #spleaf kernel dictionary
                    col_nm = gpcol

            GPobjects.append(gp)
            pargps.append(gp_x) 
            gp_colnames.append(col_nm)

    # =================RADIAL VELOCITY =========================================
    if nRV > 0: 
        if verbose: print('Setting up RV arrays ...')
    if np.any([spl.use for spl in useSpline_rv.values()]): 
        if verbose: print('Setting up Spline fitting for RVs ...')  
    
    rvGPobjects,rvGPnames,rv_pargps,rv_gpkerns,rv_gp_colnames  = [],[],[],[],[]
    rvGPparams,rvGPstepsizes,rvGPindex,rvGPprior,rvGPpriwid,rvGPlimup,rvGPlimlo = [],[],[],[],[],[],[]
    rvGPdict = {} if rv_obj is None else rv_obj._rvGP_dict
    sameRVgp = False if rv_obj is None else rv_obj._sameRVgp                                           # use spline to interpolate the light curve


    for i in range(nRV):
        # t, rv, err, bis, fwhm, contrast = np.loadtxt(rv_fpath+RVnames[i], usecols=(0,1,2,3,4,5), unpack = True)  # reading in the data
        thisRVdata = input_rvs[RVnames[i]]
        t, rv_obj, err, bis, fwhm, contrast = thisRVdata.values()

        t_arr    = np.concatenate((t_arr,t), axis=0)
        f_arr    = np.concatenate((f_arr,rv_obj), axis=0)    # ! add the RVs to the "flux" array !
        e_arr    = np.concatenate((e_arr,err), axis=0)   # ! add the RV errors to the "earr" array !
        lind     = np.concatenate((lind,np.zeros(len(t),dtype=int)+i+nphot), axis=0)   # indices
        indices  = np.where(lind==i+nphot)
        indlist.append(indices)
        Pin      = sinPs[i]

        #rv baseline 
        dcol0RV, dcol3RV, dcol4RV, dcol5RV,dsinRV,nbcRV = basecoeffRV(RVbases[i],Pin,RVbases_init[i],rvbases_lims[i])  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot = nbc_tot+nbcRV # add up the number of jumping baseline coeff
        abvar=np.concatenate(([dcol0RV[1,:],dcol3RV[1,:],dcol4RV[1,:],dcol5RV[1,:],dsinRV[1,:]]))
        abind=np.where(abvar!=0.)
        njumpRV[i] = njumpRV[i]+len(abind)
    
        if (baseLSQ == "y"):
            bvarsRV.append(abind)
            dcol0RV[1,:]=dcol3RV[1,:]=dcol4RV[1,:]=dcol5RV[1,:]=dsinRV[1,:]=0        # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters
        # append these to the respective mcmc input arrays
        params    = np.concatenate((params,   dcol0RV[0,:],dcol3RV[0,:],dcol4RV[0,:],dcol5RV[0,:],dsinRV[0,:]))
        stepsize  = np.concatenate((stepsize, dcol0RV[1,:],dcol3RV[1,:],dcol4RV[1,:],dcol5RV[1,:],dsinRV[1,:]))
        pmin      = np.concatenate((pmin,     dcol0RV[2,:],dcol3RV[2,:],dcol4RV[2,:],dcol5RV[2,:],dsinRV[2,:]))
        pmax      = np.concatenate((pmax,     dcol0RV[3,:],dcol3RV[3,:],dcol4RV[3,:],dcol5RV[3,:],dsinRV[3,:]))
        prior     = np.concatenate((prior,    np.zeros(12)))
        priorlow  = np.concatenate((priorlow, np.zeros(12)))
        priorup   = np.concatenate((priorup,  np.zeros(12)))
        pnames    = np.concatenate((pnames, [f"rv{i+1}_A0",f"rv{i+1}_B0",
                                            f"rv{i+1}_A3",f"rv{i+1}_B3",
                                            f"rv{i+1}_A4",f"rv{i+1}_B4",
                                            f"rv{i+1}_A5",f"rv{i+1}_B5",
                                            f"rv{i+1}_sin_amp",f"rv{i+1}_sin_freq",f"rv{i+1}_sin_phi",f"rv{i+1}_sin_phi2"]))            

        # calculate the weights for the lightcurves to be used for the CNM calculation later: do this in a function!
        #ewarr=grweights(earr,indlist,grnames,groups,ngroup)

        if useGPrv[i]=='n':
            rvGPobjects.append([])
            rv_pargps.append([]) 
            rv_gpkerns.append([])
            rv_gp_colnames.append([])

        if useGPrv[i] in ["ge","ce","sp"]:         #George or Celerite GP
            gp_conv  = gp_params_convert()   #class containing functions to convert gp amplitude and lengthscale to the required values for the different kernels 
            thisRVgp = rvGPdict[RVnames[i]]
            gpcols   = [thisRVgp[f"amplitude{n}"].user_data.col for n in range(thisRVgp["ngp"])]
            rv_gpkerns.append([thisRVgp[f"amplitude{n}"].user_data.kernel for n in range(thisRVgp["ngp"])])

            for n in range(thisRVgp["ngp"]):                      #loop through the number of GPs for this RV
                gpkern = thisRVgp[f"amplitude{n}"].user_data.kernel   #kernel to use for this GP
                gpcol  = thisRVgp[f"amplitude{n}"].user_data.col   #column of the data to use for this GP

                rvGPparams    = np.concatenate((rvGPparams,    [thisRVgp[f"amplitude{n}"].start_value, thisRVgp[f"lengthscale{n}"].start_value]), axis=0)
                rvGPstepsizes = np.concatenate((rvGPstepsizes, [thisRVgp[f"amplitude{n}"].step_size, thisRVgp[f"lengthscale{n}"].step_size]), axis=0)
                rvGPindex     = np.concatenate((rvGPindex,     (np.zeros(2)+i)), axis=0)
                rvGPprior     = np.concatenate((rvGPprior,     [thisRVgp[f"amplitude{n}"].prior_mean, thisRVgp[f"lengthscale{n}"].prior_mean]), axis=0)
                rvGPpriwid    = np.concatenate((rvGPpriwid,    [thisRVgp[f"amplitude{n}"].prior_width_lo, thisRVgp[f"lengthscale{n}"].prior_width_lo]), axis=0)
                rvGPlimup     = np.concatenate((rvGPlimup,     [thisRVgp[f"amplitude{n}"].bounds_hi, thisRVgp[f"lengthscale{n}"].bounds_hi]), axis=0)
                rvGPlimlo     = np.concatenate((rvGPlimlo,     [thisRVgp[f"amplitude{n}"].bounds_lo, thisRVgp[f"lengthscale{n}"].bounds_lo]), axis=0)
                if sameRVgp.flag:
                    if i == sameRVgp.first_index:
                        rvGPnames = np.concatenate((rvGPnames,     [f"GPrvSame_Amp{n+1}_{gpcol}",f"GPrvSame_len{n+1}_{gpcol}"]), axis=0)
                    else:
                        rvGPnames = np.concatenate((rvGPnames,     ['GPrvSame_Amp_None', 'GPrvSame_len_None']),axis=0)
                else:
                    rvGPnames = np.concatenate((rvGPnames,     [f"GPrv{i+1}_Amp{n+1}_{gpcol}",f"GPrv{i+1}_len{n+1}_{gpcol}"]), axis=0)


                if useGPrv[i]=="ge":  #George GP
                    assert gpkern in george_kernels.keys(), f"Invalid kernel '{gpkern}' for George GP, must be one of {list(george_kernels.keys())}"
                    ndim_gp  = len(set(gpcols))       #number of different columns used for the GP
                    axes_gp  = [gpcols.index(gpcol)]  #axes of the GP (0 or 1)

                    if n==0: 
                        kern = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels="ge_"+gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value, thisRVgp[f"lengthscale{n}"].start_value])
                        kern.set_parameter_vector([gppar1, gppar2])  
                        gp_x = thisRVdata[gpcol]  # the x values for the GP
                        col_name = gpcol

                    if n==1:                       # if this is the second GP, then add/mult the new kernel to the previous one
                        kern2 = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        gppar1, gppar2 =  gp_conv.get_values(kernels="ge_"+gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value, thisRVgp[f"lengthscale{n}"].start_value])
                        kern2.set_parameter_vector([gppar1, gppar2])
                        if thisRVgp["op"]=="+": kern += kern2
                        if thisRVgp["op"]=="*": kern *= kern2
                        
                        if ndim_gp >1: 
                            gp_x = np.vstack((gp_x, thisRVdata[gpcol])).T  #2D array with the x values for the GP
                            col_name = (col_name, gpcol)
                    
                    gp = GP(kern, mean=0)
                    col_nm = col_name
            
                if useGPrv[i]=="ce":   #Celerite GP
                    assert gpkern in celerite_kernels.keys(), f"Invalid kernel '{gpkern}' for Celerite GP, must be one of {list(celerite_kernels.keys())}"
                    if n==0: 
                        if gpkern=="sho":
                            kern  = celerite_kernels[gpkern](log_S0 =-10, log_Q=np.log(rv_Qsho), log_omega0=1)  #dummy initialization
                            kern.freeze_parameter("log_Q")   #freeze Q
                        else:
                            kern  = celerite_kernels[gpkern](-10, 1)   #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value, thisRVgp[f"lengthscale{n}"].start_value],
                                                                fixed_arg = rv_Qsho)
                        kern.set_parameter_vector([gppar1, gppar2])
                    if n==1:
                        if gpkern=="sho": 
                            kern2 = celerite_kernels[gpkern](log_S0 =-10, log_Q=np.log(rv_Qsho), log_omega0=1) 
                            kern2.freeze_parameter("log_Q")
                        else:
                            kern2 = celerite_kernels[gpkern](-10, 1)
                        #starting values of next kernel
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value, thisRVgp[f"lengthscale{n}"].start_value],
                                                                fixed_arg = rv_Qsho)
                        kern2.set_parameter_vector([gppar1, gppar2])
                        
                        if thisRVgp["op"]=="+": kern += kern2
                        if thisRVgp["op"]=="*": kern *= kern2
                    
                    gp_x = thisRVdata[gpcol] # the x values for the GP,
                    gp   = cGP(kern, mean=0, fit_mean=False)
                    col_nm = gpcol

                if useGPrv[i]=="sp":   #spleaf GP
                    assert gpkern in spleaf_kernels.keys(), f"Invalid kernel '{gpkern}' for Spleaf GP, must be one of {list(spleaf_kernels.keys())}"
                    if n==0: 
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value,
                                                                                            thisRVgp[f"lengthscale{n}"].start_value])
                        kern = spleaf_kernels[gpkern](gppar1, gppar2) if gpkern!="sho" else spleaf_kernels[gpkern](gppar1, gppar2, rv_Qsho)
                    if n==1:
                        #starting values of next kernel
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value,
                                                                                            thisRVgp[f"lengthscale{n}"].start_value])
                        kern2 = spleaf_kernels[gpkern](gppar1, gppar2) if gpkern!="sho" else spleaf_kernels[gpkern](gppar1, gppar2, rv_Qsho)

                        # if thisRVgp["op"]=="+": kern += kern2
                        # if thisRVgp["op"]=="*": kern *= kern2
                    
                    gp_x   = thisRVdata[gpcol]
                    gp     = {"k1":kern} if thisRVgp["ngp"]==1 else {"k1":kern, "k2":kern2}  #dictionary with the GP kernels
                    col_nm = gpcol

            rvGPobjects.append(gp)
            rv_pargps.append(gp_x) 
            rv_gp_colnames.append(col_nm)


    inmcmc='n'

    LCjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific lc
    #ATTENTION: pass to the lnprob function the individual subscript (of variable p) that are its jump parameters for each LC
    # which indices of p0 are referring to lc n
    if nttv>0: nt0 = 0
    

    for i in range(nphot):
        
        temp=np.ndarray([0])  # the indices of the parameters that jump for this LC
        
        tr_ind = np.array([0])   #rho_star/duration index
        tr_ind = np.append(tr_ind, np.concatenate([np.arange(1,7)+7*n for n in range(npl)])) # add index of the 6 other transit jump parameters for all planets (no K)
        lcstep = tr_ind[np.where(stepsize[tr_ind]!=0.)[0]]
        
        if (len(lcstep) > 0): 
            temp=np.copy(lcstep)

        #transit timing variation
        if ttvYN=='y':
            ttvind = 1+7*npl+ np.arange(nt0,nt0+len(ttv_conf[i].t0s))   #add indices of the t0s of this lc
            nt0 += len(ttv_conf[i].t0s)
            temp=np.concatenate((np.asarray(temp),ttvind),axis=0)

        
        # define the index in the set of filters that this LC has:
        k = np.where(filnames == filters[i])  # k is the index of the LC in the filnames array
        k = k[0].item()
        
        #transit depth variation
        if (ddfYN=='y'):    
            if temp.shape:    
                temp=np.concatenate((np.asarray(temp),np.asarray([1+7*npl+nttv+k])),axis=0)
            else:
                temp=np.asarray([1+7*npl+nttv+k])
        
        #occultation
        occind=1+7*npl+nttv+nddf+k                   # the index of the occultation parameter for this LC
        if (stepsize[occind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[occind]),axis=0)

        #nightside flux
        Fn_ind=1+7*npl+nttv+nddf+nocc+k                   # the index of the first Fn for this LC
        if (stepsize[Fn_ind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[Fn_ind]),axis=0)
        
        #phpff
        phoff_ind=1+7*npl+nttv+nddf+nocc*2+k                   # the index of the first phase offset for this LC
        if (stepsize[phoff_ind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[phoff_ind]),axis=0)

        #ev
        Aev_ind=1+7*npl+nttv+nddf+nocc*3+k                   # the index of the first ev amplitude for this LC
        if (stepsize[Aev_ind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[Aev_ind]),axis=0)

        #db
        Adb_ind=1+7*npl+nttv+nddf+nocc*4+k                   # the index of the first db amplitude for this LC
        if (stepsize[Adb_ind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[Adb_ind]),axis=0)

        #cont
        contind=1+7*npl+nttv+nddf+nocc*5+k                   # the index of the contam parameter for this LC
        if (stepsize[contind]!=0.):        # if nonzero stepsize ->it is jumping, add it to the list
            temp=np.concatenate((np.asarray(temp),[contind]),axis=0)

        #limb darkening
        q1ind=1+7*npl+nttv+nddf+nocc*6+k*2
        if (stepsize[q1ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[q1ind]),axis=0)
        
        q2ind=1+7*npl+nttv+nddf+nocc*6+k*2+1
        if (stepsize[q2ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[q2ind]),axis=0)

        #jitter
        LCjitterind = 1+7*npl +nttv+ nddf+nocc*6 + nfilt*2 + i 
        if (stepsize[LCjitterind]!=0.):           
            temp=np.concatenate((temp,[LCjitterind]),axis=0)

        #custom LC parameters
        if ncustom>0:
            customind = 1+7*npl +nttv+ nddf+nocc*6 + nfilt*2 + nphot + np.arange(ncustom) # the index of the custom parameters 
            cst_step  = customind[np.where(stepsize[customind]!=0.)[0] ]       
            temp=np.concatenate((temp,cst_step),axis=0)

        #sinuoid parameters:
        if sine_conf.flag:
            file_slct = filnames[k] if sine_conf.fit=="filt" else "same" if sine_conf.fit=="same" else LCnames[i]            
            sin_ind   = 1+7*npl +nttv+ nddf+nocc*6 + nfilt*2 + nphot + ncustom + sine_conf.index[file_slct]  #the indices of this lc/filt sinuoid parameters
            sin_step  = sin_ind[np.where(stepsize[sin_ind]!=0.)[0] ]       
            temp=np.concatenate((temp,sin_step),axis=0)
    
        #baseline
        bfstart= 1+7*npl+nttv+nddf+nocc*6+nfilt*2 + nphot + ncustom + nsin + nRV*2 + ncustomRV # the first index in the param array that refers to a baseline function    
        blind = np.asarray(list(range(bfstart+i*22,bfstart+i*22+22)))  # the indices for the coefficients for the base function   

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
        
        LCjump.append(temp)   #append parameters that jump for this LC
        
    RVjump = [] # a list where each item contain a list of the indices of params that jump and refer to this specific RV dataset
    for i in range(nRV):
        temp=np.ndarray([])
        
        rv_ind  = np.array([0])     #rho_star/duration
        rv_ind  = np.append(rv_ind, np.concatenate([np.arange(1,8)+7*n for n in range(npl)])) # the common RV jump parameters: transit + K for all planets
        rvstep1 = np.where(stepsize[rv_ind]!=0.)  
        rvstep = rvstep1[0]
        
        if (len(rvstep) > 0): 
            temp=np.copy(rvstep)

        # identify the gamma index of this RV (note: each gamma comes with a jitter, so 2 indices needed per rvdata)
        gammaind  = 1+7*npl +nttv+ nddf+nocc*6 + nfilt*2 + nphot + ncustom + nsin + i*2
        jitterind = 1+7*npl +nttv+ nddf+nocc*6 + nfilt*2 + nphot + ncustom + nsin + i*2 + 1

        if (stepsize[gammaind]!=0.):           
            # temp=np.concatenate((temp,[gammaind]),axis=0)
            temp= np.append(temp, gammaind)

        if (stepsize[jitterind]!=0.):           
            # temp=np.concatenate((temp,[jitterind]),axis=0)
            temp=np.append(temp, jitterind)

        if ncustomRV>0:
            customRVind = 1+7*npl +nttv+ nddf+nocc*6 + nfilt*2 + nphot + ncustom + nsin + nRV*2 + np.arange(ncustomRV) # the index of the custom parameters 
            cstRV_step  = customRVind[np.where(stepsize[customRVind]!=0.)[0] ]       
            temp=np.concatenate((temp,cstRV_step),axis=0)
            
        bfstart= 1+7*npl+nttv+nddf+nocc*6+nfilt*2 + nphot + ncustom + nsin + nRV*2 + ncustomRV + nphot*22  # the first index in the param array that refers to an RV baseline function    
        blind = np.asarray(list(range(bfstart+i*12,bfstart+i*12+12)))  # the indices for the coefficients for the base function    

        rvstep1 = np.where(stepsize[blind]!=0.)
        if len(rvstep1)>0:
            rvstep = rvstep1[0] 
            temp=np.concatenate((np.asarray(temp),blind[rvstep]),axis=0)

        #and also add the rv GPparams
        rvgind = np.where(rvGPindex==i)
        rvgindl = list(rvgind[0]+len(params)+len(GPparams))
        rvgind = rvgind[0]+len(params)+len(GPparams)

        if rvgindl:
            temp = np.concatenate((temp,rvgind),axis=0)
        
        RVjump.append(temp)
    # =============================== CALCULATION ==========================================

    pnames_all   = np.concatenate((pnames, GPnames,rvGPnames))
    initial      = np.concatenate((params, GPparams,rvGPparams))
    gpsteps      = np.concatenate((GPstepsizes,rvGPstepsizes))
    steps        = np.concatenate((stepsize, gpsteps))

    # for each shared parameter, update the value of the recipient and make sure it is not jumping (step=0)
    for sp in shared_params:
        for s_recip in shared_params[sp]:
            initial[pnames_all == s_recip] = f"->{sp}"#np.nan#initial[pnames_all == sp] . #specify that it takes its value from a shared parameter
            steps[pnames_all == s_recip] = 0

    priors       = np.concatenate((prior, GPprior,rvGPprior))
    priwid       = (priorup + priorlow) / 2.
    priorwids    = np.concatenate((priwid, GPpriwid,rvGPpriwid))
    lim_low      = np.concatenate((pmin, GPlimlo,rvGPlimlo))
    lim_up       = np.concatenate((pmax, GPlimup,rvGPlimup))
    ndim         = np.count_nonzero(steps)
    jumping      = np.where(steps != 0.)                                              # the indices of all the jumping parameters
    jumping_noGP = np.where(stepsize != 0.)                                           # the indices of all the jumping parameters that are not GPs
    jumping_GP   = np.nonzero(np.concatenate(([0]*len(stepsize),gpsteps)))      # the indices of all the jumping parameters that are GPs
    jumping_lcGP = np.nonzero(np.concatenate(([0]*len(stepsize),GPstepsizes)))  # the indices of all the jumping parameters that are GPs for the LCs
    jumping_rvGP = np.nonzero(np.concatenate(([0]*len(stepsize),[0]*len(GPstepsizes),rvGPstepsizes)))  # the indices of all the jumping parameters that are GPs for the RVs

    if get_parameter_names:
        return pnames_all, jumping
    
    pindices = []       #holds the indices of the jumping parameters for each lc/rv in the list contained only of jumping parameters--> pnames_all[jumping]
    for i in range(nphot):
        fullist = list(jumping[0])                          # the indices of all the jumping parameters
        lclist  = list(LCjump[i])                           # the indices of the jumping parameters for this LC
        both    = list(set(fullist).intersection(lclist))   # attention: set() makes it unordered. we'll need to reorder it
        both.sort()
        indices_A = [fullist.index(x) for x in both]
        pindices.append(indices_A)

    for i in range(nRV):
        fullist = list(jumping[0])
        rvlist  = list(RVjump[i])
        both    = list(set(fullist).intersection(rvlist))   # attention: set() makes it unordered. we'll need to reorder it
        both.sort()
        indices_A = [fullist.index(x) for x in both]
        pindices.append(indices_A)
    #pnames[jumping][pindices[0]]  #jumping params in lc1
    ewarr=np.nan#grweights(earr,indlist,grnames,groups,ngroup,nphot)


    ###### create prior distribution for the jumping parameters #####
    ijnames    = np.where(steps != 0.)    #indices of the jumping parameters
    jnames     = pnames_all[[ijnames][0]]  # jnames are the names of the jump parameters
    refl_ind   = []  # indices of the reflective parameters. for now - impact_para
    for idx, jn in enumerate(jnames):
        if jn.startswith("Impact_para"):
            refl_ind.append(idx)
    if refl_ind==[]: refl_ind = None

    norm_sigma = priorwids[jumping]
    norm_mu    = priors[jumping]
    uni_low    = lim_low[jumping]
    uni_up     = lim_up[jumping]

    uni    = lambda lowlim,uplim: uniform(lowlim, uplim-lowlim)               # uniform prior between lowlim and uplim
    t_norm = lambda a,b,mu,sig: truncnorm((a-mu)/sig, (b-mu)/sig, mu, sig) # normal prior(mu,sig) truncated  between a and b

    prior_distr = []      # list of prior distributions for the jumping parameters
    for jj in range(ndim):
        if (norm_sigma[jj]>0.):  #normal prior
            lpri = t_norm(uni_low[jj],uni_up[jj],norm_mu[jj],norm_sigma[jj])
            prior_distr.append(lpri)
        else:                    #uniform prior/loguni for rho_star,Duration,GPpars
            if (jnames[jj] in ["rho_star","Duration"]) or (jnames[jj].startswith('GP')):
                llim = loguniform(uni_low[jj] if uni_low[jj]>0 else 0.001, uni_up[jj])
            else:
                llim = uni(uni_low[jj],uni_up[jj])
            prior_distr.append(llim)

    ## plot the prior distributions
    if verbose: print("Plotting prior distributions")
    matplotlib.use('Agg')
    
    start_pars = initial[jumping]
    nrows = int(np.ceil(len(start_pars)/6))
    fig, ax = plt.subplots(nrows,6, figsize=(15, 2*nrows))
    ax = ax.reshape(-1)
    for jj in range(ndim):
        ax[jj].hist(prior_distr[jj].rvs(1000), label=jnames[jj], density=True)
        ax[jj].axvline(start_pars[jj],color="red",label=f"{start_pars[jj]:.4e}" if start_pars[jj] < 1e-3 else f"{start_pars[jj]:.4f}")
        ax[jj].set_yticks([])
        ax[jj].legend()

    for jj in range(ndim,nrows*6): ax[jj].axis("off")   #remove unused subplots
    plt.subplots_adjust(hspace=0.2)
    fig.savefig(f"{out_folder}/priors.png", bbox_inches="tight")
    plt.close("all")
    matplotlib.use(__default_backend__)



    ############################## Initial model ##################################
    if verbose: print('\nPlotting initial model(s)\n---------------------------')

    inmcmc = 'n'
    indparams = {   "custom_RVfunc": custom_RVfunc,"custom_LCfunc": custom_LCfunc,"nphot": nphot,"nRV": nRV,"sine_conf": sine_conf,
                    "filters": filters,"nfilt": nfilt,"filnames": filnames,"nddf": nddf,"nocc": nocc,"nttv": nttv,"col8_arr": _,"grprs": grprs,"ttv_conf": ttv_conf,"grnames": grnames,"groups": groups,"ngroup": ngroup,"ewarr": ewarr,
                    "inmcmc": inmcmc,"paraCNM": paraCNM,"baseLSQ": baseLSQ,"bvars": bvars,"bvarsRV": bvarsRV,"model_phasevar": lc_obj._model_phasevar,"LCnames": LCnames,"RVnames": RVnames,"lc_Qsho": lc_Qsho,"rv_Qsho": rv_Qsho,"dwCNMarr": dwCNMarr,"dwCNMind": dwCNMind,
                    "params": params,"useGPphot": useGPphot,"useGPrv": useGPrv,"GPobjects": GPobjects,"GPparams": GPparams,"GPindex": GPindex,"pindices": pindices,"jumping": jumping,"jnames": jnames,"prior_distr": prior_distr,"pnames_all": pnames_all,
                    "norm_sigma": norm_sigma,"uni_low": uni_low,"uni_up": uni_up,"rv_gp_colnames": rv_gp_colnames,"gp_colnames": gp_colnames,"gpkerns": gpkerns,"LTT": LTT,"jumping_GP": jumping_GP,"GPstepsizes": GPstepsizes,"sameLCgp": sameLCgp,
                    "npl": npl,"useSpline_lc": useSpline_lc,"useSpline_rv": useSpline_rv,"s_samp": s_samp,"rvGPobjects": rvGPobjects,"rvGPparams": rvGPparams,"rvGPindex": rvGPindex,"input_lcs": input_lcs,"input_rvs": input_rvs,
                    "RVunit": RVunit,"rv_pargps": rv_pargps,"rv_gpkerns": rv_gpkerns,"sameRVgp": sameRVgp,"fit_sampler": fit_sampler, "shared_params":shared_params }
    pickle.dump(indparams, open(out_folder+"/.par_config.pkl","wb"))

    debug_t1 = time.time()
    mval, merr,T0_init,per_init,Dur_init = logprob_multi(initial[jumping],indparams,make_outfile=True,verbose=False,debug=debug,out_folder=out_folder)
    if debug: print(f'finished logprob_multi, took {(time.time() - debug_t1)} secs')
    debug_t2 = time.time()
    fit_plots(nttv, nphot, nRV, filters, LCnames, RVnames, out_folder,'/init/init_',RVunit,initial[jumping],T0_init,per_init,Dur_init)
    if debug: print(f'finished fit_plots, took {(time.time() - debug_t2)} secs')

    ########################### MCMC run ###########################################
    print(f'\n============ Samping started ... (using {fit_sampler} [{NS_type if fit_sampler=="dynesty" else emcee_move}])======================')

    inmcmc = 'y'
    indparams["inmcmc"] = inmcmc
    print(f'No of cpus: {nproc}')
    print(f'No of dimensions: {ndim}')
    print('fitting parameters: ', pnames_all[jumping])
    if shared_params!={}: print('Shared parameters: ', shared_params)
    pool = Pool(nproc)

    if debug: 
        starting = {k:v for k,v in zip(pnames_all[jumping],initial[jumping])}
        print(f'\ninitial: {starting}')

    if init_only:
        print("Generation of initial models completed")
        return
    

    nplot     = int(np.ceil(ndim/15))                #number of chain and corner plots to make
    nplotpars = int(np.ceil(ndim/nplot))             #number of parameters to plot in each plot

    if fit_sampler == "emcee":
        if nchains < 3*ndim:
            print('WARNING: Number of chains is less than 3 times the number of dimensions. Increasing number of chains to 3*ndim')
            nchains = 3*ndim
        print('No of chains: ', nchains)
        os.environ["OMP_NUM_THREADS"] = "1"  # to avoid conflicts with multiprocessing

        # put starting points for all walkers, i.e. chains
        p0 = np.random.rand(nchains, ndim)*np.asarray(steps[jumping])*2 + (np.asarray(initial[jumping])-np.asarray(steps[jumping]))
        for i in range(nchains):
            for jj in range(ndim):
                assert np.isfinite(prior_distr[jj].logpdf(p0[i,jj])), f'start value: {jnames[jj]}={p0[i,jj]} is outside the prior distribution'
            assert np.isfinite(logprob_multi(p0[i],indparams)),f'loglikelihood of start values {p0[i]} is not finite, check that prior values are valid '

        if emcee_move == "demc":      moves = emcee.moves.DEMove()
        elif emcee_move == "snooker": moves = emcee.moves.DESnookerMove()
        else: moves = emcee.moves.StretchMove()
        
        
        if not os.path.exists(f'{out_folder}/chains_dict.pkl'):   #if chain files doesnt already exist, start sampling
            backend = emcee.backends.HDFBackend(f'{out_folder}/emcee.h5')
            
            if resume_sampling and os.path.exists(f'{out_folder}/emcee.h5'):
                niter = backend.iteration
                print(f"\nResuming emcee sampling from checkpoint file (at iteration: {niter})")
                sampler = emcee.EnsembleSampler(nchains, ndim, logprob_multi, args=(indparams,),pool=pool, 
                                                moves=moves, backend=backend)
                pos, prob, state = sampler.run_mcmc(None, nsteps-niter, progress=progress, **run_kwargs)
                print(f"Final iteration: {sampler.iteration}")
            else:
                backend.reset(nchains, ndim)
                sampler = emcee.EnsembleSampler(nchains, ndim, logprob_multi, args=(indparams,),pool=pool, 
                                                moves=moves, backend=backend)
                print("\nRunning first burn-in...")
                p0, lp, _ = sampler.run_mcmc(p0, 20, progress=progress, **run_kwargs)

                print("Running second burn-in...")
                p0 = p0[np.argmax(lp)] + steps[jumping] * np.random.randn(nchains, ndim) # this can create problems!
                sampler.reset()
                pos, prob, state = sampler.run_mcmc(p0, burnin, progress=progress, **run_kwargs)
                burnin_chains = sampler.chain
                #save burn-in chains to file
                burnin_chains_dict =  {}
                for ch in range(burnin_chains.shape[2]):
                    burnin_chains_dict[jnames[ch]] = burnin_chains[:,:,ch]
                pickle.dump(burnin_chains_dict,open(out_folder+"/"+"burnin_chains_dict.pkl","wb"))  
                print("burn-in chain written to disk")
                
                matplotlib.use('Agg')
                burn_result = load_result(out_folder, verbose=False)
                for i in range(nplot):
                    fit_pars = list(burn_result._par_names)[i*nplotpars:(i+1)*nplotpars]
                    fig      = burn_result.plot_burnin_chains(fit_pars)
                    fig.savefig(out_folder+f"/burnin_chains_{i}.png", bbox_inches="tight")
                print(f"saved {nplot} burn-in chain plot(s) as {out_folder}/burnin_chains_*.png")
                matplotlib.use(__default_backend__)
                sampler.reset()

                print("\nRunning production...")
                pos, prob, state = sampler.run_mcmc(pos, nsteps,skip_initial_state_check=True, progress=progress, **run_kwargs )
            
            bp = pos[np.argmax(prob)]
            posterior = sampler.flatchain
            chains    = sampler.chain
            print((f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}"))
            evidence  = None
        else:
            print("\nSkipping burn-in and production. Loading chains from disk")
            result     = load_result(out_folder,verbose=False)
            #try burnin chain plot
            try:
                matplotlib.use('Agg')
                for i in range(nplot):
                    fit_pars = list(result._par_names)[i*nplotpars:(i+1)*nplotpars]
                    fig      = result.plot_burnin_chains(fit_pars)
                    fig.savefig(out_folder+f"/burnin_chains_{i}.png", bbox_inches="tight")
                print(f"saved {nplot} burn-in chain plots as {out_folder}/burnin_chains_*.png")
                matplotlib.use(__default_backend__)
            except: pass

            posterior  = result.flat_posterior
            chains     = np.stack([v for k,v in result._chains.items()],axis=2)
            try: bp    = result.params.max
            except: bp = np.median(posterior,axis=0)
            try: evidence    = result.evidence
            except: evidence = None

        #save chains to file
        chains_dict =  {}
        for ch in range(chains.shape[2]):
            chains_dict[jnames[ch]] = chains[:,:,ch]
        pickle.dump(chains_dict,open(out_folder+"/"+"chains_dict.pkl","wb"))
        print(f"\nEmcee production chain written to disk as {out_folder}/chains_dict.pkl. Run `result=CONAN.load_result()` to load it.\n")  
    
        GRvals = grtest_emcee(chains)
        gr_print(jnames,GRvals,out_folder)

    else:    #dynesty sampling
        if not os.path.exists(f'{out_folder}/chains_dict.pkl') or resume_sampling:
            if not force_nlive:   # modify nlive based on the number of dimensions
                if nlive < 10*ndim :
                    print('WARNING: Number of dynesty live points is less than 10*ndim. Increasing number of live points to min(10*ndim, 1000)')
                    nlive = min(10*ndim, 1000)
            
            print('No of live points: ', nlive)

            if resume_sampling and os.path.exists(f'{out_folder}/dynesty.save'):
                print("\nResuming dynesty sampling from checkpoint file")
                sampler = dynesty.NestedSampler.restore(out_folder+f"/dynesty.save", pool=pool)
                sampler.run_nested(resume=True, checkpoint_file=out_folder+f"/dynesty.save")
            else:
                if NS_type == "static":
                    sampler = dynesty.NestedSampler(logprob_multi, prior_transform, ndim, nlive=nlive,logl_args=(indparams,),pool=pool,
                                                    ptform_args=(prior_distr,jnames), reflective=refl_ind,queue_size=max(nproc,1), **dyn_kwargs)
                    sampler.run_nested(dlogz=dlogz, checkpoint_file=out_folder+f"/dynesty.save", **run_kwargs)
                else:  #dynamic nested sampling
                    pfrac = float(NS_type.split("[")[-1][:-1])   #posterior/evidence fraction
                    if "nlive_batch" not in run_kwargs: run_kwargs["nlive_batch"] = int(nlive/4)
                    if "maxbatch" not in run_kwargs: run_kwargs["maxbatch"] = 10
                    sampler = dynesty.DynamicNestedSampler(logprob_multi, prior_transform, ndim, logl_args=(indparams,),pool=pool,
                                                            ptform_args=(prior_distr,jnames), reflective=refl_ind,queue_size=max(nproc,1), **dyn_kwargs)
                    sampler.run_nested(dlogz_init=dlogz, nlive_init=nlive, wt_kwargs={'pfrac': pfrac},
                                        checkpoint_file=out_folder+f"/dynesty.save", **run_kwargs)

            dyn_res = sampler.results
            evidence = dyn_res.logz[-1]
            #dynesty trace plot
            import dynesty.plotting as dyplot
            for i in range(nplot):
                matplotlib.use('Agg')
                fig, ax = dyplot.traceplot(dyn_res,dims=np.arange(ndim)[i*nplotpars:(i+1)*nplotpars], 
                                            labels=jnames[i*nplotpars:(i+1)*nplotpars], quantiles=[0.16,0.5,0.84])
                fig.savefig(out_folder+f"/dynesty_trace_{i}.png", bbox_inches="tight")
                matplotlib.use(__default_backend__)
            dyn_summary(dyn_res,out_folder,NS_type)   #write summary to file evidence.dat

            weights   = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])
            posterior = resample_equal(dyn_res.samples, weights)
            chains    = resample_equal(dyn_res.samples, weights)
            bp        = posterior[np.argmax(dyn_res.logl)]

        else:
            print("\nSkipping dynesty run. Loading chains from disk")
            result     = load_result(out_folder, verbose=False)
            posterior  = result.flat_posterior
            chains     = np.stack([v for k,v in result._chains.items()],axis=1)
            try: bp    = result.params.max
            except: bp = np.median(posterior,axis=0)
            try: evidence    = result.evidence
            except: evidence = None

        #save chains to file
        chains_dict =  {}
        for ch in range(chains.shape[1]):
            chains_dict[jnames[ch]] = chains[:,ch]
        pickle.dump(chains_dict,open(out_folder+"/"+"chains_dict.pkl","wb"))
        print(f"\nDynesty chain written to disk as {out_folder}/chains_dict.pkl. Run `result=CONAN.load_result()` to load it.\n")  
        
    pool.close()  #close the pool
    pool.join()   #wait for the processes to finish

    nijnames = np.where(steps == 0.)     #indices of the fixed parameters
    njnames = pnames_all[[nijnames][0]]  # njnames are the names of the fixed parameters

    exti = np.intersect1d(pnames_all,extinpars, return_indices=True)
    exti[1].sort()  #indices of pnames_all elements found in  the extinpars
    extins=np.copy(exti[1])

    print("============ Sampling Finished ==============================================\n")
    # ==== chain and corner plot ================
    matplotlib.use('Agg')
    result = load_result(out_folder,verbose=False)
    if fit_sampler == "emcee": 
        #chain plot
        for i in range(nplot):
            fit_pars = list(result._par_names)[i*nplotpars:(i+1)*nplotpars]
            fig = result.plot_chains(fit_pars)
            fig.savefig(out_folder+f"/chains_{i}.png", bbox_inches="tight") 
        print(f"\nsaved {nplot} chain plot(s) as {out_folder}/chains_*.png")
    
    
    #corner plot
    for i in range(nplot):
        fit_pars = list(result._par_names)[i*nplotpars:(i+1)*nplotpars]
        fig = result.plot_corner(fit_pars, force_plot=True)
        fig.savefig(out_folder+f"/corner_{i}.png", bbox_inches="tight")
    print(f"saved {nplot} corner plot(s) as {out_folder}/corner_*.png")
    matplotlib.use(__default_backend__)

    dim=posterior.shape
    # calculate PDFs of the stellar parameters given 
    Rs_PDF = get_PDF_Gauss(Rs_in,sRs_lo,sRs_hi,dim) if sRs_lo is not None else get_PDF_Gauss(Rs_in,1e-6,1e-6,dim)
    Ms_PDF = get_PDF_Gauss(Ms_in,sMs_lo,sMs_hi,dim) if Ms_in is not None else None
    extind_PDF = np.zeros((dim[0],len(extins)))
    for i in range(len(extinpars)):
        ind = extins[i]
        par_PDF = get_PDF_Gauss(extcens[ind],extup[ind],extlow[ind],dim)
        extind_PDF[:,i] = par_PDF

    bpfull = np.copy(initial)
    bpfull[[ijnames][0]] = bp

    try:
        medvals,maxvals,sigma1s = mcmc_outputs(posterior,jnames, ijnames, njnames, nijnames, bpfull, uwl, Rs_in, Ms_in, Rs_PDF, Ms_PDF, 
                                        nfilt, filnames, howstellar, extinpars, RVunit, extind_PDF,npl,out_folder)
    except:
        medvals,maxvals,sigma1s = mcmc_outputs(posterior,jnames, ijnames, njnames, nijnames, bpfull, uwl, Rs_in, Ms_in, Rs_PDF, Ms_PDF, 
                                        nfilt, filnames, howstellar, extinpars, RVunit, extind_PDF,npl,out_folder)

    npar=len(jnames)
    if (baseLSQ == "y"):
        # print("TODO: validate if GPs and leastsquare_for_basepar works together")
        npar = npar + nbc_tot   # add the baseline coefficients if they are done by leastsq

    medp=np.copy(initial)
    medp[[ijnames][0]]=medvals
    maxp=initial
    maxp[[ijnames][0]]=maxvals
    stdev = np.zeros_like(initial)
    stdev[[ijnames][0]] = sigma1s

    #============================== PLOTTING ===========================================
    print('\nPlotting output figures')

    inmcmc='n'
    indparams["inmcmc"] = inmcmc

    #median
    mval, merr,T0_post,p_post,Dur_post = logprob_multi(medp[jumping],indparams,make_outfile=(statistic=="median"), verbose=True,out_folder=out_folder)
    #save summary_stats and as a hidden files. can be used to run logprob_multi() to generate out_full.dat files for median posterior, max posterior and best fit values
    stat_vals = dict(med = medp[jumping], max = maxp[jumping], bf  = bpfull[jumping], stdev=stdev[jumping],
                        T0 = T0_post,  P = p_post, dur = Dur_post, evidence=evidence)
    pickle.dump(stat_vals, open(out_folder+"/.stat_vals.pkl","wb"))

    fit_plots(nttv,nphot, nRV, filters, LCnames, RVnames, out_folder,'/med/med_',RVunit,medp[jumping],T0_post,p_post,Dur_post)

    #max_posterior
    mval2, merr2, T0_post_max, p_post_max, Dur_post_max = logprob_multi(maxp[jumping],indparams,make_outfile=(statistic=="max"),verbose=False)
    fit_plots(nttv, nphot, nRV, filters, LCnames, RVnames, out_folder,'/max/max_',RVunit,maxp[jumping], T0_post_max,p_post_max,Dur_post_max)

    mod_dev = (f_arr - mval2)/merr2 if statistic != "median" else (f_arr - mval)/merr  #Akin allow statistics to be based on median of posterior
    chisq   = np.sum(mod_dev**2)
    ndata   = len(t_arr)
    get_AIC_BIC(npar,ndata,chisq,out_folder)

    rarr = f_arr-mval2  if statistic != "median" else f_arr - mval  # the full residuals
    earr = merr2 if statistic != "median" else merr  # the full errors

    if nphot > 0:
        bw, br, brt, cf, cfn = corfac(rarr, t_arr, earr, indlist, nphot, njumpphot) # get the beta_w, beta_r and CF and the factor to get redchi2=1
        of=open(out_folder+"/CF.dat",'w')
        of.write(f"{'beta_w':8s} {'beta_r':8s} {'beta_rtot':8s} {'CF':8s} {'CFerr':10s} \n")
        for i in range(nphot):   #adapt the error values
            of.write('%8.3f %8.3f %8.3f %8.3f %10.6f \n' % (bw[i], br[i], brt[i],cf[i],cfn[i]))
            if (cf_apply == 'cf'):
                earr[indlist[i][0]] = np.multiply(earr[indlist[i][0]],cf[i])
            if (cf_apply == 'rchisq'):
                earr[indlist[i][0]] = np.sqrt((earr[indlist[i][0]])**2 + (cfn[i])**2)
        of.close()

    
    print("\n")
    result = load_result(out_folder)
    matplotlib.use('Agg')

    if result.lc.names != []:
        fig = result.lc.plot_bestfit()
        fig.savefig(out_folder+"/bestfit_LC.png", bbox_inches="tight",dpi=200)
        fig = result.lc.plot_bestfit(detrend=True)
        fig.savefig(out_folder+"/bestfit_LC_detrended.png", bbox_inches="tight",dpi=200)
        fig = result.lc.plot_bestfit(detrend=True, phase_plot=1)
        fig.savefig(out_folder+"/bestfit_LC_detrended_phased.png", bbox_inches="tight",dpi=200)
    if result.rv.names != []:
        fig = result.rv.plot_bestfit()
        fig.savefig(out_folder+"/bestfit_RV.png", bbox_inches="tight",dpi=200)
        fig = result.rv.plot_bestfit(detrend=True)
        fig.savefig(out_folder+"/bestfit_RV_detrended.png", bbox_inches="tight",dpi=200)
        fig = result.rv.plot_bestfit(detrend=True, phase_plot=1)
        fig.savefig(out_folder+"/bestfit_RV_detrended_phased.png", bbox_inches="tight",dpi=200)
    matplotlib.use(__default_backend__)


    if ttvYN == "y":
        print("\nPlotting TTVs")
        matplotlib.use('Agg')
        fig=result.lc.plot_ttv()
        fig.savefig(out_folder+"/TTVs.png", bbox_inches="tight",dpi=200)

        fig=result.lc.plot_lcttv()
        fig.savefig(out_folder+"/TTVs_lc.png", bbox_inches="tight",dpi=200)
        matplotlib.use(__default_backend__)

    #make print out statement in the fashion of conan the barbarian
    print(_text_format.RED + "\nCONAN: I have now crushed your data," +\
          "\n\tthe planetary information it hides is laid bare in the results."+\
          "\n\t\tI am super ready for another quest. \n" + _text_format.END)
    return result