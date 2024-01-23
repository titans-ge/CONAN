from logging import exception
import numpy as np
from types import SimpleNamespace
import os
from multiprocessing import Pool
import pickle
import emcee, dynesty
from dynesty.utils import resample_equal
import time

from occultquad import *
from occultnl import *
from .basecoeff_setup import *
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

import celerite
from celerite import terms
from celerite import GP as cGP
from copy import deepcopy
from .utils import gp_params_convert
from scipy.stats import norm, uniform, lognorm, loguniform,truncnorm


from ._classes import _raise, fit_setup, __default_backend__, load_result
import matplotlib
matplotlib.use(__default_backend__)

import multiprocessing as mp 
mp.set_start_method('fork')
__all__ = ["run_fit"]

def prior_transform(u,prior_dst,prior_names):
    """  
    function to transform the unit cube,u, to the prior space.
    Parameters
    ----------
    u : array_like;
        unit cube, array of values between 0 and 1.
    prior_dst : list;
        list of scipy.stats prior distributions for each parameter.
    prior_names : list;
        list of parameter names.
    Returns
    -------
    x : array_like;
        array of values in the prior space.
    """
    x = np.array(u)  # copy u 
    for i, pr in enumerate(prior_dst):
        x[i] = pr.ppf(u[i]) 
        x[i] = x[i]*1e6 if (prior_names[i].startswith('GPlc') and 'Amp' in prior_names[i]) else x[i]
    return x 

def run_fit(lc_obj=None, rv_obj=None, fit_obj=None, statistic = "median", out_folder="output", progress=True,
            rerun_result=False, verbose=False, debug=False, save_burnin_chains=True, force_nlive=False, **kwargs):
    """
    function to fit the data using the light-curve object lc_obj, rv_object rv_obj, and fit_setup object fit_obj.

    Parameters
    ----------
    lc_obj : lightcurve object;
        object containing lightcurve data and setup parameters. 
        see CONAN3.load_lightcurves() for more details.

    rv_obj : rv object
        object containing radial velocity data and setup parameters. 
        see CONAN3.load_rvs() for more details.

    fit_obj : fit_setup object;
        object containing fit setup parameters. 
        see CONAN3.fit_setup() for more details.

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

#============lc_obj=========================
    #from load_lightcurves()
    fpath      = lc_obj._fpath
    names      = lc_obj._names
    filters    = lc_obj._filters
    lamdas     = lc_obj._lamdas
    bases      = lc_obj._bases
    bases_init = lc_obj._bases_init
    groups     = lc_obj._groups
    useGPphot  = lc_obj._useGPphot

    nphot      = len(names)                                                 # the number of photometry input files
    njumpphot  = np.zeros(nphot)                                            # the number of jumping parameters for each photometry input file
    filnames   = np.array(list(sorted(set(filters),key=filters.index)))     # the unique filter names
    ulamdas    = np.array(list(sorted(set(lamdas),key=lamdas.index)))       # the unique wavelengths
    grnames    = np.array(list(sorted(set(groups))))                        # the unique group names
    nfilt      = len(filnames)                                              # the number of unique filters
    ngroup     = len(grnames)                                               # the number of unique groups

    useSpline_lc  = lc_obj._lcspline                                            # use spline to interpolate the light curve
    input_lcs     = {} if lc_obj is None else lc_obj._input_lc
    s_samp        = lc_obj._ss

#============rv_obj========================== 
    # from load_rvs()
    if rv_obj is not None and rv_obj._names == []: rv_obj = None   
    RVnames  = [] if rv_obj is None else rv_obj._names
    RVbases  = [] if rv_obj is None else rv_obj._RVbases
    rv_fpath = [] if rv_obj is None else rv_obj._fpath
    rv_dict  = {} if rv_obj is None else rv_obj._rvdict
    sinPs    = [] if rv_obj is None else rv_dict["sinPs"]
    nRV      = len(RVnames)             # the number of RV input files
    njumpRV  = np.zeros(nRV)        # the number of jumping parameters for each RV input file
    useGPrv  = ["n"]*nRV if rv_obj is None else rv_obj._useGPrv 

    useSpline_rv  = [] if rv_obj is None else rv_obj._rvspline   
    input_rvs     = {} if rv_obj is None else rv_obj._input_rv  
    RVbases_init  = [] if rv_obj is None else rv_obj._RVbases_init
    RVunit        = "kms" if rv_obj is None else rv_obj._RVunit
    extinpars= []               # set up array to contain the names of the externally input parameters
    
    for i in range(nRV):
        if (float(rv_dict["gam_steps"][i]) != 0.) :
            njumpRV[i]=njumpRV[i]+1

 
#============transit and RV jump parameters===============
    #from load_lightcurves.planet_parameters()

    CP  = deepcopy(lc_obj._config_par)    #load input transit and RV parameters from dict
    npl = lc_obj._nplanet

    # # rho_star (same for all planets)
    rho_star = CP[f"pl{1}"]["rho_star"]
    if nphot > 0:    # if there is photometry, then define rho_star: can be a jumping parameter
        if rho_star.step_size != 0.: njumpphot=njumpphot+1
        if (rho_star.to_fit == 'n' and rho_star.prior == 'p'):   
            extinpars.append('rho_star')
    else:
        rho_star.to_fit == 'n'
        rho_star.step_size = 0
        
    for n in range(1,npl+1): 
        if rv_obj is None: #remove k as a free parameter
            CP[f"pl{n}"]['K'].to_fit = "n"
            CP[f"pl{n}"]['K'].step_size = 0
        if nphot == 0: #remove rprs and impact_para as a free parameter
            CP[f"pl{n}"]['RpRs'].to_fit = "n"
            CP[f"pl{n}"]['RpRs'].step_size = 0
            CP[f"pl{n}"]['Impact_para'].to_fit = "n"
            CP[f"pl{n}"]['Impact_para'].step_size = 0


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
        # for key,val in CP[f"pl{n}"]["K"].__dict__.items(): #convert K to km/s
        #     if isinstance(val, (float,int)): CP[f"pl{n}"]["K"].__dict__[key] /= 1000 
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
    drprs_op = lc_obj._ddfs.drprs_op    # drprs options --> [0., step, bounds_lo, bounds_hi, 0., width_lo, width_hi]
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


    #============phasecurve setup=============
    #from load_lightcurves.setup_phasecurve()

    DA_occ = lc_obj._PC_dict["D_occ"]
    DA_Apc = lc_obj._PC_dict["A_pc"]
    DA_off = lc_obj._PC_dict["ph_off"]

    nocc     = len(filnames) #len(DA_occ["filters_occ"])
    occ_in   = np.zeros((nocc,7))
    Apc_in   = np.zeros((nocc,7))
    phoff_in = np.zeros((nocc,7))

    for i, f in enumerate(filnames):
        k = np.where(np.array(lc_obj._filters)== f)     #  get indices where the filter name is the same as the one in the input file

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
    DA_ld = lc_obj._ld_dict

    q1_in=np.zeros((nfilt,7))
    q2_in=np.zeros((nfilt,7))

    for i in range(nfilt):
        j=np.where(filnames == filnames[i])              # make sure the sequence in this array is the same as in the "filnames" array
        k=np.where(np.array(lc_obj._filters) == filnames[i])

        q1_in[j,:] = [DA_ld["q1"][i], DA_ld["step1"][i],DA_ld["bound_lo1"][i],DA_ld["bound_hi1"][i],DA_ld["q1"][i],DA_ld["sig_lo1"][i],DA_ld["sig_hi1"][i]]
        q1_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step1"][i] == 0.) else q1_in[j,5])   #sig_lo
        q1_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step1"][i] == 0.) else q1_in[j,6])   #sig_hi
        if q1_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1


        q2_in[j,:] = [DA_ld["q2"][i], DA_ld["step2"][i],DA_ld["bound_lo2"][i],DA_ld["bound_hi2"][i],DA_ld["q2"][i],DA_ld["sig_lo2"][i],DA_ld["sig_hi2"][i]]  # the limits are -3 and 3 => very safe
        q2_in[j,5] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step2"][i] == 0.) else q2_in[j,5])
        q2_in[j,6] = (0. if (DA_ld["priors"][i] == 'n' or DA_ld["step2"][i] == 0.) else q2_in[j,6])
        if q2_in[j,1] != 0.:
            njumpphot[k]=njumpphot[k]+1

#============contamination factors=======================
    #from load_lightcurves.contamination()
    DA_cont = lc_obj._contfact_dict
    cont=np.zeros((nfilt,2))

    for i in range(nfilt):
        j = np.where(filnames == filnames[i])               # make sure the sequence in this array is the same as in the "filnames" array
        if verbose: print(j)
        cont[j,:]= [DA_cont["cont_ratio"][i][0], DA_cont["cont_ratio"][i][1]]

#=============sampling setup===============
    #from setup_fit()
    if fit_obj is None: fit_obj = fit_setup()
    DA_mc =  fit_obj._fit_dict

    nsamples    = int(DA_mc['n_steps']*DA_mc['n_chains'])   # total number of integrations
    nchains     = int(DA_mc['n_chains'])  #  number of chains
    ppchain     = int(nsamples/nchains)  # number of points per chain
    nproc       = int(DA_mc['n_cpus'])   #  number of processes
    burnin      = int(DA_mc['n_burn'])    # Length of bun-in
    emcee_move  = DA_mc['emcee_move']            # Differential Evolution? 
    fit_sampler = DA_mc['sampler']               # Which sampler to use?   
    nlive       = DA_mc["n_live"]  
    dlogz       = DA_mc["dyn_dlogz"]    
    jit_apply   = DA_mc['apply_RVjitter']       # apply rvjitter
    jit_LCapply = DA_mc['apply_LCjitter']     # apply lcjitter
    LCbase_lims = DA_mc['LCbasecoeff_lims']   # bounds of the LC baseline coefficients
    RVbase_lims = DA_mc['RVbasecoeff_lims']   # bounds of the RV baseline coefficients
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



#========= stellar properties==========================
    #from setup_fit.stellar_parameters()
    DA_stlr = fit_obj._stellar_dict
    
    Rs_in  = DA_stlr["R_st"][0]
    sRs_lo = sRs_hi = DA_stlr["R_st"][1]
    Ms_in  = DA_stlr["M_st"][0]
    sMs_lo = sMs_hi = DA_stlr["M_st"][1]

    howstellar = DA_stlr["par_input"]

    #********************************************************************
    #============Start computations as in original CONANGP===============
    #********************************************************************
        
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
        params     = np.concatenate((params,   [q1_in[i,0], q2_in[i,0]]))
        stepsize   = np.concatenate((stepsize, [q1_in[i,1], q2_in[i,1]]))
        pmin       = np.concatenate((pmin,     [q1_in[i,2], q2_in[i,2]]))
        pmax       = np.concatenate((pmax,     [q1_in[i,3], q2_in[i,3]]))
        prior      = np.concatenate((prior,    [q1_in[i,4], q2_in[i,4]]))
        priorlow   = np.concatenate((priorlow, [q1_in[i,5], q2_in[i,5]]))
        priorup    = np.concatenate((priorup,  [q1_in[i,6], q2_in[i,6]]))
        pnames     = np.concatenate((pnames,   [filnames[i]+'_q1',filnames[i]+'_q2']))

    for i in range(nphot):    #add jitter
        if (jit_LCapply=='y'):
            #in ppm
            params      = np.concatenate((params,  [-13]), axis=0)    #20ppm
            stepsize    = np.concatenate((stepsize,[0.1]), axis=0)
            pmin        = np.concatenate((pmin,    [-15]), axis=0)
            pmax        = np.concatenate((pmax,    [-4.]), axis=0)
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
    
    for i in range(nRV):
        params      = np.concatenate((params,  [rv_dict["gammas"][i]]),   axis=0)
        stepsize    = np.concatenate((stepsize,[rv_dict["gam_steps"][i]]), axis=0)
        pmin        = np.concatenate((pmin,    [rv_dict["bound_lo"][i]]), axis=0)
        pmax        = np.concatenate((pmax,    [rv_dict["bound_hi"][i]]),  axis=0)
        prior       = np.concatenate((prior,   [rv_dict["gam_pri"][i]]),   axis=0)
        priorlow    = np.concatenate((priorlow,[rv_dict["gampriloa"][i]]), axis=0)
        priorup     = np.concatenate((priorup, [rv_dict["gamprihia"][i]]), axis=0)
        pnames      = np.concatenate((pnames,  [f"rv{i+1}_gamma"]), axis=0)

        
        if (jit_apply=='y'):
            params      = np.concatenate((params,  [0.001]), axis=0)
            stepsize    = np.concatenate((stepsize,[0.0001]), axis=0)
            pmin        = np.concatenate((pmin,    [0.]), axis=0)
            pmax        = np.concatenate((pmax,    [100]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            pnames      = np.concatenate((pnames,  [f"rv{i+1}_jitter"]), axis=0)

        else:
            params      = np.concatenate((params,  [0.]), axis=0)
            stepsize    = np.concatenate((stepsize,[0.]), axis=0)
            pmin        = np.concatenate((pmin,    [0.]), axis=0)
            pmax        = np.concatenate((pmax,    [0]), axis=0)
            prior       = np.concatenate((prior,   [0.]), axis=0)
            priorlow    = np.concatenate((priorlow,[0.]), axis=0)
            priorup     = np.concatenate((priorup, [0.]), axis=0)
            pnames      = np.concatenate((pnames,  [f"rv{i+1}_jitter"]), axis=0)
    
        
    nbc_tot = np.copy(0)  # total number of baseline coefficients let to vary (leastsq OR jumping)

    #################################### GP setup #########################################
    print('Setting up photometry GPs ...')
    
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
    gpkerns     = []  # list to hold the kernel for each lc

    GPdict   = {} if lc_obj is None else lc_obj._GP_dict
    sameLCgp = False if lc_obj is None else lc_obj._sameLCgp
    
    #possible kernels
    george_kernels = dict(  expsq = george.kernels.ExpSquaredKernel, 
                            mat32 = george.kernels.Matern32Kernel,
                            mat52 = george.kernels.Matern52Kernel,
                            exp   = george.kernels.ExpKernel,
                            cos   = george.kernels.CosineKernel)
    
    celerite_kernel = dict(mat32  = celerite.terms.Matern32Term,
                            sho   = celerite.terms.SHOTerm,
                            real  = celerite.terms.RealTerm)

    # =================PHOTOMETRY =========================================
    for i in range(nphot):
        # t, flux, err, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in = np.loadtxt(fpath+names[i], usecols=(0,1,2,3,4,5,6,7,8), unpack = True)  # reading in the data
        thisLCdata = input_lcs[names[i]]
        t, flux, err, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in = thisLCdata.values()
        
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
        offset, dcol0, dcol3, dcol4, dcol5, dcol6, dcol7, dsin, dCNM, nbc = basecoeff(bases[i],useSpline_lc[i],bases_init[i],LCbase_lims)  # the baseline coefficients for this lightcurve; each is a 2D array
        nbc_tot      = nbc_tot+nbc # add up the number of jumping baseline coeff
        njumpphot[i] = njumpphot[i]+nbc   # each LC has another jump pm

        # if the least-square fitting for the baseline is turned on (baseLSQ = 'y'), then set the stepsize of the jump parameter to 0
        if (baseLSQ == "y"):
            abvar=np.concatenate(([offset[1,:],dcol0[1,:],dcol3[1,:],dcol4[1,:],dcol5[1,:],dcol6[1,:],dcol7[1,:],dsin[1,:],dCNM[1,:]]))
            abind=np.where(abvar!=0.)
            bvars.append(abind)
            offset[1,:]=dcol0[1,:]=dcol3[1,:]=dcol4[1,:]=dcol5[1,:]=dcol6[1,:]=dcol7[1,:]=dsin[1,:]=dCNM[1,:]=0      # the step sizes are set to 0 so that they are not interpreted as MCMC JUMP parameters

        # append these to the respective mcmc input arrays
        params    = np.concatenate((params,   offset[0,:],dcol0[0,:],dcol3[0,:],dcol4[0,:],dcol5[0,:],dcol6[0,:],dcol7[0,:],dsin[0,:],dCNM[0,:]))
        stepsize  = np.concatenate((stepsize, offset[1,:],dcol0[1,:],dcol3[1,:],dcol4[1,:],dcol5[1,:],dcol6[1,:],dcol7[1,:],dsin[1,:],dCNM[1,:]))
        pmin      = np.concatenate((pmin,     offset[2,:],dcol0[2,:],dcol3[2,:],dcol4[2,:],dcol5[2,:],dcol6[2,:],dcol7[2,:],dsin[2,:],dCNM[2,:]))
        pmax      = np.concatenate((pmax,     offset[3,:],dcol0[3,:],dcol3[3,:],dcol4[3,:],dcol5[3,:],dcol6[3,:],dcol7[3,:],dsin[3,:],dCNM[3,:]))
        prior     = np.concatenate((prior,    np.zeros(20)))
        priorlow  = np.concatenate((priorlow, np.zeros(20)))
        priorup   = np.concatenate((priorup,  np.zeros(20)))
        pnames   = np.concatenate((pnames, [f"lc{i+1}_off",f"lc{i+1}_A0",f"lc{i+1}_B0",f"lc{i+1}_C0",f"lc{i+1}_D0",
                                            f"lc{i+1}_A3",f"lc{i+1}_B3",
                                            f"lc{i+1}_A4",f"lc{i+1}_B4",
                                            f"lc{i+1}_A5",f"lc{i+1}_B5",
                                            f"lc{i+1}_A6",f"lc{i+1}_B6",
                                            f"lc{i+1}_A7",f"lc{i+1}_B7",
                                            f"lc{i+1}_sin_amp",f"lc{i+1}_sin_freq",f"lc{i+1}_sin_phi",
                                            f"lc{i+1}_ACNM",f"lc{i+1}_BCNM"
                                            ]))        
        # note currently we have the following parameters in these arrays:
        #   [rho_star,                                   (1)
        #   [T0,RpRs,b,per,eos, eoc,K,                   (7)*npl
        #   ddf_1, ..., ddf_n,                           (nddf)
        #   (occ_1,Apc_1,phoff_1),...,occ_n,Apc_n,phoff_n(3*nocc)
        #   q1_f1,q2_f1, q1_f2, .... , q2fn,            (2*n_filt)
        #   LC_jit                                       (nphot)
        #   Rv_gamma, RV_jit                              (2*nRVs)         
        #   baseline                                       20, ...]
        #    = 1+7*npl+nddf+nocc*3+4*n_filt+nphot+2*nRV + 20
        #    each lightcurve has 20 possible baseline jump parameters, starting with index  1+7*npl+nddf+nocc*3+4*n_filt+nphot+2*nRV

        # pargp_all = np.vstack((t, col3_in, col4_in, col5_in, col6_in, col7_in, col8_in)).T  # the matrix with all the possible inputs to the GPs

        if (useGPphot[i]=='n'):
            GPobjects.append([])
            pargps.append([])
            gpkerns.append([]) 

        elif useGPphot[i] in ['y','ce']:     #George or Celerite GP
            gp_conv  = gp_params_convert()   #class containing functions to convert gp amplitude and lengthscale to the required values for the different kernels 
            thisLCgp = GPdict[names[i]]
            gpcols   = [thisLCgp[f"amplitude{n}"].user_data[1] for n in range(thisLCgp["ngp"])]
            gpkerns.append([thisLCgp[f"amplitude{n}"].user_data[0] for n in range(thisLCgp["ngp"])])

            for n in range(thisLCgp["ngp"]):                      #loop through the number of GPs for this LC
                gpkern = thisLCgp[f"amplitude{n}"].user_data[0]   #kernel to use for this GP
                gpcol  = thisLCgp[f"amplitude{n}"].user_data[1]   #column of the data to use for this GP

                GPparams    = np.concatenate((GPparams,    [thisLCgp[f"amplitude{n}"].start_value, thisLCgp[f"lengthscale{n}"].start_value]), axis=0)
                GPstepsizes = np.concatenate((GPstepsizes, [thisLCgp[f"amplitude{n}"].step_size, thisLCgp[f"lengthscale{n}"].step_size]), axis=0)
                GPindex     = np.concatenate((GPindex,     (np.zeros(2)+i)), axis=0)
                GPprior     = np.concatenate((GPprior,     [thisLCgp[f"amplitude{n}"].prior_mean, thisLCgp[f"lengthscale{n}"].prior_mean]), axis=0)
                GPpriwid    = np.concatenate((GPpriwid,    [thisLCgp[f"amplitude{n}"].prior_width_lo, thisLCgp[f"lengthscale{n}"].prior_width_lo]), axis=0)
                GPlimup     = np.concatenate((GPlimup,     [thisLCgp[f"amplitude{n}"].bounds_hi, thisLCgp[f"lengthscale{n}"].bounds_hi]), axis=0)
                GPlimlo     = np.concatenate((GPlimlo,     [thisLCgp[f"amplitude{n}"].bounds_lo, thisLCgp[f"lengthscale{n}"].bounds_lo]), axis=0)
                if not sameLCgp.flag:
                    GPnames = np.concatenate((GPnames,     [f"GPlc{i+1}_Amp{n}_{gpcol}",f"GPlc{i+1}_len{n}_{gpcol}"]), axis=0)
                else:
                    GPnames = np.concatenate((GPnames,     [f"GPlcSame_Amp{n}_{gpcol}",f"GPlcSame_len{n}_{gpcol}"]), axis=0)

                if useGPphot[i]=="y":  #George GP
                    ndim_gp  = len(set(gpcols))       #number of different columns used for the GP
                    axes_gp  = [gpcols.index(gpcol)]  #axes of the GP (0 or 1)

                    if n==0: 
                        kern = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels="g_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value,
                                                                                                    thisLCgp[f"lengthscale{n}"].start_value])
                        kern.set_parameter_vector([gppar1, gppar2])  
                        gp_x = thisLCdata[gpcol]  # the x values for the GP

                    if n==1:                       # if this is the second GP, then add/mult the new kernel to the previous one
                        kern2 = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        gppar1, gppar2 =  gp_conv.get_values(kernels="g_"+gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value,
                                                                                                    thisLCgp[f"lengthscale{n}"].start_value])
                        kern2.set_parameter_vector([gppar1, gppar2])
                        if thisLCgp["op"]=="+": kern += kern2
                        if thisLCgp["op"]=="*": kern *= kern2
                        
                        if ndim_gp >1: gp_x = np.vstack((gp_x, thisLCdata[gpcol])).T  #2D array with the x values for the GP
                    
                    gp = GP(kern, mean=1)
                    gp.compute(x=gp_x, yerr=thisLCdata["col2"])
            
                if useGPphot[i]=="ce":   #Celerite GP
                    if n==0: 
                        if gpkern=="sho":
                            kern  = celerite_kernel[gpkern](log_S0 =-10, log_Q=np.log(1/np.sqrt(2)), log_omega0=1)  #dummy initialization
                            kern.freeze_parameter("log_Q")   #freeze Q
                        else:
                            kern  = celerite_kernel[gpkern](-10, 1)   #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value,
                                                                                                thisLCgp[f"lengthscale{n}"].start_value])
                        kern.set_parameter_vector([gppar1, gppar2])
                    if n==1:
                        if gpkern=="sho": 
                            kern2 = celerite_kernel[gpkern](log_S0 =-10, log_Q=np.log(1/np.sqrt(2)), log_omega0=1) 
                            kern2.freeze_parameter("log_Q")
                        else:
                            kern2 = celerite_kernel[gpkern](-10, 1)
                        #starting values of next kernel
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="lc", pars=[thisLCgp[f"amplitude{n}"].start_value,
                                                                                            thisLCgp[f"lengthscale{n}"].start_value])
                        kern2.set_parameter_vector([gppar1, gppar2])
                        
                        if thisLCgp["op"]=="+": kern += kern2
                        if thisLCgp["op"]=="*": kern *= kern2
                    
                    gp_x = thisLCdata[gpcol] # the x values for the GP, for celerite it is always col0 for now
                    gp   = cGP(kern, mean=1)
                    gp.compute(t=gp_x, yerr=thisLCdata["col2"])

            GPobjects.append(gp)
            pargps.append(gp_x) 

    # =================RADIAL VELOCITY =========================================
    if rv_obj is not None: print('Setting up RV arrays ...')
    
    rvGPobjects,rvGPnames,rv_pargps,rv_gpkerns = [],[],[],[]
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
        Pin       = sinPs[i]

        #rv baseline 
        dcol0RV, dcol3RV, dcol4RV, dcol5RV,dsinRV,nbcRV = basecoeffRV(RVbases[i],Pin,RVbases_init[i],RVbase_lims)  # the baseline coefficients for this lightcurve; each is a 2D array
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
        pnames   = np.concatenate((pnames, [f"rv{i+1}_A0",f"rv{i+1}_B0",
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

        if useGPrv[i] in ["y","ce"]:         #George or Celerite GP
            gp_conv  = gp_params_convert()   #class containing functions to convert gp amplitude and lengthscale to the required values for the different kernels 
            thisRVgp = rvGPdict[RVnames[i]]
            gpcols   = [thisRVgp[f"amplitude{n}"].user_data[1] for n in range(thisRVgp["ngp"])]
            rv_gpkerns.append([thisRVgp[f"amplitude{n}"].user_data[0] for n in range(thisRVgp["ngp"])])

            for n in range(thisRVgp["ngp"]):                      #loop through the number of GPs for this RV
                gpkern = thisRVgp[f"amplitude{n}"].user_data[0]   #kernel to use for this GP
                gpcol  = thisRVgp[f"amplitude{n}"].user_data[1]   #column of the data to use for this GP

                rvGPparams    = np.concatenate((rvGPparams,    [thisRVgp[f"amplitude{n}"].start_value, thisRVgp[f"lengthscale{n}"].start_value]), axis=0)
                rvGPstepsizes = np.concatenate((rvGPstepsizes, [thisRVgp[f"amplitude{n}"].step_size, thisRVgp[f"lengthscale{n}"].step_size]), axis=0)
                rvGPindex     = np.concatenate((rvGPindex,     (np.zeros(2)+i)), axis=0)
                rvGPprior     = np.concatenate((rvGPprior,     [thisRVgp[f"amplitude{n}"].prior_mean, thisRVgp[f"lengthscale{n}"].prior_mean]), axis=0)
                rvGPpriwid    = np.concatenate((rvGPpriwid,    [thisRVgp[f"amplitude{n}"].prior_width_lo, thisRVgp[f"lengthscale{n}"].prior_width_lo]), axis=0)
                rvGPlimup     = np.concatenate((rvGPlimup,     [thisRVgp[f"amplitude{n}"].bounds_hi, thisRVgp[f"lengthscale{n}"].bounds_hi]), axis=0)
                rvGPlimlo     = np.concatenate((rvGPlimlo,     [thisRVgp[f"amplitude{n}"].bounds_lo, thisRVgp[f"lengthscale{n}"].bounds_lo]), axis=0)
                if not sameRVgp.flag:
                    rvGPnames = np.concatenate((rvGPnames,     [f"GPrv{i+1}_Amp{n}_{gpcol}",f"GPrv{i+1}_len{n}_{gpcol}"]), axis=0)
                else:
                    rvGPnames = np.concatenate((rvGPnames,     [f"GPrvSame_Amp{n}_{gpcol}",f"GPrvSame_len{n}_{gpcol}"]), axis=0)

                if useGPrv[i]=="y":  #George GP
                    ndim_gp  = len(set(gpcols))       #number of different columns used for the GP
                    axes_gp  = [gpcols.index(gpcol)]  #axes of the GP (0 or 1)

                    if n==0: 
                        kern = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels="g_"+gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value,
                                                                                                    thisRVgp[f"lengthscale{n}"].start_value])
                        kern.set_parameter_vector([gppar1, gppar2])  
                        gp_x = thisRVdata[gpcol]  # the x values for the GP

                    if n==1:                       # if this is the second GP, then add/mult the new kernel to the previous one
                        kern2 = 100e-6 * george_kernels[gpkern](1, ndim=ndim_gp,axes=axes_gp)  #dummy initialization
                        gppar1, gppar2 =  gp_conv.get_values(kernels="g_"+gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value,
                                                                                                    thisRVgp[f"lengthscale{n}"].start_value])
                        kern2.set_parameter_vector([gppar1, gppar2])
                        if thisRVgp["op"]=="+": kern += kern2
                        if thisRVgp["op"]=="*": kern *= kern2
                        
                        if ndim_gp >1: gp_x = np.vstack((gp_x, thisRVdata[gpcol])).T  #2D array with the x values for the GP
                    
                        gp = GP(kern, mean=1)
                        gp.compute(x=gp_x, yerr=thisRVdata["col2"])
            
                if useGPrv[i]=="ce":   #Celerite GP
                    if n==0: 
                        if gpkern=="sho":
                            kern  = celerite_kernel[gpkern](log_S0 =-10, log_Q=np.log(1/np.sqrt(2)), log_omega0=1)  #dummy initialization
                            kern.freeze_parameter("log_Q")   #freeze Q
                        else:
                            kern  = celerite_kernel[gpkern](-10, 1)   #dummy initialization
                        # set the kernel parameters to the starting values after performing the conversion
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value,
                                                                                            thisRVgp[f"lengthscale{n}"].start_value])
                        kern.set_parameter_vector([gppar1, gppar2])
                    if n==1:
                        if gpkern=="sho": 
                            kern2 = celerite_kernel[gpkern](log_S0 =-10, log_Q=np.log(1/np.sqrt(2)), log_omega0=1) 
                            kern2.freeze_parameter("log_Q")
                        else:
                            kern2 = celerite_kernel[gpkern](-10, 1)
                        #starting values of next kernel
                        gppar1, gppar2 =  gp_conv.get_values(kernels=gpkern, data="rv", pars=[thisRVgp[f"amplitude{n}"].start_value,
                                                                                            thisRVgp[f"lengthscale{n}"].start_value])
                        kern2.set_parameter_vector([gppar1, gppar2])
                        
                        if thisRVgp["op"]=="+": kern += kern2
                        if thisRVgp["op"]=="*": kern *= kern2
                    
                        gp_x = thisRVdata[gpcol] # the x values for the GP, for celerite it is always col0 for now
                        gp   = cGP(kern, mean=1)
                        gp.compute(t=gp_x, yerr=thisRVdata["col2"])

            rvGPobjects.append(gp)
            rv_pargps.append(gp_x) 


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
        q1ind=1+7*npl+nddf+nocc*3+k*2
        if (stepsize[q1ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[q1ind]),axis=0)
        
        q2ind=1+7*npl+nddf+nocc*3+k*2+1
        if (stepsize[q2ind]!=0.):
            temp=np.concatenate((np.asarray(temp),[q2ind]),axis=0)

        LCjitterind = 1+7*npl + nddf+nocc*3 + nfilt*2 + i 
        if (stepsize[LCjitterind]!=0.):           
            temp=np.concatenate((temp,[LCjitterind]),axis=0)
    
        #baseline
        bfstart= 1+7*npl+nddf+nocc*3+nfilt*2 + nphot + nRV*2  # the first index in the param array that refers to a baseline function    
        blind = np.asarray(list(range(bfstart+i*20,bfstart+i*20+20)))  # the indices for the coefficients for the base function   
        if verbose: print(bfstart, blind, nocc, nfilt)

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
        gammaind  = 1+7*npl + nddf+nocc*3 + nfilt*2 + nphot + i*2
        jitterind = 1+7*npl + nddf+nocc*3 + nfilt*2 + nphot + i*2 + 1

        if (stepsize[gammaind]!=0.):           
            temp=np.concatenate((temp,[gammaind]),axis=0)

        if (stepsize[jitterind]!=0.):           
            temp=np.concatenate((temp,[jitterind]),axis=0)
            
        bfstart= 1+7*npl+nddf+nocc*3+nfilt*2 + nphot + nRV*2 + nphot*20  # the first index in the param array that refers to an RV baseline function    
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

    ewarr=np.nan#grweights(earr,indlist,grnames,groups,ngroup,nphot)


    ###### create prior distribution for the jumping parameters #####
    ijnames    = np.where(steps != 0.)    #indices of the jumping parameters
    jnames     = pnames_all[[ijnames][0]]  # jnames are the names of the jump parameters
    norm_sigma = priorwids[jumping]
    norm_mu    = priors[jumping]
    uni_low    = lim_low[jumping]
    uni_up     = lim_up[jumping]
    ppm        = 1e-6

    uni    = lambda lowlim,uplim: uniform(lowlim, uplim-lowlim)               # uniform prior between lowlim and uplim
    t_norm = lambda a,b,mu,sig: truncnorm((a-mu)/sig, (b-mu)/sig, mu, sig) # normal prior(mu,sig) truncated  between a and b

    prior_distr = []      # list of prior distributions for the jumping parameters

    for jj in range(ndim):
        if jnames[jj].startswith('GP'):    #if GP parameter then convert to the correct amplitude units
            gpparunit = ppm if (jnames[jj].startswith('GPlc') and 'Amp' in jnames[jj]) else 1
            if (norm_sigma[jj]>0.):   #normal prior truncated at the bounds
                lpri = t_norm(uni_low[jj]*gpparunit, uni_up[jj]*gpparunit,norm_mu[jj]*gpparunit,norm_sigma[jj]*gpparunit)
                prior_distr.append(lpri)        
            else:                     #loguniform propr
                llim = loguniform(uni_low[jj]*gpparunit, uni_up[jj]*gpparunit)
                prior_distr.append(llim)
        
        else:
            if (norm_sigma[jj]>0.):  #normal prior
                lpri = t_norm(uni_low[jj],uni_up[jj],norm_mu[jj],norm_sigma[jj])
                prior_distr.append(lpri)
            else:                    #uniform prior
                llim = uni(uni_low[jj],uni_up[jj])
                prior_distr.append(llim)

    ## plot the prior distributions
    print("Plotting priors")
    matplotlib.use('Agg')
    
    start_pars = initial[jumping]
    nrows = int(np.ceil(len(start_pars)/6))
    fig, ax = plt.subplots(nrows,6, figsize=(15, 2*nrows))
    ax = ax.reshape(-1)
    for jj in range(ndim):
        ax[jj].hist(prior_distr[jj].rvs(1000), label=jnames[jj], density=True)
        stp = start_pars[jj]*ppm if (jnames[jj].startswith('GPlc') and 'Amp' in jnames[jj]) else start_pars[jj]
        ax[jj].axvline(stp,color="red")
        ax[jj].set_yticks([])
        ax[jj].legend()
        if (jnames[jj].startswith('GPlc') and 'Amp' in jnames[jj]):
            ax[jj].set_xticklabels(ax[jj].get_xticks()*1e6)
    for jj in range(ndim,nrows*6): ax[jj].axis("off")   #remove unused subplots
    fig.savefig(f"{out_folder}/priors.png", bbox_inches="tight")
    matplotlib.use(__default_backend__)



    ############################## Initial guess ##################################
    print('\nPlotting initial guess\n---------------------------')

    inmcmc = 'n'
    indparams = [t_arr,f_arr,col3_arr,col4_arr,col6_arr,col5_arr,col7_arr,bis_arr,contr_arr, nphot, nRV, indlist, filters, nfilt, filnames,nddf,
                nocc,0,0,grprs,egrprs,grnames,groups,ngroup,ewarr, inmcmc, paraCNM, baseLSQ, bvars, bvarsRV, 
                cont,names,RVnames,e_arr,divwhite,dwCNMarr,dwCNMind,params,useGPphot,useGPrv,GPobjects,GPparams,GPindex,
                pindices,jumping,pnames_all[jumping],prior_distr,priors[jumping],priorwids[jumping],lim_low[jumping],lim_up[jumping],pargps,
                jumping_noGP,gpkerns,jit_apply,jumping_GP,GPstepsizes,sameLCgp,npl,useSpline_lc,useSpline_rv,s_samp,
                rvGPobjects,rvGPparams,rvGPindex,jumping_rvGP, jumping_lcGP,RVunit,rv_pargps,rv_gpkerns,sameRVgp,fit_sampler]
    pickle.dump(indparams, open(out_folder+"/.par_config.pkl","wb"))

    debug_t1 = time.time()
    mval, merr,T0_init,per_init,Dur_init = logprob_multi(initial[jumping],*indparams,make_outfile=True,verbose=True,debug=debug,out_folder=out_folder)
    if debug: print(f'finished logprob_multi, took {(time.time() - debug_t1)} secs')
    if not os.path.exists(out_folder+"/init"): os.mkdir(out_folder+"/init")    #folder to put initial plots    
    debug_t2 = time.time()
    mcmc_plots(mval,t_arr,f_arr,e_arr, nphot, nRV, indlist, filters, names, RVnames, out_folder+'/init/init_',RVunit,T0_init,per_init,Dur_init)
    if debug: print(f'finished mcmc_plots, took {(time.time() - debug_t2)} secs')


    ########################### MCMC run ###########################################
    print(f'\n============ Samping started ... (using {fit_sampler})======================')

    inmcmc = 'y'
    indparams[25] = inmcmc
    print('No of dimensions: ', ndim)

    if fit_sampler == "emcee":
        if nchains < 2*ndim:
            print('WARNING: Number of chains is less than twice the number of dimensions. Increasing number of chains to 2*ndim')
            nchains = 2*ndim
        print('No of chains: ', nchains)
        print('fitting parameters: ', pnames_all[jumping])
        if debug: 
            starting = {k:v for k,v in zip(pnames_all[jumping],initial[jumping])}
            print(f'initial: {starting}')


        # put starting points for all walkers, i.e. chains
        p0 = np.random.rand(ndim * nchains).reshape((nchains, ndim))*np.asarray(steps[jumping])*2 + (np.asarray(initial[jumping])-np.asarray(steps[jumping]))
        assert np.all([np.isfinite(logprob_multi(p0[i],*indparams)) for i in range(nchains)]),f'Range of start values of a(some) jump parameter(s) are outside the prior distribution'

        if emcee_move == "demc":      moves = emcee.moves.DEMove()
        elif emcee_move == "snooker": moves = emcee.moves.DESnookerMove()
        else: moves = emcee.moves.StretchMove()
        
        if not os.path.exists(f'{out_folder}/chains_dict.pkl'):   #if chain files doesnt already exist, start sampling
            sampler = emcee.EnsembleSampler(nchains, ndim, logprob_multi, args=(indparams),pool=Pool(nproc), moves=moves)
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
                    if ndim<=30:
                        nplotpars = int(np.ceil(ndim/2))
                        nplot     = 2
                    else:
                        nplotpars = 14
                        nplot = int(np.ceil(ndim/14))

                    for i in range(nplot):
                        fit_pars = list(burn_result._par_names)[i*nplotpars:(i+1)*nplotpars]
                        fig = burn_result.plot_burnin_chains(fit_pars)
                        fig.savefig(out_folder+f"/burnin_chains_{i}.png", bbox_inches="tight")
                    print(f"saved {nplot} burn-in chain plots as {out_folder}/burnin_chains_*.png")

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
            result     = load_result(out_folder)
            posterior  = result.flat_posterior
            chains     = np.stack([v for k,v in result._chains.items()],axis=2)
            try: bp    = result.params_max
            except: bp = np.median(posterior,axis=0)

        #save chains to file
        chains_dict =  {}
        for ch in range(chains.shape[2]):
            chains_dict[jnames[ch]] = chains[:,:,ch]
        pickle.dump(chains_dict,open(out_folder+"/"+"chains_dict.pkl","wb"))
        print(f"Emcee production chain written to disk as {out_folder}/chains_dict.pkl. Run `result=CONAN3.load_result()` to load it.\n")  
    
        GRvals = grtest_emcee(chains)
        gr_print(jnames,GRvals,out_folder)

    else:    #dynesty sampling
        if not os.path.exists(f'{out_folder}/chains_dict.pkl'):
            if not force_nlive:   # modify nlive based on the number of dimensions
                if nlive < ndim * (ndim + 1) // 2:
                    print('WARNING: Number of dynesty live points is less than ndim*(ndim+1)//2. Increasing number of live points to min(ndim*(ndim+1)//2, 1000)')
                    nlive = min(ndim * (ndim + 1) // 2, 1000)
            
            print('No of live points: ', nlive)
            print('fitting parameters: ', pnames_all[jumping])

            sampler = dynesty.NestedSampler(logprob_multi, prior_transform, ndim, nlive=nlive,sample="rwalk",
                                    logl_args=(indparams),ptform_args=(prior_distr,jnames),pool=Pool(nproc), queue_size=nproc-2)
            sampler.run_nested(dlogz=dlogz, **kwargs)
            dyn_res = sampler.results
            dyn_summary(dyn_res,out_folder)   #write summary to file evidence.dat

            weights   = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])
            posterior = resample_equal(dyn_res.samples, weights)
            chains    = resample_equal(dyn_res.samples, weights)
            bp        = posterior[np.argmax(dyn_res.logl)]

        else:
            print("\nSkipping dynesty run. Loading chains from disk")
            result     = load_result(out_folder)
            posterior  = result.flat_posterior
            chains     = np.stack([v for k,v in result._chains.items()],axis=1)
            try: bp    = result.params_max
            except: bp = np.median(posterior,axis=0)

        #save chains to file
        chains_dict =  {}
        for ch in range(chains.shape[1]):
            chains_dict[jnames[ch]] = chains[:,ch]
        pickle.dump(chains_dict,open(out_folder+"/"+"chains_dict.pkl","wb"))
        print(f"dynesty chain written to disk as {out_folder}/chains_dict.pkl. Run `result=CONAN3.load_result()` to load it.\n")  
        


    nijnames = np.where(steps == 0.)     #indices of the fixed parameters
    njnames = pnames_all[[nijnames][0]]  # njnames are the names of the fixed parameters

    exti = np.intersect1d(pnames_all,extinpars, return_indices=True)
    exti[1].sort()
    extins=np.copy(exti[1])

    print("==================Sampling Finished==================\n\n")

    # ==== chain and corner plot ================
    matplotlib.use('Agg')
    result = load_result(out_folder)
    if fit_sampler == "emcee":
       
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
            nplotpars = int(np.ceil(ndim/2))
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
                                    nfilt, filnames, howstellar, extinpars, RVunit, extind_PDF,npl,out_folder)

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
    indparams[25] = inmcmc

    #AKIN: save config parameters indparams and summary_stats and as a hidden files. 
    #can be used to run logprob_multi() to generate out_full.dat files for median posterior, max posterior and best fit values
    # pickle.dump(indparams, open(out_folder+"/.par_config.pkl","wb"))
    stat_vals = dict(med = medp[jumping], max=maxp[jumping], bf = bpfull[jumping])
    pickle.dump(stat_vals, open(out_folder+"/.stat_vals.pkl","wb"))

    #median
    mval, merr,T0_post,p_post,Dur_post = logprob_multi(medp[jumping],*indparams,make_outfile=(statistic=="median"), verbose=True,out_folder=out_folder)
    mcmc_plots(mval,t_arr,f_arr,e_arr, nphot, nRV, indlist, filters, names, RVnames, out_folder+'/med_',RVunit,T0_post,p_post,Dur_post)

    #max_posterior
    mval2, merr2, T0_post, p_post, Dur_post = logprob_multi(maxp[jumping],*indparams,make_outfile=(statistic=="max"),verbose=False)
    mcmc_plots(mval2,t_arr,f_arr,e_arr, nphot, nRV, indlist, filters, names, RVnames, out_folder+'/max_',RVunit, T0_post,p_post,Dur_post)


    maxresiduals = f_arr - mval2 if statistic != "median" else f_arr - mval  #Akin allow statistics to be based on median of posterior
    chisq = np.sum(maxresiduals**2/e_arr**2)

    ndat = len(t_arr)
    get_AIC_BIC(npar,ndat,chisq,out_folder)

    rarr=f_arr-mval2  if statistic != "median" else f_arr - mval  # the full residuals

    if nphot > 0:
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