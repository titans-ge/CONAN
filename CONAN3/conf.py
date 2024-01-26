from ._classes import load_lightcurves, load_rvs, fit_setup,_print_output
from .fit_data import run_fit
import numpy as np 



def fit_configfile(config_file = "input_config.dat", out_folder = "output"):
    """
        Run CONAN fit from configuration file. 
        This loads the config file and creates the required objects (lc_obj, rv_obj, fit_obj) to perform the fit.
        
        Parameters:
        -----------
        config_file: filepath;
            path to configuration file.

        out_folder: filepath;
            path to folder where output files will be saved.
    """

    lc_obj, rv_obj, fit_obj = load_configfile(config_file, verbose=True)
    result = run_fit(lc_obj, rv_obj, fit_obj,
                            out_folder=out_folder,
                            rerun_result=True)
    return result


def _skip_lines(file, n):
    """ takes an open file object and skips the reading of lines by n lines """
    for i in range(n):
        dump = file.readline()

def _prior_value(str_prior): 
    "convert string prior into float/tuple"
    str_prior = str_prior[str_prior.find("(")+1:str_prior.find(")")].split(",")
    tuple_prior = [float(v) for v in str_prior]
    tuple_prior = [(int(v) if v.is_integer() else float(v)) for v in tuple_prior]
    len_tup = len(tuple_prior)
    val = tuple_prior[0] if len_tup==1 else tuple(tuple_prior)
    return val



def create_configfile(lc_obj, rv_obj, fit_obj, filename="input_config.dat"): 
    """
        create configuration file that of lc_obj, rv_obj, amd fit_obj setup.
        
        Parameters:
        -----------
        lc_obj : object;
            Instance of CONAN.load_lightcurve() object and its attributes.

        rv_obj : object, None;
            Instance of CONAN.load_rvs() object and its attributes.
        
        fit_obj : object;
            Instance of CONAN.fit_setup() object and its attributes.
    """
    f = open(filename,"w")
    f.write("# ========================================== CONAN configuration file ============================================= \n")
    f.write("#             *********** KEYS *****************************************************************************************\n")
    f.write("#             PRIORS: *Fixed - F(val), *Normal - N(mu,std), *Uniform - U(min,start,max), *LogUniform - LU(min,start,max)\n")
    f.write("#             s_samp: supersampling - x{exp_time}\n")
    f.write("#             clip:   clip outliers - W{window_width}C{clip_sigma}\n")
    f.write("#             scl_col: scale data columns – ['med_sub','rs0to1','rs-1to1','None']\n")
    f.write("#             spline_config: spline - c{column_no}:d{degree}K{knot_spacing}\n")
    f.write("#             ***********************************************************************************************************\n")
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    f.write(f"LC_filepath: {lc_obj._fpath}\n")
    f.write(f"RV_filepath: {lc_obj._fpath}\n")
    f.write(f"n_planet: {lc_obj._nplanet}\n")
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(lc_obj,"lc_baseline",file=f)
    _print_output(lc_obj,"gp",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(rv_obj,"rv_baseline",file=f)
    _print_output(rv_obj,"rv_gp",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(lc_obj,"planet_parameters",file=f)
    _print_output(lc_obj,"limb_darkening",file=f)
    _print_output(lc_obj,"depth_variation",file=f)
    _print_output(lc_obj,"phasecurve",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(lc_obj,"contamination",file=f)
    _print_output(fit_obj,"stellar_pars",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(fit_obj, "fit",file=f)
    f.close()


def load_configfile(configfile="input_config.dat", return_fit=False, verbose=True):
    """
        configure conan from specified configfile.
        
        Parameters:
        -----------
        configfile: filepath;
            path to configuration file.

        return_fit: bool;
            whether to immediately perform the fit from this function call.
            if True, the result object from the fit is also returned

        verbose: bool;
            show print statements

        Returns:
        --------
        lc_obj, rv_obj, fit_obj. if return_fit is True, the result object of fit is also returned

        lc_obj: object;
            light curve data object generated from `conan3.load_lighturves()`.
        
        rv_obj: object;
            rv data object generated from `conan3.load_rvs()`
            
        fit_obj: object;
            fitting object generated from `conan3.fit_setup()`.

        result: object;
            result object containing chains of the mcmc fit.
    
    """

    _file = open(configfile,"r")
    _skip_lines(_file,9)                       #remove first 2 comment lines
    fpath    = _file.readline().rstrip().split()[1]           # the path where the files are
    rv_fpath = _file.readline().rstrip().split()[1]           # the path where the files are
    nplanet  = int(_file.readline().rstrip().split()[1])      # the path where the files are
    _skip_lines(_file,3)                                      #remove 3 comment lines

    # ========== Lightcurve input ====================
    _names=[]                    # array where the LC filenames are supposed to go
    _filters=[]                  # array where the filter names are supposed to go
    _lamdas=[]
    _bases=[]                    # array where the baseline exponents are supposed to go
    _groups=[]                   # array where the group indices are supposed to go
    _grbases=[]
    _useGPphot=[]
    
    _ss_lclist,_ss_exp = [],[]
    _clip_lclist, _clip, _clip_width  = [],[],[]
    _sclcol= []
    _spl_lclist,_spl_deg,_spl_par, _spl_knot=[],[],[],[]
    
    #read baseline specification for each listed light-curve file 
    dump = _file.readline() 
    while dump[0] != '#':                   # if it is not starting with # then
        _adump = dump.split()               # split it

        _names.append(_adump[0])            # append the first field to the name array
        _filters.append(_adump[1])          # append the second field to the filters array
        _lamdas.append(float(_adump[2]))    # append the second field to the filters array
        
        #supersample
        xt = _adump[3].split("|")[-1]
        if xt != "None":
            _ss_lclist.append(_adump[0])
            _ss_exp.append(float(xt.split("x")[1]))
        
        #clip_outlier
        if _adump[4]!= "None":
            _clip_lclist.append(_adump[0])
            clip_v = float(_adump[4].split("C")[1]) 
            _clip.append(int(clip_v) if clip_v.is_integer() else clip_v)                   # outlier clip value
            _clip_width.append(int(_adump[4].split("C")[0].split("W")[1])) # windown width
        
        #scale columns
        _sclcol.append(_adump[5])

        strbase=_adump[7:12]
        strbase.append(_adump[12].split("|")[0])        # string array of the baseline function coeffs
        grbase = 0
        strbase.extend([_adump[13],grbase])
        _grbases.append(grbase)
        base = [int(i) for i in strbase]
        _bases.append(base)
        
        group = int(_adump[14])
        _groups.append(group)
        _useGPphot.append(_adump[15])
        
        #LC spline
        if _adump[16] != "None":
            _spl_lclist.append(_adump[0])
            _spl_knot.append(float(_adump[16].split("k")[-1]))
            _spl_deg.append(int(_adump[16].split("k")[0].split("d")[-1]))
            _spl_par.append("col" + _adump[16].split("k")[0].split("d")[0][1])

        #move to next LC
        dump =_file.readline() 

    _skip_lines(_file,1)                                      #remove 1 comment lines
    
    # ========== GP input ====================
    gp_lclist,op = [],[]
    gp_pars, kernels, amplitude, lengthscale = [],[],[],[]

    dump =_file.readline()
    while dump[0] != "#":
        _adump = dump.split()
        gp_lclist.append(_adump[0])
        gp_pars.append(_adump[1])
        kernels.append(_adump[2])
        amplitude.append(_prior_value(_adump[3]))
        lengthscale.append(_prior_value(_adump[4]))
        
        op.append(_adump[5].strip("|"))
        if op[-1] != "--":    #if theres a second kernel 
            gp_pars[-1]     = (gp_pars[-1],_adump[6])
            kernels[-1]     = (kernels[-1],_adump[7])
            amplitude[-1]   = (amplitude[-1],_prior_value(_adump[8]))
            lengthscale[-1] = (lengthscale[-1],_prior_value(_adump[9]))

        #move to next LC
        dump =_file.readline()
    _skip_lines(_file,2)  
    
    
    # instantiate light curve object
    lc_obj = load_lightcurves(_names, fpath, _filters, _lamdas, nplanet)
    lc_obj.lc_baseline(*np.array(_bases).T, grp_id=_groups, gp=_useGPphot,verbose=False )
    lc_obj.clip_outliers(lc_list=_clip_lclist, clip=_clip, width=_clip_width,show_plot=False,verbose=False )
    lc_obj.rescale_data_columns(method=_sclcol,verbose=False)
    lc_obj.supersample(lc_list=_ss_lclist, exp_time=_ss_exp, verbose=False)
    lc_obj.add_spline(lc_list=_spl_lclist ,par=_spl_par , degree=_spl_deg,
                        knot_spacing=_spl_knot , verbose=False)
    if verbose: lc_obj.print("lc_baseline")
    lc_obj.add_GP(lc_list=gp_lclist,par=gp_pars,kernel=kernels,operation="op",
                    amplitude=amplitude,lengthscale=lengthscale,verbose=verbose)
    
    ## RV ==========================================================
    RVnames, RVbases, gammas = [],[],[]
    _RVsclcol, usegpRV,strbase = [],[],[]
    _spl_rvlist,_spl_deg,_spl_par, _spl_knot=[],[],[],[]
    
    dump =_file.readline()
    while dump[0] != '#':                   # if it is not starting with # then
        _adump = dump.split()               # split it
        RVnames.append(_adump[0])
        _RVsclcol.append(_adump[1])
        strbase=_adump[3:6]                  # string array of the baseline function coeffs
        strbase.append(_adump[6].split("|")[0])
        strbase.append(_adump[7])
        base = [int(i) for i in strbase]
        RVbases.append(base)
        usegpRV.append(_adump[8])
        
        #RV spline
        if _adump[9] != "None":
            _spl_rvlist.append(_adump[0])
            _spl_knot.append(float(_adump[9].split("k")[-1]))
            _spl_deg.append(int(_adump[9].split("k")[0].split("d")[-1]))
            _spl_par.append("col" + _adump[9].split("k")[0].split("d")[0][1])
        
        gammas.append(_prior_value(_adump[11]))
        #move to next RV
        dump =_file.readline()

    _skip_lines(_file,1)                                      #remove 1 comment lines
    
    # RV GP
    gp_rvlist,op = [],[]
    gp_pars, kernels, amplitude, lengthscale = [],[],[],[]

    dump =_file.readline()
    while dump[0] != "#":
        _adump = dump.split()
        gp_rvlist.append(_adump[0])
        gp_pars.append(_adump[1])
        kernels.append(_adump[2])
        amplitude.append(_prior_value(_adump[3]))
        lengthscale.append(_prior_value(_adump[4]))
        
        op.append(_adump[5].strip("|"))
        if op[-1] != "––":    #if theres a second kernel 
            gp_pars[-1]     = (gp_pars[-1],_adump[6])
            kernels[-1]     = (kernels[-1],_adump[7])
            amplitude[-1]   = (amplitude[-1],_prior_value(_adump[8]))
            lengthscale[-1] = (lengthscale[-1],_prior_value(_adump[9]))

        #move to next LC
        dump =_file.readline()
        
        
    rv_obj = load_rvs(RVnames,rv_fpath, nplanet=nplanet,lc_obj=lc_obj)
    rv_obj.rv_baseline(*np.array(RVbases).T, gamma=gammas,gp=usegpRV,verbose=False) 
    rv_obj.rescale_data_columns(method=_RVsclcol,verbose=False)
    rv_obj.add_spline(rv_list=_spl_rvlist ,par=_spl_par, degree=_spl_deg,
                        knot_spacing=_spl_knot, verbose=False)
    if verbose: rv_obj.print("rv_baseline")

    rv_obj.add_rvGP(rv_list=gp_rvlist,par=gp_pars,kernel=kernels,operation="op",
                        amplitude=amplitude,lengthscale=lengthscale,verbose=verbose)
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    ## Planet parameters
    dump    = _file.readline()
    _adump  = dump.split()
    pl_pars = {}
    pl_pars["rho_star"] = _prior_value(_adump[2])
    par_names = ["RpRs","Impact_para", "T_0", "Period", "Eccentricity","omega", "K"]
    for p in par_names: pl_pars[p] = []
        
    for n in range(1,nplanet+1):        #load parameters for each planet
        _skip_lines(_file,1)          #remove dashes
        for i in range(7):
            dump =_file.readline()
            _adump = dump.split()
            pl_pars[par_names[i]].append(_prior_value(_adump[2]))
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    ## limb darkening
    q1, q2 = [],[]
    for _ in range(len(lc_obj._filnames)):
        dump   = _file.readline()
        _adump = dump.split()
        q1.append(_prior_value(_adump[2]))
        q2.append(_prior_value(_adump[3]))
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    #DDFs
    dump   = _file.readline()
    _adump = dump.split()
    ddfyn,ddf_pri,div_wht  = _adump[0], _prior_value(_adump[1]), _adump[2]
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    #phase curve
    D_occ,A_pc,ph_off = [],[],[]    
    for _ in range(len(lc_obj._filnames)):
        dump   = _file.readline()
        _adump = dump.split()
        D_occ.append(_prior_value(_adump[3]))
    _skip_lines(_file,1)                                      #remove 2 comment lines
    for _ in range(len(lc_obj._filnames)):
        dump   = _file.readline()
        _adump = dump.split()
        A_pc.append(_prior_value(_adump[3]))
    _skip_lines(_file,1)                                      #remove 2 comment lines
    for _ in range(len(lc_obj._filnames)):
        dump   = _file.readline()
        _adump = dump.split()
        ph_off.append(_prior_value(_adump[3]))
    
    _skip_lines(_file,3)                                      #remove 3 comment lines
 
    #contamination factors
    cont_fac = []
    for _ in range(len(lc_obj._filnames)):
        dump   = _file.readline()
        _adump = dump.split()
        cont_fac.append(_prior_value(_adump[1]))
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    lc_obj.planet_parameters(**pl_pars,verbose=verbose)
    lc_obj.limb_darkening(q1,q2,verbose=verbose)
    lc_obj.transit_depth_variation(ddFs=ddfyn,dRpRs=ddf_pri, divwhite=div_wht,verbose=verbose)
    lc_obj.setup_phasecurve(D_occ, A_pc, ph_off, verbose=verbose)
    lc_obj.contamination_factors(cont_ratio=cont_fac, verbose=verbose)
    
    
    # stellar params
    dump    = _file.readline()
    _adump  = dump.split()
    st_rad  = _prior_value(_adump[1])
    dump    = _file.readline()
    _adump  = dump.split()
    st_mass = _prior_value(_adump[1])
    dump    = _file.readline()
    _adump  = dump.split()
    par_in  = _adump[2]
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    #fit setup
    tot_samps = int(_file.readline().split()[1])
    nchains   = int(_file.readline().split()[1])
    nsteps    = int(tot_samps/nchains)
    ncpus     = int(_file.readline().split()[1])
    nburn     = int(_file.readline().split()[1])
    nlive     = int(_file.readline().split()[1])
    dlogz     = float(_file.readline().split()[1])
    sampler   = _file.readline().split()[1]
    mc_move   = _file.readline().split()[1]
    lsq_base  = _file.readline().split()[1]
    lcjitt    = _file.readline().split()[1]
    rvjitt    = _file.readline().split()[1]
    
    _adump    = _file.readline().split()
    baselo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
    basehi    = float(_adump[2][_adump[2].find("[")+1:_adump[2].find(",")])
    lcbaselim = [baselo, basehi]
    
    _adump    = _file.readline().split()
    baselo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
    basehi    = float(_adump[2][_adump[2].find("[")+1:_adump[2].find(",")])
    rvbaselim = [baselo, basehi]
    
    fit_obj = fit_setup(R_st = st_rad, M_st = st_mass, par_input=par_in,
                        apply_LCjitter=lcjitt, apply_RVjitter=rvjitt,
                        leastsq_for_basepar=lsq_base, 
                        LCbasecoeff_lims=lcbaselim, RVbasecoeff_lims=rvbaselim,
                        verbose=verbose)
    
    fit_obj.sampling(sampler=sampler,n_cpus=ncpus, emcee_move=mc_move,
                    n_chains=nchains, n_burn   = 5000, n_steps  = nsteps, 
                    n_live=nlive, dyn_dlogz=dlogz,verbose=verbose )

    _file.close()

    if return_fit:
        from .fit_data import run_fit
        result =   run_fit(lc_obj, rv_obj, fit_obj) 
        return lc_obj,rv_obj,fit_obj,result

    return lc_obj,rv_obj,fit_obj



