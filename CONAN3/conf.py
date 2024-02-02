from ._classes import load_lightcurves, load_rvs, fit_setup,_print_output
from .fit_data import run_fit
from.utils import ecc_om_par
import numpy as np 



def fit_configfile(config_file = "input_config.dat", out_folder = "output", 
                   rerun_result= True, verbose=True):
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

    lc_obj, rv_obj, fit_obj = load_configfile(config_file, verbose=verbose)
    result = run_fit(lc_obj, rv_obj, fit_obj,out_folder=out_folder,
                        rerun_result=rerun_result,verbose=verbose)
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



def create_configfile(lc_obj=None, rv_obj=None, fit_obj=None, filename="input_config.dat"): 
    """
        create configuration file that of lc_obj, rv_obj, amd fit_obj setup.
        
        Parameters:
        -----------
        lc_obj : object,None;
            Instance of CONAN.load_lightcurve() object and its attributes.

        rv_obj : object, None;
            Instance of CONAN.load_rvs() object and its attributes.
        
        fit_obj : object;
            Instance of CONAN.fit_setup() object and its attributes.
    """
    if lc_obj is None:
        lc_obj = load_lightcurves()
    if rv_obj is None:
        rv_obj = load_rvs()
    if fit_obj is None:
        fit_obj = fit_setup()
    f = open(filename,"w")
    f.write("# ========================================== CONAN configuration file ============================================= \n")
    f.write("#             *********** KEYS *****************************************************************************************\n")
    f.write("#             PRIORS: *Fixed - F(val), *Normal - N(mu,std), *Uniform - U(min,start,max), *LogUniform - LU(min,start,max)\n")
    f.write("#             s_samp: supersampling - x{exp_time(mins)}\n")
    f.write("#             clip:   clip outliers - W{window_width}C{clip_sigma}\n")
    f.write("#             scl_col: scale data columns – ['med_sub','rs0to1','rs-1to1','None']\n")
    f.write("#             spline_config: spline - c{column_no}:d{degree}K{knot_spacing}\n")
    f.write("#             ***********************************************************************************************************\n")
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    f.write(f"\tLC_filepath: {lc_obj._fpath}\n")
    f.write(f"\tRV_filepath: {lc_obj._fpath}\n")
    f.write(f"\tn_planet: {lc_obj._nplanet}\n")
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    f.write(f"\t{'LC_auto_decorr:':15s} False       # automatically determine baseline function for the LCs\n")
    f.write(f"\t{'exclude_cols:':15s} []            # list of column numbers (e.g. [3,4]) to exclude from decorrelation.\n")
    f.write(f"\t{'enforce_pars:':15s} []            # list of decorr params (e.g. [B3, A5]) to enforce in decorrelation\n")
    _print_output(lc_obj,"lc_baseline",file=f)
    _print_output(lc_obj,"gp",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    f.write(f"\t{'RV_auto_decorr:':15s} False       # automatically determine baseline function for the RVs\n")
    f.write(f"\t{'exclude_cols:':15s} []            # list of column numbers (e.g. [3,4]) to exclude from decorrelation.\n")
    f.write(f"\t{'enforce_pars:':15s} []            # list of decorr params (e.g. [B3, A5]) to enforce in decorrelation\n")
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
    print(f"configuration file saved as {filename}")


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
    _skip_lines(_file,1)                                      #remove 3 comment lines

    #### auto decorrelation
    dump = _file.readline().rstrip().split()[1]
    assert dump in ["True","False"], f"LC_auto_decorr: must be 'True' or 'False' but {dump} given"
    use_decorr = True if dump == "True" else False
    dump = _file.readline().rstrip().split()[1]
    assert dump[0] == "[" and dump[-1] == "]", f"exclude_cols: must be a list of column numbers (e.g. [3,4]) but {dump} given"
    #convert dump to list of ints
    exclude_cols = [int(i) for i in dump[1:-1].split(",")] if dump[1]!= "]" else []
    dump = _file.readline().rstrip().split()[1]
    assert dump[0] == "[" and dump[-1] == "]", f"enforce_pars: must be a list of  pars (e.g. [B3,A5]) but {dump} given"
    #convert dump to list of strings
    enforce_pars = [i for i in dump[1:-1].split(",")] if dump[1]!= "]" else []
    _skip_lines(_file,2)                                      #remove 1 comment lines

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
        if _adump[16] != "None":   #TODO: only works for 1d spline
            _spl_lclist.append(_adump[0])
            _spl_knot.append(float(_adump[16].split("k")[-1]))
            _spl_deg.append(int(_adump[16].split("k")[0].split("d")[-1]))
            _spl_par.append("col" + _adump[16].split("k")[0].split("d")[0][1])

        #move to next LC
        dump =_file.readline() 
    

    nphot = len(_names)
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
            gp_pars[-1]     = (gp_pars[-1],_adump[7])
            kernels[-1]     = (kernels[-1],_adump[8])
            amplitude[-1]   = (amplitude[-1],_prior_value(_adump[9]))
            lengthscale[-1] = (lengthscale[-1],_prior_value(_adump[10]))

        #move to next LC
        dump =_file.readline()
    # _skip_lines(_file,1)  
    
    
    # instantiate light curve object
    lc_obj = load_lightcurves(_names, fpath, _filters, _lamdas, nplanet)
    lc_obj.lc_baseline(*np.array(_bases).T, grp_id=None, gp=_useGPphot,verbose=False )
    lc_obj.clip_outliers(lc_list=_clip_lclist , clip=_clip, width=_clip_width,show_plot=False,verbose=False )
    lc_obj.rescale_data_columns(method=_sclcol,verbose=False)
    lc_obj.supersample(lc_list=_ss_lclist, exp_time=_ss_exp, verbose=False)
    lc_obj.add_spline(lc_list=_spl_lclist ,par=_spl_par , degree=_spl_deg,
                        knot_spacing=_spl_knot , verbose=False)
    if verbose: lc_obj.print("lc_baseline")
    if gp_lclist !=[]: gp_lclist = gp_lclist[0] if gp_lclist[0]=='same' else gp_lclist
    lc_obj.add_GP(lc_list=gp_lclist,par=gp_pars,kernel=kernels,operation=op,
                    amplitude=amplitude,lengthscale=lengthscale,verbose=verbose)
    
    ## RV ==========================================================
    #### auto decorrelation
    dump = _file.readline().rstrip().split()[1]
    assert dump in ["True","False"], f"RV_auto_decorr: must be 'True' or 'False' but {dump} given"
    use_decorrRV = True if dump == "True" else False
    dump = _file.readline().rstrip().split()[1]
    assert dump[0] == "[" and dump[-1] == "]", f"RV exclude_cols: must be a list of column numbers (e.g. [3,4]) but {dump} given"
    #convert dump to list of ints
    exclude_colsRV = [int(i) for i in dump[1:-1].split(",")] if dump[1]!= "]" else []
    dump = _file.readline().rstrip().split()[1]
    assert dump[0] == "[" and dump[-1] == "]", f"RV enforce_pars: must be a list of  pars (e.g. [B3,A5]) but {dump} given"
    #convert dump to list of strings
    enforce_parsRV = [i for i in dump[1:-1].split(",")] if dump[1]!= "]" else []
    _skip_lines(_file,2)                                      #remove 1 comment lines

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
    

    nRV = len(RVnames)
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
            gp_pars[-1]     = (gp_pars[-1],_adump[7])
            kernels[-1]     = (kernels[-1],_adump[8])
            amplitude[-1]   = (amplitude[-1],_prior_value(_adump[9]))
            lengthscale[-1] = (lengthscale[-1],_prior_value(_adump[10]))

        #move to next LC
        dump =_file.readline()
        
        
    rv_obj = load_rvs(RVnames,rv_fpath, nplanet=nplanet,lc_obj=lc_obj)
    rv_obj.rv_baseline(*np.array(RVbases).T, gamma=gammas,gp=usegpRV,verbose=False) 
    rv_obj.rescale_data_columns(method=_RVsclcol,verbose=False)
    rv_obj.add_spline(rv_list=_spl_rvlist ,par=_spl_par, degree=_spl_deg,
                        knot_spacing=_spl_knot, verbose=False)
    if verbose: rv_obj.print("rv_baseline")
    if gp_rvlist !=[]: gp_rvlist = gp_rvlist[0] if gp_rvlist[0]=='same' else gp_rvlist
    rv_obj.add_rvGP(rv_list=gp_rvlist,par=gp_pars,kernel=kernels,operation=op,
                    amplitude=amplitude,lengthscale=lengthscale,verbose=verbose)
    
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    ## Planet parameters
    dump    = _file.readline()
    _adump  = dump.split()
    pl_pars = {}
    pl_pars["rho_star"] = _prior_value(_adump[2])
    par_names = ["RpRs","Impact_para", "T_0", "Period", "Eccentricity","omega", "K"]
    for p in par_names: pl_pars[p] = []
    sesinw, secosw = [],[]
        
    for n in range(1,nplanet+1):        #load parameters for each planet
        _skip_lines(_file,1)          #remove dashes
        for i in range(7):
            dump =_file.readline()
            _adump = dump.split()
            pl_pars[par_names[i]].append(_prior_value(_adump[2]))
        eos, eoc = ecc_om_par(pl_pars["Eccentricity"][-1],pl_pars["omega"][-1],
                                conv_2_obj=True,return_tuple=True)
        sesinw.append(eos)
        secosw.append(eoc)
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

    if nphot > 0:
        if not use_decorr:
            if verbose: print("\ngetting start values for LC decorrelation parameters ...")
        lc_obj.get_decorr(**pl_pars,q1=q1,q2=q2,
                            D_occ=D_occ[0] if len(D_occ)>0 else 0, 
                            A_pc=A_pc[0] if len(A_pc)>0 else 0, 
                            ph_off=ph_off[0] if len(ph_off)>0 else 0, plot_model=False,
                            setup_baseline=use_decorr,exclude_cols=exclude_cols,
                            enforce_pars=enforce_pars, verbose=verbose if use_decorr else False)
        #TODO: if not use_decorr, compare the auto decorr pars to the user-defined ones and only use start values for those
        rel_cols = [b[:6] for b in lc_obj._bases]
        _ = [b.insert(1,0) for b in rel_cols for _ in range(2)] #insert 0 to replace cols 1 and 2
        for j in range(lc_obj._nphot):
            for i,v in enumerate(rel_cols[j]):
                if i in [1,2]: continue
                if v == 0: lc_obj._bases_init[j][f"A{i}"] = lc_obj._bases_init[j][f"B{i}"] = 0
                if v >= 1: lc_obj._bases_init[j][f"A{i}"] = lc_obj._bases_init[j][f"A{i}"]
                if v == 2: lc_obj._bases_init[j][f"B{i}"] = lc_obj._bases_init[j][f"B{i}"]


    if nRV > 0:
        if not use_decorrRV:
            if verbose: print("getting start values for RV decorrelation parameters ...\n")
        rv_obj.get_decorr(T_0=pl_pars["T_0"], Period=pl_pars["Period"], K=pl_pars["K"],
                            sesinw=sesinw,secosw=secosw,
                            gamma=gammas[0] if len(gammas)>0 else 0, setup_baseline=use_decorrRV,
                            exclude_cols=exclude_colsRV, enforce_pars=enforce_parsRV, 
                            plot_model=False,verbose=verbose if use_decorrRV else False)
        rel_cols = [b[:6] for b in rv_obj._RVbases]
        _ = [b.insert(1,0) for b in rel_cols for _ in range(2)] #insert 0 to replace cols 1 and 2
        for j in range(rv_obj._nRV):
            for i,v in enumerate(rel_cols[j]):
                if i in [1,2]: continue
                if v == 0: rv_obj._RVbases_init[j][f"A{i}"] = rv_obj._RVbases_init[j][f"B{i}"] = 0
                if v >= 1: rv_obj._RVbases_init[j][f"A{i}"] = rv_obj._RVbases_init[j][f"A{i}"]
                if v == 2: rv_obj._RVbases_init[j][f"B{i}"] = rv_obj._RVbases_init[j][f"B{i}"]
        
    
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
    nsteps    = int(_file.readline().split()[1])
    nchains   = int(_file.readline().split()[1])
    ncpus     = int(_file.readline().split()[1])
    nburn     = int(_file.readline().split()[1])
    nlive     = int(_file.readline().split()[1])
    force_nl  = _file.readline().split()[1]
    force_nl  = True if force_nl == "True" else False
    dlogz     = float(_file.readline().split()[1])
    sampler   = _file.readline().split()[1]
    mc_move   = _file.readline().split()[1]
    lsq_base  = _file.readline().split()[1]
    lcjitt    = _file.readline().split()[1]
    rvjitt    = _file.readline().split()[1]

    _adump    = _file.readline().split()
    jittlo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
    jitthi    = float(_adump[2][_adump[2].find("[")+1:_adump[2].find(",")])
    lcjittlim = [jittlo, jitthi]

    _adump    = _file.readline().split()
    jittlo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
    jitthi    = float(_adump[2][_adump[2].find("[")+1:_adump[2].find(",")])
    rvjittlim = [jittlo, jitthi]

    
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
                        LCjitter_loglims=lcjittlim, RVjitter_lims=rvjittlim, 
                        verbose=verbose)
    
    fit_obj.sampling(sampler=sampler,n_cpus=ncpus, emcee_move=mc_move,
                    n_chains=nchains, n_burn   = nburn, n_steps  = nsteps, 
                    n_live=nlive, force_nlive=force_nl,
                    dyn_dlogz=dlogz,verbose=verbose )

    _file.close()

    if return_fit:
        from .fit_data import run_fit
        result =   run_fit(lc_obj, rv_obj, fit_obj) 
        return lc_obj,rv_obj,fit_obj,result

    return lc_obj,rv_obj,fit_obj



