from ._classes import load_lightcurves, load_rvs, fit_setup,_print_output
from copy import deepcopy
import numpy as np 
from .VERSION import __version__
from .misc import _compare_nested_structures, compare_objs
import inspect, os, sys, yaml, re

def new_getfile(object, _old_getfile=inspect.getfile):
    """
    Get the source file of a class method.
    This is a modified version of inspect.getfile that gets the source file of a class method.
    If the object is a class, it looks for the source file of the method that is a class method.
    If the object is a method, it gets the source file of the method.
    """
    if not inspect.isclass(object):
        return _old_getfile(object)
    
    # Lookup by parent module (as in current inspect)
    if hasattr(object, '__module__'):
        object_ = sys.modules.get(object.__module__)
        if hasattr(object_, '__file__'):
            return object_.__file__
    
    # If parent module is __main__, lookup by methods (NEW)
    for name, member in inspect.getmembers(object):
        if inspect.isfunction(member) and object.__qualname__ + '.' + member.__name__ == member.__qualname__:
            return inspect.getfile(member)
    else:
        raise TypeError('Source for {!r} not found'.format(object))


def fit_configfile(config_file = "input_config.dat", out_folder = "output", 
                    init_decorr=False, rerun_result=True, resume_sampling=False, 
                    lc_path=None, rv_path=None, verbose=False):
    """
    Run CONAN fit from configuration file. 
    This loads the config file and creates the required objects (lc_obj, rv_obj, fit_obj) to perform the fit.
    
    Parameters
    -----------
    config_file: filepath;
        path to configuration file.
    out_folder: filepath;
        path to folder where output files will be saved.
    init_decorr: bool;
        whether to run least-squares fit to determine start values of the decorrelation parameters. 
        Default is False
    rerun_result: bool;
        whether to rerun using with already exisiting result inorder to remake plots/files. Default is True
    resume_sampling: bool;
        resume sampling from last saved position 
    lc_path: str;
        path to light curve files. If None, the path in the config file is used.
    rv_path: str;
        path to radial velocity files. If None, the path in the config file is used.
    verbose: bool;
        show print statements

    Returns
    --------
    result: object;
        result object containing chains of the fit.
    """
    from .fit_data import run_fit

    lc_obj, rv_obj, fit_obj = load_configfile(config_file, init_decorr=init_decorr, 
                                                lc_path=lc_path, rv_path=rv_path, verbose=verbose)
    result = run_fit(lc_obj, rv_obj, fit_obj,out_folder=out_folder,
                        rerun_result=rerun_result,resume_sampling=resume_sampling,verbose=verbose)
    return result


def rerun_result(result_folder,  out_folder = None, resume_sampling=False, verbose=True):
    """
    rerun the fit using config_save.dat file in a previous result folder. 
    This can be to regenerate plot or continue sampling with same setup.

    Parameters
    -----------
    result_folder: str;
        path to folder containing the config_save.dat file from previous CONAN launch.
    out_folder: str;
        path to folder where output files will be saved. If None, the output files are saved in the result_folder.
    resume_sampling: bool;
        resume sampling from last saved position
    verbose: bool;
        show print statements

    Returns
    --------
    result: object;
        result object containing chains of the fit.
    """
    
    assert os.path.exists(result_folder), f"{result_folder} does not exist"
    assert os.path.exists(result_folder+"/config_save.dat") or os.path.exists(result_folder+"/config_save.yaml"), f"config_save file (.dat or .yaml) does not exist in {result_folder}"

    if out_folder is None: out_folder = result_folder
    try:
        result = fit_configfile(config_file = result_folder+"/config_save.dat", out_folder = out_folder, 
                                rerun_result=True, resume_sampling=resume_sampling, verbose=verbose)
    except Exception as e:
        print(f"Error occurred while rerunning result with config_save.dat: {e}")
        result = fit_configfile(config_file = result_folder+"/config_save.yaml", out_folder = out_folder, 
                                rerun_result=True, resume_sampling=resume_sampling, verbose=verbose)

    return result

def _skip_lines(file, n):
    """ takes an open file object and skips the reading of lines by n lines """
    for i in range(n):
        dump = file.readline()


def _prior_value(str_prior): 
    """
    convert string prior into float/tuple
    
    Parameters
    -----------
    str_prior: str;
        string representation of prior value e.g N(0.5,0.1) or F(0.5) or U(0.1,0.5,0.9) or LU(0.1,0.5,0.9)
    
    Returns
    --------
    val: float, tuple;
        value of the prior

    Examples
    ---------
    >>> _prior_value("N(0.5,0.1)")
    (0.5,0.1)
    >>> _prior_value("F(0.5)")
    0.5
    >>> _prior_value("U(0.1,0.5,0.9)")
    (0.1,0.5,0.9)
    >>> _prior_value("TN(0,1,0.2,0.1)")
    (0,1,0.2,0.1)
    """
    prior_str = str_prior.replace(" ","")   #remove spaces
    if "None" in prior_str: 
        return None
    assert '(' in  prior_str and prior_str.endswith(')'), f"prior value must be one of ['N(mu,std)','F(val)','U(min,start,max)','LU(min,start,max)'] but {str_prior} given. check parenthesis/spaces"
    prior_str   = prior_str[prior_str.find("(")+1:prior_str.find(")")].split(",")
    tuple_prior = [float(v) for v in prior_str]
    tuple_prior = [(int(v) if v.is_integer() else float(v)) for v in tuple_prior]
    if str_prior.startswith("LU("):
        tuple_prior.append("LU") #add LU str to indicate log-uniform prior

    val = tuple_prior[0] if len(tuple_prior)==1 else tuple(tuple_prior)
    return val


def create_configfile(lc_obj=None, rv_obj=None, fit_obj=None, filename="input_config.dat", both=True, verify=False): 
    """
    create configuration file that of lc_obj, rv_obj, amd fit_obj setup.
    
    Parameters
    -----------
    lc_obj : object,None;
        Instance of CONAN.load_lightcurve() object and its attributes.
    rv_obj : object, None;
        Instance of CONAN.load_rvs() object and its attributes.
    fit_obj : object;
        Instance of CONAN.fit_setup() object and its attributes.
    filename : str;
        name of the configuration file to be saved.
    verify : bool;
        whether to verify that loading from the created config file will give the same objects (lc_obj, rv_obj, fit_obj)
    """

    if lc_obj is None and rv_obj is not None:
        lc_obj=rv_obj._lcobj
    if rv_obj is None:
        rv_obj = load_rvs(verbose=False)
    if lc_obj!=None: 
        rv_obj._lcobj = lc_obj
    if fit_obj is None:
        fit_obj = fit_setup(verbose=False)

    if filename.endswith(".dat") or both==True:
        create_datfile(lc_obj=lc_obj, rv_obj=rv_obj, fit_obj=fit_obj, filename=filename.split('.')[0]+'.dat', verify=verify)

    if filename.endswith((".yaml", ".yml")) or both==True:
        create_yamlfile(lc_obj=lc_obj, rv_obj=rv_obj, fit_obj=fit_obj, filename=filename.split('.')[0]+'.yaml', verify=verify)


def create_datfile(lc_obj=None, rv_obj=None, fit_obj=None, filename="input_config.dat",verify=False): 
    """
    create configuration file that of lc_obj, rv_obj, amd fit_obj setup.
    
    Parameters
    -----------
    lc_obj : object,None;
        Instance of CONAN.load_lightcurve() object and its attributes.
    rv_obj : object, None;
        Instance of CONAN.load_rvs() object and its attributes.
    fit_obj : object;
        Instance of CONAN.fit_setup() object and its attributes.
    filename : str;
        name of the configuration file to be saved.
    verify : bool;
        whether to verify that loading from the created config file will give the same objects (lc_obj, rv_obj, fit_obj)
    """

    dirname = os.path.dirname(filename)
    dirname = "." if dirname == "" else dirname
    f = open(filename,"w")
    f.write(f"# ========================================== CONAN configuration file v{__version__} ======================================== \n")
    f.write("#      *********** KEYS *****************************************************************************************\n")
    f.write("#      PRIORS: *Fix-F(val), *Norm-N(mu,std), *Uni-U(min,start,max), *TruncNorm–(min,max,mu,std), *LogUni-LU(min,start,max)\n")
    f.write("#      Ssmp         : supersampling - x{factor} e.g. x30 to create 30 subexposures per point\n")
    f.write("#      ClipOutliers : c{column_no}:W{window_width}C{clip_sigma}n{niter} e.g. c1:W11C5n1. column_no='a' to clip in all valid columns\n")
    f.write("#      scl_col      : scale data columns – ['med_sub','rs0to1','rs-1to1','None']\n")
    f.write("#      spline       : c{column_no}:d{degree}K{knot_spacing} e.g. c0:d3K2 \n")
    f.write("#      ***********************************************************************************************************\n")
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    f.write(f"\tLC_filepath: {lc_obj._fpath}\n")
    f.write(f"\tRV_filepath: {rv_obj._fpath}\n")
    f.write(f"\tn_planet: {lc_obj._nplanet}\n")
    f.write("# -----------------------------------------PHOTOMETRY--------------------------------------------------------------------\n")
    f.write(f"\t{'LC_auto_decorr:':15s} False   | delta_BIC: -5  # automatically determine baseline function for LCs with delta_BIC=-5\n")
    f.write(f"\t{'exclude_cols:':15s} []                         # list of column numbers (e.g. [3,4]) to exclude from decorrelation.\n")
    f.write(f"\t{'enforce_pars:':15s} []                         # list of decorr params (e.g. [B3, A5]) to enforce in decorrelation\n")
    _print_output(lc_obj,"lc_baseline",file=f)
    _print_output(lc_obj,"sinusoid",file=f)
    _print_output(lc_obj,"gp",file=f)
    f.write("# -----------------------------------------RADIAL VELOCITY---------------------------------------------------------------\n")
    f.write(f"\t{'RV_auto_decorr:':15s} False   | delta_BIC: -5  # automatically determine baseline function for the RVs\n")
    f.write(f"\t{'exclude_cols:':15s} []                         # list of column numbers (e.g. [3,4]) to exclude from decorrelation.\n")
    f.write(f"\t{'enforce_pars:':15s} []                         # list of decorr params (e.g. [B3, A5]) to enforce in decorrelation\n")
    _print_output(rv_obj,"rv_baseline",file=f)
    _print_output(rv_obj,"rv_gp",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(lc_obj,"planet_parameters",file=f)
    _print_output(lc_obj,"limb_darkening",file=f)
    _print_output(lc_obj,"depth_variation",file=f)
    _print_output(lc_obj,"timing_variation",file=f)
    _print_output(lc_obj,"phasecurve",file=f)
    _print_output(lc_obj,"custom_LCfunction",file=f)
    if lc_obj._custom_LCfunc.func is not None:
        inspect.getfile = new_getfile
        cust_func_str = inspect.getsource(lc_obj._custom_LCfunc.func)  #get the source code of the custom function
        op_func_str   = inspect.getsource(lc_obj._custom_LCfunc.op_func) if lc_obj._custom_LCfunc.op_func is not None else 'None'
        with open(f"{dirname}/custom_LCfunc.py","w") as fxn:
            if lc_obj._custom_LCfunc.replace_LCmodel: 
                fxn.write("from CONAN.misc import default_LCpars_dict as LC_pars\n")
            fxn.write(cust_func_str)
            if op_func_str!='None': fxn.write(op_func_str)

    _print_output(rv_obj,"custom_RVfunction",file=f)
    if rv_obj._custom_RVfunc.func is not None:
        inspect.getfile = new_getfile
        cust_rvfunc_str = inspect.getsource(rv_obj._custom_RVfunc.func)
        op_rvfunc_str   = inspect.getsource(rv_obj._custom_RVfunc.op_func) if rv_obj._custom_RVfunc.op_func is not None else 'None'
        with open(f"{dirname}/custom_RVfunc.py","w") as fxn:
            if rv_obj._custom_RVfunc.replace_RVmodel:
                fxn.write("from CONAN.misc import default_RVpars_dict as RV_pars\n")
            fxn.write(cust_rvfunc_str)
            if op_rvfunc_str!='None': fxn.write(op_rvfunc_str)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(lc_obj,"contamination",file=f)
    _print_output(fit_obj,"stellar_pars",file=f)
    f.write("# -----------------------------------------------------------------------------------------------------------------------\n")
    _print_output(fit_obj, "fit",file=f)
    f.close()
    
    print(f"configuration file saved as {filename}")

    if verify:
        lc_obj1, rv_obj1, fit_obj1 = deepcopy(lc_obj), deepcopy(rv_obj), deepcopy(fit_obj)
        lc_obj2, rv_obj2, fit_obj2 = load_configfile(filename, verbose=False)

        ignore = [  "_lcobj","_rvobj","_fitobj", "_custom_LCfunc", "_custom_RVfunc", 
                    "_decorr_result", "_rvdecorr_result", "_tmodel", "_bases_init", "_RVbases_init",
                    "_tra_occ_pars", "_rv_pars", "_rvmodel"]

        if not compare_objs(lc_obj1,lc_obj2, ignore=ignore):   print("\nlc_obj loaded from this config file is not equal to original lc_obj")
        if not compare_objs(rv_obj1,rv_obj2, ignore=ignore):   print("rv_obj loaded from this config file is not equal to original rv_obj")
        if not compare_objs(fit_obj1,fit_obj2, ignore=ignore): print("fit_obj loaded from this config file is not equal to original fit_obj")


def create_yamlfile(lc_obj, rv_obj, fit_obj, filename="input_config.yaml", verify=False):
    """
    Create a YAML file with the current setup of the light curve and radial velocity objects.
    
    Parameters
    ----------
    lc_obj : LightCurve object
        The light curve object containing the light curve data and setup.
    rv_obj : RadialVelocity object
        The radial velocity object containing the RV data and setup.
    fit_obj : FitSetup object
        The fit setup object containing the fitting parameters.
    filename : str, optional
        The name of the YAML file to save the configuration. Default is "input_config.yaml".
    verbose : bool, optional
        If True, print additional information during the process.
    
    Returns
    -------
    dict
        A dictionary containing the YAML configuration.
    """

    abbr_reps = lambda x: x if np.iterable(x) and len(set(x))>1 else [] if np.iterable(x) and len(set(x))==0 else x[0]  # abbreviate list of items. if all same, return the first item

    sine        = lc_obj._sine_dict
    planet_pars = lc_obj._planet_pars
    rho_dur     = 'rho' if "rho_star" in planet_pars["pl1"] else 'dur'
    ecc_sesinw  = "ecc" if "Eccentricity" in planet_pars["pl1"] else "sesinw"

    # Light curve GP section
    gp_sect = []
    lc_gp   = lc_obj._GP_dict if lc_obj is not None else {}
    if lc_gp == {}: 
        val    = dict(  lc_name     = None, 
                        kernel      = None, 
                        par         = None,  
                        h1_amp      = None, 
                        h2_len_scale= None,
                        h3_other    = None, 
                        h4_period   = None,
                        op          = "",
                        gp_pck      = 'n'  
        )
        gp_sect.append(val)  
    else:
        if lc_obj._sameLCgp.filtflag:
            for f in lc_obj._sameLCgp.filters:
                lc = lc_obj._sameLCgp.LCs[f][0]

                ngp = lc_gp[lc]["ngp"]
                val    = dict(  lc_name = f, 
                                kernel      = [lc_gp[lc][f"amplitude{n}"].user_data.kernel for n in range(ngp)], 
                                par         = [lc_gp[lc][f"amplitude{n}"].user_data.col for n in range(ngp)],  
                                h1_amp      = [lc_gp[lc][f"amplitude{n}"].prior_str for n in range(ngp)], 
                                h2_len_scale= [lc_gp[lc][f"lengthscale{n}"].prior_str for n in range(ngp)],
                                h3_other    = [lc_gp[lc][f"h3{n}"].prior_str for n in range(ngp)], 
                                h4_period   = [lc_gp[lc][f"h4{n}"].prior_str for n in range(ngp)],
                                operation   = [lc_gp[lc]["op"][n-1] for n in range(1,ngp)],
                                gp_pck      = lc_obj._useGPphot[lc_obj._names.index(lc)]
                )
                gp_sect.append(val)

        else:
            if lc_obj._allLCgp:  #shortcut.  create just one gp config if all LCs have the same GP
                equal_allgp = all([_compare_nested_structures(lc_gp[list(lc_gp.keys())[0]],lc_gp[lc]) for lc in list(lc_gp.keys())[1:]])
            else:
                equal_allgp = False
            
            for lc in lc_gp.keys():
                ngp = lc_gp[lc]["ngp"]

                val    = dict(  lc_name     = 'same' if lc_obj._sameLCgp.flag else "all" if equal_allgp else lc,
                                kernel      = [lc_gp[lc][f"amplitude{n}"].user_data.kernel for n in range(ngp)],
                                par         = [lc_gp[lc][f"amplitude{n}"].user_data.col for n in range(ngp)],
                                h1_amp      = [lc_gp[lc][f"amplitude{n}"].prior_str for n in range(ngp)],
                                h2_len_scale= [lc_gp[lc][f"lengthscale{n}"].prior_str for n in range(ngp)],
                                h3_other    = [lc_gp[lc][f"h3{n}"].prior_str for n in range(ngp)],
                                h4_period   = [lc_gp[lc][f"h4{n}"].prior_str for n in range(ngp)],
                                operation   = [lc_gp[lc]["op"][n-1] for n in range(1,ngp)],
                                gp_pck      = lc_obj._useGPphot[lc_obj._names.index(lc)]
                            )

                gp_sect.append(val)
                if lc_obj._sameLCgp.flag or equal_allgp:      #dont print the other lc GPs if same_GP is True
                    break

    # RV GP section  
    rv_gp_sect = []
    rv_gp = rv_obj._rvGP_dict if rv_obj is not None else {}
    
    if rv_gp == {}:
        val = dict( rv_name     = None,
                    kernel      = None,
                    par         = None,
                    h1_amp      = None,
                    h2_len_scale= None,
                    h3_other    = None,
                    h4_period   = None,
                    h5_der_amp  = None,
                    error_col   = None,
                    operation   = "",
                    gp_pck      = 'n'
        )
        rv_gp_sect.append(val)
    else:
        if rv_obj._allRVgp:  #shortcut print just one line gp config if all RVs have same GP
            equal_allrvgp = all([_compare_nested_structures(rv_gp[list(rv_gp.keys())[0]],rv_gp[rv]) for rv in list(rv_gp.keys())[1:]])
        else:
            equal_allrvgp = False
            
        for rv in rv_gp.keys():
            ngp = rv_gp[rv]["ngp"]

            val = dict( rv_name     = 'same' if rv_obj._sameRVgp.flag else "all" if equal_allrvgp else rv,
                        kernel      = [rv_gp[rv][f"amplitude{n}"].user_data.kernel for n in range(ngp)],
                        par         = [rv_gp[rv][f"amplitude{n}"].user_data.col for n in range(ngp)],
                        h1_amp      = [rv_gp[rv][f"amplitude{n}"].prior_str for n in range(ngp)],
                        h2_len_scale= [rv_gp[rv][f"lengthscale{n}"].prior_str for n in range(ngp)],
                        h3_other    = [rv_gp[rv][f"h3{n}"].prior_str for n in range(ngp)],
                        h4_period   = [rv_gp[rv][f"h4{n}"].prior_str for n in range(ngp)],
                        h5_der_amp  = [rv_gp[rv][f"h5{n}"].prior_str for n in range(ngp)],
                        error_col   = [rv_gp[rv][f"amplitude{n}"].user_data.errcol for n in range(ngp)],
                        operation   = [rv_gp[rv]["op"][n-1] for n in range(1,ngp)],
                        gp_pck      = rv_obj._useGPrv[rv_obj._names.index(rv)]
                        )
            for k in val:
                if val[k] == ['None']*ngp:
                    val[k] = 'None'

            rv_gp_sect.append(val)
            if rv_obj._sameRVgp.flag or equal_allrvgp:  #dont print the other RV GPs if all are the same
                break

    # Template for the configuration file
    template = {
        # Header comment will be added separately
        # 'version': __version__,
        
        # General Configuration
        'general': {
            'n_planet': lc_obj._nplanet if lc_obj!=None else rv_obj._nplanet,
            },
        
        # Photometry Configuration
        'photometry': {
            
            'light_curves': {
                'filepath': lc_obj._fpath,
                'names': lc_obj._names,
                'filters': abbr_reps(lc_obj._filters),
                'wavelength_um': abbr_reps(lc_obj._wl),
                'supersampling': abbr_reps([int(c.config.split('x')[1]) if c.config[0]=='x' else c.config for c in lc_obj._ss]),
                'clip_outliers': abbr_reps(lc_obj._clipped_data.config),
                'scale_columns': abbr_reps(lc_obj._rescaled_data.config) if lc_obj._rescaled_data.config!=[] else [],
                'spline': abbr_reps([lc_obj._lcspline[n].conf for n in lc_obj._names]),
                'apply_jitter': abbr_reps(fit_obj._fit_dict['apply_LCjitter']) if  fit_obj._fit_dict['apply_LCjitter']!=[] else 'n',
                'baseline': {
                    'offset': abbr_reps(lc_obj._fit_offset),
                    'col0': abbr_reps([int(c[0]) for c in lc_obj._bases]),
                    'col3': abbr_reps([int(c[1]) for c in lc_obj._bases]),
                    'col4': abbr_reps([int(c[2]) for c in lc_obj._bases]),
                    'col5': abbr_reps([int(c[3]) for c in lc_obj._bases]),
                    'col6': abbr_reps([int(c[4]) for c in lc_obj._bases]),
                    'col7': abbr_reps([int(c[5]) for c in lc_obj._bases]),
                    'col8': abbr_reps([int(c[6]) for c in lc_obj._bases])
                    }
                },

            'sinusoid': {
                'names': [v.name  for k,v in sine.items() if v.trig!=None],
                'trig':  [v.trig for k,v in sine.items() if v.trig!=None],
                'n':     [v.n for k,v in sine.items() if v.trig!=None],
                'par':   [(v.par) for k,v in sine.items() if v.trig!=None],
                'P':     [(v.P.prior_str) for k,v in sine.items() if v.trig!=None],
                'amp':   [v.Amp.prior_str for k,v in sine.items() if v.trig!=None],
                'x0':    [v.x0.prior_str for k,v in sine.items() if v.trig!=None]
                },

            # GP section
            'gp': gp_sect,
            
            'limb_darkening': {
                'filters': list(lc_obj._filnames),
                'q1': lc_obj._ld_dict["prior_str1"],
                'q2': lc_obj._ld_dict["prior_str2"]
                },
            
            'tdv': {
                'fit_ddfs': lc_obj._ddfs.ddfYN,
                'drprs': lc_obj._ddfs.drprs.prior_str,
                'div_white': lc_obj._ddfs.divwhite 
                },
            
            'ttv': {
                'fit_ttvs': lc_obj._ttvs.to_fit,
                'dt': 'U'+str(lc_obj._ttvs.dt) if len(lc_obj._ttvs.dt)==3 else 'N'+str(lc_obj._ttvs.dt),
                'baseline': lc_obj._ttvs.baseline,
                'per_LC_T0': lc_obj._ttvs.per_LC_T0,
                'include_partial': lc_obj._ttvs.include_partial
                },
            
            'phase_curve': {
                'filters': list(lc_obj._filnames),
                'D_occ': [p.prior_str for p in lc_obj._PC_dict["D_occ"].values()],
                'Fn': [p.prior_str for p in lc_obj._PC_dict["Fn"].values()],
                'ph_off': [p.prior_str for p in lc_obj._PC_dict["ph_off"].values()],
                'A_ev': [p.prior_str for p in lc_obj._PC_dict["A_ev"].values()],
                'f1_ev': [p.prior_str for p in lc_obj._PC_dict["f1_ev"].values()],
                'A_db': [p.prior_str for p in lc_obj._PC_dict["A_db"].values()],
                'pc_model': lc_obj._pcmodel
                },
            
            'custom_LC_function': {
                'function': lc_obj._custom_LCfunc.func.__name__ if lc_obj._custom_LCfunc.func!=None else None,
                'x': lc_obj._custom_LCfunc.x,
                'func_pars': {k:v.prior_str for k,v in lc_obj._custom_LCfunc.par_dict.items()} if lc_obj._custom_LCfunc.func_args!={} else None,
                'extra_args': lc_obj._custom_LCfunc.extra_args if lc_obj._custom_LCfunc.extra_args!={} else None,
                'op_func': lc_obj._custom_LCfunc.op_func.__name__ if lc_obj._custom_LCfunc.op_func!=None else None,
                'replace_LCmodel': lc_obj._custom_LCfunc.replace_LCmodel
                },
            
            'contamination': {
                'filters': list(lc_obj._contfact_dict.keys()),
                'contam_factor': [v.prior_str for v in lc_obj._contfact_dict.values()]
                },
        
            'auto_decorr': {
                'get_decorr': False,
                'delta_bic': -5,
                'exclude_cols': [],
                'exclude_pars': [],
                'enforce_pars': []
            },        
        },
        
        # Radial Velocity Configuration
        'radial_velocity': {
            
            'rv_curves': {
                'filepath': rv_obj._fpath,
                'rv_unit': rv_obj._RVunit if rv_obj!=None else 'm/s',
                'names': rv_obj._names,
                'scale_columns': abbr_reps(rv_obj._rescaled_data.config) if rv_obj._rescaled_data.config!=[] else [],
                'spline': abbr_reps([rv_obj._rvspline[n].conf for n in rv_obj._names]),
                'apply_jitter': abbr_reps(fit_obj._fit_dict['apply_RVjitter'] if  fit_obj._fit_dict['apply_RVjitter']!=[] else 'n'),
                'baseline': {
                    'gammas': [gam.prior_str for gam in rv_obj._rvdict["gamma"]],
                    'col0': abbr_reps([int(c[0]) for c in rv_obj._RVbases]),
                    'col3': abbr_reps([int(c[1]) for c in rv_obj._RVbases]),
                    'col4': abbr_reps([int(c[2]) for c in rv_obj._RVbases]),
                    'col5': abbr_reps([int(c[3]) for c in rv_obj._RVbases]),
                }
            },
            
            'gp': rv_gp_sect,
            
            'custom_RV_function': {
                'function': rv_obj._custom_RVfunc.func.__name__ if rv_obj._custom_RVfunc.func!=None else None,
                'x': rv_obj._custom_RVfunc.x,
                'func_pars': {k:v.prior_str for k,v in rv_obj._custom_RVfunc.par_dict.items()} if rv_obj._custom_RVfunc.func_args!={} else None,
                'extra_args': rv_obj._custom_RVfunc.extra_args if rv_obj._custom_RVfunc.extra_args!={} else None,
                'op_func': rv_obj._custom_RVfunc.op_func.__name__ if rv_obj._custom_RVfunc.op_func!=None else None,
                'replace_RVmodel': rv_obj._custom_RVfunc.replace_RVmodel
            },
            
            'auto_decorr': {
                'get_decorr': False,
                'delta_bic': -5,
                'exclude_cols': [],
                'exclude_pars': [],
                'enforce_pars': []
            }
        },
        
        # Planet Parameters
        'planet_parameters': {
            'rho_star':     planet_pars["pl1"]["rho_star"].prior_str if rho_dur=='rho' else None,
            'Duration':     planet_pars["pl1"]["Duration"].prior_str if rho_dur=='dur' else None,
            'RpRs':         [v.get("RpRs").prior_str for k,v in planet_pars.items()],
            'Impact_para':  [v.get("Impact_para").prior_str for k,v in planet_pars.items()],
            'T_0':          [v.get("T_0").prior_str for k,v in planet_pars.items()],
            'Period':       [v.get("Period").prior_str for k,v in planet_pars.items()],
            'Eccentricity': [v.get("Eccentricity").prior_str for k,v in planet_pars.items()] if ecc_sesinw=='ecc' else None,
            'omega':        [v.get("omega").prior_str for k,v in planet_pars.items()] if ecc_sesinw=='ecc' else None,
            'sesinw':       [v.get("sesinw").prior_str for k,v in planet_pars.items()] if ecc_sesinw=='sesinw' else None,
            'secosw':       [v.get("secosw").prior_str for k,v in planet_pars.items()] if ecc_sesinw=='sesinw' else None,
            'K':            [v.get("K").prior_str for k,v in planet_pars.items()],
        },
        
        # Stellar Parameters
        'stellar_parameters': {
            'radius_rsun': 'N'+str(fit_obj._fit_dict["R_st"]) if fit_obj._fit_dict["R_st"] is not None else 'None',
            'mass_msun': 'N'+str(fit_obj._fit_dict["M_st"]) if fit_obj._fit_dict["M_st"] is not None else 'None',
            'input_method': fit_obj._fit_dict["par_input"]
        },
        
        # Fit Setup
        'fit_setup': {
            'sampler': fit_obj._fit_dict['sampler'],
            'number_of_processes': fit_obj._fit_dict['n_cpus'],
            'emcee_number_steps': fit_obj._fit_dict['n_steps'],
            'leastsq_for_basepar': fit_obj._fit_dict['leastsq_for_basepar'],
            'light_travel_time_correction': fit_obj._fit_dict['LTT_corr'],
            
            # EMCEE settings
            'emcee_number_steps': fit_obj._fit_dict['n_steps'],
            'emcee_number_chains': fit_obj._fit_dict['n_chains'],
            'emcee_burnin_length': fit_obj._fit_dict['n_burn'],
            'emcee_move': fit_obj._fit_dict['emcee_move'],
            
            # Dynesty settings
            'dynesty_nlive': fit_obj._fit_dict['n_live'],
            'force_nlive': fit_obj._fit_dict['force_nlive'],
            'dynesty_dlogz': fit_obj._fit_dict['dyn_dlogz'],
            'dynesty_nested_sampling': fit_obj._fit_dict['nested_sampling'],
            
            # Jitter and bounds
            # 'apply_lc_jitter': fit_obj._fit_dict['apply_LCjitter'] if  fit_obj._fit_dict['apply_LCjitter']!=[] else 'n',
            # 'apply_rv_jitter': fit_obj._fit_dict['apply_RVjitter'] if  fit_obj._fit_dict['apply_RVjitter']!=[] else 'n',
            'lc_jitter_loglims': fit_obj._fit_dict['LCjitter_loglims'],
            'rv_jitter_lims': fit_obj._fit_dict['RVjitter_lims'],
            'lc_basecoeff_lims': fit_obj._fit_dict['LCbasecoeff_lims'],
            'rv_basecoeff_lims': fit_obj._fit_dict['RVbasecoeff_lims'],
            
            # GP ndim options
            'apply_lc_gpndim_jitter': fit_obj._fit_dict['apply_LC_GPndim_jitter'],
            'apply_rv_gpndim_jitter': fit_obj._fit_dict['apply_RV_GPndim_jitter'],
            'apply_lc_gpndim_offset': fit_obj._fit_dict['apply_LC_GPndim_offset'],
            'apply_rv_gpndim_offset': fit_obj._fit_dict['apply_RV_GPndim_offset']
            }
        }
    

    # Convert any NumPy objects to native Python types
    template = convert_numpy_to_native(template)

    template = format_yaml(template)

    # Write the formatted file
    header = """# ========================================== CONAN YAML Configuration ==========================================\n"""
    header += f"""# This is a YAML configuration file for CONAN v{__version__}\n"""
    header += """# PRIORS: Fix-'F(val)', Norm-'N(mu,std)', Uni-'U(min,start,max)', TruncNorm–'TN(min,max,mu,std)', LogUni-'LU(min,start,max)'\n\n"""


    with open(filename, 'w') as f:
        f.write(header)
        f.write(template)
        f.write("\n# ============ END OF FILE ============================================\n")

    print(f"configuration file saved as {filename}")

    if verify:
        lc_obj1, rv_obj1, fit_obj1 = deepcopy(lc_obj), deepcopy(rv_obj), deepcopy(fit_obj)
        lc_obj2, rv_obj2, fit_obj2 = load_configfile(filename, verbose=False)
        ignore = [  "_lcobj","_rvobj","_fitobj", "_custom_LCfunc", "_custom_RVfunc", 
                    "_decorr_result", "_rvdecorr_result", "_tmodel", "_bases_init", "_RVbases_init",
                    "_tra_occ_pars", "_rv_pars", "_rvmodel"]

        if not compare_objs(lc_obj1,lc_obj2, ignore=ignore):   print("\nlc_obj loaded from this config file is not equal to original lc_obj")
        if not compare_objs(rv_obj1,rv_obj2, ignore=ignore):   print("rv_obj loaded from this config file is not equal to original rv_obj")
        if not compare_objs(fit_obj1,fit_obj2, ignore=ignore): print("fit_obj loaded from this config file is not equal to original fit_obj")


def load_configfile(configfile, return_fit=False, init_decorr=False, 
                    lc_path=None, rv_path=None, verbose=False):
    """
    configure CONAN from specified configfile (*.dat or *.yaml).
    
    Parameters
    -----------
    configfile: filepath;
        path to configuration file. configfile can be either .dat or .yaml
    return_fit: bool;
        whether to immediately perform the fit from this function call.
        if True, the result object from the fit is also returned
    init_decorr: bool;
        whether to run least-squares fit to determine start values of the decorrelation parameters. 
        Default is False
    lc_path: str;
        path to light curve files. If None, the path in the config file is used.
    rv_path: str;
        path to radial velocity files. If None, the path in the config file is used.
    verbose: bool;
        show print statements

    Returns
    --------
    lc_obj, rv_obj, fit_obj. if return_fit is True, the result object of fit is also returned
    lc_obj: object;
        light curve data object generated from `CONAN.load_lighturves()`.
    rv_obj: object;
        rv data object generated from `CONAN.load_rvs()`
    fit_obj: object;
        fitting object generated from `CONAN.fit_setup()`.
    result: object;
        result object containing chains of the mcmc fit.
    
    Examples
    --------
    >>> lc_obj, rv_obj, fit_obj = load_configfile( configfile  = 'Notebooks/WASP-127/WASP-127_EULER_LC/wasp127_euler_config.dat', 
    >>>                                            lc_path     = 'Notebooks/WASP-127/data/',
    >>>                                            rv_path     = 'Notebooks/WASP-127/data/',
    >>>                                            verbose     = True)
    
    """

    if configfile.endswith(".yaml") or configfile.endswith(".yml"):
        lc_obj, rv_obj, fit_obj = load_yamlfile(configfile=configfile, return_fit=return_fit, init_decorr=init_decorr,
                                                lc_path=lc_path, rv_path=rv_path, verbose=verbose)
    elif configfile.endswith(".dat"):
        lc_obj, rv_obj, fit_obj = load_datfile(configfile=configfile, return_fit=return_fit, init_decorr=init_decorr,
                                                lc_path=lc_path, rv_path=rv_path, verbose=verbose)
    else:
        raise ValueError(f"configfile must be a .dat or .yaml file but {configfile} given")
    
    return lc_obj, rv_obj, fit_obj


def load_datfile(configfile="input_config.dat", return_fit=False, init_decorr=False, 
                    lc_path=None, rv_path=None, verbose=False):
    """
    configure CONAN from legacy dat file.
    function is called from `load_configfile()`.
    """

    _file    = open(configfile,"r")
    dump     = _file.readline()
    version  = dump.split(" v")[1].split(" =")[0]
    version  = tuple(map(int, (version.split("."))))   # -> (3,3,1)
    _skip_lines(_file,8)                       #remove first 2 comment lines
    fpath    = _file.readline().rstrip().split()[1]                         # the path where the files are
    
    #TODO getting abs path may be unecessary since user can give lc_path and rv_path that overrides the one in config file
    if fpath.startswith("./") or fpath.startswith("../"):                   # if the path is relative to where the config file is get the absolute path
        if configfile!="config_save.dat":                                   # only get the absolute path if the config file is not from results folder
            fpath = os.path.dirname(os.path.abspath(configfile)) + "/" + fpath  
    
    rv_fpath = _file.readline().rstrip().split()[1]           # the path where the files are
    if rv_fpath.startswith("./") or rv_fpath.startswith("../"):    
        if configfile!="config_save.dat":
            rv_fpath = os.path.dirname(os.path.abspath(configfile)) + "/" + rv_fpath
    
    fpath    = fpath if lc_path is None else lc_path
    rv_fpath = rv_fpath if rv_path is None else rv_path

    nplanet  = int(_file.readline().rstrip().split()[1])      # the path where the files are
    _skip_lines(_file,1)                                      #remove 3 comment lines

    #### auto decorrelation
    dump   = _file.readline().rstrip()
    _adump = dump.split()
    assert _adump[1] in ["True","False"], f"LC_auto_decorr: must be 'True' or 'False' but {_adump[1]} given"
    use_decorr = True if _adump[1] == "True" else False
    del_BIC = float(_adump[4]) if len(_adump) > 4 else -5
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
    _names      = []                    # array where the LC filenames are supposed to go
    _filters    = []                  # array where the filter names are supposed to go
    _wl         = []
    _bases      = []                    # array where the baseline exponents are supposed to go
    _bsin       = []                      # whether to include a sin term in the baseline
    _groups     = []                   # array where the group indices are supposed to go
    _grbases    = []
    _useGPphot  = []
    
    _ss_lclist,_ss_fac = [],[]
    _clip_cols, _clip_lclist, _clip, _clip_width,_clip_niter  = [],[],[],[],[]
    _offset, _sclcol= [], []
    _spl_lclist,_spl_deg,_spl_par, _spl_knot=[],[],[],[]
    
    #read baseline specification for each listed light-curve file 
    dump = _file.readline() 
    while dump[0] != '#':                   # if it is not starting with # then
        _adump = dump.split()               # split it

        _names.append(_adump[0])            # append the first field to the name array
        _filters.append(_adump[1])          # append the second field to the filters array
        _wl.append(float(_adump[2]))    # append the second field to the filters array
        
        #supersample
        xt = _adump[3].split("|")[-1]
        if xt != "None":
            _ss_lclist.append(_adump[0])
            _ss_fac.append(float(xt.split("x")[1]))
        
        #clip_outlier
        if _adump[4]!= "None":
            _clip_lclist.append(_adump[0])
            _clip_niter.append(int(_adump[4].split("n")[1])) 
            clip_v = float(_adump[4].split("n")[0].split("C")[1]) 
            _clip.append(int(clip_v) if clip_v.is_integer() else clip_v)                   # outlier clip value
            _clip_width.append(int(_adump[4].split("n")[0].split("C")[0].split("W")[1])) # windown width
            col_nos = _adump[4].split(":")[0][1:]
            if col_nos == "a": col_nos = "135678"
            _clip_cols.append(col_nos)
        #scale columns
        _sclcol.append(_adump[5].split("|")[0])
        _offset.append(_adump[7])

        strbase=_adump[8:14]    #col0 - col7
        strbase.append(_adump[14].split("|")[0])        #col8|sin. add col8 to strbase
        base = [int(i) for i in strbase]            # convert to int
        _bases.append(base)

        _bsin.append(_adump[14].split("|")[1])           #col8|sin. sin-y/n
        group = int(_adump[15])
        _groups.append(group)
        _useGPphot.append(_adump[16])
        
        #LC spline
        if _adump[17] != "None": 
            _spl_lclist.append(_adump[0])
            if "|" not in _adump[17]:   #1D spline
                k1 = _adump[17].split("k")[-1]
                _spl_knot.append((int(k1) if float(k1).is_integer() else float(k1)) if k1 != "r" else k1)
                _spl_deg.append(int(_adump[17].split("k")[0].split("d")[-1]))
                _spl_par.append("col" + _adump[17].split("d")[0][1])
            else: #2D spline
                sp    = _adump[17].split("|")  #split the diff spline configs
                k1,k2 = sp[0].split("k")[-1], sp[1].split("k")[-1]
                k_1   = (int(k1) if float(k1).is_integer() else float(k1)) if k1 != "r" else k1
                k_2   = (int(k2) if float(k2).is_integer() else float(k2)) if k2 != "r" else k2
                _spl_knot.append( (k_1,k_2) )
                _spl_deg.append( (int(sp[0].split("k")[0].split("d")[-1]),int(sp[1].split("k")[0].split("d")[-1])) )
                _spl_par.append( ("col"+sp[0].split("d")[0][1],"col"+sp[1].split("d")[0][1]) ) 
        #move to next LC
        dump =_file.readline() 
    

    nphot = len(_names)
    _skip_lines(_file,1)                                      #remove 1 comment lines

    # ====== sinsuoid input ====================
    sin_lclist, trig, sin_n, sin_par, sin_Amp, sin_Per, sin_x0 = [],[],[],[],[],[],[]
    dump =_file.readline()
    while dump[0] != "#":
        _adump = dump.split()
        sin_lclist.append(_adump[0])
        trig.append(_adump[1])
        sin_n.append(int(_adump[2]))
        sin_par.append(_adump[3])
        sin_Amp.append(_prior_value(_adump[4]))
        sin_Per.append(_prior_value(_adump[5]))
        sin_x0.append(_prior_value(_adump[6]))
        
        #move to next LC
        dump =_file.readline()

    if len(sin_lclist)==1:
        sin_lclist = sin_lclist[0]  if sin_lclist[0] in ["all","same","filt"] else sin_lclist
    elif len(sin_lclist)>1:
        if set(sin_lclist) == set(_names):  #if all light curves have a sin term
            sin_lclist = "all" if len(sin_lclist) > 1 else sin_lclist[0]

    _skip_lines(_file,1)                                      #remove 1 comment lines

    # ========== GP input ====================
    gp_lclist,op = [],[]
    gp_pars, kernels, amplitude, lengthscale, h3, h4 = [],[],[],[],[],[]
    # Helper function to add element to tuple
    def _add_to_tuple(ngp, current, new_value):
        return (current,new_value) if ngp==2 else current+(new_value,)

    dump =_file.readline()
    if version < (3,3,11):
        while dump[0] != "#":
            _adump = dump.split()
            gp_lclist.append(_adump[0])
            gp_pars.append(_adump[1])
            kernels.append(_adump[2])
            amplitude.append(_prior_value(_adump[3]))
            lengthscale.append(_prior_value(_adump[4]))
            h3.append(None)
            h4.append(None)

            op.append(_adump[5].strip("|"))
            if op[-1] != "--":    #if theres a second kernel 
                gp_pars[-1]     = (gp_pars[-1],_adump[7])
                kernels[-1]     = (kernels[-1],_adump[8])
                amplitude[-1]   = (amplitude[-1],_prior_value(_adump[9]))
                lengthscale[-1] = (lengthscale[-1],_prior_value(_adump[10]))
                h3.append(None)
                h4.append(None)

            #move to next LC
            dump =_file.readline()

    else:
        while dump[0] != "#":
            ngp    = 1
            _adump = dump.split()
            gp_lclist.append(_adump[0])
            kernels.append(_adump[1])
            gp_pars.append(_adump[2])
            amplitude.append(_prior_value(_adump[3]))
            lengthscale.append(_prior_value(_adump[4]))
            h3.append(_prior_value(_adump[5]))
            h4.append(_prior_value(_adump[6]))

            #move to next line
            dump   = _file.readline()
            _adump = dump.split()
            while _adump[0][0] in ["|","*","+"]: #if so, file has a additional kernels
                ngp += 1
                if ngp == 2:  #if this is the second kernel
                    op.append(_adump[0].strip("|"))
                elif ngp >2 :
                    op[-1]      =  (op[-1], _adump[0].strip("|"))
                kernels[-1]     = _add_to_tuple(ngp, kernels[-1], _adump[1])
                gp_pars[-1]     = _add_to_tuple(ngp, gp_pars[-1], _adump[2])
                amplitude[-1]   = _add_to_tuple(ngp, amplitude[-1], _prior_value(_adump[3]))
                lengthscale[-1] = _add_to_tuple(ngp, lengthscale[-1], _prior_value(_adump[4]))
                h3[-1]          = _add_to_tuple(ngp, h3[-1], _prior_value(_adump[5]))
                h4[-1]          = _add_to_tuple(ngp, h4[-1], _prior_value(_adump[6]))
                #move to next gp line    
                dump   = _file.readline()
                _adump = dump.split()
            # move to next lc
        if op == []:
            op.append("")

    #check that the elements of _clip_cols are the same
    if len(_clip_cols) > 1:
        assert len(set(_clip_cols)) == 1, f"all columns to clip must be the same for all files but {_clip_cols} given"
    if _clip_cols!=[]: _clip_cols = [f"col{c}" for c in _clip_cols[0]]  #convert to list of col{c} format

    # instantiate light curve object
    lc_obj = load_lightcurves(_names, data_filepath=fpath, filters=_filters, wl=_wl, nplanet=nplanet, verbose=verbose)
    lc_obj.lc_baseline(_offset.copy(), *np.array(_bases).T, sin=_bsin, grp_id=None, gp=_useGPphot,verbose=False )
    lc_obj.add_sinusoid(lc_list=sin_lclist, trig=trig, n=sin_n, par=sin_par, Amp = sin_Amp, P=sin_Per, x0=sin_x0, verbose=False)
    lc_obj.clip_outliers(lc_list=_clip_lclist , clip=_clip, width=_clip_width,select_column=_clip_cols,niter=_clip_niter, show_plot=False,verbose=False )
    lc_obj.rescale_data_columns(method=_sclcol,verbose=False)
    lc_obj.supersample(lc_list=_ss_lclist, ss_factor=_ss_fac, verbose=False)
    lc_obj.add_spline(lc_list=_spl_lclist ,par=_spl_par , degree=_spl_deg,
                        knot_spacing=_spl_knot , verbose=False)
    if verbose: lc_obj.print("lc_baseline")
    if gp_lclist !=[]: 
        gp_lclist = gp_lclist[0] if gp_lclist[0] in ['same','all'] else gp_lclist
    gp_pck = [_useGPphot[lc_obj._names.index(lc)] for lc in lc_obj._gp_lcs()]if _useGPphot!=[] else []
    lc_obj.add_GP(lc_list=gp_lclist,par=gp_pars,kernel=kernels,operation=op,
                    amplitude=amplitude,lengthscale=lengthscale,h3=h3,h4=h4,gp_pck=gp_pck,
                    verbose=verbose)
    lc_obj._fit_offset = _offset

    ## RV ==========================================================
    #### auto decorrelation
    dump = _file.readline().rstrip()
    _adump = dump.split()
    assert _adump[1] in ["True","False"], f"RV_auto_decorr: must be 'True' or 'False' but {_adump[1]} given"
    use_decorrRV = True if _adump[1] == "True" else False
    rvdel_BIC = float(_adump[4]) if len(_adump) > 4 else -5
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
    RVunit = "km/s"
    
    dump =_file.readline()
    while dump[0] != '#':                   # if it is not starting with # then
        _adump = dump.split()               # split it
        RVnames.append(_adump[0])
        RVunit = _adump[1]
        _RVsclcol.append(_adump[2])
        strbase=_adump[4:7]                  # string array of the baseline function coeffs
        strbase.append(_adump[7].split("|")[0])
        strbase.append(_adump[8])
        base = [int(i) for i in strbase]
        RVbases.append(base)
        usegpRV.append(_adump[9])
        
        #RV spline
        if _adump[10] != "None":
            _spl_rvlist.append(_adump[0])
            if "|" not in _adump[10]:
                _spl_knot.append(float(_adump[10].split("k")[-1]))
                _spl_deg.append(int(_adump[10].split("k")[0].split("d")[-1]))
                _spl_par.append("col" + _adump[10].split("d")[0][1])
            else:
                sp = _adump[10].split("|") #split the diff spline configs
                _spl_knot.append( (float(sp[0].split("k")[-1]),float(sp[1].split("k")[-1])) )
                _spl_deg.append( (int(sp[0].split("k")[0].split("d")[-1]),int(sp[1].split("k")[0].split("d")[-1])) )
                _spl_par.append( ("col"+sp[0].split("d")[0][1],"col"+sp[1].split("d")[0][1]) )
        
        gammas.append(_prior_value(_adump[12]))
        #move to next RV
        dump =_file.readline()
    

    nRV = len(RVnames)
    _skip_lines(_file,1)                                      #remove 1 comment lines
    
    # RV GP
    gp_rvlist,op = [],[]
    gp_pars, kernels, amplitude, lengthscale, h3, h4, h5, err_col = [],[],[],[],[],[],[],[]

    dump =_file.readline()
    if version < (3,3,11):
        while dump[0] != "#":
            _adump = dump.split()
            gp_rvlist.append(_adump[0])
            gp_pars.append(_adump[1])
            kernels.append(_adump[2])
            amplitude.append(_prior_value(_adump[3]))
            lengthscale.append(_prior_value(_adump[4]))
            h3.append(None)
            h4.append(None)
            h5.append(None)
            err_col.append(None)
            
            op.append(_adump[5].strip("|"))
            if op[-1] != "--":    #if theres a second kernel 
                gp_pars[-1]     = (gp_pars[-1],_adump[7])
                kernels[-1]     = (kernels[-1],_adump[8])
                amplitude[-1]   = (amplitude[-1],_prior_value(_adump[9]))
                lengthscale[-1] = (lengthscale[-1],_prior_value(_adump[10]))
                h3.append(None)
                h4.append(None)
                h5.append(None)
                err_col.append(None)

            #move to next RV
            dump =_file.readline()

    else:
        while dump[0] != "#":
            ngp    = 1
            _adump = dump.split()
            gp_rvlist.append(_adump[0])
            kernels.append(_adump[1])
            gp_pars.append(_adump[2])
            amplitude.append(_prior_value(_adump[3]))
            lengthscale.append(_prior_value(_adump[4]))
            h3.append(_prior_value(_adump[5]))
            h4.append(_prior_value(_adump[6]))
            if version >= (3,3,12):  #if then h5 and err_col are present for GP derivative amp and err
                h5.append(_prior_value(_adump[8]))
                err_col.append(_adump[9])

            
            #move to next line
            dump   = _file.readline()
            _adump = dump.split()
            while _adump[0][0] in ["|","*","+"]: #if so, file has additional kernels
                ngp += 1
                if ngp == 2:  #if this is the second kernel
                    op.append(_adump[0].strip("|"))
                elif ngp >2 :
                    op[-1]      = (op[-1], _adump[0].strip("|"))
                kernels[-1]     = _add_to_tuple(ngp, kernels[-1], _adump[1])
                gp_pars[-1]     = _add_to_tuple(ngp, gp_pars[-1], _adump[2])
                amplitude[-1]   = _add_to_tuple(ngp, amplitude[-1], _prior_value(_adump[3]))
                lengthscale[-1] = _add_to_tuple(ngp, lengthscale[-1], _prior_value(_adump[4]))
                h3[-1]          = _add_to_tuple(ngp, h3[-1], _prior_value(_adump[5]))
                h4[-1]          = _add_to_tuple(ngp, h4[-1], _prior_value(_adump[6]))
                if version >= (3,3,12):  #if then h5 and err_col are present for GP derivative amp and err
                    h5[-1]      = _add_to_tuple(ngp, h5[-1], _prior_value(_adump[8]))
                    err_col[-1] = _add_to_tuple(ngp, err_col[-1], _adump[9])
                #move to next rv file    
                dump   = _file.readline()
                _adump = dump.split()
        if op == []:
            op.append("")
        if h5==[]: h5 = None
        if err_col==[] : err_col=None
    
    rv_obj = load_rvs(RVnames,rv_fpath, nplanet=nplanet,rv_unit=RVunit,lc_obj=lc_obj, verbose=verbose)
    rv_obj.rv_baseline(*np.array(RVbases).T, gamma=gammas,gp=usegpRV,verbose=False) 
    rv_obj.rescale_data_columns(method=_RVsclcol,verbose=False)
    rv_obj.add_spline(rv_list=_spl_rvlist ,par=_spl_par, degree=_spl_deg,
                        knot_spacing=_spl_knot, verbose=False)
    if verbose: rv_obj.print("rv_baseline")
    if gp_rvlist !=[]: 
        gp_rvlist = gp_rvlist[0] if gp_rvlist[0] in ['same','all'] else gp_rvlist
    gp_pck = [usegpRV[rv_obj._names.index(rv)] for rv in rv_obj._gp_rvs()] if usegpRV!=[] else []
    rv_obj.add_rvGP(rv_list=gp_rvlist,par=gp_pars,kernel=kernels,operation=op,
                    amplitude=amplitude,lengthscale=lengthscale,h3=h3,h4=h4,h5=h5, err_col=err_col,gp_pck=gp_pck,
                    verbose=verbose)
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    ## Planet parameters
    dump    = _file.readline()
    _adump  = dump.split()
    pl_pars = {}
    rho_dur = _adump[0]    # rho_star/[Duration]
    #select string in rho_dur with []
    rho_dur = rho_dur[rho_dur.find("[")+1:rho_dur.find("]")]
    pl_pars[rho_dur] = _prior_value(_adump[2])
    
    par_names = ["RpRs","Impact_para", "T_0", "Period", "Eccentricity","omega", "K"]
        
    for n in range(1,nplanet+1):        #load parameters for each planet
        lbl = f"_{n}" if nplanet>1 else ""
        _skip_lines(_file,1)          #remove dashes
        for pn in par_names:
            dump =_file.readline()
            _adump = dump.split()
            if pn in [ "Eccentricity"+lbl,"omega"+lbl]:
                ecc_par = _adump[0]   # "[Eccentricity]/sesinw" or "[omega]/secosw"
                if ecc_par == pn: ecc_par = '['+pn+']'   #backwards compatibility 
                ecc_par = ecc_par[ecc_par.find("[")+1:ecc_par.find("]")]   #select option in []
                if n==1: pl_pars[ecc_par] = []
                pl_pars[ecc_par].append(_prior_value(_adump[2]))
            else:
                if n==1: pl_pars[pn] = []
                pl_pars[pn].append(_prior_value(_adump[2]))
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    ## limb darkening
    q1, q2 = [],[]
    dump   = _file.readline()
    while dump[0] != "#":
        _adump = dump.split()
        q1.append(_prior_value(_adump[2]))
        q2.append(_prior_value(_adump[3]))
        dump = _file.readline()
    assert len(q1) == len(lc_obj._filnames), f"number of q1 values must be equal to number of unique filters({len(lc_obj._filnames)}) but len(q1)={len(q1)}"
    _skip_lines(_file,1)                                      #remove 2 comment lines

    #DDFs
    dump   = _file.readline()
    _adump = dump.split()
    ddfyn,ddf_pri,div_wht  = _adump[0], _prior_value(_adump[1]), _adump[2]
    _skip_lines(_file,2)                                      #remove 2 comment lines
    
    #TTVS
    dump   =_file.readline()
    _adump = dump.split()
    ttvs, dt, base = _adump[0], _prior_value(_adump[1]), float(_adump[2])
    per_LC_T0, incl_partial = True if _adump[3]=="True" else False, True if _adump[4]=="True" else False
    _skip_lines(_file,2)                                      #remove 2 comment lines

    #phase curve
    D_occ,Fn,ph_off,A_ev,f1_ev,A_db,pc_model = [],[],[],[],[],[],[]    
    dump   = _file.readline()
    while dump[0] != "#":
        _adump = dump.split()
        D_occ.append(_prior_value(_adump[1]))
        Fn.append(_prior_value(_adump[2]))
        ph_off.append(_prior_value(_adump[3]))
        A_ev.append(_prior_value(_adump[4]))
        if version < (3,3,11):
            A_db.append(_prior_value(_adump[5]))
            f1_ev.append(0)
            pc_model.append("cosine")
        else:
            f1_ev.append(_prior_value(_adump[5]))
            A_db.append(_prior_value(_adump[6]))
            pc_model.append(_adump[7])

        dump = _file.readline()
    assert len(D_occ) == len(lc_obj._filnames), f"number of D_occ values must be equal to number of unique filters({len(lc_obj._filnames)}) but len(D_occ)={len(D_occ)}"


    #custom LC function
    dump   = _file.readline()
    _adump = dump.split()
    if _adump[0] == "function":   #custom function lines
        func_name = _adump[2]
        if func_name!='None':
            # directly import function named func_name from custom_LCfunc.py
            import custom_LCfunc
            custom_lcfunc = getattr(custom_LCfunc, func_name)

            func_x    = _file.readline().split()[2]
            _adump    = _file.readline().split()
            func_args = {}
            if _adump[2]!='None':
                str_pars  = _adump[2].split("),")
                str_pars  = [s if s[-1]==')' else s+')' for s in str_pars]   # add ')' to the all elements lacking closing bracket
                for p in str_pars:
                    p_name, p_prior = p.split(":")
                    func_args[p_name] = _prior_value(p_prior)
            dump   = _file.readline()
            _adump = dump.split()
            extra_args = {}
            if _adump[2]!='None':
                str_expars  = _adump[2].split(",")
                for p in str_expars:
                    p_name, p_val = p.split(":")
                    extra_args[p_name] = p_val
            dump   = _file.readline()
            _adump = dump.split()
            opfunc_name = _adump[2]
            if opfunc_name!='None':
                op_func = getattr(custom_LCfunc, opfunc_name)
            else: op_func = None
            dump   = _file.readline()
            _adump = dump.split()
            replace_LCmodel = True if _adump[2] == "True" else False
        else:   #skip remaining custom function lines (4) since func_name is None
            _skip_lines(_file,5)
            custom_lcfunc,func_x,func_args,extra_args,op_func,replace_LCmodel = None,None,{},{},None,False
        
        _skip_lines(_file,1)                                      #remove 3 comment lines

    #custom RV function
    dump   = _file.readline()
    _adump = dump.split()
    if _adump[0] == "function":   #custom function lines
        func_name = _adump[2]
        if func_name!='None':
            # directly import function named func_name from custom_RVfunc.py
            import custom_RVfunc
            custom_rvfunc = getattr(custom_RVfunc, func_name)

            rvfunc_x    = _file.readline().split()[2]
            _adump    = _file.readline().split()
            rvfunc_args = {}
            if _adump[2]!='None':
                str_pars  = _adump[2].split("),")
                str_pars  = [s if s[-1]==')' else s+')' for s in str_pars]   # add ')' to the all elements lacking closing bracket
                for p in str_pars:
                    p_name, p_prior = p.split(":")
                    rvfunc_args[p_name] = _prior_value(p_prior)
            dump   = _file.readline()
            _adump = dump.split()
            rvextra_args = {}
            if _adump[2]!='None':
                str_expars  = _adump[2].split(",")
                for p in str_expars:
                    p_name, p_val = p.split(":")
                    rvextra_args[p_name] = p_val
            dump   = _file.readline()
            _adump = dump.split()
            opfunc_name = _adump[2]
            if opfunc_name!='None':
                op_rvfunc = getattr(custom_RVfunc, opfunc_name)
            else: op_rvfunc = None
            dump   = _file.readline()
            _adump = dump.split()
            replace_RVmodel = True if _adump[2] == "True" else False
        else:   #skip remaining custom function lines (4) since func_name is None
            _skip_lines(_file,5)
            custom_rvfunc,rvfunc_x,rvfunc_args,rvextra_args,op_rvfunc,replace_RVmodel = None,None,{},{},None,False
        
        _skip_lines(_file,3)                                      #remove 3 comment lines

    #contamination factors
    cont_fac = []
    dump   = _file.readline()
    while dump[0] != "#":
        _adump = dump.split()
        cont_fac.append(_prior_value(_adump[1]))
        dump = _file.readline()
    assert len(cont_fac) == len(lc_obj._filnames), f"number of contamination factors must be equal to number of unique filters({len(lc_obj._filnames)}) but len(cont_fac)={len(cont_fac)}"
    _skip_lines(_file,1)                                      #remove 2 comment lines
    
    lc_obj.planet_parameters(**pl_pars, verbose=verbose)
    lc_obj.limb_darkening(q1,q2,verbose=verbose)
    lc_obj.transit_depth_variation(ddFs=ddfyn,dRpRs=ddf_pri, divwhite=div_wht,verbose=verbose)
    lc_obj.transit_timing_variation(ttvs=ttvs, dt=dt, baseline_amount=base,include_partial=incl_partial,per_LC_T0=per_LC_T0,verbose=verbose,print_linear_eph=False)
    lc_obj.phasecurve(D_occ=D_occ, Fn=Fn, ph_off=ph_off, A_ev=A_ev, f1_ev=f1_ev, A_db=A_db, pc_model=pc_model,verbose=verbose)
    lc_obj.add_custom_LC_function(func=custom_lcfunc,x=func_x,func_args=func_args,extra_args=extra_args,op_func=op_func,replace_LCmodel=replace_LCmodel,verbose=verbose)
    rv_obj.add_custom_RV_function(func=custom_rvfunc,x=rvfunc_x,func_args=rvfunc_args,extra_args=rvextra_args,op_func=op_rvfunc,replace_RVmodel=replace_RVmodel,verbose=verbose)
    lc_obj.contamination_factors(cont_ratio=cont_fac, verbose=verbose)

    if nphot > 0:
        if use_decorr or init_decorr:
            if init_decorr and verbose: print("\ngetting start values for LC decorrelation parameters ...")
            lc_obj.get_decorr(**pl_pars,q1=q1,q2=q2,
                                D_occ=D_occ[0] if len(D_occ)>0 else 0, 
                                Fn=Fn[0] if len(Fn)>0 else 0, 
                                ph_off=ph_off[0] if len(ph_off)>0 else 0, 
                                A_ev=A_ev[0] if len(A_ev)>0 else 0,
                                f1_ev=f1_ev[0] if len(f1_ev)>0 else 0, 
                                A_db=A_db[0] if len(A_db)>0 else 0, 
                                pc_model = pc_model, plot_model=False,
                                setup_baseline=use_decorr,exclude_cols=exclude_cols,delta_BIC=del_BIC,
                                enforce_pars=enforce_pars, verbose=verbose if use_decorr else False)
            if init_decorr:  #if not use_decorr, compare the  get_decorr pars to the user-defined ones and only use start values for user-defined ones
                rel_cols = [b[:6] for b in lc_obj._bases]
                _ = [b.insert(1,0) for b in rel_cols for _ in range(2)] #insert 0 to replace cols 1 and 2
                for j in range(lc_obj._nphot):
                    for i,v in enumerate(rel_cols[j]):
                        if i in [1,2]: continue
                        if v == 0: lc_obj._bases_init[j][f"A{i}"] = lc_obj._bases_init[j][f"B{i}"] = 0
                        if v >= 1: lc_obj._bases_init[j][f"A{i}"] = lc_obj._bases_init[j][f"A{i}"]
                        if v == 2: lc_obj._bases_init[j][f"B{i}"] = lc_obj._bases_init[j][f"B{i}"]

    if nRV > 0:
        if use_decorrRV or init_decorr:
            if init_decorr and verbose: print("\ngetting start values for RV decorrelation parameters ...\n")
            rv_obj.get_decorr(  T_0=pl_pars["T_0"], Period=pl_pars["Period"], K=pl_pars["K"],
                                Eccentricity=pl_pars["Eccentricity"], omega=pl_pars["omega"],
                                gamma=gammas[0] if len(gammas)>0 else 0, setup_baseline=use_decorrRV,
                                exclude_cols=exclude_colsRV, enforce_pars=enforce_parsRV, delta_BIC=rvdel_BIC,
                                plot_model=False,verbose=verbose if use_decorrRV else False)
            if init_decorr:  #if not use_decorr, compare the  get_decorr pars to the user-defined ones and only use start values for user-defined ones
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
    dyn_samp  = _file.readline().split()[1]
    lsq_base  = _file.readline().split()[1]
    lcjitt    = _file.readline().split()[1:] #get all inputs for lcjitter
    lcjitt    = " ".join(lcjitt).replace("[","").replace("]","").replace(" ","").replace("'","").replace("\"","").split(",")
    rvjitt    = _file.readline().split()[1:] #get all inputs for rvjitter
    rvjitt    = " ".join(rvjitt).replace("[","").replace("]","").replace(" ","").replace("'","").replace("\"","").split(",")


    #lcjittlims
    _adump    = _file.readline().split()
    assert len(_adump)==2, f"LCjitter_loglims: must be a list of 2 values (e.g. [0.1,0.5], with no space between limits) or 'auto' but {_adump[1:]} given" 
    if "auto" in _adump[1]:
        lcjittlim = "auto"
    else:
        jittlo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
        jitthi    = float(_adump[1][_adump[1].find(",")+1:_adump[1].find("]")])
        lcjittlim = [jittlo, jitthi]
    #rvjittlims
    _adump    = _file.readline().split()
    assert len(_adump)==2, f"RVjitter_lims: must be a list of 2 values (e.g. [0.1,0.5], with no space between limits) or 'auto' but {_adump[1:]} given"
    if "auto" in _adump[1]:
        rvjittlim = "auto"
    else:
        jittlo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
        jitthi    = float(_adump[1][_adump[1].find(",")+1:_adump[1].find("]")])
        rvjittlim = [jittlo, jitthi]
    
    #LCbasecoeff_lims
    _adump    = _file.readline().split()
    assert len(_adump)==2, f"LCbasecoeff_lims: must be a list of 2 values (e.g. [0.1,0.5], with no space between limits) or 'auto' but {_adump[1:]} given"
    if "auto" in _adump[1]:
        lcbaselim = "auto"
    else:
        baselo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
        basehi    = float(_adump[1][_adump[1].find(",")+1:_adump[1].find("]")])
        lcbaselim = [baselo, basehi]
    #RVbasecoeff_lims
    _adump    = _file.readline().split()
    assert len(_adump)==2, f"RVbasecoeff_lims: must be a list of 2 values (e.g. [0.1,0.5], with no space between limits) or 'auto' but {_adump[1:]} given"
    if "auto" in _adump[1]:
        rvbaselim = "auto"
    else:
        baselo    = float(_adump[1][_adump[1].find("[")+1:_adump[1].find(",")])
        basehi    = float(_adump[1][_adump[1].find(",")+1:_adump[1].find("]")])
        rvbaselim = [baselo, basehi]

    ltt     = _file.readline().split()[1]
    LC_GPndim_jitt = _file.readline().split()[1] if version >= (3,3,12) else "n"
    RV_GPndim_jitt = _file.readline().split()[1] if version >= (3,3,12) else "n" 
    LC_GPndim_off  = _file.readline().split()[1] if version >= (3,3,12) else "n"
    RV_GPndim_off  = _file.readline().split()[1] if version >= (3,3,12) else "n"

    fit_obj = fit_setup(R_st = st_rad, M_st = st_mass, par_input=par_in,
                        apply_LCjitter=lcjitt, apply_RVjitter=rvjitt,
                        leastsq_for_basepar=lsq_base, 
                        LCbasecoeff_lims=lcbaselim, RVbasecoeff_lims=rvbaselim,
                        LCjitter_loglims=lcjittlim, RVjitter_lims=rvjittlim, LTT_corr=ltt,
                        apply_LC_GPndim_jitter=LC_GPndim_jitt, apply_RV_GPndim_jitter=RV_GPndim_jitt,
                        apply_LC_GPndim_offset=LC_GPndim_off, apply_RV_GPndim_offset=RV_GPndim_off,
                        verbose=verbose)
    
    fit_obj.sampling(sampler=sampler,n_cpus=ncpus, emcee_move=mc_move,
                    n_chains=nchains, n_burn   = nburn, n_steps  = nsteps, 
                    n_live=nlive, force_nlive=force_nl, nested_sampling=dyn_samp,
                    dyn_dlogz=dlogz,verbose=verbose )

    _file.close()

    if return_fit:
        from .fit_data import run_fit
        result = run_fit(lc_obj, rv_obj, fit_obj) 
        return lc_obj,rv_obj,fit_obj,result

    return lc_obj,rv_obj,fit_obj


def load_yamlfile(configfile, return_fit=False, init_decorr=False, 
                    lc_path=None, rv_path=None, verbose=False):
    """
    configure CONAN from yaml file.
    
    Parameters
    -----------
    configfile: filepath;
        path to .yaml configuration file.
    return_fit: bool;
        whether to immediately perform the fit from this function call.
        if True, the result object from the fit is also returned
    init_decorr: bool;
        whether to run least-squares fit to determine start values of the decorrelation parameters. 
        Default is False
    lc_path: str;
        path to light curve files. If None, the path in the config file is used.
    rv_path: str;
        path to radial velocity files. If None, the path in the config file is used.
    verbose: bool;
        show print statements
    Returns
    --------
    lc_obj, rv_obj, fit_obj. if return_fit is True, the result object of fit is also returned
    lc_obj: object;
        light curve data object generated from `CONAN.load_lighturves()`.
    rv_obj: object;
        rv data object generated from `CONAN.load_rvs()`
    fit_obj: object;
        fitting object generated from `CONAN.fit_setup()`.
    result: object;
        result object containing chains of the mcmc fit.
    """

    _file    = yaml.safe_load(open(configfile,'r'))
    for k in _file:
        assert k in ['general','photometry','radial_velocity','planet_parameters',
                        'stellar_parameters','fit_setup']
        if k == 'general':
            for k1 in _file[k]:
                assert k1 in ['n_planet']
        if k == 'photometry':
            for k1 in _file[k]:
                assert k1 in ['light_curves','sinusoid','gp','limb_darkening','tdv','ttv',
                                'phase_curve','custom_LC_function','contamination','auto_decorr']
        if k == 'radial_velocity':
            for k1 in _file[k]:
                assert k1 in ['rv_curves','gp','custom_RV_function','auto_decorr']
        if k == 'planet_parameters':
            for k1 in _file[k]:
                assert k1 in ['rho_star','Duration','RpRs','Impact_para','T_0','Period',
                                'Eccentricity','omega','sesinw','secosw','K']
        if k == 'stellar_parameters':
            for k1 in _file[k]:
                assert k1 in ['radius_rsun','mass_msun','input_method']
        if k == 'fit_setup':
            for k1 in _file[k]:
                assert k1 in ['sampler','number_of_processes','emcee_number_steps',
                                'leastsq_for_basepar','light_travel_time_correction',
                                'emcee_number_chains','emcee_burnin_length',
                                'emcee_move','dynesty_nlive','force_nlive',
                                'dynesty_dlogz','dynesty_nested_sampling',
                                'lc_jitter_loglims','rv_jitter_lims',
                                'lc_basecoeff_lims','rv_basecoeff_lims',
                                'apply_lc_gpndim_jitter','apply_rv_gpndim_jitter',
                                'apply_lc_gpndim_offset','apply_rv_gpndim_offset']


    nplanet  = _file["general"].get("n_planet", 1)

    assert nplanet > 0, f"n_planet must be > 0 but {nplanet} given"

    if "photometry" in _file:

        # ========== Lightcurve input ====================
        lcs         = _file["photometry"]["light_curves"]
        lc_fpath    = lcs.get("filepath", None)                         # the path where the files are
        lc_fpath    = lc_fpath if lc_path is None else lc_path

        if isinstance(lcs["names"], str):  # if it is a string, convert to list
            lcs["names"] = [lcs["names"]]
        elif isinstance(lcs["names"], list):
            assert len(set(lcs["names"])) == len(lcs["names"]), "Light curve names must be given in a list and must be unique"

        nphot = len(lcs["names"])
    else:
        nphot = 0

    
    if nphot>0:
        for k,v in lcs.items():
            if k != 'baseline':
                if isinstance(v, (str, int, float)):
                    v = [v] * nphot
                elif isinstance(v, list):
                    if len(v)==1: # if it is a single value, convert to list of nphot
                        v = [v[0]] * nphot
                    else:
                        assert len(v)==nphot, f'{k} must be a list of length {nphot} but {len(v)} given'
                lcs[k] = v
            elif k == 'baseline' and lcs["baseline"]!='None':
                for bk, bv in v.items():
                    if isinstance(bv, (str, int, float)):
                        bv = [bv] * nphot
                    elif isinstance(bv, list):
                        if len(bv)==1: # if it is a single value, convert to list of nphot
                            bv = [bv[0]] * nphot
                        else:
                            assert len(bv)==nphot, f'{k}:{bk} must be a list of length {nphot} but {len(bv)} given'
                    lcs[k][bk] = bv

        _names      = lcs["names"]                      # list of light curve names
        _filters    = lcs.get("filters", ['V']*nphot)   # list of light curve filters
        _wl         = lcs.get("wavelength_um", None)    # list of light curve central wavelengths
        _filtnames  = list(sorted(set(_filters),key=_filters.index))
        nfilt       = len(set(_filters))                # number of filters

        # instantiate light curve object
        # assert os.path.exists(lc_fpath), f"Light curve file path {lc_fpath} does not exist"
        lc_obj = load_lightcurves(_names, data_filepath=lc_fpath, filters=_filters, wl=_wl, nplanet=nplanet, verbose=verbose)

        # baseline
        if "baseline" in lcs and lcs["baseline"]!='None':
            _offset = lcs["baseline"]["offset"]  # list of offsets for each light curve
            _bases  = {}
            for c in lcs["baseline"]:
                if c.startswith('col'):
                    _bases[f'd{c}'] = lcs["baseline"][c]
            lc_obj.lc_baseline(_offset.copy(), **_bases,  gp=lc_obj._useGPphot,verbose=False )
        else:
            _offset = ["y"] * nphot
            lc_obj.lc_baseline( verbose=False)

        # supersampling
        if 'supersampling' in lcs:
            _ss_lclist  = []  # list of light curve names to supersample
            _ss_fac     = []
            for n,sf in zip(_names, lcs["supersampling"]):
                if sf!='None':
                    _ss_lclist.append(n)
                    _ss_fac.append(sf)
            lc_obj.supersample(lc_list=_ss_lclist, ss_factor=_ss_fac, verbose=False)

        # clip_outlier
        if 'clip_outliers' in lcs:
            _clip_cols, _clip_lclist, _clip, _clip_width, _clip_niter  = [],[],[],[],[]
            for n,co in zip(_names, lcs["clip_outliers"]):
                if co != "None":
                    _clip_lclist.append(n)
                    _clip_niter.append(int(co.split("n")[1])) 
                    clip_v = float(co.split("n")[0].split("C")[1]) 
                    _clip.append(int(clip_v) if clip_v.is_integer() else clip_v)                   # outlier clip value
                    _clip_width.append(int(co.split("n")[0].split("C")[0].split("W")[1])) # windown width
                    col_nos = co.split(":")[0][1:]  # column numbers to clip
                    if col_nos == "a": col_nos = "135678"  # clip all valid columns
                    _clip_cols.append(col_nos)  # append the column numbers

            # check that the elements of _clip_cols are the same
            if len(_clip_cols) > 1:
                assert len(set(_clip_cols)) == 1, f"all columns to clip must be the same for all files but {_clip_cols} given"
            if _clip_cols!=[]: 
                _clip_cols = [f"col{c}" for c in _clip_cols[0]]  #convert to list of col{c} format

            lc_obj.clip_outliers(lc_list=_clip_lclist , clip=_clip, width=_clip_width,select_column=_clip_cols,niter=_clip_niter, show_plot=False,verbose=False )

        # scale columns
        if 'scale_columns' in lcs:
            _sclcol     = lcs["scale_columns"]      # list of config for scaling the  columns
            lc_obj.rescale_data_columns(method=_sclcol,verbose=False)

        # spline
        if 'spline' in lcs:
            _spl_lclist = []  # list of light curves with spline
            _spl_deg    = []  # list of spline degrees
            _spl_par    = []  # list of spline parameters
            _spl_knot   = []  # list of spline knots    
            for n,spl in zip(_names, lcs["spline"]):
                if spl != "None":
                    _spl_lclist.append(n)
                    if "|" not in spl:   # 1D spline
                        k1 = spl.split("k")[-1]
                        _spl_knot.append((int(k1) if float(k1).is_integer() else float(k1)) if k1 != "r" else k1)
                        _spl_deg.append(int(spl.split("k")[0].split("d")[-1]))
                        _spl_par.append("col" + spl.split("d")[0][1])
                    else: # 2D spline
                        sp    = spl.split("|")  # split the diff spline configs
                        k1,k2 = sp[0].split("k")[-1], sp[1].split("k")[-1]
                        k_1   = (int(k1) if float(k1).is_integer() else float(k1)) if k1 != "r" else k1
                        k_2   = (int(k2) if float(k2).is_integer() else float(k2)) if k2 != "r" else k2
                        _spl_knot.append( (k_1,k_2) )
                        _spl_deg.append( (int(sp[0].split("k")[0].split("d")[-1]),int(sp[1].split("k")[0].split("d")[-1])) )
                        _spl_par.append( ("col"+sp[0].split("d")[0][1],"col"+sp[1].split("d")[0][1]) ) 

            lc_obj.add_spline(lc_list=_spl_lclist ,par=_spl_par , degree=_spl_deg,
                                knot_spacing=_spl_knot , verbose=False)

        if 'apply_jitter' not in lcs:
            lcs['apply_jitter'] = 'y'

        # sinusoid
        if 'sinusoid' in _file["photometry"]:
            sin_lclist, trig, sin_n, sin_par, sin_Amp, sin_Per, sin_x0 = [],[],[],[],[],[],[]
            sin = _file["photometry"]["sinusoid"]
            if isinstance(sin["names"], str):  # if it is a string, convert to list
                sin_lclist = sin["names"]
                trig = sin["trig"]
                sin_n = sin["n"]
                sin_par = sin["par"]    
                sin_Amp = sin["amp"]
                sin_Per = sin["P"]
                sin_x0 = sin["x0"]
            elif isinstance(sin["names"], list):  
                for i,nm in enumerate(sin["names"]):
                    sin_lclist.append(nm)
                    trig.append(sin['trig'][i])
                    sin_n.append(sin['n'][i])
                    sin_par.append(sin['par'][i])
                    sin_Amp.append(_prior_value(sin['amp'][i]))
                    sin_Per.append(_prior_value(sin["P"][i]))
                    sin_x0.append(_prior_value(sin["x0"][i]))

            if len(sin_lclist)==1:
                sin_lclist = sin_lclist[0]  if sin_lclist[0] in ["all","same","filt"] else sin_lclist
            elif len(sin_lclist) > 1:
                if set(sin_lclist) == set(_names):  #if all light curves have a sin term
                    sin_lclist = "all" if len(sin_lclist) > 1 else sin_lclist[0]

            lc_obj.add_sinusoid(lc_list=sin_lclist, trig=trig, n=sin_n, par=sin_par, Amp = sin_Amp, P=sin_Per, x0=sin_x0, verbose=False)

        # ========== GP input ====================
        if 'gp' in _file["photometry"]:
            gp_lclist,op, _useGPphot = [],  [], ['n']*nphot
            gp_pars, kernels, amplitude, lengthscale, h3, h4 = [],[],[],[],[],[]
            gp = _file["photometry"]["gp"]

            for thisgp in gp:
                if thisgp['lc_name']=='None':
                    pass
                else:
                    gp_lclist.append(thisgp["lc_name"])

                ngp = len(thisgp["kernel"]) if isinstance(thisgp["kernel"], list) else 1
                assert isinstance(thisgp["gp_pck"],str) and thisgp["gp_pck"] in ["ge","ce","sp", "n"], f"gp: gp_pck must be one of ['ge','ce','sp', 'n'] but {thisgp['gp_pck']} given"
                if thisgp["lc_name"] in ["all","same"]:
                    _useGPphot = [thisgp["gp_pck"]]*nphot
                elif thisgp["lc_name"] in _names:
                    _useGPphot[_names.index(thisgp["lc_name"])] = thisgp["gp_pck"]
                elif thisgp["lc_name"] in _filtnames:
                    for j in np.where(np.array(_filters)==thisgp["lc_name"])[0]:
                        _useGPphot[j] = thisgp["gp_pck"]
                elif thisgp["lc_name"]=='None':
                    pass
                else:
                    raise ValueError(f"gp: lc_name {thisgp['lc_name']} not found in light curve names {_names} or filters {_filtnames} and not 'all' or 'same' ")

                if ngp ==1:  # if there is only one kernel
                    for p in ['kernel','par','h1_amp', 'h2_len_scale', 'h3_other', 'h4_period']:
                        if isinstance(thisgp[p], list):
                            assert len(thisgp[p]) == 1, f"if there is only one kernel, {p} must be a single value but {thisgp[p]} given"
                            thisgp[p] = thisgp[p][0]  # convert to single value
                        elif thisgp[p] == 'None':  # if the prior is None, set it to None
                            pass

                    gp_pars.append(thisgp["par"])
                    kernels.append(thisgp["kernel"])
                    amplitude.append(_prior_value(thisgp["h1_amp"]))
                    lengthscale.append(_prior_value(thisgp["h2_len_scale"]))
                    h3.append(_prior_value(thisgp["h3_other"]))
                    h4.append(_prior_value(thisgp["h4_period"]))
                    op.append("")

                else:  # if there are multiple kernels
                    for p in ['h1_amp', 'h2_len_scale', 'h3_other', 'h4_period']:
                        if not isinstance(thisgp[p], list):  # if the prior is None, set it to None
                            thisgp[p] = [thisgp[p]] * ngp

                    gp_pars.append(tuple(thisgp["par"]))
                    kernels.append(tuple(thisgp["kernel"]))
                    amplitude.append( tuple([_prior_value(p) for p in thisgp["h1_amp"]] ))
                    lengthscale.append(tuple([_prior_value(p) for p in thisgp["h2_len_scale"]] ))
                    h3.append(tuple([_prior_value(p) for p in thisgp["h3_other"]] ))
                    h4.append(tuple([_prior_value(p) for p in thisgp["h4_period"]] ))
                    op.append(tuple([p for p in thisgp["operation"]]))

            if gp_lclist !=[]: 
                gp_lclist = gp_lclist[0] if gp_lclist[0] in ['same','all'] else gp_lclist
            gp_pck = _useGPphot #[_useGPphot[lc_obj._names.index(lc)] for lc in lc_obj._gp_lcs()] if _useGPphot!=[] else []
            lc_obj.add_GP(lc_list=gp_lclist,par=gp_pars,kernel=kernels,operation=op,
                            amplitude=amplitude,lengthscale=lengthscale,h3=h3,h4=h4,gp_pck=gp_pck,
                            verbose=verbose)
        
        
        lc_obj._fit_offset = _offset
        if verbose:
            lc_obj.print("lc_baseline")

        ## limb darkening
        if 'limb_darkening' in _file["photometry"]:
            ldc = _file["photometry"]["limb_darkening"]
            assert len(ldc["filters"]) == len(lc_obj._filnames), f"number of filters in limb darkening ({len(ldc['filters'])}) must be equal to number of unique light curve filters ({len(lc_obj._filnames)})"

            for i,f in enumerate(ldc["filters"]):
                assert f in lc_obj._filnames, f"LD filter {f} not found in light curve filters {lc_obj._filnames}"

            f_indc = [list(lc_obj._filnames).index(f) for f in ldc["filters"]]
            q1 = [_prior_value(ldc["q1"][i]) for i in f_indc]
            q2 = [_prior_value(ldc["q2"][i]) for i in f_indc]
            
            lc_obj.limb_darkening(q1,q2,verbose=verbose)

        # phase curve
        if 'phase_curve' in _file["photometry"]:
            pc = _file["photometry"]["phase_curve"]
            for k,v in pc.items():
                if isinstance(v, (str, int, float)):
                    v = [v] * nfilt
                elif isinstance(v, list):
                    if len(v)==1: # if it is a single value, convert to list of nphot
                        v = [v[0]] * nfilt
                    else:
                        assert len(v)==nfilt, f'phase curve: {k} must be a equal to the number of unique filters {nfilt} but {len(v)} given'
                pc[k] = v

            for i,f in enumerate(pc["filters"]):
                assert f in lc_obj._filnames, f"phase curve filter {f} not found in light curve filters {lc_obj._filnames}"

            f_indc = [list(lc_obj._filnames).index(f) for f in pc["filters"]]

            D_occ   = [_prior_value(pc["D_occ"][i])  for i in f_indc]
            Fn      = [_prior_value(pc["Fn"][i])     for i in f_indc]
            ph_off  = [_prior_value(pc["ph_off"][i]) for i in f_indc]
            A_ev    = [_prior_value(pc["A_ev"][i])   for i in f_indc]
            f1_ev   = [_prior_value(pc["f1_ev"][i])  for i in f_indc]
            A_db    = [_prior_value(pc["A_db"][i])   for i in f_indc]
            pc_model= [pc["pc_model"][i]             for i in f_indc]

            lc_obj.phasecurve(  D_occ=D_occ, Fn=Fn, ph_off=ph_off, A_ev=A_ev, f1_ev=f1_ev, 
                                A_db=A_db, pc_model=pc_model,verbose=verbose)

            #to be passed to get_decorr. but only allows one value for now
            #TODO allow prior per filter
            pc_pars = dict(     D_occ   = D_occ[0]    if len(D_occ)>0    else 0, 
                                Fn      = Fn[0]       if len(Fn)>0       else 0, 
                                ph_off  = ph_off[0]   if len(ph_off)>0   else 0, 
                                A_ev    = A_ev[0]     if len(A_ev)>0     else 0,
                                f1_ev   = f1_ev[0]    if len(f1_ev)>0    else 0, 
                                A_db    = A_db[0]     if len(A_db)>0     else 0,
                                pc_model= pc_model[0] if len(pc_model)>0 else 0)
        else:
            pc_pars = {}


        # custom LC function
        if "custom_LC_function" in _file["photometry"]:
            cf = _file["photometry"]["custom_LC_function"]
            func_name = cf["function"]
            if func_name!='None':
                # directly import function named func_name from custom_LCfunc.py
                import custom_LCfunc
                custom_lcfunc = getattr(custom_LCfunc, func_name)

                func_x    = cf["x"]
                func_args = {}
                if cf["func_pars"]!='None':
                    for p_name, p_prior in cf["func_pars"].items():
                        func_args[p_name] = _prior_value(p_prior)

                extra_args  = cf["extra_args"] if cf["extra_args"]!='None' else {}
                opfunc_name = cf["op_func"]
                if opfunc_name!='None':
                    op_func = getattr(custom_LCfunc, opfunc_name)
                else:
                    op_func = None

                replace_LCmodel = cf["replace_LCmodel"]
            else:   
                custom_lcfunc,func_x,func_args,extra_args,op_func,replace_LCmodel = None,None,{},{},None,False

            lc_obj.add_custom_LC_function(  func=custom_lcfunc,x=func_x,func_args=func_args,
                                            extra_args=extra_args,op_func=op_func,replace_LCmodel=replace_LCmodel,
                                            verbose=verbose)

        # contamination factors
        cont_fac = 0
        if 'contamination' in _file["photometry"]:
            cont_fac = []
            contam = _file["photometry"]["contamination"]
            for k,v in contam.items():
                if isinstance(v, (str, int, float)):
                    v = [v] * nfilt
                elif isinstance(v, list):
                    if len(v)==1: # if it is a single value, convert to list of nfilt
                        v = [v[0]] * nfilt
                    else:
                        assert len(v)==nfilt, f'contamination: {k} must be a equal to the number of unique filters {nfilt} but {len(v)} given'
                contam[k] = v

            for i,f in enumerate(contam["filters"]):
                assert f in lc_obj._filnames, f"contamination filter {f} not found in LC filters {lc_obj._filnames}"

            f_indc = [list(lc_obj._filnames).index(f) for f in contam["filters"]]

            cont_fac = [_prior_value(contam["contam_factor"][i]) for i in f_indc]

            lc_obj.contamination_factors(cont_ratio=cont_fac, verbose=verbose)

    else:
        lc_obj = load_lightcurves(nplanet=nplanet, verbose=False)

    
    
    ## RV ==========================================================
    if "radial_velocity" in _file:

        # ========== RV input ====================
        rvs      = _file["radial_velocity"]["rv_curves"]
        rv_fpath = rvs.get("filepath",None)                        # the path where the files are
        rv_fpath = rv_fpath if rv_path is None else rv_path
        RVunit   = rvs.get("rv_unit","m/s")  #default is m/s


        if isinstance(rvs["names"],str):
            rvs["names"] = [rvs["names"]]
        elif isinstance(rvs["names"], list):
            assert len(set(rvs["names"])) == len(rvs["names"]), "RV names must be given in a list and must be unique"

        nRV = len(rvs["names"])
    else:
        nRV = 0


    if nRV > 0:
        for k,v in rvs.items():
            if k != 'baseline':
                if isinstance(v, (str, int, float)):
                    v = [v] * nRV
                elif isinstance(v, list):
                    if len(v)==1: # if it is a single value, convert to list of nRV
                        v = [v[0]] * nRV
                    else:
                        assert len(v)==nRV, f'{k} must be a list of length {nRV} but {len(v)} given'
                rvs[k] = v
            else:
                for bk, bv in v.items():
                    if isinstance(bv, (str, int, float)):
                        bv = [bv] * nRV
                    elif isinstance(bv, list):
                        if len(bv)==1: # if it is a single value, convert to list of nRV
                            bv = [bv[0]] * nRV
                        else:
                            assert len(bv)==nRV, f'{k}:{bk} must be a list of length {nRV} but {len(bv)} given'
                    rvs[k][bk] = bv

        RVnames   = rvs["names"]                # list of RV names
        gammas    = [_prior_value(gam) for gam in rvs["baseline"]["gammas"]]  # list of RV gammas

        # instantiate the rv object
        # assert os.path.exists(rv_fpath), f"Radial velocity file path {rv_fpath} does not exist"
        rv_obj = load_rvs(RVnames,rv_fpath, nplanet=nplanet,rv_unit=RVunit,lc_obj=lc_obj, verbose=verbose)

        # baseline
        RVbases    = {}
        for c in rvs["baseline"]:
            if c.startswith('col'):
                RVbases[f'd{c}'] = rvs["baseline"][c]
        rv_obj.rv_baseline(**RVbases, gamma=gammas,gp='n',verbose=False) 

        # scale columns
        if 'scale_columns' in rvs:
            _RVsclcol     = rvs["scale_columns"]      # list of config for scaling the RV columns
            rv_obj.rescale_data_columns(method=_RVsclcol,verbose=False)

        # spline
        if 'spline' in rvs:
            _spl_rvlist,_spl_deg,_spl_par, _spl_knot=[],[],[],[]
            for n,spl in zip(RVnames, rvs["spline"]):
                if spl != "None":
                    _spl_rvlist.append(n)
                    if "|" not in spl:   # 1D spline
                        k1 = spl.split("k")[-1]
                        _spl_knot.append((int(k1) if float(k1).is_integer() else float(k1)) if k1 != "r" else k1)
                        _spl_deg.append(int(spl.split("k")[0].split("d")[-1]))
                        _spl_par.append("col" + spl.split("d")[0][1])
                    else: # 2D spline
                        sp    = spl.split("|")  # split the diff spline configs
                        k1,k2 = sp[0].split("k")[-1], sp[1].split("k")[-1]
                        k_1   = (int(k1) if float(k1).is_integer() else float(k1)) if k1 != "r" else k1
                        k_2   = (int(k2) if float(k2).is_integer() else float(k2)) if k2 != "r" else k2
                        _spl_knot.append( (k_1,k_2) )
                        _spl_deg.append( (int(sp[0].split("k")[0].split("d")[-1]),int(sp[1].split("k")[0].split("d")[-1])) )
                        _spl_par.append( ("col"+sp[0].split("d")[0][1],"col"+sp[1].split("d")[0][1]) )      

            rv_obj.add_spline(rv_list=_spl_rvlist ,par=_spl_par, degree=_spl_deg,
                                knot_spacing=_spl_knot, verbose=False)
            
        if 'apply_jitter' not in rvs:
            rvs['apply_jitter'] = 'y'

        # RV GP ====================
        if 'gp' in _file["radial_velocity"]:
            gp_rvlist,op,usegpRV = [],[],['n']*nRV
            gp_pars, kernels, amplitude, lengthscale, h3, h4, h5, err_col = [],[],[],[],[],[],[],[]
            gp = _file["radial_velocity"]["gp"]

            for thisgp in gp:
                if thisgp["rv_name"]=='None':
                    pass
                else:
                    gp_rvlist.append(thisgp["rv_name"])

                ngp = len(thisgp["kernel"]) if isinstance(thisgp["kernel"], list) else 1
                assert isinstance(thisgp["gp_pck"],str) and thisgp["gp_pck"] in ["ge","ce","sp", "n"], f"gp: gp_pck must be one of ['ge','ce','sp', 'n'] but {thisgp['gp_pck']} given"
                if thisgp["rv_name"] in ["all","same"]:
                    usegpRV = [thisgp["gp_pck"]]*nRV
                elif thisgp["rv_name"] in RVnames:
                    usegpRV[RVnames.index(thisgp["rv_name"])] = thisgp["gp_pck"]
                elif thisgp['rv_name']=='None':
                    pass
                else:
                    raise ValueError(f"gp: rv_name {thisgp['rv_name']} not found in RV names {RVnames} and not 'all' or 'same' ")

                if ngp ==1:  # if there is only one kernel
                    for p in ['kernel','par','h1_amp', 'h2_len_scale', 'h3_other', 'h4_period','h5_der_amp', 'error_col']:
                        if isinstance(thisgp[p], list):
                            assert len(thisgp[p]) == 1, f"if there is only one kernel, {p} must be a single value but {thisgp[p]} given"
                            thisgp[p] = thisgp[p][0]  # convert to single value
                        elif thisgp[p] == 'None':  # if the prior is None, set it to None
                            pass

                    gp_pars.append(thisgp["par"])
                    kernels.append(thisgp["kernel"])
                    amplitude.append(_prior_value(thisgp["h1_amp"]))
                    lengthscale.append(_prior_value(thisgp["h2_len_scale"]))
                    h3.append(_prior_value(thisgp["h3_other"]))
                    h4.append(_prior_value(thisgp["h4_period"]))
                    h5.append(_prior_value(thisgp["h5_der_amp"]))
                    err_col.append(thisgp["error_col"])
                    op.append("")

                else:  # if there are multiple kernels
                    for p in ['h1_amp', 'h2_len_scale', 'h3_other', 'h4_period', 'h5_der_amp', 'error_col']:
                        if isinstance(thisgp[p], list):
                            if len(thisgp[p])==1:  # if it is a single value, convert to list of ngp
                                thisgp[p] = [thisgp[p][0]] * ngp
                            else:
                                assert len(thisgp[p])==ngp, f'{p} must be a list of length {ngp} but {len(thisgp[p])} given'
                        elif not isinstance(thisgp[p], list):  # if the prior is None, set it to None
                            thisgp[p] = [thisgp[p]] * ngp   

                    gp_pars.append(tuple(thisgp["par"]))
                    kernels.append(tuple(thisgp["kernel"]))
                    amplitude.append( tuple([_prior_value(p) for p in thisgp["h1_amp"]] ))
                    lengthscale.append(tuple([_prior_value(p) for p in thisgp["h2_len_scale"]] ))
                    h3.append(tuple([_prior_value(p) for p in thisgp["h3_other"]] ))
                    h4.append(tuple([_prior_value(p) for p in thisgp["h4_period"]] ))
                    h5.append(tuple([_prior_value(p) for p in thisgp["h5_der_amp"]] ))
                    err_col.append(tuple([p for p in thisgp["error_col"]]))
                    op.append(tuple([p for p in thisgp["operation"]]))

            if gp_rvlist !=[]: 
                gp_rvlist = gp_rvlist[0] if gp_rvlist[0] in ['same','all'] else gp_rvlist
            gp_pck = usegpRV #[usegpRV[rv_obj._names.index(rv)] for rv in rv_obj._gp_rvs()] if usegpRV!=[] else []
            rv_obj.add_rvGP(rv_list=gp_rvlist,par=gp_pars,kernel=kernels,operation=op,
                            amplitude=amplitude,lengthscale=lengthscale,h3=h3,h4=h4,h5=h5, err_col=err_col,gp_pck=gp_pck,
                            verbose=verbose)

        if verbose: 
            rv_obj.print("rv_baseline")

        # custom RV function
        if "custom_RV_function" in _file["radial_velocity"]:
            cf = _file["radial_velocity"]["custom_RV_function"]
            func_name = cf["function"]
            if func_name!='None':
                # directly import function named func_name from custom_RVfunc.py
                import custom_RVfunc
                custom_rvfunc = getattr(custom_RVfunc, func_name)

                rvfunc_x    = cf["x"]
                rvfunc_args = {}
                if cf["func_pars"]!='None':
                    for p_name, p_prior in cf["func_pars"].items():
                        rvfunc_args[p_name] = _prior_value(p_prior)

                rvextra_args = cf["extra_args"] if cf["extra_args"]!='None' else {}
                opfunc_name  = cf["op_func"]
                if opfunc_name!='None':
                    op_rvfunc = getattr(custom_RVfunc, opfunc_name)
                else: 
                    op_rvfunc = None

                replace_RVmodel = cf["replace_RVmodel"]
            else:   
                custom_rvfunc,rvfunc_x,rvfunc_args,rvextra_args,op_rvfunc,replace_RVmodel = None,None,{},{},None,False

            rv_obj.add_custom_RV_function(func=custom_rvfunc,x=rvfunc_x,func_args=rvfunc_args,extra_args=rvextra_args,op_func=op_rvfunc,replace_RVmodel=replace_RVmodel,verbose=verbose)

    else:
        rv_obj = load_rvs(nplanet=nplanet, lc_obj=lc_obj, verbose=False)

    
    
    ## Planet parameters ===============================
    assert 'planet_parameters' in _file, f'planet_parameters must be defined in .yaml file'
    planet = _file["planet_parameters"]
    pl_pars = {}

    for k,v in planet.items():
        if k in ["rho_star", "Duration"]:
            assert isinstance(v, str), f"planet parameters: {k} must be a string but {type(v)} given"
            if v == 'None':
                pl_pars[k] = None
            else:
                assert v.startswith(('F(','U(','LU(','N(','TN(')), f"planet parameters: {k} must be a prior string starting with one of ['F','U','LU','N','TN'] but {v} given"
                pl_pars[k] = _prior_value(v)
        else:
            if isinstance(v, str):
                if v == 'None':
                    pl_pars[k] = None
                else:
                    assert v.startswith(('F(','U(','LU(','N(','TN(')), f"planet parameters: {k} must be a prior string starting with one of ['F','U','LU','N','TN'] but {v} given"
                    pl_pars[k] = [_prior_value(v)] * nplanet
            elif isinstance(v, list):
                if len(v)==1: # if it is a single value, convert to list of nplanet
                    pl_pars[k] = [_prior_value(v[0])] * nplanet
                else:
                    assert len(v)==nplanet, f'planet parameters: {k} must be a list of length {nplanet} but {len(v)} given'
                    pl_pars[k] = [_prior_value(v[i]) for i in range(nplanet)]

    if pl_pars["rho_star"]==None:
        assert pl_pars["Duration"] != None
    if pl_pars["Duration"]==None:
        assert pl_pars["rho_star"] != None

    lc_obj.planet_parameters(**pl_pars, verbose=verbose)

    # ========== get decorrelation parameters ====================
    if nphot>0:
        # DDFs
        if 'tdv' in _file["photometry"]:
            tdv = _file["photometry"]["tdv"]
            ddfyn,ddf_pri,div_wht  = tdv["fit_ddfs"], _prior_value(tdv["drprs"]), tdv["div_white"]
            lc_obj.transit_depth_variation(ddFs=ddfyn,dRpRs=ddf_pri, divwhite=div_wht,verbose=verbose)

        # TTVS
        if 'ttv' in _file["photometry"]:
            ttv = _file["photometry"]["ttv"]
            ttvs, dt, base = ttv["fit_ttvs"], _prior_value(ttv["dt"]), ttv["baseline"]
            per_LC_T0, incl_partial = ttv["per_LC_T0"], ttv["include_partial"]
            
            lc_obj.transit_timing_variation(ttvs=ttvs, dt=dt, baseline_amount=base,include_partial=incl_partial,
                                            per_LC_T0=per_LC_T0,verbose=verbose,print_linear_eph=False)


        #### auto decorrelation
        if "auto_decorr" in _file["photometry"]:
            auto_decorr  = _file["photometry"]["auto_decorr"]
            use_decorr   = auto_decorr.get("get_decorr", False)
            del_BIC      = auto_decorr.get("delta_bic", -5)
            exclude_cols = auto_decorr.get("exclude_cols", [])
            exclude_pars = auto_decorr.get("exclude_pars", [])
            enforce_pars = auto_decorr.get("enforce_pars", [])
        else:
            use_decorr   = False

        if use_decorr or init_decorr:
            if init_decorr and verbose: print("\ngetting start values for LC decorrelation parameters ...")
            lc_obj.get_decorr(**pl_pars,**pc_pars, q1=q1,q2=q2,
                                plot_model=False,fit_offset = _offset, cont=cont_fac,
                                ttv =  np.where(_file["photometry"].get('ttv','n')=='n', True, False).item(),
                                setup_baseline=use_decorr,exclude_cols=exclude_cols,delta_BIC=del_BIC,
                                enforce_pars=enforce_pars, exclude_pars=exclude_pars,verbose=verbose if use_decorr else False)
            if init_decorr:  #if not use_decorr, compare the  get_decorr pars to the user-defined ones and only use start values for user-defined ones
                rel_cols = [b[:6] for b in lc_obj._bases]
                _ = [b.insert(1,0) for b in rel_cols for _ in range(2)] #insert 0 to replace cols 1 and 2
                for j in range(lc_obj._nphot):
                    for i,v in enumerate(rel_cols[j]):
                        if i in [1,2]: continue
                        if v == 0: lc_obj._bases_init[j][f"A{i}"] = lc_obj._bases_init[j][f"B{i}"] = 0
                        if v >= 1: lc_obj._bases_init[j][f"A{i}"] = lc_obj._bases_init[j][f"A{i}"]
                        if v == 2: lc_obj._bases_init[j][f"B{i}"] = lc_obj._bases_init[j][f"B{i}"]

    if nRV > 0:
        #### auto decorrelation
        if "auto_decorr" in _file["radial_velocity"]:
            RV_auto_decorr = _file["radial_velocity"]["auto_decorr"]
            use_decorrRV   = RV_auto_decorr.get("get_decorr", False)
            rvdel_BIC      = RV_auto_decorr.get("delta_bic", -5)
            exclude_colsRV = RV_auto_decorr.get("exclude_cols", [])
            exclude_parsRV = RV_auto_decorr.get("exclude_pars", [])
            enforce_parsRV = RV_auto_decorr.get("enforce_pars", [])
        else:
            use_decorrRV   = False

        if use_decorrRV or init_decorr:
            if init_decorr and verbose: print("\ngetting start values for RV decorrelation parameters ...\n")
            rv_obj.get_decorr(  T_0=pl_pars["T_0"], Period=pl_pars["Period"], K=pl_pars["K"],
                                Eccentricity=pl_pars["Eccentricity"], omega=pl_pars["omega"],
                                gamma=gammas[0] if len(gammas)>0 else 0, setup_baseline=use_decorrRV,
                                exclude_cols=exclude_colsRV, enforce_pars=enforce_parsRV, exclude_pars=exclude_parsRV,
                                delta_BIC=rvdel_BIC, plot_model=False,verbose=verbose if use_decorrRV else False)
            if init_decorr:  #if not use_decorr, compare the  get_decorr pars to the user-defined ones and only use start values for user-defined ones
                rel_cols = [b[:6] for b in rv_obj._RVbases]
                _ = [b.insert(1,0) for b in rel_cols for _ in range(2)] #insert 0 to replace cols 1 and 2
                for j in range(rv_obj._nRV):
                    for i,v in enumerate(rel_cols[j]):
                        if i in [1,2]: continue
                        if v == 0: rv_obj._RVbases_init[j][f"A{i}"] = rv_obj._RVbases_init[j][f"B{i}"] = 0
                        if v >= 1: rv_obj._RVbases_init[j][f"A{i}"] = rv_obj._RVbases_init[j][f"A{i}"]
                        if v == 2: rv_obj._RVbases_init[j][f"B{i}"] = rv_obj._RVbases_init[j][f"B{i}"]

    
    # stellar parameters =========================
    if 'stellar_parameters' in _file:
        star     = _file["stellar_parameters"]
        st_rad   = _prior_value(star['radius_rsun'])
        st_mass  = _prior_value(star['mass_msun'])
        par_in   = star.get('input_method', 'Rrho')
    else:
        st_rad, st_mass, par_in = None, None, 'Rrho'


    # ========== fit setup ====================
    fit = _file["fit_setup"]
    # fit setup
    nsteps          = fit.get("emcee_number_steps",1000)
    nchains         = fit.get("emcee_number_chains",64)
    ncpus           = fit.get("number_of_processes",1)
    nburn           = fit.get("emcee_burnin_length",500)
    nlive           = fit.get("dynesty_nlive",500)
    force_nl        = fit.get("force_nlive",False)
    dlogz           = fit.get("dynesty_dlogz",0.1)
    sampler         = fit.get("sampler", "emcee")
    mc_move         = fit.get("emcee_move", "stretch")
    dyn_samp        = fit.get("dynesty_nested_sampling", 'static')
    lsq_base        = fit.get("leastsq_for_basepar", 'n')
    lcjitt          = lcs.get("apply_jitter", 'n') if 'photometry' in _file else 'n'
    rvjitt          = rvs.get("apply_jitter", 'n') if 'radial_velocity' in _file else 'n'
    lcjittlim       = fit.get("lc_jitter_loglims",'auto')
    rvjittlim       = fit.get("rv_jitter_lims",'auto')
    lcbaselim       = fit.get("lc_basecoeff_lims",'auto')
    rvbaselim       = fit.get("rv_basecoeff_lims",'auto')
    ltt             = fit.get("light_travel_time_correction",'n')
    LC_GPndim_jitt  = fit.get("apply_lc_gpndim_jitter", 'y')
    RV_GPndim_jitt  = fit.get("apply_lc_gpndim_jitter", 'y')
    LC_GPndim_off   = fit.get("apply_lc_gpndim_offset", 'y')
    RV_GPndim_off   = fit.get("apply_lc_gpndim_offset", 'y')

    fit_obj = fit_setup(R_st = st_rad, M_st = st_mass, par_input=par_in,
                        apply_LCjitter=lcjitt, apply_RVjitter=rvjitt,
                        leastsq_for_basepar=lsq_base, 
                        LCbasecoeff_lims=lcbaselim, RVbasecoeff_lims=rvbaselim,
                        LCjitter_loglims=lcjittlim, RVjitter_lims=rvjittlim, LTT_corr=ltt,
                        apply_LC_GPndim_jitter=LC_GPndim_jitt, apply_RV_GPndim_jitter=RV_GPndim_jitt,
                        apply_LC_GPndim_offset=LC_GPndim_off, apply_RV_GPndim_offset=RV_GPndim_off,
                        verbose=verbose)

    fit_obj.sampling(sampler=sampler,n_cpus=ncpus, emcee_move=mc_move,
                    n_chains=nchains, n_burn   = nburn, n_steps  = nsteps, 
                    n_live=nlive, force_nlive=force_nl, nested_sampling=dyn_samp,
                    dyn_dlogz=dlogz,verbose=verbose )

    if return_fit:
        from .fit_data import run_fit
        result = run_fit(lc_obj, rv_obj, fit_obj) 
        return lc_obj,rv_obj,fit_obj,result

    return lc_obj,rv_obj,fit_obj


def convert_numpy_to_native(obj):
    """Recursively convert NumPy objects to native Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_native(item) for item in obj)
    else:
        return obj


def format_yaml(data, indent_size=2):
    """Create aligned YAML with proper spacing and formatting rules"""
    
    def format_value(value):
        """Format individual values according to specific rules"""
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, str):
            # Check if string starts with prior patterns and ends with )
            if re.match(r'^(U|F|N|TN|LU)\(.*\)$', value):
                return f"'{value}'"
            return value
        elif isinstance(value, list):
            return format_list(value)
        else:
            return str(value)
    
    def format_list(lst):
        """Format list as inline with proper spacing and formatting rules"""
        if not lst:
            return "[]"
        
        formatted_items = []
        for item in lst:
            if item is None:
                formatted_items.append("None")
            elif isinstance(item, bool):
                formatted_items.append("True" if item else "False")
            elif isinstance(item, str):
                # Check if string starts with prior patterns and ends with )
                if re.match(r'^(U|F|N|TN|LU)\(.*\)$', item) or item in ["*","+"]:
                    formatted_items.append(f"'{item}'")
                else:
                    formatted_items.append(item)
            else:
                formatted_items.append(str(item))
        
        return f"[{', '.join(formatted_items)}]"
    
    def format_inline_dict(d):
        """Format dictionary as inline for 'func_pars','extra_args' in custom functions"""
        if not d:
            return "{}"
        
        items = []
        for k, v in d.items():
            formatted_v = format_value(v)
            items.append(f"{k}: {formatted_v}")
        return "{" + ", ".join(items) + "}"

    def format_dict(d, level=0, is_top_level=False):
        """Format dictionary with alignment"""
        lines = []
        if not d:
            return lines
        
        # Calculate max key length at this level for alignment
        max_key_len = max(len(str(k)) for k in d.keys()) if d else 0
        
        dict_items = list(d.items())
        
        for i, (key, value) in enumerate(dict_items):
            indent = ' ' * (level * indent_size)
            key_str = str(key)
            
            if isinstance(value, dict):
                # Check if this key is 'func_pars','extra_args' from "custom" - if so, format as single line
                if key_str.lower() in ['func_pars','extra_args']:
                    aligned_key = f"{key_str:<{max_key_len}}"
                    formatted_dict = format_inline_dict(value)
                    lines.append(f"{indent}{aligned_key}: {formatted_dict}")
                else:
                    # For regular nested dictionaries
                    aligned_key = f"{key_str}:"
                    lines.append(f"{indent}{aligned_key}")
                    lines.extend(format_dict(value, level + 1))
                
                # Add spacing after top-level sections (except the last one)
                if is_top_level and i < len(dict_items) - 1:
                    lines.append("")
                    
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                # For lists containing dictionaries (like gp sections) - keep multiline
                aligned_key = f"{key_str}:"
                lines.append(f"{indent}{aligned_key}")
                
                for j, list_item in enumerate(value):
                    if isinstance(list_item, dict):
                        dict_items_inner = list(list_item.items())
                        if dict_items_inner:
                            # Calculate max key length for this inner dict
                            max_inner_key_len = max(len(str(k)) for k in list_item.keys())
                            
                            # Hyphen is indented one level beyond the parent key
                            hyphen_indent = ' ' * ((level + 1) * indent_size)
                            
                            # First key-value pair goes on the same line as the hyphen
                            first_key, first_value = dict_items_inner[0]
                            aligned_first_key = f"{first_key:<{max_inner_key_len}}"
                            formatted_first_value = format_value(first_value)
                            lines.append(f"{hyphen_indent}- {aligned_first_key}: {formatted_first_value}")
                            
                            # Remaining key-value pairs align with the first key (2 spaces from hyphen)
                            key_alignment_indent = f"{hyphen_indent}  "
                            for inner_key, inner_value in dict_items_inner[1:]:
                                aligned_inner_key = f"{inner_key:<{max_inner_key_len}}"
                                formatted_inner_value = format_value(inner_value)
                                lines.append(f"{key_alignment_indent}{aligned_inner_key}: {formatted_inner_value}")
                
                # Add spacing after top-level sections
                if is_top_level and i < len(dict_items) - 1:
                    lines.append("")
                    
            else:
                # For simple key-value pairs - align the colons
                aligned_key = f"{key_str:<{max_key_len}}"
                formatted_value = format_value(value)
                lines.append(f"{indent}{aligned_key}: {formatted_value}")
        
        return lines
    
    # Format the entire document
    lines = format_dict(data, level=0, is_top_level=True)
    return '\n'.join(lines)
