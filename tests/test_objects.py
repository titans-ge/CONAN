import pytest
from CONAN import load_lightcurves,load_rvs,fit_setup, load_configfile, run_fit, create_configfile
from CONAN.get_files import get_parameters
from CONAN.misc import compare_objs
import os


def test_empty_init(verbose=False):
    try:
        lc_obj  = load_lightcurves(verbose=verbose)
        rv_obj  = load_rvs(verbose=verbose)
        fit_obj = fit_setup(verbose=verbose)
        assert True
    except:
        assert False

def test_WASP127_LC_RV_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP127_LC_RV/wasp127_lcrv_config.dat',
                                                lc_path     = 'Notebooks/WASP-127/data/',
                                                rv_path     = 'Notebooks/WASP-127/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-127/WASP127_LC_RV/wasp127_lcrv_config.yaml',both=True)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP127_LC_RV/wasp127_lcrv_config.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."    

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass

def test_WASP127_LC_RV_decorr(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-127/yaml_decorr/wasp127_euler_decorr_config.yaml', 
                                                lc_path     = 'Notebooks/WASP-127/data/',
                                                rv_path     = 'Notebooks/WASP-127/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-127/yaml_decorr/wasp127_euler_decorr_config.dat',both=False, verify=True)

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass

def test_WASP121_pc_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-121_phasecurve/WASP121_TESS_config.dat',
                                                lc_path     = 'Notebooks/WASP-121_phasecurve/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-121_phasecurve/WASP121_TESS_config.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/WASP-121_phasecurve/WASP121_TESS_config.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."    

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass

def test_2D_george_fit(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-103/WASP103_CHEOPS_2D_GP_george.dat', 
                                                lc_path     = 'Notebooks/WASP-103/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-103/WASP103_CHEOPS_2D_GP_george.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/WASP-103/WASP103_CHEOPS_2D_GP_george.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass

def test_customLC_fit(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-103/WASP103_CHEOPS_custom_func.dat', 
                                                lc_path     = 'Notebooks/WASP-103/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-103/WASP103_CHEOPS_custom_func.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/WASP-103/WASP103_CHEOPS_custom_func.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass

def test_3D_spleaf_fit(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/K2-233_spleaf_3D_GP/k2-233_lcrv_spleaf_multiGP.dat', 
                                                lc_path     = 'Notebooks/K2-233_spleaf_3D_GP/data/',
                                                rv_path     = 'Notebooks/K2-233_spleaf_3D_GP/data/',
                                                verbose     = verbose)
    
    # test .dat and .yaml file creation and loading give same config
    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/K2-233_spleaf_3D_GP/k2-233_lcrv_spleaf_multiGP.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/K2-233_spleaf_3D_GP/k2-233_lcrv_spleaf_multiGP.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."   

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass


def test_WASP127_eulerLC_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP-127_EULER_LC/wasp127_euler_config.dat', 
                                                lc_path     = 'Notebooks/WASP-127/data/',
                                                rv_path     = 'Notebooks/WASP-127/data/',
                                                verbose     = verbose)
    # test .dat and .yaml file creation and loading give same config
    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-127/WASP-127_EULER_LC/wasp127_euler_config.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP-127_EULER_LC/wasp127_euler_config.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass


def test_WASP127_RV_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP127_RV/wasp127_rv_config.dat', 
                                                lc_path     = 'Notebooks/WASP-127/data/',
                                                rv_path     = 'Notebooks/WASP-127/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/WASP-127/WASP127_RV/wasp127_rv_config.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP127_RV/wasp127_rv_config.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."             
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass

def test_TTV_TOI_216_init(init_only=True, verbose=False):
    """TESTS GP and ttv"""
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/TOI-216/TOI216_ttvconfig.dat',
                                                lc_path     = 'Notebooks/TOI-216/data/', 
                                                verbose     = verbose)
    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/TOI-216/TOI216_ttvconfig.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/TOI-216/TOI216_ttvconfig.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass

def test_TOI_469_init(init_only=True, verbose=False):
    """TESTS GP, multiplanet, for lc and rv"""
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/TOI469/TOI469_lc_rvconfig.dat', 
                                                lc_path     = 'Notebooks/TOI469/data/',
                                                rv_path     = 'Notebooks/TOI469/data/',
                                                verbose     = verbose)
    
    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/TOI469/TOI469_lc_rvconfig.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/TOI469/TOI469_lc_rvconfig.yaml')
    
    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass

def test_KELT20_gp_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/KELT-20/gp_config.dat', 
                                                lc_path     = 'Notebooks/KELT-20/data/',
                                                rv_path     = 'Notebooks/KELT-20/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/KELT-20/gp_config.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/KELT-20/gp_config.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass

def test_KELT20_sin_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/KELT-20/sine_config.dat', 
                                                lc_path     = 'Notebooks/KELT-20/data/',
                                                rv_path     = 'Notebooks/KELT-20/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/KELT-20/sine_config.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/KELT-20/sine_config.yaml')
    
    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass

def test_KELT20_spline_init(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/KELT-20/spl_config.dat', 
                                                lc_path     = 'Notebooks/KELT-20/data/',
                                                rv_path     = 'Notebooks/KELT-20/data/',
                                                verbose     = verbose)

    # create_configfile(lc_obj, rv_obj, fit_obj, 'Notebooks/KELT-20/spl_config.yaml',both=False)
    lc_obj2, rv_obj2, fit_obj2 = load_configfile(  configfile  = 'Notebooks/KELT-20/spl_config.yaml')

    assert compare_objs(lc_obj,lc_obj2, ignore=["_fpath"]), "lc_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(rv_obj,rv_obj2, ignore=["_fpath","_lcobj"]), "rv_obj loaded from .dat file does not match that loaded from .yaml file."
    assert compare_objs(fit_obj,fit_obj2, ignore=["_lcobj","_rvobj","_fitobj"]), "fit_obj loaded from .dat file does not match that loaded from .yaml file."

    init_pass = run_fit(lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose);   #rerun result even to use existing chains to remake plots
    os.system("rm -r dummy_result")
    assert init_pass

if __name__ == "__main__":
    test_empty_init(verbose=True)
    test_WASP127_LC_RV_init(init_only=True, verbose=True)
    test_WASP127_eulerLC_init(init_only=True, verbose=True)
    test_WASP127_RV_init(init_only=True, verbose=True)
    test_TTV_TOI_216_init(init_only=True, verbose=True)
    test_TOI_469_init(init_only=True, verbose=True)
    test_2D_george_fit(init_only=True, verbose=True)
    test_3D_spleaf_fit(init_only=True, verbose=True)
    test_KELT20_gp_init(init_only=True, verbose=True)
    test_KELT20_sin_init(init_only=True, verbose=True)
    test_KELT20_spline_init(init_only=True, verbose=True)
    print("All tests passed successfully!")

