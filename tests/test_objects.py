import pytest
from CONAN import load_lightcurves,load_rvs,fit_setup, load_configfile, run_fit
from CONAN.get_files import get_parameters
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
                                                verbose     = verbose)

    init_pass = run_fit(   lc_obj      = lc_obj,
                        rv_obj      = rv_obj,
                        fit_obj     = fit_obj,
                        out_folder  = "dummy_result",
                        init_only   = init_only,
                        rerun_result=True,
                        verbose     = verbose); 
    os.system("rm -r dummy_result")  #remove created output folder
    assert init_pass

def test_2D_george_fit(init_only=True, verbose=False):
    lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/george_2D_GP/WASP103_CHEOPS_2D_GP_george.dat', 
                                                lc_path     = 'Notebooks/george_2D_GP/data/',
                                                verbose     = verbose)

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

