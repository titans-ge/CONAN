import pytest
from CONAN3 import load_lightcurves,load_rvs,fit_setup, load_configfile, run_fit
from CONAN3.get_files import get_parameters
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
    try:
        lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP127_LC_RV/wasp127_lcrv_config.dat', 
                                                    verbose     = verbose)

        result = run_fit(   lc_obj      = lc_obj,
                            rv_obj      = rv_obj,
                            fit_obj     = fit_obj,
                            out_folder  = "result_wasp127_lcrv_fit",
                            init_only   = init_only,
                            rerun_result=True,
                            verbose     = verbose); 
        assert True
        os.system("rm -r result_wasp127_lcrv_fit")  #remove created output folder
    except:
        assert False

def test_WASP127_eulerLC_init(init_only=True, verbose=False):
    try:
        lc_obj, rv_obj, fit_obj = load_configfile(  configfile  = 'Notebooks/WASP-127/WASP-127_EULER_LC/wasp127_euler_config.dat', 
                                                    verbose     = verbose)
        result = run_fit(   lc_obj      = lc_obj,
                            rv_obj      = rv_obj,
                            fit_obj     = fit_obj,
                            out_folder  = "result_wasp127_euler_fit",
                            init_only   = init_only,
                            rerun_result=True,
                            verbose     = verbose);   #rerun result even to use existing chains to remake plots
        assert True
        os.system("rm -r result_wasp127_euler_fit")
    except:
        assert False


if __name__ == "__main__":
    test_empty_init(verbose=True)
    test_WASP127_LC_RV_init(init_only=True, verbose=True)
    test_WASP127_eulerLC_init(init_only=True, verbose=True)
