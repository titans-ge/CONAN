__all__ = ["load_lightcurves", "load_rvs", "fit_setup", "run_fit", "create_configfile","load_configfile","fit_configfile"]

from ._classes import (load_lightcurves, load_rvs, fit_setup, load_result, 
                       get_parameter_names, compare_results, __default_backend__)
from .conf import create_configfile, load_configfile,fit_configfile
from .fit_data import run_fit
from .VERSION import __version__

# from os import path
# here = path.abspath(path.dirname(__file__))
# with open(path.join(here, 'VERSION.dat')) as version_file:
# 	__version__ = version_file.read().strip()

