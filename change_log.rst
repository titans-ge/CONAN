12-Feb-2024: version 3.1.3
~~~~~~~~~~~~
* added estimate of rms and jitter for lc_obj and rv_obj
* "auto" option for the limits of the lc and rv paramteric baseline parameters. 

8-Feb-2024: version 3.1.2
~~~~~~~~~~~~
* fit configfile from command line: >>> conanfit config_file.dat
* added notebook for occultation fitting (KELT20b)



2-Feb-2024: version 3.1.0
~~~~~~~~~~~~
* added function get_decorr() now available for lc and rv objects
* get_decorr() can now use spline. useful when analysing CHEOPS data or to remove other long-term trend in the data
* added function clip_outliers() and rescale_column_data() to lc object
* download data directly from TESS archive using function get_tess_data()
* changed transit_rv_pars() function of the lc object to planet_parameters()
* get system parameters from NASA exoplanet archive
* limb darkening model now uses kipping parameterization.
* get LD priors from phoenix
* automatically fit .FITS and .dat files from EULER pipeline
* supersampling of long cadence data 
* improved results object that allows to re-evaluate the model at user-defined times and make plots
* phase curve fitting
* sampling also with dynesty, now the default sampler
* reimplementation of GP, for lc and rv, including more kernels and combinations of kernels
* support for multiplanet system in lc and rv. no defined limit on the number of planets
* see example implementations in the Notebooks folder

20-11-2023: version 3.0.5(dev)
~~~~~~~~~~~~
* added functions clip_outliers() and split_transits to the lc object.
* get_decorr() now uses same planet parameter names ["Period", "Duration","Impact_para","RpRs",...]
* get_decorr() also allows to exclude some columns from the decorrelation analysis while enforcing others
* load_chain() has been renamed as load_result() which returns an object that contains all the results from the MCMC run.
* improved results object allows to reevaluate the bestfit model at user-given times
* use can now specify output directory for the results using the "out_folder" argument of CONAN.fit_data().
* columns of the input data are now described by col0, col3, col4, etc. instead of the legacy xshift, yshift which dont necessarily describe the input columns
* added shoTerm gp for celerite
* added function get_decorr() to the rv object to find best baseline model for each rv data.
* added 2D spline fit for decorrelation of lc and rv data
* support for multiplanet system in lc and rv

28-12-2022: version 3.0.5
~~~~~~~~~~~~
* fixed bug in fit_data.py when dt is set for RV
* from Andreas: removed function call to grweights in fit_data.py, caused errors for TDVs
* from Andreas: added some attributes (TO,P,dur) to the result object

25-07-2022: version 3.0.4
~~~~~~~~~~~~
* added phases to the output files
* fixed problem with RV jitter and gamma indexes when jit_apply = "n"
* fixed RV filepath
* added spline for roll-angle decorrelation (added roll and spline_fit columns to output file)
* can obtain priors for limb darkening using ldtk
* smooth sampled transit model in decorr plot
* allow setting up lc object without any lc file.
* create bin_data function in plots.py

22-07-2022: version 3.0.3
~~~~~~~~~~~
* Modified automatic steps assignment in planet_parameterss() function
* allow fitting single LDC while keeping the other fixed
* correction in celerite fitting
* corrected setup_occultation() which erroneously showed fit="y" when not fitting a lc for occultation
* specify delta_BIC threshold for selecting parameters in get_decorr() function
* reduced bounds on the offset parameter -> [0.9, 1.2]
* white noise in celerite uses bounds: [-5,-12]
* burn-in chains now saved as .png before running the production chain
* increased max number of parameters for making cornerplot from 12 to 14
* increase maximum of impact parameter to 1.5
* pointing input errors back to the concerned functions/methods
* fixed issue with RV jitter not jumping
* corrected error for celerite when WN="n"

10-07-2022: version 3.0.2
~~~~~~~~~~~~
 * changed format of quadratic limb darekening: now allows either gaussian or uniform priors. Gaussian still recommended
 * included 1D GP fitting using Celerite (~5X faster than with George)
    to use celerite gp for a lightcurve, use "ce" instead of "y" in function lc_baseline().
 * added function get_decorr()  for light curves object to find best baseline model for each lc.
 * added function plot_burnin_chains() to the results object to see how the chains evolved during burn-in.
 * added function load_result_array() to load result array for customized plotting.
 * some  notebooks in example folder
