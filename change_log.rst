22-07-2022: version 3.0.3
~~~~~~~~~~~
* Modified automatic steps assignment in setup_transit_rvs() function
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
