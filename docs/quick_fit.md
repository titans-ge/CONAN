(doc:quickfit)=

# Quickstart
`CONAN` can be run interactively in python shell/Jupyter notebook (see Tutorials section), but can also be quickly launched from [configuration files](doc:configfiles) (.dat or .yaml).


## Lightcurve fit
Let's fit transit light curves of WASP-127b taken with the **EulerCam** at the Swiss telescope in La Silla.

To do this, we need **input files** for each light curve. These are saved as `lc4bjd.dat`, `lc6bjd.dat`, and `lc7bjd.dat` which are text files with up to 9 columns representing the measurements at each timestamp.  

```
    time         flux       error       xshift  yshift  airmass   fwhm     sky      ...
    7833.540364  1.000585   0.000941    -0.04   0.21    1.43      14.51    38.45    ...
    7833.540988  1.000088   0.000941    -1.43   0.26    1.43      14.44    38.21    ...
    7833.541569  1.000092   0.000997    -1.32  -0.04    1.42      13.84    33.30    ...
    7833.542130  1.000181   0.000997    -1.53   0.04    1.42      13.94    33.28    ...
```

`CONAN` does not take note of the headers of these columns but refers to them only as column numbers [col0, col1,...,col8]. However the first 3 columns need to be 
- `col0` - time in BJD (or modified)
- `col1` - normalized flux
- `col2` - flux uncertainty

Next, we need a **config file** that instructs `CONAN` on how to ingest the input files and the parameters and priors of the fit. The new configuration file in `CONAN` is in [.yaml](https://yaml.org/) format allowing users to configuration to only the relevant parameters for a  specific fit.

To fit the 3 light curves from **EulerCam**. We use the minimal configuration below. 

```{code-block} yaml
:lineno-start: 1

# ========================================== CONAN YAML Configuration ==========================================
# This is a YAML configuration file for CONAN v3.3.12
# PRIORS: Fix-'F(val)', Norm-'N(mu,std)', Uni-'U(min,start,max)', TruncNorm–'TN(min,max,mu,std)', LogUni-'LU(min,start,max)'

general:
    n_planet   : 1                                          # number of planets in system

photometry:
  light_curves:
    filepath     : Notebooks/WASP-127/data/                 # Light curve data directory
    names        : [lc4bjd.dat, lc6bjd.dat, lc7bjd.dat]     # List of LC filenames (must be in lc_filepath directory)
    filters      : R                                        # List of Filter names for each LC (for wavelength-dependent analysis)
    wavelength_um: 0.6                                      # Central wavelengths in microns for each filter                             
    clip_outliers: c1:W11C4n1                               # Outlier clipping for each LC (format: 'c[columns]:W[width]C[sigma]n[iterations]',)
    scale_columns: med_sub                                  # rescale data (med_sub=subtract median, rs0to1=rescale 0-1, rs-1to1=rescale -1 to 1, None=no rescaling) 
    baseline     : None                                     # Baseline model configuration - polynomial orders for each data column of each LC

  limb_darkening:                                           # Limb darkening coefficients (using Kipping parameterization)
    filters: [R]                                            # list of unique filter names
    q1     : ['N(0.4223,0.0240)']                           # list of linear LD coefficients prior for each filter
    q2     : ['N(0.3961,0.0157)']                           # list of quadratic LD coefficients prior for each LC
  
  auto_decorr:                                  # Automatic decorrelation settings - finds best baseline model using statistical comparison (BIC)
    get_decorr  : True                          # Set to True to automatically find best decorrelation parameters
    delta_bic   : -5                            # BIC improvement threshold for parameter selection (more negative = more conservative)

planet_parameters:
  rho_star    : 'N(0.565,0.035)'                # Stellar density prior (g/cm³) - alternative to duration parameterization
  Duration    : None                            # Transit duration (alternative to rho_star) - set to None if using rho_star
  RpRs        : ['U(0.05,0.108,0.17)']          # prior Planet-to-star radius ratio. make into list for multiplanet
  Impact_para : ['U(0,0.29,1)']                 # pripr on impact parameter, list for multiplanet
  T_0         : ['N(6776.621239999775,0.001)']  # Mid-transit time (BJD)
  Period      : ['F(4.17806203)']               # Orbital period (days)
  Eccentricity: ['F(0)']                        # prior on eccentricity. list for multiplanet
  omega       : ['F(90)']                       # prior on argument of periastron (degrees: 0-360)    
  K           : ['F(0)']                        # prior on RV semi amplitude. in same unit as rv data

stellar_parameters:                             # optional. Only used post-fit to convert radius and mass to actual units
  radius_rsun : 'N(1.33, 0.03)'                 # stellar radius in Rsun 
  mass_msun   : 'N(0.95, 0.02)'                 # stellar mass in Msun

fit_setup:
  sampler                     : dynesty         # Sampling algorithm: 'dynesty' (nested sampling) or 'emcee' (MCMC)
  number_of_processes         : 10              # Number of CPU cores for parallel processing
  dynesty_nlive               : 300             # Number of live points
# ============ END OF FILE ============================================
```



Note that several possible setups are omitted in this **.yaml** config since they are not needed. A [full yaml config](doc:yamlconf) file can be downloaded [here](https://github.com/titans-ge/CONAN/blob/main/sample_config.yaml). An [equivalent .dat config](docs:datconf) file is also available [here](https://github.com/titans-ge/CONAN/blob/main/sample_config.dat).

See from the congiration file that we do not specify any setup to model the baseline from cotrending column vector. Instead, we turned on the `get_decorr` functionality that finds the best combination of column vectors and the polynomial order to detrend the data.

### Running the fit
Fitting from a config file can be launched within `python` or from the `command line`

- Within `python`
    ```
    from CONAN import fit_configfile
    result = fit_configfile("input_config.yaml", out_folder="output")
    ```
- from `command line`: 
    ```
    conanfit input_config.yam output_folder 
    ```

    to see commandline help use:
    ``` 
    conanfit -h  
    ```

## Adding Radial velocity data

Including radial velocity data in the fit requies adding a `radial_velocity` section, specifing the input files (with up to 6 columns). The first 3 columns are:
- `col0` - time in BJD (or modified)
- `col1` - RV measurements 
- `col2` - RV uncertainty

while the other columns can contain e.g., `fwhm`,`BIS`,`S_HK`.

We also need to include a prior for the RV semi-amplitude, `K`. For the baseline here, we chose to specify directly the polynomial order to use for each cotrending column vector which in this case are the 

```{code-block} yaml
:lineno-start: 1

# ========================================== CONAN YAML Configuration ==========================================
# This is a YAML configuration file for CONAN v3.3.12
# PRIORS: Fix-'F(val)', Norm-'N(mu,std)', Uni-'U(min,start,max)', TruncNorm–'TN(min,max,mu,std)', LogUni-'LU(min,start,max)'

general:
    n_planet   : 1                                          # number of planets in system

photometry:
  light_curves:
    filepath     : Notebooks/WASP-127/data/                 # Light curve data directory
    names        : [lc4bjd.dat, lc6bjd.dat, lc7bjd.dat]     # List of LC filenames (must be in lc_filepath directory)
    filters      : R                                        # List of Filter names for each LC (for wavelength-dependent analysis)
    wavelength_um: 0.6                                      # Central wavelengths in microns for each filter                             
    clip_outliers: c1:W11C4n1                               # Outlier clipping for each LC (format: 'c[columns]:W[width]C[sigma]n[iterations]',)
    scale_columns: med_sub                                  # rescale data (med_sub=subtract median, rs0to1=rescale 0-1, rs-1to1=rescale -1 to 1, None=no rescaling) 
    baseline     : None                                     # Baseline model configuration - polynomial orders for each data column of each LC

  limb_darkening:                                           # Limb darkening coefficients (using Kipping parameterization)
    filters: [R]                                            # list of unique filter names
    q1     : ['N(0.4223,0.0240)']                           # list of linear LD coefficients prior for each filter
    q2     : ['N(0.3961,0.0157)']                           # list of quadratic LD coefficients prior for each LC
  
  auto_decorr:                                  # Automatic decorrelation settings - finds best baseline model using statistical comparison (BIC)
    get_decorr  : True                          # Set to True to automatically find best decorrelation parameters
    delta_bic   : -5                            # BIC improvement threshold for parameter selection (more negative = more conservative)

# ==== RADIAL VELOCITY CONFIGURATION ====
radial_velocity:   
    rv_curves:
        filepath     : Notebooks/WASP-127/data/ # directory of rv files
        rv_unit      : km/s                     # unit for radial velocity data: m/s or km/s 
        names        : [rv1.dat, rv2.dat]
        scale_columns: [med_sub, med_sub]       # rescale data (med_sub=subtract median, rs0to1=rescale 0-1, rs-1to1=rescale -1 to 1, None=no rescaling)
        spline       : [None, None]             # c{column_no}:d{degree}K{knot_spacing} e.g. c0:d3K2 or c0:d3K2|c4:d3K2 for two dimensional
        apply_jitter : [y, y]                   # Whether to fit jitter term for each rv ('y'=yes, 'n'=no)
        baseline:
            gammas: ['N(-9.22924,0.1)', 'N(-9.2105,0.1)']  # systemic velocity priors 
            col0  : [2, 0]                      # Time trend (0=none, 1=linear, 2=quadratic)
            col3  : [1, 0]                      # Column 3 decorrelation order
            col4  : [1, 0]                      # Column 4 decorrelation order
            col5  : [0, 0]                      # Column 5 decorrelation order

planet_parameters:
  rho_star    : 'N(0.565,0.035)'                # Stellar density prior (g/cm³) - alternative to duration parameterization
  Duration    : None                            # Transit duration (alternative to rho_star) - set to None if using rho_star
  RpRs        : ['U(0.05,0.108,0.17)']          # prior Planet-to-star radius ratio. make into list for multiplanet
  Impact_para : ['U(0,0.29,1)']                 # pripr on impact parameter, list for multiplanet
  T_0         : ['N(6776.621239999775,0.001)']  # Mid-transit time (BJD)
  Period      : ['F(4.17806203)']               # Orbital period (days)
  Eccentricity: ['F(0)']                        # prior on eccentricity. list for multiplanet
  omega       : ['F(90)']                       # prior on argument of periastron (degrees: 0-360)    
  K           : ['U(0,0.01,0.05)']              #prior on RV semi amplitude. in same unit as rv data

stellar_parameters:                             # optional. Only used post-fit to convert radius and mass to actual units
  radius_rsun : 'N(1.33, 0.03)'                 # stellar radius in Rsun 
  mass_msun   : 'N(0.95, 0.02)'                 # stellar mass in Msun

fit_setup:
  sampler                     : dynesty         # Sampling algorithm: 'dynesty' (nested sampling) or 'emcee' (MCMC)
  number_of_processes         : 10              # Number of CPU cores for parallel processing
  dynesty_nlive               : 300             # Number of live points
# ============ END OF FILE ============================================
```

