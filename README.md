[![Python package](https://github.com/titans-ge/CONAN/actions/workflows/python-package.yml/badge.svg)](https://github.com/titans-ge/CONAN/actions/workflows/python-package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/conan-exoplanet)
[![Upload Python Package](https://github.com/titans-ge/CONAN/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/titans-ge/CONAN/actions/workflows/python-publish-pypi.yml)
![PyPI - Version](https://img.shields.io/pypi/v/conan-exoplanet)


# CONAN
_**CO**de for exopla**N**et **AN**alysis: A flexible bayesian framework for modeling heterogeneous exoplanet data_


---

`CONAN` is an open-source Python package designed for comprehensive analysis of exoplanetary systems. 
It provides a unified framework for simultaneous modeling of diverse observational data including
photometric transit light curves, occultations, phase curves, and radial velocity measurements. 
It is designed to be flexible, easy to use, and fast. 

It is developed and maintained at the 
Observatory of Geneva, Switzerland under the MIT license.

Key features:
-------------
- **Multi-dataset analysis**: Seamless analysis of combined lightcurve (LC) and radial velocity (RV) datasets from various instruments.
- **Multiplanet support:** Simultaneous fit to multiple planets in a single system.
- **Comprehensive photometric modeling**: Robust modeling of transits, occultations, and phase curves, including effects such as ellipsoidal variations and Doppler beaming ([see Model definition](https://github.com/titans-ge/CONAN/wiki/LC-and-RV-models)).
- **Support for modeling light curve variations**: Analysis of transit timing variations (TTVs) and  transit depth variations (transmission spectroscopy).
- **Flexible baseline and noise modeling**: Selection of one or combination of Polynomial, sinusoidal, Gaussian Processes (GP), and spline functions for data detrending.
- **Extensible and customizable modeling**: Users can easily incorporate new LC and RV models or modify default ones to suit specific needs, e.g., modeling the transit of non-spherical planets, Rossiter–McLaughlin signals, or even non-planetary signals.
- **Robust Bayesian inference**: Parameter estimation via MCMC (`emcee`) or nested sampling (`dynesty`)
- **Derivation of priors limb darkening coefficients**: Incorporation of `ldtk` to derive priors for the quadratic limb darkening coefficients from the stellar parameters.
- **Automated selection of parametric model parameters**: Uses the Bayesian Information Criterion to suggest best combination of vectors to use in decorrelating the data.
- **Science data download**: Built-in support for downloading data from various instruments (including TESS, CHEOPS, and Kepler) and also system parameters from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
- **Quick result visualization and manipulation**: Instant plot of the best-fit model and a result object that can be easily manipulated for customized analysis.

The full documentation can be accessed at [https://conan-exoplanet.readthedocs.io](https://conan-exoplanet.readthedocs.io/en/latest/)


## Installation
To avoid requirement conflicts with other packages, it is better to create a new environment (or clone a current environment) to install CONAN


To create a new environment:
```bash
conda create -n conan_env python=3.10
```
then
```
conda activate conan_env
```


### CONAN can be installed using different methods: 

#### 1) Installing from PyPI:
```
pip install conan-exoplanet
```

#### 2) Downloading the source files from github: 
```
git clone https://github.com/titans-ge/CONAN.git
cd CONAN 
```

then running
```
pip install .
```

#### 3) Installing directly from github using pip
```
pip install git+https://github.com/titans-ge/CONAN.git
```
Note that a folder 'src' is created where the CONAN source files are downloaded to before installation.



---

If having troubles compiling the fortran code used for the transit model, set `NO_FORTRAN=True` in terminal before pip installing. This uses a python implementation of the fortran code (which is ~30X slower)

```
export NO_FORTRAN=True
pip install git+https://github.com/titans-ge/CONAN.git
```

-------------------------
## Recent changes
See [change_log.rst](https://github.com/titans-ge/CONAN/blob/main/change_log.rst)


## Run fit from config file 
Fit can be launched from a config file within `python` or from the `command line`

- Within `python`
    ```
    from CONAN import fit_configfile
    result = fit_configfile("input_config.dat", out_folder="output")
    ```
- from `command line`: 
    ```
    conanfit path/to/config_file output_folder 
    ```

    to see commandline help use:
    ``` 
    conanfit -h  
    ```

## Attribution

If you find `CONAN` useful in your research, please reference the GitHub
repository. The first implementations of CONAN have been descibed in a few papers, kindly cite them using the following BibTeX entries:
```
@ARTICLE{2017A&A...606A..18L,
        author = {{Lendl}, M. and {Cubillos}, P.~E. and 
                    {Hagelberg}, J. and 
                    {M{\"u}ller}, A. and 
                    {Juvan}, I. and 
                    {Fossati}, L.},
            title = "{Signs of strong Na and K absorption in the transmission spectrum of WASP-103b}",
        journal = {\aap},
            year = 2017,
            month = sep,
        volume = {606},
            eid = {A18},
            pages = {A18},
            doi = {10.1051/0004-6361/201731242},
    archivePrefix = {arXiv},
        eprint = {1708.05737},
    primaryClass = {astro-ph.EP},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2017A&A...606A..18L},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


@ARTICLE{2020MNRAS.492.1761L,
       author = {{Lendl}, Monika and 
                {Bouchy}, Fran{\c{c}}ois and 
                {Gill}, Samuel and 
                {Nielsen}, Louise D. and 
                {Turner}, Oliver and 
                {Stassun}, Keivan and 
                {Acton}, Jack S. and 
                {Anderson}, David R. Edward, 
                et al},
        title = "{TOI-222: a single-transit TESS candidate revealed to be a 34-d eclipsing binary with CORALIE, EulerCam, and NGTS}",
      journal = {\mnras},
         year = 2020,
        month = feb,
       volume = {492},
       number = {2},
        pages = {1761-1769},
          doi = {10.1093/mnras/stz3545},
archivePrefix = {arXiv},
       eprint = {1910.05050},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.1761L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


```