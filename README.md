[![Python package](https://github.com/titans-ge/CONAN/actions/workflows/python-package.yml/badge.svg)](https://github.com/titans-ge/CONAN/actions/workflows/python-package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/conan-exoplanet)
[![Upload Python Package](https://github.com/titans-ge/CONAN/actions/workflows/python-publish-pypi.yml/badge.svg)](https://github.com/titans-ge/CONAN/actions/workflows/python-publish-pypi.yml)
![PyPI - Version](https://img.shields.io/pypi/v/conan-exoplanet)


# CONAN
**CO**de for exopla**N**et **AN**alysis: A flexible bayesian framework for modeling heterogeneous exoplanet data

### Installation
To avoid requirement conflicts with other packages, it is better to create a new environment (or clone a current environment) to install CONAN


To create a new environment:
```bash
conda create -n conan_env python=3.10
```
then
```
conda activate conan_env
```

CONAN can be installed using different methods: 

- (1) Installing from PyPI:
    ```
    pip install conan-exoplanet
    ```
or

- (2) Downloading the source files from github: 
    ```
    git clone https://github.com/titans-ge/CONAN.git
    cd CONAN 
    ```

    then running
    ```
    pip install .
    ```

or 

- (3) directly using pip to install from github
    ```
    pip install git+https://github.com/titans-ge/CONAN.git
    ```
    Note that a folder 'src' is created where the CONAN source files are downloaded to before installation.


If having troubles compiling the fortran code used for the transit model, set `NO_FORTRAN=True` in terminal before pip installing. This uses a python implementation of the fortran code (which is ~30X slower)

```
export NO_FORTRAN=True
pip install git+https://github.com/titans-ge/CONAN.git
```

-------------------------
See recent changes in [change_log.rst](https://github.com/titans-ge/CONAN/blob/main/change_log.rst)


### Fit from config file 
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
