# CONAN
COde for exoplaNet ANalysis

### Installation
To avoid requirement conflicts with other packages, it is better to create a new environment (or clone a current environment) to install CONAN


To create a new environment:
```bash
conda create -n conan_env numpy=2.1.3 python=3.10
```

or to clone an existing environment with python>=3.10:
```
conda create -n conan_env --clone my_old_env
```

then
```
conda activate conan_env
```

CONAN can be installed by: 

- (1) downloading the source files from github: 
```
git clone https://github.com/mlendl42/CONAN3.git
cd CONAN3 
```

then running
```pip install -e .```

- (2) directly using pip to install from github
```pip install -e git+https://github.com/mlendl42/CONAN3.git#egg=CONAN3```

Note that a folder 'src' is created where the CONAN source files are downloaded to before installation.

if having troubles compiling the fortran code used for the transit model, set `NO_FORTRAN=True` in terminal before pip installing. This uses a python implementation of the fortran code (which is ~30X slower)

```
export NO_FORTRAN=True
pip install -e git+https://github.com/mlendl42/CONAN3.git#egg=CONAN3
```

-------------------------
See recent changes in change_log.rst


### Fit from config file 
Fit can be launched from config file within `python` or from the `command line`

- Within `python`

```
from CONAN3 import fit_configfile
result = fit_configfile("input_config.dat", out_folder="output")
```
- from `command line`: 

```
conanfit path/to/config_file output_folder 

conanfit -h   # to see the help
```
