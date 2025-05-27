(doc:install)=

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