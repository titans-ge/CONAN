# CONAN
COde for exoplaNet ANalysis

### Installation
To avoid requirement conflicts with other packages, it is better to create a new environment to install CONAN

```bash
conda create -n conan_env numpy python = 3.10

conda activate conan_env
```

CONAN can be installed by: 

- downloading the source files from github: 

```
git clone https://github.com/mlendl42/CONAN3.git
cd CONAN3

python setup.py develop

```

- or directly using pip to install from github
```
pip install -e git+https://github.com/mlendl42/CONAN3.git#egg=CONAN3
```

See recent changes in change_log.rst

