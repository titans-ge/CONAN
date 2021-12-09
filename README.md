# CONAN
COde for exoplaNet ANalysis

### Installation
```bash
git clone https://github.com/mlendl42/CONAN.git
cd CONAN
python setup.py develop
```

Python package requirements:
numpy, scipy, matplotlib, mc3, batman

Fortran packages:
Routines from Mandel & Agol 2002, compiled with:
```bash
f2py -m occultquad -c occultquad.f
f2py -m occultnl -c occultnl.f
```

