# from setuptools import setup, Extension
from numpy.distutils.core import setup, Extension

with open('CONAN3/VERSION.dat') as version_file:
      __version__ = version_file.read().strip()

extOccultnl   = Extension('occultnl',  sources=['occultnl.f'])
extOccultquad = Extension('occultquad',sources=['occultquad.f'])

setup(name='CONAN3',
      version=__version__,
      description='COde for exoplaNet ANalysis with Gaussian Process',
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/mlendl42/CONAN3',
      author='Monika Lendl',
      author_email='monika.lendl@unige.ch',
      license='MIT',
      packages=['CONAN3'],
      install_requires=['numpy', 'scipy','pandas','lmfit','dynesty',
                        'batman-package','celerite','corner','lightkurve',
                        'matplotlib','emcee','george','ldtk','tqdm'],
      ext_modules=[extOccultnl,extOccultquad]
      )
