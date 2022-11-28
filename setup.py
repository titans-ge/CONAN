from setuptools import setup
from CONAN3.__version__ import __version__

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
      install_requires=['numpy', 'scipy','pandas','lmfit',
                        'mc3', 'batman-package','celerite','corner',
                        'matplotlib','emcee','george','ldtk','tqdm',],
)
