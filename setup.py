# from setuptools import setup, Extension
from numpy.distutils.core import setup, Extension
import os

with open('CONAN3/VERSION.dat') as version_file:
      __version__ = version_file.read().strip()

extOccultnl   = Extension('occultnl',  sources=['occultnl.f'])
extOccultquad = Extension('occultquad',sources=['occultquad.f'])

setup(name='CONAN3',
      version=__version__,
      description='COde for exoplaNet ANalysis',
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      url='https://github.com/mlendl42/CONAN3',
      author='Monika Lendl, Babatunde Akinsanmi',
      author_email='monika.lendl@unige.ch',
      license='MIT',
      packages=['CONAN3'],
      install_requires=['numpy', 'scipy','pandas','lmfit','dynesty', 'astropy',
                        'batman-package','celerite','corner','lightkurve',
                        'matplotlib','emcee','george','ldtk==1.7.0','tqdm'],
      ext_modules=[extOccultnl,extOccultquad]
      )

#copy command line script to home directory
print("\ncopying command line script to home directory")
os.system("cp conan_cmd_fit.py ~/conan_fit.py")

#add function in .zshrc/.bashrc to call conan_fit.py from anywhere
if os.path.exists(os.path.expanduser("~/.zshrc")):
      if os.system("grep -q 'function conanfit()' ~/.zshrc"):  #check if function already exists
            print("adding conanfit function to ~/.zshrc")
            os.system("echo 'function conanfit() { python ~/conan_fit.py $@; }' >> ~/.zshrc")
            os.system("source ~/.zshrc")
      else:
            print("conanfit function already exists in ~/.zshrc")
      

if os.path.exists(os.path.expanduser("~/.bashrc")):
      if os.system("grep -q 'function conanfit()' ~/.bashrc"): #check if function already exists
            print("adding conanfit function to ~/.bashrc")
            os.system("echo 'function conanfit() { python ~/conan_fit.py $@; }' >> ~/.bashrc")
            os.system("source ~/.bashrc")
      else:
            print("conanfit function already exists in ~/.bashrc")

print("\n")