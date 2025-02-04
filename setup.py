#works for normal installation but not for editable installation unless 
import os
from  numpy.distutils.core import setup, Extension
# os.system("f2py -c -m occultquad occultquad.f")
# os.system("f2py -c -m occultnl occultnl.f")

# # Check if the NO_FORTRAN environment variable is set
no_fortran = os.getenv('NO_FORTRAN', 'False').lower() in ('true', '1', 't')  

if no_fortran:
      print("Skipping compilation of fortran code, python equivalent of transit model will be used")
      ext_modules = []
else:
      print("Compiling fortran code")     
      extOccultnl = Extension('CONAN.occultnl', sources=['CONAN/occultnl.f'])
      extOccultquad = Extension('CONAN.occultquad', sources=['CONAN/occultquad.f'])
      ext_modules = [extOccultnl, extOccultquad]


setup(
      long_description=open("README.md").read(),
      long_description_content_type='text/markdown',
      packages=['CONAN'],
      ext_modules=ext_modules,
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