.._installation:

Installation
============

Python package requirements:
``numpy``, ``scipy``, ``matplotlib``, ``batman``, ``emcee``, ``corner``, ``pandas``
Fortran packages: Routines from Mandel & Agol 2002, compiled with f2py:

To avoid requirement conflicts with other packages, it is better to create a new environment to install ``CONAN``

.. code-block:: bash

    conda create -n conan_env python=3.10
    conda activate conan_env

``CONAN`` can be installed in this new environment by: 

- (1) Installing from PyPI:
    .. code-block:: bash

        pip install conan-exoplanet

- (2) downloading the source files from github: 

.. code-block:: bash

    git clone https://github.com/titans-ge/CONAN.git
.. code-block:: bash

    cd CONAN    
    pip install .

- (3) pip installing directly from the github repository:
.. code-block:: bash

    pip install git+https://github.com/titans-ge/CONAN.git

Note that a folder 'src' is created where the CONAN source files are downloaded to before installation.

If having troubles compiling the fortran code used for the transit model, set `NO_FORTRAN=True` in terminal before pip installing. 
This uses a python implementation of the fortran code (which is ~30X slower)

.. code-block:: bash

    export NO_FORTRAN=True
    pip install git+https://github.com/titans-ge/CONAN.git