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

- downloading the source files from github: 

.. code-block:: bash

    git clone https://github.com/tundeakins/CONAN3.git
.. code-block:: bash

    cd CONAN3    
    pip install .

- or pip installing directly from the github repository:
.. code-block:: bash

    pip install -e git+https://github.com/tundeakins/CONAN3.git#egg=CONAN3

Note that a folder 'src' is created where the CONAN source files are downloaded to before installation.