.._installation:

Installation
============

Python package requirements:
``numpy``, ``scipy``, ``matplotlib``, ``batman``, ``emcee``, ``corner``, ``pandas``

Fortran packages: Routines from Mandel & Agol 2002, compiled with f2py:

.. code-block:: bash

    git clone https://github.com/mlendl42/CONAN3.git
.. code-block:: bash

    cd CONAN3
.. code-block:: bash

    pip install -r requirements.txt
.. code-block:: bash

    f2py -m occultquad -c occultquad.f
.. code-block:: bash

    f2py -m occultnl -c occultnl.f
.. code-block:: bash
    
    python setup.py develop
