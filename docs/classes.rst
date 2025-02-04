.. _api:

API
===

``CONAN`` has 3 major classes that are used to store information about the input files and also perform computations.
They are the ``load_lightcurves``, ``load_rvs``, and ``fit_setup`` which are described in detail below.

.. autoclass:: CONAN._classes.load_lightcurves
    :members:
    :inherited-members:
    :private-members:
    :show-inheritance:

.. autoclass:: CONAN._classes.load_rvs
    :members:
    :inherited-members:
    :private-members:
    :show-inheritance:

.. autoclass:: CONAN._classes.fit_setup
    :members:
    :inherited-members:
    :private-members:
    :show-inheritance:

These classes return objects to be passed into the ``run_fit`` function.

.. autofunction:: CONAN.run_fit

The results of the fit are stored in the ``result`` object. 
After a run, the results can be reloaded into memory using the ``load_result`` class. 
This class has a number of methods to access the results.

.. autoclass:: CONAN._classes.load_result
    :members:
    :undoc-members:
    :inherited-members:
    :private-members:
    :show-inheritance: