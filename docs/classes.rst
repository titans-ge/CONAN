.. _api:

API
===

``CONAN3`` has 3 major classes that are used to store information about the input files and also perform computations.
They are the ``load_lightcurves``, ``load_rvs``, and ``mcmc_setup`` which are described in detail below.

.. autoclass:: CONAN3._classes.load_lightcurves
    :members:

.. autoclass:: CONAN3._classes.load_rvs
    :members:

.. autoclass:: CONAN3._classes.mcmc_setup
    :members:


These classes return objects to be passed into the ``fit_data`` function.

.. autofunction:: CONAN3.fit_data

The results of the fit are stored in the ``result`` object. 
After a run, the results can be reloaded into memory using the ``load_result`` class. 
This class has a number of methods to access the results.

.. autoclass:: CONAN3._classes.load_result
    :members: