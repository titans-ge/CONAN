.. _api:

API
===

``CONAN3`` has 3 major classes that are used to store information about the input files and also perform computations.
They are the ``load_lightcurves``, ``load_rvs``, and ``fit_setup`` which are described in detail below.

.. autoclass:: CONAN3._classes.load_lightcurves
    :members:

.. autoclass:: CONAN3._classes.load_rvs
    :members:

.. autoclass:: CONAN3._classes.fit_setup
    :members:


These classes return objects to be passed into the ``run_fit`` function.

.. autofunction:: CONAN3.run_fit

The results of the fit are stored in the ``result`` object. 
After a run, the results can be reloaded into memory using the ``load_result`` class. 
This class has a number of methods to access the results.

.. autoclass:: CONAN3._classes.load_result
    :members:
    :undoc-members:

Results of different runs can be compared using the ``compare_results`` class.  

.. autoclass:: CONAN3._classes.compare_results
    :members:
    :undoc-members:
