(doc:api)=

# Top-level API

These pages contain the API reference for `CONAN`.

``CONAN`` has 3 major classes that are used to store information about the input files and also perform computations.
They are:

- [load_lightcurves](CONAN._classes.load_lightcurves): ingest lightcurve files and creates an object that is used to configure baseline and model parameters. It contains several methods to configure the LCs for fitting. 
- [load_rvs](CONAN._classes.load_rvs): ingest RV files and creates an object that is used to configure baseline and model parameters. It contains methods to configure the RVs for fitting

- [fit_setup](CONAN._classes.fit_setup): object to setup sampling of the parameter space


These classes return objects to be passed into the [run_fit](CONAN.fit_data.run_fit) function to start sampling using the defined configurations. The results of the fit are stored in the ``result`` object. 


After a run, the results can be reloaded into memory using the [load_result](CONAN._classes.load_result) class. This class has a number of methods to make plots and compute statistics from the samples.

