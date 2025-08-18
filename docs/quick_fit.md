(doc:quickfit)=

## Fitting data
`CONAN` can be run interactively in python shell/Jupyter notebook, but can also be quickly launched from configuration files (.dat or .yaml).

Here are sample [.dat](https://github.com/titans-ge/CONAN/blob/main/sample_config.dat) and [.yaml](https://github.com/titans-ge/CONAN/blob/main/sample_config.yaml) configfiles for fitting the lightcurves and RVs of WASP-127b.

Fitting from a config file can be launched within `python` or from the `command line`

- Within `python`
    ```
    from CONAN import fit_configfile
    result = fit_configfile("input_config.dat", out_folder="output")
    ```
- from `command line`: 
    ```
    conanfit path/to/config_file output_folder 
    ```

    to see commandline help use:
    ``` 
    conanfit -h  
    ```