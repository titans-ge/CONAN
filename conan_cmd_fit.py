if __name__ == '__main__':

    import argparse
    import CONAN

    # Create the parser
    parser = argparse.ArgumentParser(description="Run CONAN fit from configuration file.")

    # Add the arguments
    parser.add_argument("config_file", metavar="config_file", type=str, help="path to configuration file")
    parser.add_argument("out_folder", metavar="out_folder", type=str, help="path to folder where output files will be saved")
    parser.add_argument("--lc_path", type=str, default=None, help="path to light curve files. Overwrites path in configfile")
    parser.add_argument("--rv_path", type=str, default=None, help="path to radial velocity files. Overwrites path in configfile")
    parser.add_argument("--rerun_result", action='store_true', help="flag to rerun result")
    parser.add_argument("--resume_sampling", action='store_true', help="flag to continue sampling from last iteration")
    parser.add_argument("--verbose", action='store_true', help="flag to print verbose output")

    #config_file and out_folder are positional arguments, which means they are required. 
    #rerun_result and verbose are optional flags, when present their values will be set to True; otherwise, their values will be False.

    # Parse the arguments
    args = parser.parse_args()

    print(f"\nLoading config file: '{args.config_file}' and saving result to directory: '{args.out_folder}'")

    # Now you can use the arguments
    result = CONAN.fit_configfile(args.config_file, out_folder=args.out_folder, init_decorr=False,
                                    rerun_result=args.rerun_result, resume_sampling=args.resume_sampling, 
                                    lc_path=args.lc_path, rv_path=args.rv_path, verbose= args.verbose)
