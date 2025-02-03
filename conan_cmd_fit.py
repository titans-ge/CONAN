import argparse
import CONAN3

# Create the parser
parser = argparse.ArgumentParser(description="Run CONAN fit from configuration file.")

# Add the arguments
parser.add_argument("config_file", metavar="config_file", type=str, help="path to configuration file")
parser.add_argument("out_folder", metavar="out_folder", type=str, help="path to folder where output files will be saved")
parser.add_argument("--rerun_result", action='store_true', help="flag to rerun result")
parser.add_argument("--resume_sampling", action='store_false', help="flag to continue sampling from last iteration")
parser.add_argument("--verbose", action='store_true', help="flag to print verbose output")

#config_file and out_folder are positional arguments, which means they are required. 
#rerun_result and verbose are optional arguments, which means they are not required. 
#If the user provides these arguments, their values will be True; otherwise, their values will be False.

# Parse the arguments
args = parser.parse_args()

print(f"\nLoading config file: '{args.config_file}' and saving result to directory: '{args.out_folder}'")

# Now you can use the arguments
result = CONAN3.fit_configfile(args.config_file, out_folder=args.out_folder, init_decorr=False,
rerun_result=args.rerun_result, resume_sampling=args.resume_sampling, verbose= args.verbose)
