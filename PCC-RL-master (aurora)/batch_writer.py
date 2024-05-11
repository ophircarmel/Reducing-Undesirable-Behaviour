# reward modifier
downscaling = [1, 0.8, 0.6, 0.5, 0.4, 0.2, 0.1, 0.05, 0.01]
run_number = 2100
mes = "Less than 1.2 ruleset, for timestamp measuring"
"""
ARGUMENTS:
sys.argv[1] - 1 if it is run on cluster, 0 if on my cpu
sys.argv[2] - run number
sys.argv[3] - place in array
sys.argv[4] - ruleset name
sys.argv[5] - featureset name
sys.argv[6] - treefile_name
sys.argv[7] - message
sys.argv[8] - alpha downscaling  - reward modifier
sys.argv[9] - training_steps
sys.argv[10] - time_steps
"""

SPACE = " "
arg1 = "1"
arg_mid = "0"
arg2 = str(run_number)
arg4 = "less_than_1_pnt_2"
arg5 = "all_with_res"
arg6 = "less_than_1_pnt_2"
arg7 = "\"" + mes + "\""

# for running on a cluster with sbatch
for i in range(len(downscaling)):
    arg2 = str(run_number + 10*i)
    arg8 = str(downscaling[i])
    to_print = ["sbatch", "train_mod.sbatch", arg1, arg2, arg4, arg5, arg6, arg7, arg8]
    to_print = " ".join(to_print)
    print(to_print)
    print()