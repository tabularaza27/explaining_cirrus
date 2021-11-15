### parallelize backtrajectory calculation ###
# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx <file>`

import multiprocessing as mp
import glob
import sys
import os

BACKTRAJECTORY_SCRIPT = "/net/n2o/wolke/kjeggle/Repos/cirrus/src/preprocess/bash_scripts/calc_backtrajectories.sh"
START_FILE_DIR = "/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/start_files"  # get dir of config id
OUT_FILE_DIR = "/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/outfiles"

# todo make dynamic → potentially use with config

def process_singlefile(date_hour):
    """call preprocessing bash script for given file pTH

    Args:
        filepath (str):
        config_id (config): config_id (str) config determines resolutions and location of load/save directories

    Returns:
    """
    print("Call Bash Script for date {}".format(date_hour))
    os.system("{} {}".format(BACKTRAJECTORY_SCRIPT, date_hour))


def parallel_caltra(n_workers, year, month):
    """call bash script that calculates backtrajectories using lagranto in parallel using python multiprocessing

    Args:
        n_workers (int):
        year (int):
        month (int):

    """
    print("Start parallel merra preprocessing for month {} for year {} with {} workers".format(month, year,
                                                                                               n_workers))
    # link startfiles to output dir
    # todo now I just link all available startfiles
    os.system("ln -s {}/* {}".format(START_FILE_DIR, OUT_FILE_DIR))

    filepaths = glob.glob("{}/*{}{:02d}*".format(START_FILE_DIR, year, month))

    pool = mp.Pool(n_workers)
    for filepath in filepaths:
        date_hour = filepath.split("startf_")[1]
        pool.apply_async(process_singlefile, args=(date_hour,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly
    if len(sys.argv) == 4:
        parallel_caltra(n_workers=int(sys.argv[1]), year=int(sys.argv[2]), month=int(sys.argv[3]))
    else:
        raise ValueError("Provide valid arguments. E.g.: python calc_backtrajectories.py <#workers> <year> <month>")
