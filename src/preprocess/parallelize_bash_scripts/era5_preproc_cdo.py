# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx era_preproc_single_file.sh `

import multiprocessing as mp
import glob
import sys
import os

from src.preprocess.helpers.constants import ERA_CDO_SCRIPT_PATH, ERA_INCOMING_DIR


def process_singlefile(filepath, config_id):
    """call preprocessing bash script for given file path

    Args:
        filepath (str):
        config_id (config): config_id (str) config determines resolutions and location of load/save directories

    Returns:

    """
    print("Call Bash Script for file {}".format(filepath))
    os.system("{} {} {}".format(ERA_CDO_SCRIPT_PATH, filepath, config_id))


def parallel_preproc(n_workers, year, config_id):
    """call bash script that preprocesses era5 data with cdo in parallel using python multiprocessing for given config

    Args:
        n_workers (int):
        year (int):
        config_id (str):

    Returns:

    """
    print("Start parallel era preprocessing of config {} for year {} with {} workers".format(config_id, year,
                                                                                               n_workers))
    filepaths = glob.glob("{}/era5_date_{}_*_time_*.grb".format(ERA_INCOMING_DIR, year))

    if len(filepaths) == 0:
        print("no ERA5 Files have been downloaded for that year yet")
        return None
    else:
        print("Start horizontal regridding for {} files".format(len(filepaths)))

    pool = mp.Pool(n_workers)
    for filepath in filepaths:
        pool.apply_async(process_singlefile, args=(filepath, config_id, ))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly → use parsing package
    if len(sys.argv) == 4:
        parallel_preproc(n_workers=int(sys.argv[1]), year=int(sys.argv[2]), config_id=sys.argv[3])
    else:
        raise ValueError("Provide valid arguments. E.g.: python era5_preproc_cdo.py <#workers> <year> <config_id>")
