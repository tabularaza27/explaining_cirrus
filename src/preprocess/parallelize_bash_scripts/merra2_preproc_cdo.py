### parallelize cdo preprocessing ###
# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx era_preproc_single_file.sh `

from src.scaffolding.scaffolding import get_data_product_dir

import multiprocessing as mp
import glob
import sys
import os

from src.preprocess.helpers.constants import MERRA_CDO_SCRIPT_PATH, MERRA_INCOMING_DIR


def process_singlefile(filepath, config_id):
    """call preprocessing bash script for given file pTH

    Args:
        filepath (str):
        config_id (config): config_id (str) config determines resolutions and location of load/save directories

    Returns:
    """
    print("Call Bash Script for file {}".format(filepath))
    os.system("{} {} {}".format(MERRA_CDO_SCRIPT_PATH, filepath, config_id))


def parallel_preproc(n_workers, config_id, year=None):
    """call bash script that preprocesses era5 data with cdo in parallel using python multiprocessing

    Args:
        n_workers (int):
        year (int|None): if year is None run preproc for all existing files in directory
        config_id (str):

    """
    print("Start parallel merra preprocessing of config {} for year {} with {} workers".format(config_id, year,
                                                                                               n_workers))
    merra_incoming_dir = get_data_product_dir(config_id, MERRA_INCOMING_DIR)  # get dir of config id
    if year is None:
        filepaths = glob.glob("{}/*.nc4".format(merra_incoming_dir, year))
    else:
        filepaths = glob.glob("{}/*{}*.nc4".format(merra_incoming_dir, year))

    pool = mp.Pool(n_workers)
    for filepath in filepaths:
        pool.apply_async(process_singlefile, args=(filepath, config_id,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly
    if len(sys.argv) == 3:
        parallel_preproc(n_workers=int(sys.argv[1]), config_id=sys.argv[2])
    elif len(sys.argv) == 4:
        parallel_preproc(n_workers=int(sys.argv[1]), config_id=sys.argv[2], year=int(sys.argv[3]))
    else:
        raise ValueError("Provide valid arguments. E.g.: python merra2_preproc_cdo.py <#workers> <config_id> <year>")
