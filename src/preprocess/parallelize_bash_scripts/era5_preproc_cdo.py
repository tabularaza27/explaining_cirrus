# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx era_preproc_single_file.sh `

import multiprocessing as mp
import subprocess
import glob
import sys
import os

Source_File_Directory = '/net/n2o/wolke_scratch/kjeggle/ERA5/ECMWF_incoming'
Script_Path = '/home/kjeggle/cirrus/src/preprocess/bash_scripts/era_preproc_single_file.sh'


def process_singlefile(filepath):
    """call preprocessing bash script for given file pTH"""
    print("Call Bash Script for file {}".format(filepath))
    os.system("{} {}".format(Script_Path,filepath))


def parallel_preproc(n_workers, year):
    """call bash script that preprocesses era5 data with cdo in parallel using python multiprocessing"""
    print("Start parallel preprocessing for year {} with {} workers".format(year, n_workers))
    filepaths = glob.glob("{}/era5_date_{}_*_time_*.grb".format(Source_File_Directory, year))

    pool = mp.Pool(n_workers)
    for filepath in filepaths:
        pool.apply_async(process_singlefile, args=(filepath,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly
    if len(sys.argv) == 3:
        parallel_preproc(n_workers=int(sys.argv[1]), year=int(sys.argv[2]))
    else:
        raise ValueError("Provide valid arguments. E.g.: python era5_preproc_cdo.py <#workers> <year>")
