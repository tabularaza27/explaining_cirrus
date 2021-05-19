### parallelize cdo preprocessing ###
# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx era_preproc_single_file.sh `

import multiprocessing as mp
import subprocess
import glob
import sys
import os

Source_File_Directory = '/net/n2o/wolke_scratch/kjeggle/MERRA2/incoming'
Script_Path = '/home/kaijeggle/dev/DataAnalysis/cirrus/src/preprocess/bash_scripts/merra_horizontal_remap.sh'


def process_singlefile(filepath):
    """call preprocessing bash script for given file pTH"""
    print("Call Bash Script for file {}".format(filepath))
    os.system("{} {}".format(Script_Path,filepath))


def parallel_preproc(n_workers=8):
    """call bash script that preprocesses era5 data with cdo in parallel using python multiprocessing"""
    print("Start parallel preprocessing with {} workers".format(n_workers))
    filepaths = glob.glob("{}/*".format(Source_File_Directory))

    pool = mp.Pool(n_workers)
    for filepath in filepaths:
        pool.apply_async(process_singlefile, args=(filepath,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly
    if len(sys.argv) == 2:
        parallel_preproc(n_workers=int(sys.argv[1]))
    elif len(sys.argv) == 1:
        parallel_preproc()
        #filepaths = glob.glob("{}/era5_date_*_time_*.grb".format(Source_File_Directory))
        #print(filepaths[0])
        #process_singlefile(filepaths[0])
    else:
        raise ValueError("Provide valid arguments. E.g.: python merra2_preproc_cdo.py <#workers>")
