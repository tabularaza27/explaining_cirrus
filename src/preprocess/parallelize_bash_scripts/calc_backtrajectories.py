### parallelize backtrajectory calculation ###
# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx <file>`

import multiprocessing as mp
import glob
import sys
import os
import re
import datetime
import gc
import time
import subprocess
import pandas as pd
import numpy as np
import argparse

from src.scaffolding.scaffolding import get_data_product_dir
from src.preprocess.helpers.constants import BACKTRAJECTORIES, BACKTRAJ_OUTFILES, BACKTRAJ_STARTFILES

# BACKTRAJECTORY_SCRIPT = "/net/n2o/wolke/kjeggle/Repos/cirrus/src/preprocess/bash_scripts/calc_backtrajectories.sh"
BACKTRAJECTORY_SCRIPT = "/net/n2o/wolke/kjeggle/Repos/cirrus/src/preprocess/bash_scripts/tmux_caltra.sh"

# class ParallelCaltra:
#     def __init__(self, n_workers, year):
#         self.n_workers = n_workers
#         self.year = year
#
#         self.BLOCKED_TIMES = []
#         self.FILEPATHS = glob.glob("{}/{}/*{}*_*".format(START_FILE_DIR, year, year))
#         # remove old tmp files
#         os.system("rm {}/*tmp_{}*".format(OUT_FILE_DIR, year))
#         print("removed intermediate leftover files")
#
#         # link startfiles to output dir
#         # todo now I just link all available startfiles
#         # os.system("ln -sf {}/* {}".format(START_FILE_DIR, OUT_FILE_DIR))
#
#     def process_singlefile(self, date_hour):
#         """call preprocessing bash script for given file pTH
#
#         Args:
#             date_hour (str): yyyymmdd_hh
#
#         Returns:
#         """
#         print("Call Bash Script for date {}".format(date_hour))
#
#         target_filename = "tra_traced_{}.1".format(date_hour)
#
#         yyyy = date_hour[0:4]
#         mm = date_hour[4:6]
#         dd = date_hour[6:8]
#
#         # todo also check if exists in new dir structure
#         if os.path.isfile(os.path.join(OUT_FILE_DIR, target_filename)) or os.path.isfile(
#                 os.path.join(OUT_FILE_DIR, yyyy, mm, dd, target_filename)):
#             print(date_hour, "file already exists")
#         else:
#             # run caltra
#             start_time = datetime.datetime.now()
#             os.system("{} {}".format(BACKTRAJECTORY_SCRIPT, date_hour))
#
#         # check for finished file
#         while True:
#             if os.path.isfile(os.path.join(OUT_FILE_DIR, target_filename)) or os.path.isfile(
#                     os.path.join(OUT_FILE_DIR, yyyy, mm, dd, target_filename)):
#                 duration = datetime.datetime.now() - start_time
#                 print("Finished Caltra for", date_hour, "in", str(duration))
#                 break
#             else:
#                 time.sleep(1)
#
#         # # after successful run remove blocked times and file path from the global variables
#         # d = datetime.datetime.strptime(date_hour, "%Y%m%d_%H")
#         # self.BLOCKED_TIMES = ParallelCaltra.remove_blocked_times(d, self.BLOCKED_TIMES)
#         # self.FILEPATHS = [file for file in self.FILEPATHS if date_hour not in file]
#
#     def run_next_caltra(self):
#         LocalProcRandGen = np.random.RandomState()
#         file = LocalProcRandGen.choice(self.FILEPATHS)
#         date_hour = file.split("startf_")[1]
#         # d = datetime.datetime.strptime(date_hour, "%Y%m%d_%H")
#         # get times that need to be locked for this files
#         # file_block_times = ParallelCaltra.get_blocked_times(d)
#         # check if these times are not locked
#         # if len([t for t in file_block_times if t in self.BLOCKED_TIMES]) == 0:
#         #     print("All times for {} are available → run now".format(date_hour))
#         #     self.BLOCKED_TIMES += file_block_times
#         self.process_singlefile(date_hour)
#         # else:
#         #     print("Dates for {} are occupied, try another file".format(date_hour))
#
#     def parallel_caltra(self):
#         """call bash script that calculates backtrajectories using lagranto in parallel using python multiprocessing
#
#         """
#         n_workers = self.n_workers
#         year = self.year
#
#         print("Start parallel caltra for year {} with {} workers".format(year, n_workers))
#
#         pool = mp.Pool(n_workers)
#         # randomly select filepath
#         while len(self.FILEPATHS) > 0:
#             pool.apply_async(self.run_next_caltra)
#
#         # for filepath in self.FILEPATHS:
#         #     date_hour = filepath.split("startf_")[1]
#         #     pool.apply_async(self.process_singlefile, args=(date_hour,))
#
#         pool.close()
#         pool.join()
#
#     @staticmethod
#     def get_blocked_times(d, steps=60):
#         return pd.date_range(d + datetime.timedelta(hours=-(steps + 1)), periods=steps + 2, freq="1H").tolist()
#
#     @staticmethod
#     def remove_blocked_times(d, blocked_times, steps=60):
#         temp = ParallelCaltra.get_blocked_times(d, steps)
#         return [t for t in blocked_times if t not in temp]


# def run(n_workers, year):
#     print("run")
#     pc = ParallelCaltra(n_workers, year)
#     pc.parallel_caltra()

def get_number_of_caltra_sessions():
    """returns number of currently executing caltra runs in tmux sessions"""
    output = subprocess.check_output("tmux ls", shell=True).decode("utf-8")
    tmux_sessions = output.split("\n")
    parallel_caltra_sessions = [sess for sess in tmux_sessions if "caltra_" in sess]
    n_sessions = len(parallel_caltra_sessions)
    return n_sessions

def process_singlefile(date_hour , config_id, max_concurrency=32):
    """call preprocessing bash script for given file pTH

    Args:
        date_hour (str): yyyymmdd_hh
        config_id (str):
        max_concurrency (int): maximum number of parallel caltra executions

    Returns:
        date_hour (str) : return date_hour str so it can be removed from the filepaths list
    """

    print("Start processiong for date {}".format(date_hour))
    print("Checking for free capacity for {}".format(date_hour))
    start_queue_time = datetime.datetime.now()
    while get_number_of_caltra_sessions() >= max_concurrency:
        time.sleep(10)
    duration = datetime.datetime.now() - start_queue_time
    print("Queueing time for", date_hour, ":", str(duration))

    print("Call Bash Script for date {}".format(date_hour))

    out_file_dir = get_data_product_dir(config_id, BACKTRAJ_OUTFILES)
    target_filename = "tra_traced_{}.1".format(date_hour)

    yyyy = date_hour[0:4]
    mm = date_hour[4:6]
    dd = date_hour[6:8]

    if os.path.isfile(os.path.join(out_file_dir, target_filename)) or os.path.isfile(
            os.path.join(out_file_dir, yyyy, mm, dd, target_filename)):
        print(date_hour, "file already exists")
        return date_hour
    else:
        # run caltra
        start_time = datetime.datetime.now()
        os.system("{} {} {}".format(BACKTRAJECTORY_SCRIPT, config_id, date_hour))
        return date_hour

    # # check for finished file
    # while True:
    #     if os.path.isfile(os.path.join(out_file_dir, target_filename)) or os.path.isfile(
    #             os.path.join(out_file_dir, yyyy, mm, dd, target_filename)):
    #         duration = datetime.datetime.now() - start_time
    #         print("Finished Caltra for", date_hour, "in", str(duration))
    #         return date_hour
    #     else:
    #         time.sleep(1)


# class Filepaths:
#     def __init__(self, config_id, year):
#         self.start_file_dir = get_data_product_dir(config_id, BACKTRAJ_STARTFILES)
#         self.year=year
#         self.filepath_list = glob.glob("{}/{}/*{}*_*".format(self.start_file_dir, self.year, self.year))
#         self.already_calculated=[]
#
#     def get_random_file(self):
#         """get random file from files that were not already calculated"""
#         remaining_files = [file for file in self.filepath_list if file not in self.already_calculated]
#         LocalProcRandGen = np.random.RandomState()
#         file = LocalProcRandGen.choice(remaining_files)
#         return file
#
#     def update_filepaths(self, date_hour):
#         """adds startfile with corresponding date_hour to already calculated list
#
#         is called once the caltra for this date hour was calculated
#         """
#         fp = os.path.join(self.start_file_dir, str(self.year), "startf_{}".format(date_hour))
#
#         if fp not in self.already_calculated:
#             self.already_calculated.append(fp)
#             print("Added {} to already calculated files".format(date_hour))
#         else:
#             print("{} is already part of calculated files".format(date_hour))
#
#         remaining_files = [file for file in self.filepath_list if file not in self.already_calculated]
#         print("Remaining Caltra Startfiles: {}".format(len(remaining_files)))

def parallel_caltra(n_workers, year, config_id):
    """call bash script that calculates backtrajectories using lagranto in parallel using python multiprocessing

    Args:
        n_workers (int):
        year (int):
    """
    print("Start parallel caltra preprocessing for year {} with {} workers".format(year, n_workers))

    # os.system("rm {}/*tmp_{}{:02d}*".format(OUT_FILE_DIR, year, month))
    # print("removed intermediate leftover files")

    pool = mp.Pool(n_workers)

    # check caltras that have already been calculted
    out_file_dir = get_data_product_dir(config_id,BACKTRAJ_OUTFILES)
    already_calculated = glob.glob(os.path.join(out_file_dir,  "{}/**/**/tra_traced*".format(year)), recursive=False)
    calculated_caltras = [re.findall("[0-9]{8}_[0-9]{2}", file)[0] for file in already_calculated] # list of date_hour of already calculated caltras

    # get start files
    start_file_dir = get_data_product_dir(config_id, BACKTRAJ_STARTFILES)
    startfile_list = glob.glob("{}/{}/*{}*_*".format(start_file_dir, year, year))

    # caltras to be calculated: start_files - already calculated caltras
    filepaths = [file for file in startfile_list if re.findall("[0-9]{8}_[0-9]{2}",file)[0] not in calculated_caltras]
    np.random.shuffle(filepaths) # randomize order so parallel caltras are less likely to use same input files

    print("total startfiles: {}, alread calculated: {}, to be calculated: {}".format(len(startfile_list), len(calculated_caltras), len(filepaths)))

    for file in filepaths:
        date_hour = file.split("startf_")[1]  #
        pool.apply_async(process_singlefile, args=(date_hour, config_id, n_workers))

    # while len(filepaths.filepath_list) > 0:
    #     # todo I think while loop is leaking memory, check if process is available befor apply_async
    #     # todo implement check for when all backtrajectories are calculated
    #     # todo implement function that not 2 backtrajectories have to access same source files
    #     file = filepaths.get_random_file()
    #     date_hour = file.split("startf_")[1] #
    #     pool.apply_async(process_singlefile, args=(date_hour, config_id), callback=filepaths.update_filepaths)

    pool.close()
    pool.join()


if __name__ == "__main__":
    # python calc_backtrajectories.py --config_id north_atlantic --n_workers 8 --year 2008
    CLI = argparse.ArgumentParser()

    CLI.add_argument(
        "--config_id",
        type=str
    )

    CLI.add_argument(
        "--n_workers",
        type=int,
        default=6
    )

    CLI.add_argument(
        "--year",
        type=int
    )
    args = CLI.parse_args()

    config_id = args.config_id
    n_workers = args.n_workers
    year = args.year

    print("n_workers: ", n_workers, "year: ", year)
    parallel_caltra(n_workers, year, config_id)

# FILEPATHS=glob.glob("{}/{}/*{}*_*".format(OUT_FILE_DIR,year, year))
# BLOCKED_TIMES=[]

# parallel_caltra(n_workers=n_workers, year=year)
