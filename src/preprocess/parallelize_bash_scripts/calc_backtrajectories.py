### parallelize backtrajectory calculation ###
# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx <file>`

import multiprocessing as mp
import glob
import sys
import os
import datetime
import time
import pandas as pd
import numpy as np
import argparse

BACKTRAJECTORY_SCRIPT = "/net/n2o/wolke/kjeggle/Repos/cirrus/src/preprocess/bash_scripts/calc_backtrajectories.sh"
START_FILE_DIR = "/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/start_files"  # get dir of config id
OUT_FILE_DIR = "/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/outfiles"

### parallelize backtrajectory calculation ###
# To be able to execute bash script from python they need to be executable, i.e:
# `$ chmod u=rwx <file>`

import multiprocessing as mp
import glob
import sys
import os
import datetime
import time
import pandas as pd
import numpy as np
import argparse

BACKTRAJECTORY_SCRIPT = "/net/n2o/wolke/kjeggle/Repos/cirrus/src/preprocess/bash_scripts/calc_backtrajectories.sh"
START_FILE_DIR = "/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/start_files"  # get dir of config id
OUT_FILE_DIR = "/net/n2o/wolke_scratch/kjeggle/BACKTRAJECTORIES/outfiles"


# todo make dynamic → potentially use with config

class ParallelCaltra:
    def __init__(self, n_workers, year):
        self.n_workers = n_workers
        self.year = year

        self.BLOCKED_TIMES = []
        self.FILEPATHS = glob.glob("{}/{}/*{}*_*".format(START_FILE_DIR, year, year))
        # remove old tmp files
        os.system("rm {}/*tmp_{}*".format(OUT_FILE_DIR, year))
        print("removed intermediate leftover files")

        # link startfiles to output dir
        # todo now I just link all available startfiles
        #os.system("ln -sf {}/* {}".format(START_FILE_DIR, OUT_FILE_DIR))

    def process_singlefile(self, date_hour):
        """call preprocessing bash script for given file pTH

        Args:
            date_hour (str): yyyymmdd_hh

        Returns:
        """
        print("Call Bash Script for date {}".format(date_hour))

        target_filename = "tra_traced_{}.1"

        if os.path.isfile(target_filename):
            print("file already exists")
        else:
            # run caltra
            os.system("{} {}".format(BACKTRAJECTORY_SCRIPT, date_hour))

        # after successful run remove blocked times and file path from the global variables
        d = datetime.datetime.strptime(date_hour, "%Y%m%d_%H")
        self.BLOCKED_TIMES = ParallelCaltra.remove_blocked_times(d, self.BLOCKED_TIMES)
        self.FILEPATHS = [file for file in self.FILEPATHS if date_hour not in file]

    def run_next_caltra(self):
        time.sleep(np.random.randint(2,7))
        file = np.random.choice(self.FILEPATHS)
        date_hour = file.split("startf_")[1]
        d = datetime.datetime.strptime(date_hour, "%Y%m%d_%H")
        # get times that need to be locked for this files
        file_block_times = ParallelCaltra.get_blocked_times(d)
        # check if these times are not locked
        if len([t for t in file_block_times if t in self.BLOCKED_TIMES]) == 0:
            print("All times for {} are available → run now".format(date_hour))
            self.BLOCKED_TIMES += file_block_times
            self.process_singlefile(date_hour)
        else:
            print("Dates for {} are occupied, try another file".format(date_hour))

    def parallel_caltra(self):
        """call bash script that calculates backtrajectories using lagranto in parallel using python multiprocessing

        """
        n_workers = self.n_workers
        year = self.year

        print("Start parallel caltra for year {} with {} workers".format(year, n_workers))

        pool = mp.Pool(n_workers)
        # randomly select filepath
        while len(self.FILEPATHS) > 0:
            pool.apply_async(self.run_next_caltra)

        pool.close()
        pool.join()



    @staticmethod
    def get_blocked_times(d, steps=60):
        return pd.date_range(d + datetime.timedelta(hours=-(steps + 1)), periods=steps + 2, freq="1H").tolist()

    @staticmethod
    def remove_blocked_times(d, blocked_times, steps=60):
        temp = ParallelCaltra.get_blocked_times(d, steps)
        return [t for t in blocked_times if t not in temp]

def run(n_workers, year):
    print("run")
    pc = ParallelCaltra(n_workers, year)
    pc.parallel_caltra()

# def parallel_caltra(n_workers, year, month):
#     """call bash script that calculates backtrajectories using lagranto in parallel using python multiprocessing
#
#     Args:
#         n_workers (int):
#         year (int):
#         month (int):
#
#     """
#     print("Start parallel merra preprocessing for month {} for year {} with {} workers".format(month, year,
#                                                                                                n_workers))
#
#     os.system("rm {}/*tmp_{}{:02d}*".format(OUT_FILE_DIR,year,month))
#     print("removed intermediate leftover files")
#
#     # link startfiles to output dir
#     # todo now I just link all available startfiles
#     os.system("ln -sf {}/* {}".format(START_FILE_DIR, OUT_FILE_DIR))
#
#     filepaths = glob.glob("{}/{}/*{}{:02d}*".format(START_FILE_DIR,year, year, month))
#
#     pool = mp.Pool(n_workers)
#     for filepath in filepaths:
#         date_hour = filepath.split("startf_")[1]
#         pool.apply_async(process_singlefile, args=(date_hour,))
#     pool.close()
#     pool.join()

if __name__ == "__main__":
    # python calc_backtrajectories.py --n_workers 8 --year 2008 #--months 1 2 3
    CLI = argparse.ArgumentParser()
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

    n_workers = args.n_workers
    year = args.year

    print("n_workers: ", n_workers, "year: ", year)
    run(n_workers, year)

# FILEPATHS=glob.glob("{}/{}/*{}*_*".format(OUT_FILE_DIR,year, year))
# BLOCKED_TIMES=[]

# parallel_caltra(n_workers=n_workers, year=year)
