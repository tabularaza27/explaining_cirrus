import datetime
import sys

import pandas as pd
import os
import re
import glob
import gc
import numpy as np
import dask
import xarray as xr
import hvplot.pandas  # noqa
import hvplot.xarray
import matplotlib.pyplot as plt
import holoviews as hv
import dask.dataframe as dd
from sklearn.decomposition import PCA
import glob
import multiprocessing as mp

from src.preprocess.helpers.io_helpers import exists
# from src.preprocess.era5_preproc import calc_rh_ice, calc_e_sat_i, calc_e_sat_w
from src.preprocess.merra2_preproc import calc_plevs
from src.preprocess.helpers.constants import *
from src.scaffolding.scaffolding import get_data_product_dir


def trace_merra_daily(config_id, year, month):
    """write daily files of merra traced trajectories

    reason: monthly files didnt fit in memory for all sky trajectories

    """
    file_type = "ftr"
    load_fcn = pd.read_feather

    if year == 2008:
        file_type = "pickle"
        load_fcn = pd.read_pickle

    # directories specs
    merra_dir = get_data_product_dir(config_id, MERRA_PRE_PROC_DIR)  # merra2 reanalysis preproc dir
    traced_traj_df_dir = get_data_product_dir("larger_domain_high_res", BACKTRAJ_DF_DIR)  # traced trajectory dataframes
    merra_traced_output_dir = get_data_product_dir(config_id, BACKTRAJ_MERRATRACED)  # output dir

    # get trajectory df
    traj_df_fname = os.path.join(traced_traj_df_dir,
                                 "{}_{}{:02d}.{}".format(BACKTRAJ_DF_FILESTUMPY, year, month, file_type))
    traj_df = load_fcn(traj_df_fname)
    traj_df = traj_df[["date", "time", "lat", "lon", "p", "trajectory_id", "T"]]
    # create column with date of each timestep
    traj_df["timestep_date"] = traj_df["date"] + pd.to_timedelta(traj_df["time"], unit="hours")
    # round date and horizontal variable to resolution of merra2
    traj_df["latr"] = np.round((np.round(traj_df.lat * (1 / 0.25)) * 0.25).astype('float64'), 4)
    traj_df["lonr"] = np.round((np.round(traj_df.lon * (1 / 0.25)) * 0.25).astype('float64'), 4)
    traj_df["dater"] = traj_df.date.dt.round("3H")
    traj_df["timestep_dater"] = traj_df.timestep_date.dt.round("3H")

    print("loaded traj df")

    # identify unique days of trajectory start points
    unique_days = traj_df.date.dt.date.unique()
    unique_days = np.sort(unique_days)

    for day in unique_days:
        print(day)

        # check if file already exists
        if exists(day, "merra_traced_df", merra_traced_output_dir, date_fmt_str="%Y%m%d", file_fmt="ftr"):
            print("File already exists for: {}".format(day))
            continue

        day_traj_df = traj_df[traj_df.date.dt.date == day]

        # get merra files of given day + previous three days
        merra_date_range = [day - datetime.timedelta(days=i) for i in range(0, 4)]
        merra_files = [merra_dir + "/all_merra2_date_{}{:02d}{:02d}.nc".format(dt.year, dt.month, dt.day) for dt in
                       merra_date_range]
        merra_files = [f for f in merra_files if os.path.isfile(f)]  # check if file is available
        mds = xr.open_mfdataset(merra_files)  # ,preprocess=calc_plevs)
        # mds=mds.load()
        print("loaded merra dataset")

        # create plev_center variable
        # mds = calc_plevs(mds)
        # plev_center = mds.plev_edge.rolling(lev_edge=2,center=True).mean()
        # mds = mds.assign(plev_center=(["time", "lev", "lat", "lon"], plev_center.isel(lev_edge=slice(1,73)).values))
        # print("created plev center for merra dataset")

        traced_df = trace_merra(day_traj_df, mds)

        # save to file
        traced_df.to_feather(os.path.join(merra_traced_output_dir,
                                          "merra_traced_df_{}{:02d}{:02d}.ftr".format(day.year, day.month, day.day)))
        print("saved to file")


def trace_merra(trajectory_df, merra_ds):
    print("start tracing merra reanalyis for {} backtrajectory rows".format(trajectory_df.date.count()))

    # stack the dimensions of merra data set
    mds_stack = merra_ds.stack(mul=["time", "lat", "lon"])
    # drop redundant lev_edge dimension
    # mds_stack = mds_stack.drop_dims("lev_edge")

    # create column in trajectory data frame with (time, lat, lon) tuple. This can be then compared against the multi index in the merra ds
    trajectory_df["mul_index"] = list(
        zip(trajectory_df["timestep_dater"], trajectory_df["latr"], trajectory_df["lonr"]))

    # get unique (time,lat,lon) gridcells of trajectory df
    unique_mul_index = trajectory_df.mul_index.unique()

    # create set of occuring (time,lat,on) gridcells in merra
    available_grid_cells = set(mds_stack.mul.values.reshape(mds_stack.mul.values.size))

    # note that not all gridcells available in trajectory df are available in merra ds. Reasons: out of spatial domain or out of time
    # get grid_cells available in both merra ds an trajectory df
    both_available_grid_cell_bool = [grid_cell in available_grid_cells for grid_cell in unique_mul_index]
    both_available_grid_cell = unique_mul_index[both_available_grid_cell_bool]

    # select gridcells from dataset and create dataframe from it
    mdf_df = mds_stack.sel(mul=both_available_grid_cell).to_dataframe()
    mdf_df = mdf_df.reset_index()

    # remove mds stack from memory
    mds_stack = None

    # only select level that are within the range of the trajectory data
    min_p, max_p = trajectory_df.p.min() * 100, trajectory_df.p.max() * 100
    print("min,max pressure of traj_df [Pa]: {}/{}".format(min_p, max_p))
    mdf_df_red = mdf_df.query("(PL >={}) & (PL <= {})".format(trajectory_df.p.min() * 100, trajectory_df.p.max() * 100))

    # unique identifier in time and space for each timestep of each trajectory
    trajectory_df["grid_box"] = list(
        zip(trajectory_df["timestep_dater"], trajectory_df["latr"], trajectory_df["lonr"], trajectory_df["p"],
            trajectory_df["trajectory_id"], trajectory_df["time"]))

    # merge trajectory df with merra df on (lat,lon,time)
    # on each row of trajectory df 72 rows are joined, i.e. the height levels
    merge = pd.merge(trajectory_df, mdf_df_red, how="inner", left_on=["latr", "lonr", "timestep_dater"],
                     right_on=["lat", "lon", "time"], suffixes=("_traj", "_merra"))

    print("merged trajectory with merra")

    # calculate difference between trajectory pressure level and pressure levels of merra
    merge["p_diff"] = np.abs(merge["p"] * 100 - merge["PL"])  # *100: hPa → Pa

    # select rows (i.e. height levels) with lowest pressure difference between merra and trajectory
    final = merge.sort_values(["grid_box", "p_diff"]).drop_duplicates("grid_box", keep="first")

    print("sorted df")

    # join traced data back to trajectory dataframe
    # where we don't have merra tracer, there will be np.nans
    traj_df_merge = pd.merge(trajectory_df, final[
        ["grid_box", "DU001", "DU002", "DU003", "DU004", "DU005", "BCPHILIC", "BCPHOBIC", "DMS", "OCPHILIC", "OCPHOBIC",
         "SO4", "T_merra", "PL"]], on="grid_box", how="left")

    print("traced merra data for {}/{} trajectory data points/rows".format(traj_df_merge.T_merra.count(),
                                                                           traj_df_merge["T"].count()))

    return traj_df_merge


def run_parallel(n_workers=5, config_id="merra_extended", year=2007):
    """run regridding process in parallel year or whole directory

        Args:
            n_workers:
            config_id
            year (int): if none run for all available merra2 files

        Returns:
    """

    pool = mp.Pool(n_workers)

    for month in range(1, 13):
        print(month)
        pool.apply_async(trace_merra_daily, args=(config_id, year, month))

    pool.close()
    pool.join()


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_parallel(n_workers=int(sys.argv[1]), config_id=sys.argv[2], year=int(sys.argv[3]))
    else:
        raise ValueError(
            "Provide valid arguments. E.g.: python merra_traj_tracing.py <#workers> <config_id> <year>")
