import numpy as np
import pandas as pd
import xarray as xr
import datetime
import os
import glob
import merra2_preproc
import era5_preproc
from helpers.io_helpers import exists,save_file

MIN_LON = -75
MAX_LON = -15.25
MIN_LAT = 0
MAX_LAT = 59.75
MIN_LEV = 1020
MAX_LEV = 20040

DARDAR_SOURCE_DIR = "/net/n2o/wolke_scratch/kjeggle/DARDAR_NICE/gridded/hourly"
DESTINATION_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/pre_proc"
DATA_CUBE_FILESTUMPY = "data_cube_perproc"


def get_file_paths(dates, source_dir, date_str="%Y_%m_%d", file_format="nc"):
    """get filepaths for given date

    Args:
        dates:
        source_dir:
        date_str:
        file_format:

    Returns:

    """
    print("retrieve filepaths for following dates", dates)
    paths = []
    for date in dates:
        dt_str = date.strftime(date_str)
        filepaths = glob.glob("{}/*{}*.{}".format(source_dir, dt_str, file_format))
        if filepaths:
            paths.append(filepaths)
        else:
            print("no files availables for", str(date))
    paths = np.array(paths, dtype=object).flatten()

    return paths


def create_dardar_masks(ds):
    """

    Args:
        ds: dardar nice dataset

    Returns:

    """
    # observations mask
    obs = ds.iwc.sum(dim="lev", skipna=True, min_count=400).persist()  # grid points without observations are set to nan
    observation_mask = np.isfinite(obs)

    # add mask as coordniate, so I can easily apply `where()`
    ds.coords["observation_mask"] = (("time", "lat", "lon"), observation_mask)
    ds.observation_mask.attrs.update(
        {"long_name": "1 for grid cells (time,lat,lon) where there is a calipso/cloudsat observation, 0 else"
         })

    # data mask
    data_mask = obs > 0
    ds.coords["data_mask"] = (("time", "lat", "lon"), data_mask)
    ds.observation_mask.attrs.update({
        "long_name": "1 for grid cells (time,lat,lon) where there is a calipso/cloudsat observation with observed iwc, 0 else"
    })

    # observation vicinity mask (gridpoint that are an observation or where an observation will be in the next 3 hours)
    observation_vicinity_mask = np.zeros(obs.shape)
    time_range = 4  # current hour + 3
    for time_idx in range(0, ds.dims["time"]):
        observation_vicinity_mask[time_idx, :, :] = observation_mask[time_idx:time_idx + time_range, :, :].sum(
            dim="time").values

    observation_vicinity_mask = observation_vicinity_mask > 0  # set all values to True that are in the vicinity of an observation

    # add mask as coordniate, so I can easily apply `where()`
    ds.coords["observation_vicinity_mask"] = (("time", "lat", "lon"), observation_vicinity_mask)
    ds.observation_vicinity_mask.attrs.update({
        "long_name": "1 for grid cells (time,lat,lon) where there is a calipso/cloudsat observation or where an observation will be in the next 3 hours, 0 else"
    })

    # timestep mask
    # create mask for timesteps (timestep with at least one observations: 1, timestep with no observations:0)
    t_mask = 1 * np.ones((ds.dims["time"])) * (np.sum(ds.observation_mask, axis=(1, 2)) > 0)
    ds.coords["timestep_observation_mask"] = (("time"), t_mask)
    ds.timestep_observation_mask.attrs.update(
        {"long_name": "1 for timesteps where there is at least 1 calipso/cloudsat observation, 0 else"
         })

    return ds


def crop_ds(ds, min_date, max_date, min_lon=MIN_LON, max_lon=MAX_LON, min_lat=MIN_LAT, max_lat=MAX_LAT, min_lev=MIN_LEV,
            max_lev=MAX_LEV):
    """crops dataset to given dimensions

    Args:
        ds:
        min_date:
        max_date:
        min_lon:
        max_lon:
        min_lat:
        max_lat:
        min_lev:
        max_lev:

    Returns:

    """
    ds = ds.sel(time=slice(min_date, max_date), lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat),
                lev=slice(max_lev, min_lev))
    return ds


def check_dimensions(dardar, era, merra):
    for dim in ["time", "lat", "lon", "lev"]:
        # check for same length
        if np.logical_and(dardar.dims[dim] == era.dims[dim], era.dims[dim] == merra.dims[dim]):
            # check for same contents
            if np.logical_and((dardar[dim] == era[dim]).all(), (era[dim] == merra[dim]).all()):
                print("Dimension {} is the same for all datasets".format(dim))
            else:
                raise ValueError("Dimension {} has same length but alters for datasets".format(dim))
        else:
            raise ValueError(
                "Dimension {} has different lengths and potentially different contents for datasets".format(dim))


def merge_one_day(date):
    """merge data sources for given date

    Args:
        date:

    Returns:

    """
    print("Start merging for", str(date))
    # datestrings
    min_date_str = str(date)
    max_date_str = str(date + datetime.timedelta(hours=23))
    np_dt = np.datetime64(date)  # numpy date time

    # load dardar data for given data + next day ( for vicinity mask )
    paths = get_file_paths([date, date + datetime.timedelta(days=1)], DARDAR_SOURCE_DIR)
    dardar_ds = xr.open_mfdataset(paths, concat_dim="time")
    dardar_ds = dardar_ds.transpose("time", "lev", "lat", "lon")
    print("loaded dardar data")

    # create observation,data mask etc
    dardar_ds = create_dardar_masks(dardar_ds)
    print("created masks on dardar data")

    # crop dardar dataset
    dardar_ds = crop_ds(dardar_ds, min_date_str, max_date_str)
    print("cropped dardar data")

    # retrieve and transform reanalysis data online
    era = era5_preproc.run_preprocess_pipeline(np_dt)
    print("loaded  and transformed era data")
    era = crop_ds(era, min_date_str, max_date_str)
    print("cropped era data")
    merra = merra2_preproc.run_preprocess_pipeline(np_dt)
    print("loaded  and transformed merra data")
    merra = crop_ds(merra, min_date_str, max_date_str)
    print("cropped merra data")

    # add observation vicinity mask
    era.coords["observation_vicinity_mask"] = (("time", "lat", "lon"), dardar_ds.observation_vicinity_mask)
    merra.coords["observation_vicinity_mask"] = (("time", "lat", "lon"), dardar_ds.observation_vicinity_mask)
    # mask all values with nan that don't have an observation in their vicinity (cause we do not need them)
    era_reduced = era.where(era.observation_vicinity_mask == 1)
    merra_reduced = merra.where(merra.observation_vicinity_mask == 1)
    print("added observation vicinity mask and dropped all unnecessary data from reanalysis data")

    # check if all dimensions are the same
    check_dimensions(dardar_ds, era_reduced, merra_reduced)

    # merge datasets
    merged = xr.merge([dardar_ds, era_reduced, merra_reduced])
    print("merged datasets")

    return merged # , dardar_ds, era_reduced, merra_reduced

def merge_and_save(date):
    """merges and saves for one day

    Args:
        date:

    Returns:

    """
    # run merging
    merged = merge_one_day(date)

    # load into memory
    merged = merged.load()
    print("loaded into memory")

    save_file(DESTINATION_DIR, DATA_CUBE_FILESTUMPY, merged, date, complevel=4)
    print("saved file")

def run_merging()
    files = glob.glob("{}/*.nc".format(DARDAR_SOURCE_DIR))

    for file in files:
        # extract date string from file
        date_str = file.split("dardar_nice_")[-1].split(".")[0]
        date = datetime.datetime.strptime(date_str, "%Y_%m_%d")

        # check if file already exists
        if exists(date, DATA_CUBE_FILESTUMPY, DESTINATION_DIR):
            logger.info("File already exists for: {}".format(date))

        merge_and_save(date)


if __name__ == "__main__":
    run_merging()



