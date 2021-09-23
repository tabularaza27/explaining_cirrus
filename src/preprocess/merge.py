import numpy as np
import xarray as xr
import datetime
import sys
import glob
import multiprocessing as mp

from src.preprocess import merra2_preproc
from src.preprocess import era5_preproc
from src.scaffolding.scaffolding import get_data_product_dir, get_config
from src.preprocess.helpers.io_helpers import exists, save_file
from src.preprocess.helpers.constants import DARDAR_GRIDDED_DIR, DATA_CUBE_PRE_PROC_DIR, DATA_CUBE_PRE_PROC_FILESTUMPY


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


def crop_ds(ds, min_date, max_date, config_id):
    """crops dataset to given dimensions

    Args:
        ds (xr.Dataset): dataset to crop
        min_date (str): format 2008-06-24T21:00:00
        max_date (str): format 2008-06-24T21:00:00
        config_id (str) config determines resolutions and location of load/save directories

    Returns:

    """

    # get config info
    config = get_config(config_id)

    latmin = config["latmin"]
    latmax = config["latmax"]
    lonmin = config["lonmin"]
    lonmax = config["lonmax"]
    altmin = config["altitude_min"]
    altmax = config["altitude_max"]

    ds = ds.sel(time=slice(min_date, max_date), lon=slice(lonmin, lonmax), lat=slice(latmin, latmax),
                lev=slice(altmax, altmin))
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


def merge_one_day(date, config_id):
    """merge data sources for given date

    Args:
        date:
        config_id (str) config determines resolutions and location of load/save directories

    Returns:

    """
    print("Start merging for {}; config: {}".format(str(date), config_id))

    # datestrings
    min_date_str = str(date)
    max_date_str = str(date + datetime.timedelta(hours=23))
    np_dt = np.datetime64(date)  # numpy date time

    # load dardar data for given data + next day ( for vicinity mask )
    paths = get_file_paths([date, date + datetime.timedelta(days=1)],
                           get_data_product_dir(config_id, DARDAR_GRIDDED_DIR))
    dardar_ds = xr.open_mfdataset(paths, concat_dim="time")
    dardar_ds = dardar_ds.transpose("time", "lev", "lat", "lon")
    print("loaded dardar data")

    # create observation,data mask etc
    dardar_ds = create_dardar_masks(dardar_ds)
    print("created masks on dardar data")

    # crop dardar dataset
    dardar_ds = crop_ds(ds=dardar_ds, min_date=min_date_str, max_date=max_date_str, config_id=config_id)
    print("cropped dardar data")

    # retrieve and transform reanalysis data online
    era = era5_preproc.run_preprocess_pipeline(date=np_dt, config_id=config_id)
    print("loaded  and transformed era data")
    era = crop_ds(era, min_date_str, max_date_str, config_id)
    print("cropped era data")
    merra = merra2_preproc.run_preprocess_pipeline(np_dt, config_id)
    print("loaded  and transformed merra data")
    merra = crop_ds(merra, min_date_str, max_date_str, config_id)
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

    return merged  # , dardar_ds, era_reduced, merra_reduced


def merge_and_save(date, config_id):
    """merges and saves for one day

    Args:
        date (datetime.datetime):
        config_id (str) config determines resolutions and location of load/save directories

    Returns:

    """
    # run merging
    merged = merge_one_day(date, config_id)

    # load into memory
    # amerged = merged.load()
    # print("loaded into memory")

    save_file(get_data_product_dir(config_id, DATA_CUBE_PRE_PROC_DIR), DATA_CUBE_PRE_PROC_FILESTUMPY, merged, date,
              complevel=4)
    print("saved file")


def run_merging(config_id, n_workers=4, year=None):
    """run merging process in parallel

    Args:
        config_id (str) config determines resolutions and location of load/save directories
        n_workers:
        year (int): if none run for all available dardar files

    Returns:
    """
    pool = mp.Pool(n_workers)

    dardar_gridded_dir = get_data_product_dir(config_id, DARDAR_GRIDDED_DIR)
    data_cube_preproc_dir = get_data_product_dir(config_id, DATA_CUBE_PRE_PROC_DIR)

    if year:
        print("run merging for {}".format(year))
        files = glob.glob("{}/*{}*.nc".format(dardar_gridded_dir, year))
    else:
        print("run merging for all available dardar data")
        files = glob.glob("{}/*.nc".format(dardar_gridded_dir))

    for file in files:
        # extract date string from file
        date_str = file.split("dardar_nice_")[-1].split(".")[0]
        date = datetime.datetime.strptime(date_str, "%Y_%m_%d")

        # check if file already exists
        if exists(date, DATA_CUBE_PRE_PROC_FILESTUMPY, data_cube_preproc_dir):
            print("File already exists for: {}".format(date))
            continue

        pool.apply_async(merge_and_save, args=(date, config_id,))

    pool.close()
    pool.join()


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_merging(config_id=sys.argv[1], n_workers=int(sys.argv[2]), year=int(sys.argv[3]))
    if len(sys.argv) == 3:
        run_merging(config_id=sys.argv[1], n_workers=int(sys.argv[2]))
    elif len(sys.argv) == 2:
        run_merging(config_id=sys.argv[1])
    else:
        raise ValueError(
            "Provide valid arguments. E.g.: python merge.py <config_id> <#workers> or python merge.py <config_id> <#workers> <year>")
