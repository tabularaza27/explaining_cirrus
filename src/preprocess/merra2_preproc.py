"""
Preprocess MERRA2 files that have already been preprocessed with cdo (i.e. horizontal remapping)

Steps:
1. Calculate pressure levels of layer edges using PTOP=1 Pa + DELP (pressure thickness) as described in 4.2 of [Merra Spec Files](https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf)
2. Calculate height levels with hypsometric equation
3. Unit Transformation from mass mixing ratio to volume mixing ratio
4. Interpolate to Height Levels
5. Resample to hourly grid (optional)
"""
import os
import sys
import datetime
import glob
import multiprocessing as mp

import xarray as xr
import xgcm
import numpy as np
import pandas as pd

from src.preprocess.helpers.common_helpers import check_for_nans
from src.preprocess.helpers.io_helpers import exists, save_file
from src.preprocess.helpers.constants import R, g
from src.preprocess.helpers.constants import MERRA_PRE_PROC_DIR, MERRA_REGRID_DIR, MERRA_REGRID_FILESTUMPY

from src.scaffolding.scaffolding import get_data_product_dir, get_height_levels, get_config

# pressure at top of atmosphere is fixed constant 0.01 hPa = 1 Pa
PTOP = 1

# variables of our interest
VARIABLES = ["DU00{}".format(i) for i in range(1, 6)]
VARIABLES += ["SO4", "SO2"]

# HLEVS = pd.read_csv("/home/kjeggle/cirrus/src/config_files/height_levels.csv", index_col=0) # todo make dynamic
# TARGET_LEVEL_CENTER = HLEVS["lev"].dropna() # todo
# TARGET_LEVEL_EDGE = HLEVS["lev_edge"] # todo


def load_ds(date, config_id):
    """loads file for given date and adds first timestamp of next day (needed for the hourly upsampling)

    Args:
        date (numpy.datetime64):
        config_id (str): config determines resolutions and location of load/save directories
    """
    next_day = date + np.timedelta64(1, "D")

    date_str = "%Y%m%d"
    day_str = pd.to_datetime(str(date)).strftime(
        date_str)  # convert to pandas datetime first to be able to use strftime
    next_day_str = pd.to_datetime(str(next_day)).strftime(date_str)

    day_ds = xr.open_dataset(os.path.join(get_data_product_dir(config_id,MERRA_PRE_PROC_DIR), "all_merra2_date_{}.nc".format(day_str)))
    next_day_ds = xr.open_dataset(os.path.join(get_data_product_dir(config_id, MERRA_PRE_PROC_DIR), "all_merra2_date_{}.nc".format(next_day_str)))

    ds = xr.concat([day_ds, next_day_ds.isel(time=0)], dim="time")

    assert ds.dims["time"] == 9, "time dim of dataset needs to have length 9 to cover whole day, has {}".format(
        ds.dims["time"])

    return ds


def calc_plevs(ds):
    """Calculate pressure levels of layer edges using PTOP=1 Pa + DELP (pressure thickness) as described in 4.2 of Merra Spec Files."""

    # create level edge coordinate
    ds = ds.assign_coords(lev_edge=np.linspace(0.5, 72.5, 73))
    # PTOP with 1 Pa is not included yet, i.e. first index is top of second layer. top of first layer is PTOP
    plev_edge = np.cumsum(ds.DELP[:, :, :, :], axis=1).values + 1
    # add PTOP as first index to each gridcell
    plev_edge = np.insert(arr=plev_edge, obj=0, values=PTOP, axis=1)
    # add edge pressure to dataset
    ds = ds.assign(plev_edge=(["time", "lev_edge", "lat", "lon"], plev_edge))

    return ds


def calc_hlevs(ds):
    """calculate geometric heights at level center and level edges via hypsometric equation

    Use Hypsometric equation:
    $\Delta z = (z_2 - z_1) = \frac{R}{g} * T * ln(\frac{p_1}{p_2})$

    [source](http://tornado.sfsu.edu/Geosciences/classes/e260/Hypsometric/Hypsometric%20Equation.pdf)
    """

    # calculate 𝑙𝑛(𝑝1𝑝2), where p1 is pressure at lower interface of level an p2 is pressure at upper interface of level
    rolling = ds.plev_edge.rolling(lev_edge=2)
    rolling_da = rolling.construct(lev_edge="z_win")  # construct returns data array view of rolling object
    log_div = np.log(rolling_da.isel(z_win=1) / rolling_da.isel(z_win=0))
    ds = ds.assign(log_div=(["time", "lev", "lat", "lon"], log_div.isel(lev_edge=slice(1, ds.dims["lev_edge"])).values))

    # calculate geometric layer thickness with hypsometric equation
    delta_z = R / g * ds.T * ds.log_div

    # set layer thickness of top layer to 5000 (doesnt really matter what value it is, since we are not interested in data at those heights but currently it is infinity which fucks up the cumsum)
    delta_z[:, 0, :, :] = np.ones((ds.dims["time"], ds.dims["lat"], ds.dims["lon"])) * 5000
    ds["delta_z"] = delta_z

    # add geometric height at surface (geopotential / gravitational acceleration)
    surface_geometric_height = np.expand_dims(ds.PHIS.values, 1) / g
    delta_z = np.append(arr=delta_z, values=surface_geometric_height, axis=1)

    # calculate heights on top of levels
    hlev_edge = np.flip(
        np.cumsum(np.flip(delta_z), axis=1))  # need to flip for cumsum cause index 0 is top level in dataset

    # assign to dataset
    ds = ds.assign(hlev_edge=(["time", "lev_edge", "lat", "lon"], hlev_edge))

    # calculate heights at level center by taking rolling mean of heights at level interfaces
    hlev_center = ds.hlev_edge.rolling(lev_edge=2).mean().dropna(dim="lev_edge")
    ds = ds.assign(hlev_center=(["time", "lev", "lat", "lon"], hlev_center))

    return ds


def calc_vol_mixing_ratio(ds):
    """calculates volume mixing ratio from mass mixing ratio and air density"""
    # variables to transform
    for var in VARIABLES:

        if ds[var].attrs["units"] == "kg m**-3":
            print('{} is already given in volume mixing ration'.format(var))
            continue

        ds[var] = ds[var] * ds.AIRDENS
        ds[var].attrs.update({"units": "kg m**-3"})

    return ds


def vert_trafo(ds, altitude_min, altitude_max, layer_thickness, linear=False):
    """vertical coordinate transformation from model to height levels
    all variables are conservative for MERRA2

    Args:
        ds (xr.Dataset): dataset with hybrid sigma pressure levels
        altitude_min (int): minimum altitude of dataset after transformation
        altitude_max (int): maximum altitude of dataset after transformation
        layer_thickness (int): vertical resolution after transformation
        linear (bool): If True also transform linearly

    Returns:
        xr.Dataset: dataset on height levels

    Target Levels need to be in ascending order for conservative regrid (seems to be some bug in xgcm)
    """
    target_level_center = get_height_levels(altitude_min, altitude_max, layer_thickness,
                                            position="center")  # height levels for linear trafo on level center
    target_level_edge = get_height_levels(altitude_min, altitude_max, layer_thickness,
                                            position="center")  # height levels for conservative trafo on level edge

    ### create extensive variables ###
    for var in VARIABLES:
        new_var = "{}_ext".format(var)

        ds[new_var] = ds[var] * ds.delta_z
        ds[new_var].attrs.update({"units": "kg m**-3 m"})

    # select variables to be transformed
    var_dict = dict()

    # create grid
    grid = xgcm.Grid(
        ds,
        periodic=False,
        coords={'Z': {'center': 'lev', 'outer': 'lev_edge'}}
    )

    ### conservative regridding ###

    # transform vertical coordinate for each variable
    for var_name in VARIABLES:
        var_name = "{}_ext".format(var_name)

        da = grid.transform(
            ds[var_name],
            'Z',
            np.flip(target_level_edge),
            target_data=ds.hlev_edge,
            method="conservative",
        )
        da.attrs.update(ds[var_name].attrs)
        var_dict[var_name] = da

    # create new dataset with transformed variables
    ds_hlev = xr.Dataset(var_dict)
    ds_hlev = ds_hlev.reindex(hlev_edge=np.flip(ds_hlev.hlev_edge))  # reindex so level is in descending order again
    ds_hlev = ds_hlev.rename({"hlev_edge": "lev"})
    ds_hlev.lev.attrs.update({"units": "m",
                              "standard_name": "altitude",
                              "long_name": "altitude at level center",
                              "axis": "Z"
                              })

    ds_hlev = ds_hlev.transpose("time", "lev", "lat", "lon")

    ### convert extensive variables to intensive variables again ###

    # (values /  thickness)
    for var in VARIABLES:
        ext_var = "{}_ext".format(var)

        ds_hlev[var] = ds_hlev[ext_var] / layer_thickness
        ds_hlev[var].attrs.update({"units": "kg m**-3"})

    ### linear regridding - just for fun ###
    if linear:
        for var_name in VARIABLES:
            da = grid.transform(
                ds[var_name],
                'Z',
                np.flip(target_level_center),
                target_data=ds.hlev_center,
            )
            # da = da.reindex(hlev_center=np.flip(da.hlev_center))
            da.attrs.update(ds[var_name].attrs)
            da = da.rename({"hlev_center": "lev"})
            ds_hlev["{}_lin".format(var_name)] = da

    ds_hlev["PHIS"] = ds["PHIS"]
    ds_hlev["PS"] = ds["PS"]

    return ds_hlev


def temporal_upsampling(ds):
    """Upsample from 3hourly to 1 hourly"""
    ds_hourly = ds.resample(time="1h").nearest(tolerance="1h")  # maximum distance is 1 hour, so this is the tolerance
    return ds_hourly


def run_preprocess_pipeline(date, config_id):
    """run preprocessing pipeline for given day

    Args:
        date (numpy.datetime64):
        config_id (str) config determines resolutions and location of load/save directories
    Returns:
        xr.Dataset: vertically regridded and to hourly upsampled dataset

    """

    config = get_config(config_id)
    altitude_min = config["altitude_min"]
    altitude_max = config["altitude_max"]
    layer_thickness = config["layer_thickness"]

    # load datases
    ds = load_ds(date, config_id)
    # checks for nans
    check_for_nans(ds)
    # calculate pressure levels
    ds = calc_plevs(ds)
    # calculate height levels
    ds = calc_hlevs(ds)
    # calculate volume mixing ratio
    ds = calc_vol_mixing_ratio(ds)
    # vertical regrid to height levels
    ds_hlev = vert_trafo(ds, altitude_min, altitude_max, layer_thickness)
    # temporal upsampling to hourly data
    ds_hourly = temporal_upsampling(ds_hlev)
    # select only specified day (right now also 0 of following day is in dataset)
    dt_str = pd.to_datetime(str(date)).strftime("%Y-%m-%d")
    final_ds = ds_hourly.sel(time=dt_str)

    # select data variables
    final_ds = final_ds[VARIABLES + ["PHIS", "PS"]]

    return final_ds


# not used directly, since run_preprocessing_pipeline is called directly by merge.py
def run_and_save(date, config_id):
    """run preprocessing pipeline for given day and save regridded file to disk

    Args:
        date (datetime.datetime):
        config_id (str) config determines resolutions and location of load/save directories

    Returns:

    """
    np_date = np.datetime64(date)

    print("start regridding for {}".format(date))
    merra_regridded = run_preprocess_pipeline(np_date, config_id)

    save_file(get_data_product_dir(config_id, MERRA_REGRID_DIR), MERRA_REGRID_FILESTUMPY, merra_regridded, date, complevel=4)
    print("save file: {}".format(date))


# not used directly, since run_preprocessing_pipeline is called directly by merge.py
def run_parallel(n_workers=4, year=None):
    """run regridding process in parallel year or whole directory

        Args:
            n_workers:
            year (int): if none run for all available merra2 files

        Returns:
    """
    pool = mp.Pool(n_workers)

    if year:
        print("run regridding for {}".format(year))
        files = glob.glob("{}/*{}*.nc".format(MERRA_PRE_PROC_DIR, year))
    else:
        print("run regridding for all available merra data")
        files = glob.glob("{}/*.nc".format(MERRA_PRE_PROC_DIR))

    print("{} files found".format(len(files)))

    for file in files:
        # extract date string from file
        date_str = file.split("all_merra2_date_")[-1].split(".")[0]
        date = datetime.datetime.strptime(date_str, "%Y%m%d")

        # check if file already exists
        if exists(date, MERRA_REGRID_FILESTUMPY, MERRA_REGRID_DIR):
            print("File already exists for: {}".format(date))
            continue
        pool.apply_async(run_and_save, args=(date,))

    pool.close()
    pool.join()

if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_parallel(n_workers=int(sys.argv[1]),year=int(sys.argv[2]))
    if len(sys.argv) == 2:
        run_parallel(n_workers=int(sys.argv[1]))
    elif len(sys.argv) == 1:
        run_parallel()
    else:
        raise ValueError("Provide valid arguments. E.g.: python merra2_preproc.py <#workers> or python merge.py <#workers> <year>")