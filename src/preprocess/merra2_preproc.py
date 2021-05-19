"""
Preprocess MERRA2 files that have already been preprocessed with cdo (i.e. horizontal remapping)

Steps:
1. Calculate pressure levels of layer edges using PTOP=1 Pa + DELP (pressure thickness) as described in 4.2 of [Merra Spec Files](https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf)
2. Calculate height levels with hypsometric equation
3. Unit Transformation from mass mixing ratio to volume mixing ratio
4. Interpolate to Height Levels
5. Resample to hourly grid (optional)
"""

import xarray as xr
import xgcm
import numpy as np
import pandas as pd

from helpers.common_helpers import check_for_nans

# pressure at top of atmosphere is fixed constant 0.01 hPa = 1 Pa
PTOP = 1

# variables of our interest
VARIABLES = ["DU00{}".format(i) for i in range(1, 6)]
VARIABLES += ["SO4", "SO2"]

HLEVS = pd.read_csv("src/data/height_levels.csv", index_col=0)
TARGET_LEVEL_CENTER = HLEVS["lev"].dropna()
TARGET_LEVEL_EDGE = HLEVS["lev_edge"]

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

    R = 287.058
    g = 9.806

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

    # calculate heights on top of levels
    hlev_edge = np.flip(
        np.cumsum(np.flip(delta_z), axis=1)).values  # need to flip for cumsum cause index 0 is top level in dataset
    # insert 0 as surface height level
    hlev_edge = np.append(arr=hlev_edge, values=np.zeros((ds.dims["time"], 1, ds.dims["lat"], ds.dims["lon"])), axis=1)
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


def vert_trafo(ds):
    """vertical coordinate transformation from model to height levels
    all variables are conservative for MERRA2

    Target Levels need to be in ascending order for conservative regrid (seems to be some bug in xgcm)
    """

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
            np.flip(TARGET_LEVEL_EDGE),
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
    layer_thickness = 60
    for var in VARIABLES:
        ext_var = "{}_ext".format(var)

        ds_hlev[var] = ds_hlev[ext_var] / 60
        ds_hlev[var].attrs.update({"units": "kg m**-3"})

    ### linear regridding - just for fun ###

    for var_name in VARIABLES:
        da = grid.transform(
            ds[var_name],
            'Z',
            np.flip(TARGET_LEVEL_CENTER),
            target_data=ds.hlev_center,
        )
        # da = da.reindex(hlev_center=np.flip(da.hlev_center))
        da.attrs.update(ds[var_name].attrs)
        da = da.rename({"hlev_center": "lev"})
        ds_hlev["{}_lin".format(var_name)] = da

    return ds_hlev


def temporal_upsampling(ds):
    """Upsample from 3hourly to 1 hourly"""
    ds_hourly = ds.resample(time="1h").nearest(tolerance="1h")  # maximum distance is 1 hour, so this is the tolerance
    return ds_hourly


def run_preprocess_pipeline(filepath):
    """

    Args:
        filepath: filepath to horizontally regridded file

    Returns:
        xr.Dataset: vertically regridded and to hourly upsampled dataset

    """
    ds = xr.open_dataset(filepath)
    check_for_nans(ds)
    ds = calc_plevs(ds)
    ds = calc_hlevs(ds)
    ds = calc_vol_mixing_ratio(ds)
    ds_hlev = vert_trafo(ds)
    ds_hourly = temporal_upsampling(ds_hlev)

    return ds_hourly