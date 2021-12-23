"""
Preprocess MERRA2 files that have already been preprocessed with cdo (i.e. horizontal remapping)

Steps:
1. Calculate pressure levels of layer edges using ybrid model level constants a and b
2. Calculate height levels with hypsometric equation
3. Calculate rh_ice
4. Interpolate to Height Levels
"""

import xarray as xr
import xgcm
import numpy as np
import pandas as pd

from src.preprocess.helpers.common_helpers import check_for_nans
from src.preprocess.helpers.constants import R, g
from src.preprocess.helpers.constants import ERA_PRE_PROC_DIR
from src.scaffolding.scaffolding import get_data_product_dir, get_height_levels, get_config, get_alt_range

# HLEVS = pd.read_csv("/home/kjeggle/cirrus/src/config_files/height_levels.csv", index_col=0) # todo make dynamic
# TARGET_LEVEL_CENTER = HLEVS["lev"].dropna()


def load_ds(date, config_id):
    """loads file for given date and adds first timestamp of next day (needed for the hourly upsampling)

    Args:
        date (numpy.datetime64):
        config_id (str) config determines of load directories
    """
    date_str = pd.to_datetime(str(date)).strftime("%Y_%m_%d")
    era_pre_proc_dir = get_data_product_dir(config_id, ERA_PRE_PROC_DIR)
    path = "{}/all_era5_date_{}_time_*.nc".format(era_pre_proc_dir, date_str)
    ds = xr.open_mfdataset(path)

    min_lev = 45  #
    max_lev = 137  # closer to ground → min/max in terms of pressure
    ds = ds.sel(lev=slice(min_lev, max_lev), nhyi=slice(min_lev - 1, max_lev + 1), nhym=slice(min_lev - 1, max_lev))

    ds = ds.load()  # load from dask arrays into memory

    return ds


def calc_plevs(ds):
    """calculate pressures at level center and level edges via hybrid model level constants a and b"""

    surface_pressure = np.exp(
        ds.lnsp).persist()  # convert log surface pressure to surface pressure, use persist to keep in memory for dask distributed execution
    surface_pressure.attrs.update({"long_name": "surface pressure"})

    # create stacked pressure level variable to be able to multiply with a nd b constants
    surface_pressure_stacked_m = np.hstack([surface_pressure for i in range(0, ds.dims["nhym"])])  # for level centers
    surface_pressure_stacked_i = np.hstack([surface_pressure for i in range(0, ds.dims["nhyi"])])  # for level edges

    ds = ds.assign(sp_stacked_m=(["time", "nhym", "lat", "lon"], surface_pressure_stacked_m))  # level centers
    ds = ds.assign(sp_stacked_i=(["time", "nhyi", "lat", "lon"], surface_pressure_stacked_i))  # level edges

    # calculate pressure level at layer centers
    # set `lev` as vertical coordinate
    plev_center = ds.hyam + ds.sp_stacked_m * ds.hybm  # p = ap + b * surface pressure
    plev_center = plev_center.rename({"nhym": "lev"})
    plev_center = plev_center.transpose("time", "lev", "lat", "lon")
    ds["plev_center"] = plev_center
    ds.plev_center.attrs.update(
        {"standard_name": "air_pressure", "long_name": "air pressure on level centers", "units": "Pa"})

    # calculate pressure at layer edges
    # use nhyi as vertical coordinate (has one more tick than level center coordinate)
    plev_edge = ds.hyai + ds.sp_stacked_i * ds.hybi
    ds["plev_edge"] = plev_edge.transpose("time", "nhyi", "lat", "lon")
    ds.plev_edge.attrs.update(
        {"standard_name": "air_pressure", "long_name": "air pressure on level edges", "units": "Pa"})

    return ds


def calc_hlevs(ds):
    """calculate geometric heights at level center and level edges via hypsometric equation

    Use Hypsometric equation
    $\Delta z = (z_2 - z_1) = \frac{R}{g} * T * ln(\frac{p_1}{p_2})$

    temperature in Kelvin

    [source](http://tornado.sfsu.edu/Geosciences/classes/e260/Hypsometric/Hypsometric%20Equation.pdf)
    """

    # calculate 𝑙𝑛(𝑝1𝑝2), where p1 is pressure at lower interface of level an p2 is pressure at upper interface of level
    rolling = ds.plev_edge.rolling(nhyi=2)
    rolling_da = rolling.construct(nhyi="z_win")  # construct returns data array view of rolling object
    #     log_div = np.log(rolling_da.isel(z_win=1)) - np.log(rolling_da.isel(z_win=0)) # 𝑙𝑛(𝑝1𝑝2) == ln(p1)-ln(p2)
    log_div = np.log(rolling_da.isel(z_win=1) / rolling_da.isel(z_win=0))
    ds = ds.assign(log_div=(["time", "lev", "lat", "lon"], log_div.isel(nhyi=slice(1, ds.dims["nhyi"])).values))

    # calculate geometric layer thickness with hypsometric equation
    delta_z = R / g * ds.t * ds.log_div #  todo use virtual temperature
    ds["delta_z"] = delta_z

    # set layer thickness of top layer to 5000 (doesnt really matter what value it is, since we are not interested in data at those heights but currently it is infinity which fucks up the cumsum)
    # delta_z[:, 0, :, :] = np.ones((ds.dims["time"], ds.dims["lat"], ds.dims["lon"])) * 5000

    # add geometric height at surface (geopotential / gravitational acceleration)
    surface_geometric_height = (ds.z / g).values
    delta_z = np.append(arr=delta_z, values=surface_geometric_height, axis=1)

    # calculate heights on top of levels
    hlev_edge = np.flip(
        np.cumsum(np.flip(delta_z), axis=1))  # need to flip for cumsum cause index 0 is top level in dataset
    
    # assign to dataset
    ds = ds.assign(hlev_edge=(["time", "nhyi", "lat", "lon"], hlev_edge[:, :, :, :]))

    # calculate heights at level center by taking rolling mean of heights at level interfaces
    hlev_center = ds.hlev_edge.rolling(nhyi=2).mean().dropna(dim="nhyi")
    ds = ds.assign(hlev_center=(["time", "lev", "lat", "lon"], hlev_center))

    return ds


def calc_e_sat_w(tk):
    """
    calculate saturation vapor pressure w.r.t. water [hPa] for given temp [K]

    args:
        tk: temperature in Kelvin
    """
    tc = tk - 273.15  # convert K to C
    e_sat_w = 6.112 * np.exp(17.62 * tc / (243.12 + tc))

    return e_sat_w


def calc_e_sat_i(tk):
    """calculate saturation vapor pressure w.r.t. ice [hPa] for given temp [K]

    args:
        tk: temperature in Kelvin

    """
    tc = tk - 273.15  # convert K to C
    e_sat_w = 6.112 * np.exp(22.46 * tc / (272.62 + tc))

    return e_sat_w

# todo double check
def calc_rh(ds):
    """calculate relative humidity w.r.t. water and ice for given temp

    RH_{ice} = RH_{w} * \frac{e_{sat_w}}{e_{sat_i}}$

    use [2] and [14] from [here](https://www.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html) to calculate saturation vapor pressures

    [2]: ew = 6.112 e(17.62 t/(243.12 + t))
    with t in [°C] and ew in [hPa]


    [14] ei = 6.112 e(22.46 t/(272.62 + t))
    with t in [°C] and ei in [hPa]

    °C = K - 273.15

    args:
        rh_w: relative humidity w.r.t water
        tk: temperature in Kelvin
    """
    e_sat_w = calc_e_sat_w(ds["t"])
    e_sat_i = calc_e_sat_i(ds["t"])

    # calculate partial pressure of water (https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html)
    e = ds["q"] * ds["plev_center"] / (0.622 + 0.378 * ds["q"])

    rh_w = e/e_sat_w
    rh_ice = rh_w * e_sat_w / e_sat_i

    ds["rh"] = rh_w
    ds["rh_ice"] = rh_ice

    return ds

def calc_iwc(ds):
    """transform specific iwc (ciwc)  [kg kg**-1] to iwc [kg m**-3]

    https://glossary.ametsoc.org/wiki/Virtual_temperature

    1. calculate virtual temperature: T_v = T(1+r_v/ε)/(1+r_v); ε = 0.622, r_v = approximated by specific humidity
    2. calculate air density via ideal gas law: air_density=air_pressure/(virtual_air_temperature*R_air) [kg m-3]
    3. calculate iwc: iwc = ciwc * air_density
    """
    epsilon = 0.622  # ratio of the gas constants of air and water vapor
    ds["t_v"] = ds.t * (1 + ds.q / epsilon) / (1 + ds.q)  # calculate virtual temp
    ds["airdens"] = ds.plev_center / (R * ds["t_v"])
    ds["era_iwc"] = ds["ciwc"] * ds["airdens"]

    ds.era_iwc.attrs.update({"units": "kg m**-3",
                             "standard_name": "ice water content"
                             })

    return ds


def calc_trans_w(ds):
    """transform vertical velocity from Pa s**-1 to m s**-1

    1. calculate air density: $\rho = \frac{p}{RT}$ (https://confluence.ecmwf.int/pages/viewpage.action?pageId=153391710)
    2. calculate w_trans: $w_{trans} = - (w/ (\rho * g))$
    """
    ds["airdens"] = ds.plev_center / (R * ds.t)

    ds.airdens.attrs.update({"units": "kg m**-3",
                             "standard_name": "air_density",
                             "long_name": "air density at level center"
                             })

    ds["w_trans"] = -ds.w / (ds.airdens * g)

    ds.w_trans.attrs.update({"units": "m s**-1",
                             "standard_name": "upward_air_velocity",
                             "long_name": "vertical velocity"
                             })

    return ds


def vert_trafo(ds, altitude_min, altitude_max, layer_thickness):
    """vertical coordinate transformation from model to pressure levels

    cloud cover and iwc are transformed conservative, all other variables are transformed linear

    Args:
        ds (xr.Dataset): dataset with hybrid sigma pressure levels
        altitude_min (int): minimum altitude of dataset after transformation
        altitude_max (int): maximum altitude of dataset after transformation
        layer_thickness (int): vertical resolution after transformation

    Returns:
        xr.Dataset: dataset with geometric height as vertical coordinate
    """

    cons_vars = ["cc","era_iwc"]
    lin_vars = ["etadot", "t", "w", "w_trans", "u", "v", "rh", "rh_ice", "plev_center"]

    target_level_center = get_height_levels(altitude_min, altitude_max, layer_thickness,
                                            position="center")  # height levels for linear trafo on level center
    target_level_edge = get_height_levels(altitude_min, altitude_max, layer_thickness,
                                          position="edge")  # height levels for conservative trafo on level edge

    ### conservative transformation ###
    for var in cons_vars:
        new_var = "{}_ext".format(var)

        ds[new_var] = ds[var] * ds.delta_z
        ds[new_var].attrs.update({"units": "{} m".format(ds[var].attrs["units"])})

    # select variables to be transformed
    var_dict = dict()

    # create grid for vertical transformation
    grid = xgcm.Grid(
        ds,
        periodic=False,
        coords={'Z': {'center': 'lev', 'outer': 'nhyi'}}
    )

    for var_name in cons_vars:
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

    # (values /  thickness) → convert extensive variables to intensive variables again
    for var in cons_vars:
        ext_var = "{}_ext".format(var)

        ds_hlev[var] = ds_hlev[ext_var] / layer_thickness
        ds_hlev[var].attrs.update({"units": ds[var].attrs["units"]})

    ### linear transformation ###

    for var_name in lin_vars:
        da = grid.transform(
            ds[var_name],
            'Z',
            target_level_center,
            target_data=ds.hlev_center,
            method="linear"
        )
        da.attrs.update(ds[var_name].attrs)
        da = da.rename({"hlev_center": "lev"})
        ds = da.transpose("time", "lev", "lat", "lon")
        ds_hlev["{}".format(var_name)] = da
        # var_dict[var_name] = da

    # # create hlev dataset from transformed dataarrays and set standard coordinate order and attributes
    # ds_hlev = xr.Dataset(var_dict).transpose("time", "hlev_center", "lat", "lon")
    #
    # ds_hlev = ds_hlev.rename({"hlev_center": "lev"})
    # ds_hlev.lev.attrs.update({"units": "m",
    #                           "standard_name": "altitude",
    #                           "long_name": "altitude at level center",
    #                           "axis": "Z"
    #                           })

    # add 2D variables
    ds_hlev["lnsp"] = ds["lnsp"]
    ds_hlev["z"] = ds["z"]

    return ds_hlev


def run_preprocess_pipeline(date, config_id):
    """run preprocess pipeline for one day

    Args:
        date (numpy.datetime64):
        config_id (str) config determines resolutions and location of load/save directories
    Returns:
        xr.Dataset: vertically regridded dataset

    """
    config = get_config(config_id)
    layer_thickness=config["layer_thickness"]
    altitude_min, altitude_max = get_alt_range(config_id)

    ds = load_ds(date, config_id)
    check_for_nans(ds)
    ds = calc_plevs(ds)
    ds = calc_trans_w(ds)
    ds = calc_iwc(ds)
    ds = calc_hlevs(ds)
    ds = calc_rh(ds)
    ds_hlev = vert_trafo(ds, altitude_min, altitude_max, layer_thickness)

    return ds_hlev
