from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy as np
import pickle
import sys
import glob
import dask

from src.preprocess.helpers.constants import DATA_CUBE_PRE_PROC_DIR, DATA_CUBE_DF_DIR, DATA_CUBE_FILTERED_DIR
from src.scaffolding.scaffolding import get_data_product_dir

VAR_PROPERTIES = {
    'iwc': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'reffcli': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'icnc_5um': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'icnc_100um': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'clm_v2': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'clm': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'cloud_cover': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'dz_top': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    'instrument_flag': {'scale': 'lin', 'source': 'dardar', 'type': 'cirrus_var', 'load': True},
    't': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'ta': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'w': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'u': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'v': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'etadot': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'rh_ice': {'scale': 'lin', 'source': 'era5', 'type': 'driver', 'load': True},
    'DU001': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
    'DU002': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
    'DU003': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
    'DU004': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
    'DU005': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
    'SO4': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
    'SO2': {'scale': 'lin', 'source': 'merra2', 'type': 'driver', 'load': True},
}


def get_load_variables():
    """return list of variables to load based on VAR_PROPERTIES"""
    return [var for var in VAR_PROPERTIES if VAR_PROPERTIES[var]["load"] == True]


def get_drop_vars():
    """return list of variables not to load based on VAR_PROPERTIES"""
    return [var for var in VAR_PROPERTIES if VAR_PROPERTIES[var]["load"] == False]


def get_month_files(year, month, config_id):
    """return list of files for given month and year of datacube preproc

    Args:
        year:
        month:
        config_id:

    Returns:
        list: list of file paths
    """
    month = str(month).zfill(2)
    data_cube_pre_proc_dir = get_data_product_dir(config_id, DATA_CUBE_PRE_PROC_DIR)
    files = glob.glob("{}/data_cube_perproc_{}_{}*.nc".format(data_cube_pre_proc_dir, year, month))
    return files


def filter_and_save_months(year, months, filter_type, config_id):
    """create data frame and filtered dataset for given year and months. Filters data based on observation_mask or data_mask.

    observation_mask: True, where we have dardar retrievals
    data_mask: True, where we have iwc > 0

    Args:
        year (int):
        months (list[int]): e.g. [1,2,3]
        filter_type (str): one of the following ['data','observations','observation_vicinity']
        config_id (str) config determines resolutions and location of load/save directories

    Returns:

    """
    print("start processing: ", year, str(months))

    initial_month_str = str(months[0]).zfill(2)

    data_cube_df_dir = get_data_product_dir(config_id, DATA_CUBE_DF_DIR)
    data_cube_filtered_dir = get_data_product_dir(config_id, DATA_CUBE_FILTERED_DIR)

    if filter_type == "data":
        df_filename = "{}/ice_in_cloud_df_{}_{}.pickle".format(data_cube_df_dir, year, initial_month_str)
        filtered_cube_filename = "{}/data_only_{}_{}.pickle".format(data_cube_filtered_dir, year, initial_month_str)
        mask_var = "data_mask"
    elif filter_type == "observations":
        df_filename = "{}/observations_df_{}_{}.pickle".format(data_cube_df_dir, year, initial_month_str)
        filtered_cube_filename = "{}/observations_{}_{}.pickle".format(data_cube_filtered_dir, year, initial_month_str)
        mask_var = "observation_mask"
    elif filter_type == "observation_vicinity":
        df_filename = "{}/observation_vicinity_df_{}_{}.pickle".format(data_cube_df_dir, year, initial_month_str)
        filtered_cube_filename = "{}/observation_vicinity_{}_{}.pickle".format(data_cube_filtered_dir, year,
                                                                               initial_month_str)
        mask_var = "observation_vicinity_mask"
    else:
        raise ValueError(
            "filter_type needs to be one of the following ['data','observations','observation_vicinity'], was {}".format(
                filter_type))

    # check if file already exists
    if len(glob.glob(df_filename)) > 0:
        print(year, str(months), "already exists")
        return

    # get data cube files
    files = []
    for month in months:
        files += get_month_files(year, month, config_id)

    # load into xarray
    drop_vars = get_drop_vars()
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.open_mfdataset(files,
                               parallel=True,
                               concat_dim="time",
                               drop_variables=drop_vars,
                               # Attempt to auto-magically combine the given datasets into one by using dimension
                               # coordinates.
                               combine="by_coords",
                               # Only data variables in which the dimension already appears are included.
                               data_vars="minimal",
                               # Only coordinates in which the dimension already appears are included.
                               coords="minimal",
                               # Skip comparing and pick variable from first dataset.
                               compat="override")
        print("loaded ds")

    # stack time lat lon to multiindex
    ds_stack = ds.stack(mul=["time", "lat", "lon"])

    # filter for days with data
    filtered_cube = ds_stack.where(ds_stack[mask_var] == True, drop=True)

    filtered_cube = filtered_cube.compute()
    print("computed")

    # filter for ice cloud

    # create dataframe
    df = filtered_cube.to_dataframe()
    # reset multiindex, i.e. flatten data
    df = df.reset_index()

    # create helper columns for time related variables
    df["hour"] = df.time.dt.hour
    df["month"] = df.time.dt.month
    df["season"] = ((df.time.dt.month % 12 + 3) // 3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})
    drop = ["lev_2", "observation_mask", "data_mask"]
    df.drop(columns=drop, inplace=True)

    if filter_type == "data":
        # filter for cloud cover
        df = df[df.cloud_cover > 0]

    # write dataframe and data only ds to pickle
    df.to_pickle(df_filename)

    with open(filtered_cube_filename, 'wb') as handle:
        pickle.dump(filtered_cube, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_year(config_id, year, filter_type):
    """create data frame and data_only dataset for given year. one file each 3 months

    Args:
        config_id (str) config determines resolutions and location of load/save directories
        year:
        filter_type (str): one of the following ['data','observations']

    Returns:

    """

    month_ranges = np.arange(1, 13, 1).reshape(4, 3)
    # month_ranges = np.arange(7,13,1).reshape(6,1)

    cluster = LocalCluster(processes=True)
    with Client(cluster) as client:
        dashboard_port = client.scheduler_info()['services']['dashboard']
        print("execute on local terminal to connect to dashboard: \n`ssh -L 8787:localhost:{} n2o`".format(
            dashboard_port))

        for month_range in month_ranges:
            filter_and_save_months(year, month_range, filter_type, config_id)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_year(config_id=sys.argv[1], year=int(sys.argv[2]), filter_type=str(sys.argv[3]))
    else:
        raise ValueError("Provide valid arguments. E.g.: python data_cube_filters.py <config_id> <year> <filter_type>")
