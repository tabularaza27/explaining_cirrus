from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy as np
import pickle
import sys
import glob
import dask

TEMP_THRES = 235.15
ICE_CLOUD_MASKS = [1, 2, 9, 10]  # maybe add 10 → top of convective towers

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

SOURCE_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/pre_proc"
DF_FINAL_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/dataframes"  # 2d data frame with all ice cloud ovservations
FILTERED_DF_FINAL_DIR = "/net/n2o/wolke_scratch/kjeggle/DATA_CUBE/filtered_cube"  # contains only entries with data mask true


def get_load_variables():
    """return list of variables to load based on VAR_PROPERTIES"""
    return [var for var in VAR_PROPERTIES if VAR_PROPERTIES[var]["load"] == True]


def get_drop_vars():
    """return list of variables not to load based on VAR_PROPERTIES"""
    return [var for var in VAR_PROPERTIES if VAR_PROPERTIES[var]["load"] == False]


def get_month_files(year, month):
    """return list of files for given month and year of datacube preproc"""
    month = str(month).zfill(2)
    files = glob.glob("{}/data_cube_perproc_{}_{}*.nc".format(SOURCE_DIR, year, month))
    return files


def filter_and_save_months(year, months):
    """create data frame and data_only dataset for given year and months

    Args:
        year (int):
        months (list[int]): e.g. [1,2,3]

    Returns:

    """
    print("start processing: ", year, str(months))

    initial_month_str = str(months[0]).zfill(2)
    df_filename = "{}/ice_in_cloud_df_{}_{}.pickle".format(DF_FINAL_DIR, year, initial_month_str)
    data_only_filename = "{}/data_only_{}_{}.pickle".format(year, initial_month_str)

    # check if file already exists
    if len(glob.glob(df_filename)) > 0:
        print(year, str(months), "already exists")
        return

    # get data cube files
    files = []
    for month in months:
        files += get_month_files(year, month)

    # load into xarray
    drop_vars = get_drop_vars()
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        ds = xr.open_mfdataset(files, parallel=True, concat_dim="time", drop_variables=drop_vars)

    # stack time lat lon to multiindex
    ds_stack = ds.stack(mul=["time", "lat", "lon"])

    # filter for days with data
    data_only = ds_stack.where(ds_stack.data_mask == True, drop=True)

    data_only = data_only.compute()
    print("computed")

    # filter for ice cloud
    ice_in_cloud = data_only.where(data_only.clm_v2.isin(ICE_CLOUD_MASKS))

    # create dataframe
    df = ice_in_cloud.to_dataframe()
    # reset multiindex, i.e. flatten data
    df = df.reset_index()

    # create helper columns for time related variables
    df["hour"] = df.time.dt.hour
    df["month"] = df.time.dt.month
    df["season"] = ((df.time.dt.month % 12 + 3) // 3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})
    drop = ["lev_2", "observation_mask", "data_mask", "observation_vicinity_mask", "timestep_observation_mask"]
    df.drop(columns=drop, inplace=True)

    # filter data frame
    df = df[df.clm_v2.isin(ICE_CLOUD_MASKS)]
    df = df.query("ta <= {}".format(TEMP_THRES))

    df.to_pickle(df_filename)

    with open(data_only_filename, 'wb') as handle:
        pickle.dump(data_only, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_year(year):
    """create data frame and data_only dataset for given year. one file each 3 months

    Args:
        year:

    Returns:

    """

    month_ranges = np.arange(1, 13, 1).reshape(4, 3)

    cluster = LocalCluster()
    with Client(cluster) as client:
        dashboard_port = client.scheduler_info()['services']['dashboard']
        print("execute on local terminal to connect to dashboard: \n`ssh -L 8787:localhost:{} n2o`".format(
            dashboard_port))

        for month_range in month_ranges:
            filter_and_save_months(year, month_range)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_year(year=int(sys.argv[1]))
    else:
        raise ValueError("Provide valid arguments. E.g.: python data_cube_filters.py <year>")
