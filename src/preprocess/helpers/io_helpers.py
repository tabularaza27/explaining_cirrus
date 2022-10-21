import glob
import os
import datetime

import numpy as np


# io helpers
def get_day_files(date, dir_path, file_format="hdf", time_range="day"):
    """returns sorted array of filepaths of given day

    Args:
        date (datetime.datetime):
        dir_path (str): path to DARDAR Files
        file_format (str): format of L2 files, e.g. hdf, nc
        time_range (str): ddy | month

    Returns:
        list: list of filepaths
    """
    if time_range== "day":
        date_str = date.strftime("%Y_%m_%d")
    elif time_range== "month":
        date_str = date.strftime("%Y_%m_*")
    if file_format == "hdf":
        # dardar cloud
        date_dir = os.path.join(dir_path, date_str, "*.{}".format(file_format))
    elif file_format == "nc":
        # dardar nice
        date_dir = os.path.join(dir_path, str(date.year), date_str, "*.{}".format(file_format))
    filepaths = glob.glob(date_dir)
    filepaths = np.sort(filepaths)  # sort files in ascending order

    return filepaths


def get_filepaths(date, dir_path, file_format="hdf", time_range="day"):
    """load filepaths for given date + last file from previous/next day/month

    Args:
        date (datetime.datetime): date
        dir_path (str): path to DARDAR Files
        file_format (str): format of L2 files, e.g. hdf, nc
        time_range (str): day | month

    Returns:
        list|None: list of filepaths to load. If no files exist for that day return None
    """
    # get filepaths for day and add last file from previous day and first file from next day
    filepaths = get_day_files(date, dir_path, file_format, time_range)
    prev_day_paths = get_day_files(date + datetime.timedelta(days=-1), dir_path, file_format, time_range)
    next_day_paths = get_day_files(date + datetime.timedelta(days=1), dir_path, file_format, time_range)

    if filepaths.size == 0:
        return None

    if prev_day_paths.size == 0:
        print("info: no data for prev day available")
    else:
        filepaths = np.insert(filepaths, 0, prev_day_paths[-1])

    if next_day_paths.size == 0:
        print("info: not data for next day available")
    else:
        filepaths = np.insert(filepaths, filepaths.shape[0], next_day_paths[0])

    return filepaths


def save_file(dir_path, file_name, ds, date, time_range="day", complevel=4):
    """compresses and saves file

    Args:
        dir_path (str): target dir to save files
        file_name (str): file name to save the file to
        ds (xarray.Dataset):
        date (datetime.datetime):
        time_range (str): day | month
        complevel (int): compression level
    """

    if time_range == "day":
        date_str = date.strftime("%Y_%m_%d")
    elif time_range == "month":
        date_str = date.strftime("%Y_%m")
    else:
        raise ValueError("specify valid time range, got {}".format(time_range))
    filepath = os.path.join(dir_path, "{}_{}.nc".format(file_name, date_str))

    # compress all data variables
    comp = dict(zlib=True, complevel=complevel)
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(filepath, encoding=encoding)


def exists(date, file_name, dir, date_fmt_str="%Y_%m_%d"):
    """checks if file already exists

    Args:
        date (datetime.datetime):
        file_name (str): file name. the full file name is `file_name`_`datestr`.nc
        dir (str): str of directory in which ii will be checked
        date_fmt_str (str): e.g. "%Y_%m_%d"

    Returns:
        bool: True if file already exists

    """
    datestr = date.strftime(date_fmt_str)
    filepath = os.path.join(dir, "{}_{}.nc".format(file_name, datestr))

    if len(glob.glob(filepath)) > 0:
        return True
    else:
        return False