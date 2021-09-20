import json
import os

import numpy as np
import xarray as xr

from src.preprocess.helpers.constants import *


# todo: docstrings

def get_config(config_id):
    """returns dict of config"""
    with open(CONFIGS) as f:
        configs = json.load(f)
        config = configs[config_id]

    return config


def get_config_base_dir(config_id):
    """returns base path of config"""
    config_base_dir = os.path.join(BASE_DIRECTORY, config_id)
    return config_base_dir


def get_data_product_dir(config_id, relative_dir):
    """returns absolute path of a config sub directory

    e.g. of MERRA_INCOMING

    Args:
        config_id:
        relative_dir (str): relative path of subdirectory

    Returns:

    """
    config_base_dir = get_config_base_dir(config_id)

    absoulte_path = os.path.join(config_base_dir, relative_dir)
    # check if exists
    if os.path.isdir(absoulte_path):
        return absoulte_path
    else:
        raise ValueError("Directory `{}` doesnt exist in config  `{}` yet".format(relative_dir, config_id))


def get_abs_file_path(config_id, rel_file_path):
    """returns absolute filepath given config id and relative filepath

    e.g. for template.nc

    Args:
        config_id:
        rel_file_path:

    Returns:
        str: absolute file path
    """
    config_base_dir = get_config_base_dir(config_id)

    absoulte_path = os.path.join(config_base_dir, rel_file_path)
    # check if exists
    if os.path.isfile(absoulte_path):
        return absoulte_path
    else:
        raise ValueError("Directory `{}` doesnt exist in config  `{}` yet".format(rel_file_path, config_id))


def create_directory(config_id, dir_path, base_directory=BASE_DIRECTORY):
    path = os.path.join(base_directory, config_id, dir_path)
    try:
        os.makedirs(path, mode=0o755)
        print("created:", path)
    except FileExistsError:
        print("directory `{}` already exists for config `{}`".format(dir_path, config_id))


def setup_config_directories(config_id):
    """create all directories for config"""

    dirs = [CONFIG_FILE_DIR,
            MERRA_INCOMING_DIR,
            MERRA_PRE_PROC_DIR,
            MERRA_REGRID_DIR,
            ERA_PRE_PROC_DIR,
            ERA_IM_DIR,
            DARDAR_GRIDDED_DIR,
            DATA_CUBE_PRE_PROC_DIR,
            DATA_CUBE_FILTERED_DIR,
            DATA_CUBE_DF_DIR,
            DATA_CUBE_FEATURE_ENGINEERED_DF_DIR
            ]

    print("##### start creating directories for config {}".format(config_id))
    for path in dirs:
        create_directory(config_id, path)
    print("##### created all directories for config {}".format(config_id))


def create_horizontal_template(config_id):
    """creates and saves horizontal template used for horizontal regridding"""
    config = get_config(config_id)
    config_base_dir = get_config_base_dir(config_id)
    config_file_path = os.path.join(config_base_dir, CONFIG_FILE_DIR, "template.nc")

    lon_values = np.arange(config["lonmin"], config["lonmax"] + config["horizontal_resolution"],
                           config["horizontal_resolution"])
    lat_values = np.arange(config["latmin"], config["latmax"] + config["horizontal_resolution"],
                           config["horizontal_resolution"])
    random_place_holder_values = np.random.rand(lon_values.shape[0], lat_values.shape[0])

    # create dataset
    template = xr.Dataset(data_vars={"random": (("lat", "lon"), random_place_holder_values)},
                          coords={"lon": lon_values, "lat": lat_values})

    # set attributes
    template.lon.attrs = {'standard_name': 'longitude',
                          'long_name': 'longitude',
                          'units': 'degrees_east',
                          'axis': 'X'}
    template.lat.attrs = {'standard_name': 'latitude',
                          'long_name': 'latitude',
                          'units': 'degrees_north',
                          'axis': 'Y'}

    template.to_netcdf(config_file_path)

    print("created and saved horizontal template file (template.nc) for config `{}`".format(config_id))

    # todo save template in correct dir


def get_height_levels(min_level, max_level, layer_thickness, position="center"):
    """returns array of height levels"""
    if position == "center":
        hlevs = np.arange(min_level, max_level, layer_thickness)
    elif position == "edge":
        lower_edge = min_level - layer_thickness / 2
        upper_edge = max_level + layer_thickness / 2
        hlevs = np.arange(lower_edge, upper_edge, layer_thickness)

    hlevs = np.flip(hlevs)  # flip levels, so that highest altitude level has index 0

    return hlevs


def scaffolding(config_id):
    """sets up grid config files based on config and creates directories for config

    * template.nc
    * create directories
    """
    setup_config_directories(config_id)
    create_horizontal_template(config_id)


# todo
def verify_config(config_id):
    """verifies config"""


# todo
def list_directories(config_id):
    """lists directories of given config"""
