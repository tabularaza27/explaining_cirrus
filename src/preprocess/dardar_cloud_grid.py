import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import datetime
import cis
import iris
import os
import glob
from copy import deepcopy


class DardarCloud:
    # variables to retrieve from DARDAR Cloud L2 Data
    VARIABLES_3D = ["iwc",
                    "ln_iwc_error",
                    "effective_radius",
                    "ln_effective_radius_error",
                    "extinction",
                    "ln_extinction_error",
                    "temperature"]

    VARIABLES_2D = ["Tropopause_Height",
                    "land_water_mask",
                    "day_night_flag"]

    # meta data for spatial resolution
    # horizontal
    LATMIN = 0  # 0
    LATMAX = 60  # 60
    LONMIN = -75  # -75
    LONMAX = -15  # -15
    HOR_RES = 0.25  # horizontal resolution [degree]
    TEMP_RES = 3  # temporal resolution in hours

    # vertical; vertical resolution is kept at 60m for now
    ALTMIN = 0
    ALTMAX = 25080
    ALTINTERVAL = 60
    ALTLEVELS = 419

    DIR_PATH = "/wolke_scratch/nedavid/DARDAR_CLOUD"

    def __init__(self, date):
        self.date = date
        ### set up 3D grid ###

        # temporal
        temp_delta = datetime.timedelta(hours=self.TEMP_RES)
        self.intervals_per_day = int(24 / self.TEMP_RES)
        # all intervals include 1 time interval before and after the current day, cause the data in the day dir can contain also these data points
        all_intervals = np.array([self.date + temp_delta * i for i in range(1, self.intervals_per_day + 2)])
        self.all_intervals = cis.time_util.convert_datetime_to_std_time(all_intervals)
        # daily intervals
        daily_intervals = np.array([self.date + temp_delta * i for i in range(1, self.intervals_per_day + 1)])
        self.daily_intervals = cis.time_util.convert_datetime_to_std_time(daily_intervals)

        # horizontal
        latrange = self.LATMAX - self.LATMIN
        lonrange = self.LONMAX - self.LONMIN
        self.nlat = int(latrange / self.HOR_RES)
        self.nlon = int(lonrange / self.HOR_RES)

        # create altitude level vector
        self.alt_levels = np.arange(self.ALTMIN, self.ALTMAX + self.ALTINTERVAL, self.ALTINTERVAL)

        # create latitude and longitude vectors of grid
        self.latgr = np.round(np.arange(self.LATMIN, self.LATMAX, self.HOR_RES), 4)  # need to round for float errors
        self.longr = np.round(np.arange(self.LONMIN, self.LONMAX, self.HOR_RES), 4)

        ### setup coord and data vectors ###

        # create empty arrays to store coords and variables of each file
        self.latv = []
        self.lonv = []
        self.altv = []
        self.timev = []
        self.var_dict = dict()
        self.agg_dict = dict()

        # init 3d variables
        for var in self.VARIABLES_3D:
            self.var_dict[var] = []
        # init 2d variables
        for var in self.VARIABLES_2D:
            self.var_dict[var] = []

    def load_files(self):
        """load L2 files for given date and concat coord and data vectors"""

        # load L2 file paths
        files = get_filepaths(self.date, dir_path=self.DIR_PATH)

        # load each file from disk and extract coord and data vectors
        for file in files:
            # 3d variable data
            dardar_3d = cis.read_data_list(file, self.VARIABLES_3D,
                                           product="DARDAR_CLOUD")  # cis.read_data(file, 'iwc', product="DARDAR_CLOUD")
            # coords
            self.latv.append(dardar_3d.coord("latitude").data)
            self.lonv.append(dardar_3d.coord("longitude").data)
            self.altv.append(dardar_3d.coord("altitude").data.astype(
                np.int32))  # cast as int, cause sometimes there are values like 16079.999
            self.timev.append(dardar_3d.coord("time").data)

            for var in dardar_3d:
                # some variable data come as masked data, some as normal ndarrays. Since the mask is not used in the correct way in the dardar data anyway, we only want the data vector of the masked array.
                if issubclass(type(var.data), np.ma.core.MaskedArray):
                    self.var_dict[var.var_name].append(var.data.data)
                else:
                    self.var_dict[var.var_name].append(var.data)

                    # 2d variable data
            # loading 2d variable data directly raises an error in the `post_process` method in cis. That's why we use `retrieve_raw_data`
            for var_2d in self.VARIABLES_2D:
                dardar_2d = cis.read_data(file, var_2d, product="DARDAR_CLOUD")
                data_2d = dardar_2d.retrieve_raw_data(dardar_2d._data_manager[0]).data

                # make sure shapes are correct
                assert data_2d.shape[0] == int(
                    dardar_3d[0].shape[0] / self.ALTLEVELS), "2d variable shape doesnt match 3d variable shape"
                self.var_dict[var_2d].append(data_2d)

    def concatenate_file_vectors(self):
        # concatenate data from files to one array for coords and variables
        self.latv = np.concatenate(self.latv)
        self.lonv = np.concatenate(self.lonv)
        self.altv = np.concatenate(self.altv)
        self.timev = np.concatenate(self.timev)

        for var_name, varv in self.var_dict.items():
            nan_value = -999.
            if var_name == "land_water_mask":
                nan_value = -9.
            elif var_name == "day_night_flag":
                nan_value = -1.
            else:
                nan_value = -999.
            v_con = np.concatenate(varv)  # concatenate variable vectors of each file
            v = np.where(v_con == nan_value, np.nan, v_con)  # replace -999. with nan
            self.var_dict[var_name] = v

    def grid_and_aggregate(self):
        """creates a evenly spaced grid and fills where available with aggregates of data vectors"""

        ### grid meta data ###

        ### find nearest neighbour gridcell for L2 coordinates ###

        # round coords to grid resolution
        self.latr = np.round((np.round(self.latv * (1 / self.HOR_RES)) * self.HOR_RES).astype('float64'), 4)
        self.lonr = np.round((np.round(self.lonv * (1 / self.HOR_RES)) * self.HOR_RES).astype('float64'), 4)
        # "round" to time interval. the right side of the interval is given as identifier for interval. e.g. 3am for 0-3am
        self.timer = np.array(self.all_intervals)[np.searchsorted(self.all_intervals, self.timev, side="left")]
        # 2d coordinate vectors
        nobs = int(self.latr.shape[0] / self.ALTLEVELS)
        self.latr_2d = dim_reduction(self.latr, nobs, self.ALTLEVELS)
        self.lonr_2d = dim_reduction(self.lonr, nobs, self.ALTLEVELS)
        self.timer_2d = dim_reduction(self.timer, nobs, self.ALTLEVELS)

        ### find coordinates with observations/realizations

        # retrieve lat lon, time combinations and how often they occur
        (self.lats_all, self.lons_all, self.times_all), counts_all = np.unique(np.array([self.latr, self.lonr, self.timer]),
                                                                               axis=1, return_counts=True)
        self.lats_all = np.round(self.lats_all.astype('float64'), 4)  # float rounding errors
        self.lons_all = np.round(self.lons_all.astype('float64'), 4)

        # retrieve lat lon, time combinations and how often they occur for observations with data
        data_mask = self.var_dict["iwc"] > 0
        (self.lats_data, self.lons_data, self.times_data), counts_data = np.unique(
            np.array([self.latr[data_mask], self.lonr[data_mask], self.timer[data_mask]]), axis=1,
            return_counts=True)  # lat/lon/time combo with iwc at at at least on altitude level
        self.lats_data = np.round(self.lats_data.astype('float64'), 4)  # float rounding errors
        self.lons_data = np.round(self.lons_data.astype('float64'), 4)

        ### aggregate observations to coordinates

        ## setup grid for 3d/2d variables
        means = np.empty(
            (self.nlon, self.nlat, self.ALTLEVELS, self.intervals_per_day))  ## ! watch out lat/lon reihenfolge
        means[:, :, :, :] = np.nan  # set all grid points to nan

        means_2d = np.empty((self.nlon, self.nlat, self.intervals_per_day))
        means_2d[:, :, :] = np.nan

        # set all grid points with observations to 0 - all other gridpoints are nan. i.e. only the swath is set to 0
        for lat, lon, timestamp in zip(self.lats_all, self.lons_all, self.times_all):
            if timestamp not in self.daily_intervals:
                # not from today
                continue

            if (lat < self.LATMIN) | (lat >= self.LATMAX) | (lon < self.LONMIN) | (lon >= self.LONMAX):
                # print("out of area")
                continue

            # get indices
            latidx = np.where(np.round(self.latgr, 4) == lat)[0][0]
            lonidx = np.where(np.round(self.longr, 4) == lon)[0][0]
            timeidx = np.where(timestamp == self.daily_intervals)[0][0]

            means[lonidx, latidx, :, timeidx] = np.zeros(shape=(self.ALTLEVELS,), dtype=np.float64)

        # deepcopy is quite compute intensive, consider create function that outsources creation of initial mean tensor
        # create mean arrays for all variables
        for var in self.var_dict:
            if var in self.VARIABLES_3D:
                self.agg_dict[var] = deepcopy(means)
            elif var in self.VARIABLES_2D:
                self.agg_dict[var] = deepcopy(means_2d)
            else:
                raise ValueError("variable {} is not specified".format(var))
        self.agg_dict["cloud_cover"] = deepcopy(means)

        ## calculate aggregates for each gridpoint with data
        for lon, lat, timestamp in zip(self.lons_data, self.lats_data, self.times_data):
            if timestamp not in self.daily_intervals:
                # not from today
                continue

            if (lat < self.LATMIN) | (lat >= self.LATMAX) | (lon < self.LONMIN) | (lon >= self.LONMAX):
                # print("out of area")
                continue

            # calc aggregate for each variable
            self.aggregate(lon, lat, timestamp)

    # aggregate helpers
    def aggregate(self, lon, lat, timestamp):
        # get lat/lon/time index in grid
        lonidx, latidx, timeidx = get_grid_idx(self.longr, self.latgr, self.daily_intervals, lon, lat, timestamp)
        data_vector_idxs = self.get_vector_idxs(lon, lat, timestamp)
        data_vector_idxs_2d = self.get_vector_idxs(lon, lat, timestamp, three_d=False)

        # in cloud mask: masks all values with iwc == 0
        in_cloud_mask = self.get_in_cloud_mask(data_vector_idxs)

        # calc aggregate for each variable
        for var in self.agg_dict:
            if var == "cloud_cover":
                # calc cloud coverage for grid cell (#observations with iwc / #all observations)
                d = self.get_observations("iwc", data_vector_idxs)
                nobs = d.shape[0]  # number of observations at this gridpoint
                iwc_obs = np.count_nonzero(d, axis=0)  # number of observations with iwc for each altitude level
                self.agg_dict[var][lonidx, latidx, :, timeidx] = iwc_obs / nobs
            elif var in self.VARIABLES_3D:
                # calc in cloud mean
                in_cloud_means = self.calc_in_cloud_means(var, data_vector_idxs, in_cloud_mask)
                self.agg_dict[var][lonidx, latidx, :, timeidx] = in_cloud_means
            elif var in self.VARIABLES_2D:
                isCategorical = False if var == 'Tropopause_Height' else True
                agg = self.calc_2d_agg(var, data_vector_idxs_2d, categorical=isCategorical)
                self.agg_dict[var][lonidx, latidx, timeidx] = agg
            else:
                raise ValueError("variable {} is not specified".format(var))

    def get_observations(self, var_name, data_vector_idxs):
        """get observations and reshape to (# observations, altitude levels)

        Args:
            var_name (str): variable to retrieve
            data_vector_idxs (np.ndarray): indices of matching observations
        """
        obs = self.var_dict[var_name][data_vector_idxs]

        if var_name in self.VARIABLES_3D:
            nobs = int(obs.shape[0] / self.ALTLEVELS)  # there are 419 altitude levels
            obs = obs.reshape(nobs, self.ALTLEVELS)
            obs = np.flip(obs,
                          axis=1)  # flip altitude levels so first index is lowest altitude level at each observation

        return obs

    def get_vector_idxs(self, lon, lat, time, three_d=True):
        """get indices of data vectors that match requested gridpoint/time combi

        Args:
            lon (float):
            lat (float):
            time (float):
            three_d (bool): if yes retrieve idx of 3d coord vector, otherwise from 2d coor vector


        Example:
        -------

        get_vector_idxs(180.0, 80.0, 152218.0)

        returns: array([False, False, False, ..., False, False, False])
        """
        if three_d:
            idxs = (lat == np.round(self.latr, 4)) & (lon == np.round(self.lonr, 4)) & (time == self.timer)
        else:
            # 2d coord idx
            idxs = (lat == np.round(self.latr_2d, 4)) & (lon == np.round(self.lonr_2d, 4)) & (time == self.timer_2d)

        return idxs

    def get_in_cloud_mask(self, data_vector_idxs):
        """True for masked values"""

        obs = self.get_observations("iwc", data_vector_idxs)
        in_cloud_mask = np.ma.masked_values(obs, 0.).mask  # create mask with all values that are 0 masked

        return in_cloud_mask

    def calc_in_cloud_means(self, var_name, data_vector_idxs, in_cloud_mask):
        """calculates in cloud mean (don't consider alt levels with observations with no iwc)

        if var_name is `iwc` determine in_cloud_mask based on variable data
        else use provided in_cloud_mask
        """
        obs = self.get_observations(var_name, data_vector_idxs)
        obs[in_cloud_mask] = np.nan
        in_cloud_mean = np.nanmean(obs, axis=0)

        return in_cloud_mean

    def calc_2d_agg(self, var_name, data_vector_idxs, categorical=False):
        """calculates aggregate for 2d variables

        Args:
            var_name (str): variable name
            data_vector_idxs (np.ndarray): vector with matching observations set to True
            categorical (bool): if True take value with highest occurance, else calculate mean

        Returns:
            int|float: aggregated value
        """
        obs = self.get_observations(var_name, data_vector_idxs)

        if categorical:
            vals, counts = np.unique(obs, return_counts=True)
            max_occurance = np.nanargmax(counts)
            agg = int(vals[max_occurance])
        else:
            agg = np.nanmean(obs)

        return agg


# grid helpers
def dim_reduction(vec_3d, nobs, alt_levels):
    """extracts coordinate vector of 2d variable from coordinate vector of 3d variable

    coordinate vector of 3d variables have shape (#observations*altitude levels,), i.e. always 419 consecutive values are equal
    for 2d variables we need shape (#observations,)

    Args:
        vec_3d (np.ndarray): coordinate vector of 3d variable
        nobs (int): # of observations
        alt_levels (int): # alt levels

    Returns
        coordinate vector of 2d variable
    """
    vec_2d = vec_3d.reshape(nobs, alt_levels)
    vec_2d = vec_2d[:, 0]  # we only need first value of every row

    assert vec_2d.shape[0] == nobs

    return vec_2d


def get_grid_idx(longr, latgr, daily_intervals, lon, lat, timestamp=None):
    """return idx of lon/lat/time combo in grid

    if time is not specified return only lon, lat
    
    Args:
        longr (np.array): lon grid vector
        latgr (np.array): lat grid vector
        daily_intervals (np.array): time grid vector
        lon (float):
        lat (float):
        timestamp (float): timestamp

    Returns:
        tuple (int,int,int): lonidx, latidx, timeidx

    """
    latidx = np.where(np.round(latgr, 4) == lat)[0][0]
    lonidx = np.where(np.round(longr, 4) == lon)[0][0]

    if timestamp:
        timeidx = np.where(timestamp == daily_intervals)[0][0]
        return lonidx, latidx, timeidx
    else:
        return lonidx, latidx


# io helpers
def get_day_files(date, dir_path):
    """returns sorted array of filepaths of given day

    Args:
        date (datetime.datetime):
        dir_path (str): path to DARDAR Files

    Returns:
        list: list of filepaths
    """
    date_str = date.strftime("%Y_%m_%d")
    date_dir = os.path.join(dir_path, date_str, "*.hdf")
    filepaths = glob.glob(date_dir)
    filepaths = np.sort(filepaths)  # sort files in ascending order

    return filepaths

def get_filepaths(date, dir_path):
    """load filepaths for given date + last file from previous day

    Args:
        date (datetime.datetime): date
        dir_path (str): path to DARDAR Files

    Returns:
        list|None: list of filepaths to load. If no files exist for that day return None
    """
    # get filepaths for day and add last file from previous day
    filepaths = get_day_files(date, dir_path)
    prev_day_paths = get_day_files(date + datetime.timedelta(days=-1), dir_path)

    if filepaths.size == 0:
        print("no data available")
        return None

    if prev_day_paths.size == 0:
        print("no data for prev day available")
    else:
        filepaths = np.insert(filepaths, 0, prev_day_paths[-1])

    return filepaths
