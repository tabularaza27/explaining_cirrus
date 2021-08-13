import numpy as np
import pandas as pd
import xarray as xr
import datetime
import cis
import os
import glob
import gc
import logging
import sys
import multiprocessing as mp
from copy import deepcopy
from cis import time_util
from dateutil.relativedelta import relativedelta
import warnings
import socket
from src.preprocess.helpers.io_helpers import save_file
from src.preprocess.helpers.io_helpers import get_filepaths
from src.preprocess.helpers.io_helpers import exists
from src.preprocess.helpers.constants import DARDAR_INCOMING_DIR, DARDAR_GRIDDED_DIR

warnings.filterwarnings('ignore')  # because of zero/nan divide warnings

# setup logger - see: https://docs.python.org/3/howto/logging-cookbook.html
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
# log file
fh = logging.FileHandler("{}.log".format("dardar_nice"), mode="w")
fh.setLevel(logging.DEBUG)
# console output
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# format
formatter = logging.Formatter('%(asctime)-15s %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add handlers
logger.addHandler(fh)
logger.addHandler(ch)

# meta data for spatial resolution
# horizontal
LATMIN = 0  # 0
LATMAX = 60  # 60
LONMIN = -75  # -75
LONMAX = -15  # -15
HOR_RES = 0.25  # horizontal resolution [degree]
TEMP_RES = "1H"  # temporal resolution in hours

# vertical; vertical resolution is kept at 60m for now
ALTMIN = 0
ALTMAX = 25080
ALTINTERVAL = 60
ALTLEVELS = 419

CONT = "continuous"
CAT = "categorical"

# variables that will be gridded

CONT_VAR_NAMES = ['ps',
                  'plev',
                  'ta',
                  'iwc',
                  'iwc_error',
                  'reffcli',
                  'icnc_5um',
                  'icnc_100um',
                  'icnc_5um_error',
                  'icnc_100um_error',
                  'dz_top',
                  ]

CAT_VAR_NAMES = [
    'land_water_mask',
    'clm',
    'clm_v2',
    'nightday_flag',
    'mixedphase_flag',
    'instrument_flag'
]

class DardarNiceGrid:
    def __init__(self, date, time_range="day"):
        """

        Args:
            date (datetime.datetime):
            time_range (str): day | month
        """
        self.time_range = time_range
        self.start_date = date

        if time_range == "day":
            self.end_date = date + relativedelta(days=1)
        elif time_range == "month":
            self.end_date = date + relativedelta(months=1)
        else:
            raise ValueError("Specify correct timerange. got {}".format(time_range))

        logger.info("Created Gridder with following specs {}".format(self.get_specs()))
        self.l2_ds = load_files(date, self.time_range)
        if self.l2_ds is None:
            raise NoDataError(
                "No Level 2 data found in specified source dir {} for Gridder with specs: {}".format(DARDAR_INCOMING_DIR,
                                                                                                     self.get_specs()))
        logger.info("loaded l2 files")

        self.l2_ds = remove_bad_quality_data(self.l2_ds)
        logger.info("removed bad quality retrievals")

        self.l3_ds = create_empty_grid(start_date=str(self.start_date), end_date=str(self.end_date))
        logger.info("created empty grid")

        # get combinations of lat/lon/time  with observations
        (self.lats_all, self.lons_all, self.times_all), counts_all = np.unique(
            np.array([self.l2_ds.latr.values, self.l2_ds.lonr.values, self.l2_ds.timer.values]),
            axis=1, return_counts=True)
        self.lats_all = np.round(self.lats_all.astype('float64'), 4)  # float rounding errors
        self.lons_all = np.round(self.lons_all.astype('float64'), 4)

        # obervation has data if at least one vertical level contains iwc>0
        iwc = self.l2_ds["iwc"].values
        vertical_sum = np.nansum(iwc, 1)
        data_mask = vertical_sum > 0

        # retrieve lat lon, time combinations and how often they occur for observations with data
        (self.lats_data, self.lons_data, self.times_data), counts_data = np.unique(
            np.array([self.l2_ds.latr.values[data_mask], self.l2_ds.lonr.values[data_mask],
                      self.l2_ds.timer.values[data_mask]]),
            axis=1,
            return_counts=True)  # lat/lon/time combo with iwc at at at least on altitude level
        self.lats_data = np.round(self.lats_data.astype('float64'), 4)  # float rounding errors
        self.lons_data = np.round(self.lons_data.astype('float64'), 4)

        # l3_ds with empty mean tensors for each data variable
        self.prep_l3()
        logger.info("filled l3 files with empty tensors")

    def get_specs(self):
        """returns string with specifications of this gridder"""
        specs_string = "Start Date:{}  End Date: {} Time Range: {} ".format(self.start_date,
                                                                            self.end_date,
                                                                            self.time_range)
        return specs_string

    def prep_l3(self):
        """fill level 3 data set with empty tensors for each variable.

        all gridpoints are set to np.nan except for gridpoints with observations in the level 2 data
        (i.e. the satellite swath) are set to 0"""

        # create empty means tensor
        means = np.empty(
            (self.l3_ds.sizes["lon"], self.l3_ds.sizes["lat"], self.l3_ds.sizes["lev"], self.l3_ds.sizes["time"]))
        means[:, :, :, :] = np.nan  # set all grid points to nan

        # set all grid points with observations to 0 - all other gridpoints are nan. i.e. only the swath is set to 0
        for lat, lon, timestamp in zip(self.lats_all, self.lons_all, self.times_all):
            cf_timestamp = cis.time_util.convert_std_time_to_datetime(timestamp)
            if cf_timestamp not in self.l3_ds.time:
                # not from today
                continue

            if (lat < LATMIN) | (lat >= LATMAX) | (lon < LONMIN) | (lon >= LONMAX):
                # logger.info("out of area")
                continue

            # get indices
            lonidx, latidx, timeidx = get_grid_idx(lon, lat, cf_timestamp, self.l3_ds)
            means[lonidx, latidx, :, timeidx] = np.zeros(shape=(self.l3_ds.sizes["lev"],), dtype=np.float64)

        # for 2d variables, i.e. without height coord
        means_2d = means[:, :, 0, :]

        var_dict = {}
        for var, var_info in self.l2_ds.items():
            if var not in CONT_VAR_NAMES + CAT_VAR_NAMES:
                continue

            attrs = var_info.attrs
            # add variable type
            if var in CONT_VAR_NAMES:
                attrs["var_type"] = CONT
            elif var in CAT_VAR_NAMES:
                attrs["var_type"] = CAT

            # set coordinates
            if var_info.dims == ('time',):
                coords = ["lon", "lat", "time"]
                values = deepcopy(means_2d)
            elif var_info.dims == ('time', 'height'):
                coords = ["lon", "lat", "lev", "time"]
                values = deepcopy(means)
            else:
                raise ValueError("{} are unknown coords".format(var_info.dims))

            # add to dict
            var_dict[var] = (coords, values, attrs)

        # add cloud cover as variable
        cc_attrs = {"units": "1",
                    "long_name": "cloud cover per vertical coordinate",
                    "description": "percentage of observations in gridbox that had observations containing iwc",
                    "var_type": CONT,
                    "valid_range": [0, 1]
                    }

        var_dict["cloud_cover"] = (["lon", "lat", "lev", "time"], deepcopy(means), cc_attrs)

        self.l3_ds = self.l3_ds.assign(var_dict)

    def aggregate(self):
        """aggregates whole lon/lat/timestamp data for each gridbox that contains at least
         one altitude level with positive iwc"""

        for lon, lat, timestamp in zip(self.lons_data, self.lats_data, self.times_data):
            cf_timestamp = cis.time_util.convert_std_time_to_datetime(timestamp)
            if cf_timestamp not in self.l3_ds.time:
                # not from today
                continue

            if (lat < LATMIN) | (lat >= LATMAX) | (lon < LONMIN) | (lon >= LONMAX):
                # logger.info("out of area")
                continue

            # calc aggregate for each variable
            logger.debug("Grid: {} , {} , {}".format(lon, lat, timestamp))
            self.aggregate_gridbox(lon, lat, timestamp)

    def aggregate_gridbox(self, lon, lat, timestamp):
        """aggregates l2 data for given lon/lat/timestamp

        Args:
            lon (float):
            lat (float):
            timestamp (float): std time

        Returns:

        """
        # idx in l3 grid
        lonidx, latidx, timeidx = get_grid_idx(lon, lat, cis.time_util.convert_std_time_to_datetime(timestamp),
                                               self.l3_ds)
        # idxs of data in l2 ds
        data_vector_idxs = get_data_vector_idxs(lon, lat, timestamp, self.l2_ds)
        in_cloud_mask = get_in_cloud_mask(self.l2_ds, data_vector_idxs)  # needed for cont 3d aggregation

        for var_name in self.l3_ds:
            dims = len(self.l3_ds[var_name].dims)
            if var_name == "cloud_cover":
                obs = get_observations(self.l2_ds, "iwc", 4, data_vector_idxs)
                nobs = obs.shape[0]
                iwc_obs = np.count_nonzero(obs, axis=0)  # observations with iwc >=0
                agg = iwc_obs / nobs
            else:
                agg = self.calc_in_cloud_agg(var_name, data_vector_idxs, in_cloud_mask)

            # write result in l3 grid
            if dims == 4:
                self.l3_ds[var_name][lonidx, latidx, :, timeidx] = agg
            else:
                self.l3_ds[var_name][lonidx, latidx, timeidx] = agg

    def calc_in_cloud_agg(self, var_name, data_vector_idxs, in_cloud_mask=None):
        """

        Args:
            in_cloud_mask (np.ndarray): only needed for continuous 3d
        """
        dims = len(self.l3_ds[var_name].dims)
        var_type = self.l3_ds[var_name].attrs["var_type"]
        # get observations
        obs = get_observations(self.l2_ds, var_name, dims, data_vector_idxs)

        # continuous 3d
        if dims == 4 and var_type == CONT:
            if in_cloud_mask is None:
                raise ValueError("provide a `in_cloud_mask` for cont 3d variables")

            # for environmental variables we want the average of the whole gridcell and especially all levels
            # not the incloud mean
            if var_name in ["plev", "ta"]:
                agg = np.nanmean(obs, axis=0)

            # calculate in-cloud means
            else:
                obs_no_data = np.all(obs == 0, 0)  # save inidices that have no data
                obs[in_cloud_mask] = np.nan  # set all observations with 0 iwc to nan, so we can apply nanmean
                agg = np.nanmean(obs, axis=0)
                agg[obs_no_data] = 0.  # replace nan with 0 where all observations were 0

        # categorical 3d
        elif dims == 4 and var_type == CAT:
            # catergorical "mean" is value with max occurance
            agg = np.empty(obs.shape[1], )
            for i in range(obs.shape[1]):
                vals, counts = np.unique(obs[:, i], return_counts=True)
                max_occurance = np.nanargmax(counts)
                agg[i] = vals[max_occurance]

        # continuous 2d
        elif dims == 3 and var_type == CONT:
            agg = np.nanmean(obs)

        # categorical 2d
        elif dims == 3 and var_type == CAT:
            # catergorical "mean" is value with max occurance
            vals, counts = np.unique(obs, return_counts=True)
            max_occurance = np.nanargmax(counts)
            agg = vals[max_occurance]

        else:
            raise ValueError(
                "variable dims needs to be in [3,4] and variable type in ['categorical','continuous']. Is dim: {}; type: {}".format(
                    dims, var_type))

        return agg


class NoDataError(Exception):
    """Raised when no L2 data exists"""
    pass


def create_empty_grid(start_date,
                      end_date,
                      freq=TEMP_RES,
                      lonmin=LONMIN,
                      lonmax=LONMAX,
                      latmin=LATMIN,
                      latmax=LATMAX,
                      hor_res=HOR_RES,
                      altmin=ALTMIN,
                      altmax=ALTMAX,
                      alt_res=ALTINTERVAL,
                      ):
    """create empty grid for l3 dataset

    Args:
        start_date:
        end_date:
        freq:
        lonmin:
        lonmax:
        latmin:
        latmax:
        hor_res:
        altmin:
        altmax:
        alt_res:

    Returns:

    """
    # load dardar nice L3 dataset to copy the coordinate attributes
    dardar_l3_files = glob.glob(os.path.join(ROOT_DIR, 'wolke_scratch/kjeggle/DARDAR_NICE/L3/2006/*'))
    dar_nice = xr.open_dataset(dardar_l3_files[3])

    timegr = xr.cftime_range(start=start_date, end=end_date, freq=freq, closed="left")  # closed to the left 00 - 23
    latgr = np.round(np.arange(latmin, latmax, hor_res), 4)  # need to round for float errors
    longr = np.round(np.arange(lonmin, lonmax, hor_res), 4)
    altgr = np.flip(np.arange(altmin, altmax + alt_res, alt_res))  # highest hight at index 0

    ds = xr.Dataset(
        coords=dict(
            lon=(["lon"], longr, dar_nice.lon.attrs.copy()),
            lat=(["lat"], latgr, dar_nice.lat.attrs.copy()),
            lev=(["lev"], altgr, {"units": "m", "axis": "Z", "long_name": "Altitude Level"}),
            time=(["time"], timegr, {"axis": "T", "long_name": "time"})
        )
    )

    return ds


def get_grid_idx(lon, lat, timestamp, ds):
    """returns indices in grid"""

    lonidx = np.where(np.round(ds.lon, 4) == lon)[0][0]
    latidx = np.where(np.round(ds.lat, 4) == lat)[0][0]
    timeidx = np.where(timestamp == ds.time)[0][0]

    return lonidx, latidx, timeidx


### load l2 data + helpers ###

def add_timestamp_coord(ds):
    """add time timestamp as coordinate to dataset"""
    base_time = ds.base_time.values  # reference time
    time_deltas = pd.to_timedelta(ds.dtime.values, 's')  # dtime
    pixel_time = base_time + time_deltas  # timestamp
    ds = ds.assign_coords({'time': pixel_time})
    return ds


def sel_pos_heights(ds):
    """returns dataset for positive heights only"""
    return ds.sel(height=slice(25080, 0))


def preproc_height_coord(ds):
    """casts height coord as int
    cause some height values are floats"""
    ds["height"] = ds.height.values.astype(np.int)
    return ds


def preprocess(ds):
    """preprocess dataset

    function is passed as argument in xr.open_mfdataset()
    """
    ds = add_timestamp_coord(ds)
    ds = sel_pos_heights(ds)
    ds = preproc_height_coord(ds)
    return ds


def load_files(date, time_range="day"):
    """loads l2 files of dardar nice dataset for given date

    Args:
        date (datetime.datetime):
        time_range (str): day|month. If month, then load all files for the month. date is still specified as "%Y_%m_%d"

    Returns:

    """
    files = get_filepaths(date, DARDAR_INCOMING_DIR, file_format="nc", time_range=time_range)

    if files is None:
        return None

    # use preprocess to add timestamp coordinate (this is required so we can cocatenate)
    ds = xr.open_mfdataset(files, preprocess=preprocess, concat_dim="time")

    # create data variables with rounded lat/lon/time
    ds = ds.assign(latr=lambda x: np.round((np.round(x.lat * (1 / HOR_RES)) * HOR_RES).astype('float64'), 4))
    ds = ds.assign(lonr=lambda x: np.round((np.round(x.lon * (1 / HOR_RES)) * HOR_RES).astype('float64'), 4))
    ds = ds.assign(timer=ds.time.dt.round(TEMP_RES))

    # convert timer to std datetime, so it can be used in np.unique
    # a bit over complicated, but havent found better way so far
    a = pd.to_datetime(ds["timer"].values)  # pandas datetime
    b = a.to_pydatetime()  # python datetime
    c = cis.time_util.convert_datetime_to_std_time(b)  # convert to std time, necessary for np.unique
    ds["timer"].values = c

    return ds


def remove_bad_quality_data(l2_ds):
    """remove retrievals with bad quality flags

    The number of iteration is used here as a proxy to avoid any strong influence of a priori assumptions on the
    retrievals; cloud products associated with niter < 2 are excluded from this study [Sourdeval et al. 2018]

    quality Flag is `iteration_flag`

    Args:
        l2_ds:

    Returns:
        xr.Dataset: l2 data set with only good quality retrievals
    """
    good_mask = l2_ds.iteration_flag == 1
    good_l2_ds = l2_ds.isel(time=good_mask)

    return good_l2_ds


def get_data_vector_idxs(lon, lat, timestamp, l2_ds):
    """get indices of data vectors that match requested gridpoint/time combi

        Args:
            lon (float):
            lat (float):
            timestamp (float): standard time
            l2_ds (xr.Dataset): level 2 dataset

        Example:
        -------

        get_vector_idxs(180.0, 80.0, 152218.0)

        returns: array([False, False, False, ..., False, False, False])
    """

    data_vector_idxs = (lat == np.round(l2_ds.latr, 4)) & (lon == np.round(l2_ds.lonr, 4)) & (timestamp == l2_ds.timer)
    return data_vector_idxs


def get_observations(l2_ds, var_name, var_dims, data_vector_idxs):
    """get observations for variable from l2 data set for specifies indices

    Args:
        l2_ds (xr.Dataset): l2 data set
        var_name (str):
        var_dims (int): 4 if variable has three spatial and one temporal dimension, 3 for 2 spatial dimenstions
        data_vector_idxs (np.array): array with True/False values indicating which values to return

    Returns:
        np.array: observations of
    """
    # get observations
    if var_dims == 4:
        obs = l2_ds[var_name][data_vector_idxs, :].values  # 3d
    elif var_dims == 3:
        obs = l2_ds[var_name][data_vector_idxs].values  # 3d
    return obs


def get_in_cloud_mask(l2_ds, data_vector_idxs):
    """returns vector that has all values masked that are 0.

    True for masked values"""

    obs = get_observations(l2_ds, "iwc", 4, data_vector_idxs)
    in_cloud_mask = np.ma.masked_values(obs, 0.).mask  # create mask with all values that are 0 masked

    return in_cloud_mask


### run gridding ###

def grid_one_day(date):
    """runs gridding process for DARDAR NICE L2 data for 1 day

    Args:
        date (datetime.datetime):

    Returns:
        None if file doesnt exist or gridded file already exists. True if successfully gridded

    """
    if exists(date, "dardar_nice", TARGET_DIR):
        logger.info("File already exists for: {}".format(date))
        return None
    logger.info("Start Gridding: {}".format(date))
    try:
        dn = DardarNiceGrid(date, "day")
    except NoDataError as err:
        logger.info(err)
        print(err)
    dn.aggregate()
    save_file(TARGET_DIR, "dardar_nice", dn.l3_ds, date=date)
    logger.info("saved file")
    gc.collect()  # garbage collection
    return True


def run_gridding(start_date, end_date, n_workers=10):
    """runs gridding process for DARDAR CLOUD L2 data for given time period in parallel, starts one gridding process
    per day

    Args:
        start_date (str): YYYY-mm-dd
        end_date (str): YYYY-mm-dd
        n_workers (int): nuber of cpus to use for gridding

    """

    pool = mp.Pool(n_workers)
    logger.info("{} {} {} {}".format(hostname, ROOT_DIR, DARDAR_INCOMING_DIR, TARGET_DIR))
    logger.info("++++++++++++++ Start new gridding process with {} workers ++++++++++++++".format(n_workers))
    logger.info("gridding period: {} - {}".format(start_date, end_date))
    daterange = pd.date_range(start=start_date, end=end_date)
    for date in daterange:
        date = date.to_pydatetime()
        pool.apply_async(grid_one_day, args=(date,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly
    if len(sys.argv) == 4:
        run_gridding(start_date=sys.argv[1], end_date=sys.argv[2], n_workers=int(sys.argv[3]))
    elif len(sys.argv) == 3:
        run_gridding(start_date=sys.argv[1], end_date=sys.argv[2])
    else:
        raise ValueError("Provide valid arguments. E.g.: python dardar_nice_grid '2016-01-01' '2016-01-31' <#workers>")
