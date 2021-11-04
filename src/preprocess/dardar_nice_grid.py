import numpy as np
import pandas as pd
import xarray as xr
import datetime
import cis
import gc
import logging
import sys
import multiprocessing as mp
from copy import deepcopy
from cis import time_util
from dateutil.relativedelta import relativedelta
import warnings
from src.preprocess.dardar_l2_preproc import run_l2_preproc
from src.preprocess.helpers.io_helpers import save_file
from src.preprocess.helpers.io_helpers import get_filepaths
from src.preprocess.helpers.io_helpers import exists
from src.preprocess.helpers.constants import DARDAR_INCOMING_DIR, DARDAR_GRIDDED_DIR
from src.preprocess.helpers.common_helpers import custom_mode
from src.scaffolding.scaffolding import get_config, get_data_product_dir

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

# todo clean up
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
                  'dz_top_v2',
                  'cloud_cover'
                  ]

CAT_VAR_NAMES = [
    'land_water_mask',
    'clm',
    'clm_v2',
    'nightday_flag',
    'instrument_flag',
    'cloud_layer',
    'cloud_top',
    'cloud_bottom',
    'liquid_origin',
    'data_mask'
]


class DardarNiceGrid:
    def __init__(self, date, config_id, time_range="day"):
        """
        Args:
            date (datetime.datetime):
            config_id (str): config determines resolutions and location of load/save directories
            time_range (str): day | month
        """
        # get configuration file
        self.config = get_config(config_id)

        # define spatial/temporal extent and resolutions
        # horizontal
        self.latmin = self.config["latmin"]
        self.latmax = self.config["latmax"]
        self.lonmin = self.config["lonmin"]
        self.lonmax = self.config["lonmax"]
        self.hor_res = self.config["horizontal_resolution"]  # horizontal resolution [degree]
        self.temp_res = self.config["temporal_resolution"]  # temporal resolution in hours

        # vertical; vertical resolution is kept at 60m for now
        self.altmin = self.config["altitude_min"]
        self.altmax = self.config["altitude_max"]
        self.layer_thickness = self.config["layer_thickness"]

        # set start date and time range
        self.time_range = time_range
        self.start_date = date

        if time_range == "day":
            self.end_date = date + relativedelta(days=1)
        elif time_range == "month":
            self.end_date = date + relativedelta(months=1)
        else:
            raise ValueError("Specify correct timerange. got {}".format(time_range))

        logger.info("Created Gridder with following specs {}".format(self.get_specs()))
        self.l2_ds = load_files(date=date, hor_res=self.hor_res, temp_res=self.temp_res, time_range=self.time_range)
        if self.l2_ds is None:
            raise NoDataError(
                "No Level 2 data found in specified source dir {} for Gridder with specs: {}".format(
                    DARDAR_INCOMING_DIR,
                    self.get_specs()))
        logger.info("loaded l2 files")

        self.l2_ds = remove_bad_quality_data(self.l2_ds)
        logger.info("removed bad quality retrievals")

        # filter feature engineer and regrid l2 dataset
        self.l2_ds = run_l2_preproc(self.l2_ds, self.altmax, self.altmin, layer_thickness=self.layer_thickness)

        self.l3_ds = create_empty_grid(start_date=str(self.start_date),
                                       end_date=str(self.end_date),
                                       temp_res=self.temp_res,
                                       lonmin=self.lonmin,
                                       lonmax=self.lonmax,
                                       latmin=self.latmin,
                                       latmax=self.latmax,
                                       hor_res=self.hor_res,
                                       altmin=self.altmin,
                                       altmax=self.altmax,
                                       layer_thickness=self.layer_thickness)

        logger.info("created empty grid")

        # get combinations of lat/lon/time  with observations
        # todo filter for lats/lons of given domain before to reduce overhead
        (self.lats_all, self.lons_all, self.times_all), counts_all = np.unique(
            np.array([self.l2_ds.latr.values, self.l2_ds.lonr.values, self.l2_ds.timer.values]),
            axis=1, return_counts=True)
        self.lats_all = np.round(self.lats_all.astype('float64'), 4)  # float rounding errors
        self.lons_all = np.round(self.lons_all.astype('float64'), 4)

        # obervation has data if at least one vertical level contains cloud cover > 0
        data_mask = (self.l2_ds.data_mask == 1).values
        # iwc = self.l2_ds["iwc"].values
        # vertical_sum = np.nansum(iwc, 1)
        # data_mask = vertical_sum > 0

        # retrieve lat lon, time combinations and how often they occur for observations with data
        (self.lats_data, self.lons_data, self.times_data), counts_data = np.unique(
            np.array([self.l2_ds.latr.values[data_mask], self.l2_ds.lonr.values[data_mask],
                      self.l2_ds.timer.values[data_mask]]),
            axis=1,
            return_counts=True)  # lat/lon/time combo with iwc at at at least on altitude level
        self.lats_data = np.round(self.lats_data.astype('float64'), 4)  # float rounding errors
        self.lons_data = np.round(self.lons_data.astype('float64'), 4)

        logger.info("{} columns with data".format(len(self.lats_data)))

        # l3_ds with empty mean tensors for each data variable
        self.prep_l3()
        logger.info("filled l3 files with empty tensors")

    def get_specs(self):
        """returns string with specifications of this gridder"""
        specs_string = "Start Date:{}  End Date: {} Time Range: {} \n" \
                       "Config: {} ".format(self.start_date,
                                            self.end_date,
                                            self.time_range,
                                            self.config)
        return specs_string

    def prep_l3(self):
        """fill level 3 data set with empty tensors for each variable.

        all gridpoints are set to np.nan except for gridpoints with observations in the level 2 data
        (i.e. the satellite swath) are set to 0"""

        # create empty means tensor
        means = np.empty(
            (self.l3_ds.sizes["lon"], self.l3_ds.sizes["lat"], self.l3_ds.sizes["lev"], self.l3_ds.sizes["time"]))
        means[:, :, :, :] = np.nan  # set all grid points to nan

        # for 2d variables, i.e. without height coord
        means_2d = means[:, :, 0, :]

        # create observation mask variable
        observation_mask = np.zeros((self.l3_ds.sizes["lon"], self.l3_ds.sizes["lat"], self.l3_ds.sizes["time"]))
        for lat, lon, timestamp in zip(self.lats_all, self.lons_all, self.times_all):
            cf_timestamp = cis.time_util.convert_std_time_to_datetime(timestamp)
            if cf_timestamp not in self.l3_ds.time:
                # not from today
                continue

            if (lat < self.latmin) | (lat >= self.latmax) | (lon < self.lonmin) | (lon >= self.lonmax):
                # logger.info("out of area")
                continue

            # get indices
            lonidx, latidx, timeidx = get_grid_idx(lon, lat, cf_timestamp, self.l3_ds)
            observation_mask[lonidx, latidx, timeidx] = 1

        # create data variables
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

        # add observation mask as variable
        om_attrs = {"units": "1",
                    "long_name": "flags atmospheric columns with satellite observations",
                    "description": "1 if atmospheric column was overflown by satellite",
                    "var_type": CAT,
                    "valid_range": [0, 1]
                    }

        var_dict["observation_mask"] = (["lon", "lat", "time"], observation_mask, om_attrs)

        # add cloud cover as variable
        # cc_attrs = {"units": "1",
        #             "long_name": "cloud cover per vertical coordinate",
        #             "description": "percentage of observations in gridbox that had observations containing iwc",
        #             "var_type": CONT,
        #             "valid_range": [0, 1]
        #             }
        #
        # var_dict["cloud_cover"] = (["lon", "lat", "lev", "time"], deepcopy(means), cc_attrs)

        self.l3_ds = self.l3_ds.assign(var_dict)

    def aggregate(self):
        """aggregates whole lon/lat/timestamp data for each gridbox that contains at least
         one altitude level with positive iwc"""

        for lon, lat, timestamp in zip(self.lons_data, self.lats_data, self.times_data):
            cf_timestamp = cis.time_util.convert_std_time_to_datetime(timestamp)
            if cf_timestamp not in self.l3_ds.time:
                # not from today
                continue

            if (lat < self.latmin) | (lat >= self.latmax) | (lon < self.lonmin) | (lon >= self.lonmax):
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
        l2_time_indexes = get_data_vector_idxs(lon, lat, timestamp, self.l2_ds)
        timestamp_array = l2_time_indexes.where(l2_time_indexes == True, drop=True).time.values

        # select timestamps
        grid_cell = self.l2_ds.sel(time=timestamp_array)

        # in_cloud_mask = get_in_cloud_mask(self.l2_ds, data_vector_idxs)  # needed for cont 3d aggregation

        # cloud_cover weighted mean for continuous variables
        cc_weighted_mean = (grid_cell[CONT_VAR_NAMES] * grid_cell["cloud_cover"]).sum(dim="time", keep_attrs=True) / grid_cell.cloud_cover.sum(
        dim="time", keep_attrs=True)
        cc_weighted_mean = cc_weighted_mean.drop_vars(["cloud_cover", "plev", "ta"])

        # normal mean for cloud cover, plev, ta
        cc_mean = grid_cell[["cloud_cover", "plev", "ta"]].mean(dim="time", keep_attrs=True)

        # mode for flag variables + categorical variables (drop cloud masks, as they are encoded in cloud cover)
        hor_mode = grid_cell[CAT_VAR_NAMES].reduce(custom_mode, dim="time", keep_attrs=True)

        hor_agg_merge = xr.merge([cc_weighted_mean, cc_mean, hor_mode], compat="override")
        hor_agg_merge = hor_agg_merge.load()

        for var_name in self.l3_ds:
            dims = len(self.l3_ds[var_name].dims)
            # if var_name == "cloud_cover":
            #     obs = get_observations(self.l2_ds, "iwc", 4, data_vector_idxs)  # todo cloudmask
            #     nobs = obs.shape[0]
            #     iwc_obs = np.count_nonzero(obs, axis=0)  # observations with iwc >=0 # todo cloudmask
            #     agg = iwc_obs / nobs
            # else:
            #     agg = self.calc_in_cloud_agg(var_name, data_vector_idxs, in_cloud_mask)

            # write result in l3 grid
            if dims == 4:
                self.l3_ds[var_name][lonidx, latidx, :, timeidx] = hor_agg_merge[var_name].values
            else:
                self.l3_ds[var_name][lonidx, latidx, timeidx] = hor_agg_merge[var_name].values

    def calc_in_cloud_agg(self, var_name, data_vector_idxs, in_cloud_mask=None):
        """

        Args:
            var_name (str)
            data_vector_idxs (np.array): array with True/False values indicating which values to return. shape: (n_obs, )
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
                      temp_res,
                      lonmin,
                      lonmax,
                      latmin,
                      latmax,
                      hor_res,
                      altmin,
                      altmax,
                      layer_thickness
                      ):
    """create empty grid for l3 dataset

    Args:
        start_date (str or cf.datetime):
        end_date (str or cf.datetime):
        temp_res (str): e.g. 1H for hourly
        lonmin (int):
        lonmax (int):
        latmin (int):
        latmax (int):
        hor_res (float): horizontal resolutin in degrees
        altmin (int):
        altmax (int):
        layer_thickness (int): thickness of atmospheric layer

    Returns:

    """
    timegr = xr.cftime_range(start=start_date, end=end_date, freq=temp_res, closed="left")  # closed to the left 00 - 23
    latgr = np.round(np.arange(latmin, latmax, hor_res), 4)  # need to round for float errors
    longr = np.round(np.arange(lonmin, lonmax, hor_res), 4)
    altgr = np.flip(np.arange(altmin, altmax + layer_thickness, layer_thickness))  # highest hight at index 0

    ds = xr.Dataset(
        coords=dict(
            lon=(["lon"], longr, {'units': 'degree_east',
                                  'standard_name': 'longitude',
                                  'valid_range': [-180., 180.],
                                  'axis': 'X'}),
            lat=(["lat"], latgr, {'units': 'degree_north',
                                  'standard_name': 'latitude',
                                  'valid_range': [-90., 90.],
                                  'axis': 'Y'}),
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


def load_files(date, hor_res, temp_res, time_range="day"):
    """loads l2 files of dardar nice dataset for given date

    Args:
        date (datetime.datetime):
        hor_res (float): horizontal resolution in degrees
        temp_res (str): e.g. 1H for hourly
        time_range (str): day|month. If month, then load all files for the month. date is still specified as "%Y_%m_%d"

    Returns:

    """
    files = get_filepaths(date, DARDAR_INCOMING_DIR, file_format="nc", time_range=time_range)  # todo replace global

    if files is None:
        return None

    # use preprocess to add timestamp coordinate (this is required so we can cocatenate)
    ds = xr.open_mfdataset(files, preprocess=preprocess, concat_dim="time")

    # create data variables with rounded lat/lon/time
    ds = ds.assign(latr=lambda x: np.round((np.round(x.lat * (1 / hor_res)) * hor_res).astype('float64'),
                                           4))
    ds = ds.assign(lonr=lambda x: np.round((np.round(x.lon * (1 / hor_res)) * hor_res).astype('float64'),
                                           4))
    ds = ds.assign(timer=ds.time.dt.round(temp_res))

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

    # todo use correct cloud mask

    True for masked values"""

    obs = get_observations(l2_ds, "iwc", 4, data_vector_idxs)
    in_cloud_mask = np.ma.masked_values(obs, 0.).mask  # create mask with all values that are 0 masked

    return in_cloud_mask


### run gridding ###

def grid_one_day(date, config_id):
    """runs gridding process for DARDAR NICE L2 data for 1 day

    # todo consider case where temporal resolution is larger then 1 day

    Args:
        date (datetime.datetime):
        config_id (str) config determines resolutions and location of load/save directories

    Returns:
        None if file doesnt exist or gridded file already exists. True if successfully gridded

    """
    target_dir = get_data_product_dir(config_id, DARDAR_GRIDDED_DIR)
    if exists(date, "dardar_nice", target_dir):
        logger.info("File already exists for: {}".format(date))
        return None
    logger.info("Start Gridding: {}".format(date))
    try:
        dn = DardarNiceGrid(date=date, config_id=config_id, time_range="day")
    except NoDataError as err:
        logger.info(err)
        print(err)
    dn.aggregate()
    save_file(target_dir, "dardar_nice", dn.l3_ds, date=date)
    logger.info("saved file")
    gc.collect()  # garbage collection
    return True


def run_gridding(start_date, end_date, config_id, n_workers=10):
    """runs gridding process for DARDAR CLOUD L2 data for given time period in parallel, starts one gridding process
    per day

    Args:
        start_date (str): YYYY-mm-dd
        end_date (str): YYYY-mm-dd
        config_id (str) config determines resolutions and location of load/save directories
        n_workers (int): nuber of cpus to use for gridding

    """
    # todo check if config_id is valid
    pool = mp.Pool(n_workers)
    logger.info(
        "++++++++++++++ Start new gridding process for config {} with {} workers ++++++++++++++".format(config_id,
                                                                                                        n_workers))
    logger.info("gridding period: {} "
                "- {}".format(start_date, end_date))
    daterange = pd.date_range(start=start_date, end=end_date)
    for date in daterange:
        date = date.to_pydatetime()
        pool.apply_async(grid_one_day, args=(date, config_id,))
    pool.close()
    pool.join()


if __name__ == "__main__":
    # todo make user friendly
    if len(sys.argv) == 5:
        run_gridding(start_date=sys.argv[1], end_date=sys.argv[2], config_id=sys.argv[3], n_workers=int(sys.argv[4]))
    else:
        raise ValueError("Provide valid arguments. E.g.: python dardar_nice_grid '2016-01-01' '2016-01-31' "
                         "<config_id> <#workers>")
