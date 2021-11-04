"""feature engineering and vertical regrid for dardar l2 data"""

import xarray as xr
import numpy as np
import pandas as pd

from src.preprocess.helpers.common_helpers import custom_mode
from src.preprocess.helpers.constants import TEMP_THRES

CAT_VARIABLES = ['clm', 'clm_v2', 'instrument_flag']  # use value with max occurance
CONT_VARIABLES = [
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
    'cloud_thickness',
    'cloud_cover',
    'cloud_layer']  # use mean
FLAG_VARIABLES = ['cloud_top', 'cloud_bottom', 'liquid_origin', 'cloud_layer']  # take max


def l2_feature_engineering(ds):
    """

    Args:
        ds (xr.Dataset): L2 dataset of Dardar nice

    Returns:
        xr.Dataset: feature engineered L2 dataset
    """
    # define cloud masks
    clm_v2_cloud_masks = [1, 2, 3, 4, 9,
                          10]  # 3,4, contains super cooled water, is used for calculating cloud thickness and dz_top; will be filtered out later when filtering for t thres
    # clm_v2_ice_masks = [1, 2, 9, 10]

    # kick everything out that is above melting layer (tip from Odran → see his email)
    melting_layer = 270
    ds = ds.where(ds.ta < melting_layer)  # kickout everything above melting layer

    # fill missing values
    ds["clm_v2"] = ds.clm_v2.fillna(0)  # fill na values with clear sky

    # cloud cover, if clm v2 has a cloud flag
    ds["cloud_cover"] = (ds.clm_v2.isin(clm_v2_cloud_masks)).astype(int)

    # flag layers that are at the top of a cloud (a nwe cloud layer starts if gap is more than 400m)
    ds["cloud_free"] = ds.cloud_cover.where(ds.cloud_cover == 0) + 1
    ds["cum_layer_distance"] = ds["cloud_free"].cumsum("height") - ds.cloud_free.cumsum("height").where(
        ds.cloud_free.isnull()).ffill("height").fillna(0).astype(int)
    ds["cum_layer_distance_diff"] = ds.cum_layer_distance.diff("height")
    ds["cloud_top"] = (ds.cum_layer_distance_diff <= -7).astype(
        int)  # identify cloud top, if layers are more then 7 layers apart (7*60 m)
    ds["first_cloud_top"] = (ds.cloud_cover.where(ds.cloud_cover == 1).idxmax(dim="height") == ds.height).astype(int)
    ds["cloud_top"] = (ds["first_cloud_top"] | ds["cloud_top"]).astype(int)

    # flag cloud bottom, → same procedure as for cloud top with reversed index
    ds_reindex = ds.cloud_free.reindex(height=ds.height[::-1])
    cum_layer_distance = ds_reindex.cumsum("height") - ds_reindex.cumsum("height").where(ds_reindex.isnull()).ffill(
        "height").fillna(0).astype(int)
    cum_layer_distance_diff = cum_layer_distance.diff("height")
    cloud_bottom = cum_layer_distance_diff <= -7
    cloud_bottom = cloud_bottom.reindex(height=cloud_bottom.height[::-1])
    ds["cloud_bottom"] = cloud_bottom.astype(int)

    # enumerate cloud layers
    ds["cloud_layer"] = ds.cloud_top.cumsum(dim="height").where(ds.cloud_cover == 1)

    # calculate distance to cloud top based on clm v2
    a = (ds.height * ds.cloud_top + ds.cloud_cover - ds.cloud_top)
    cloud_top_height = a.where((a > 1)).ffill(dim="height") * ds.cloud_cover  # cloud top height for every layer
    dz_top = (cloud_top_height - ds.height + 60) * ds.cloud_cover  # cloud top height - height of layer
    ds["dz_top_v2"] = dz_top.where(dz_top)

    # set iwc values to nan when clm_v2 == 1 and iwc == 0; iwc of layer will be discarded for mean iwc, icnc_5um, icnc_100um, reffcli, but still contributes to cloud_cover
    # Reason for this phenomenon is that retrievals were calculated based on clm v1.
    cloud_no_iwc = ds.cloud_cover.where((ds.cloud_cover == 1) & (ds.iwc > 0))
    cirrus_properties = ["iwc", "icnc_5um", "icnc_100um", "reffcli", "iwc_error", "icnc_5um_error", "icnc_100um_error"]

    for prop in cirrus_properties:
        attrs = ds[prop].attrs
        ds[prop] = ds[prop] * cloud_no_iwc
        ds[prop].attrs = attrs

    # calculate cloud thickness
    ds["cloud_thickness"] = ds.dz_top_v2.where(ds.cloud_bottom == 1).bfill("height").where(ds.cloud_cover == 1)

    # flag clouds that extend beyond cirrus regime temps
    ds["liquid_origin"] = (ds.cloud_bottom - ds.cloud_bottom.where(ds.ta < TEMP_THRES, other=0)).where(
        ds.cloud_bottom == 1).bfill("height").where(ds.cloud_cover == 1)

    # set values for flag variables to nan where I dont have observations. Needed for horizontal aggregation later
    # resulting an an in-cloud instrument flag variable
    ds["instrument_flag"] = ds["instrument_flag"].where(ds.cloud_cover == 1)
    ds["cloud_top"] = ds["cloud_top"].where(ds.cloud_cover == 1)
    ds["cloud_bottom"] = ds["cloud_bottom"].where(ds.cloud_cover == 1)

    # transpose dimens
    ds = ds.transpose("time", "height")

    return ds


def l2_vertical_regrid(ds, layer_thickness):
    """aggregates l2 data variables to new layerthickness

    Args:
        ds:
        layer_thickness:

    Returns:

    """

    agg_layers = layer_thickness / 60
    assert agg_layers % 1 == 0, "layer_thickness must be multiple of 60, is {}".format(layer_thickness)
    agg_layers = int(agg_layers)

    # mean for continous variables
    coarse_cont = ds[CONT_VARIABLES].coarsen(height=agg_layers, boundary="trim", side="right", coord_func="mean").mean(
        keep_attrs=True)
    print("cont coarse")

    # mode for categorical variables
    coarse_cat = ds[CAT_VARIABLES].coarsen(height=agg_layers, boundary="trim", side="right", coord_func="mean").reduce(
        custom_mode, keep_attrs=True)
    print("cat coarse")

    # max for flag variables
    coarse_flag = ds[FLAG_VARIABLES].coarsen(height=agg_layers, boundary="trim", side="right", coord_func="mean").max(
        keep_attrs=True)
    print("flag coarse")

    merge = xr.merge([coarse_cat, coarse_cont, coarse_flag])
    print("merged")

    # flag columns with clouds
    merge["data_mask"] = (merge.cloud_cover.sum(dim="height") > 0).astype(int)

    return merge


def l2_filter_cirrus_regime(ds, altmax, altmin):
    # all values for all variables will be set to nan
    # + select only viable altitude range
    ice = ds.sel(height=slice(altmax, altmin)).where(ds.ta <= TEMP_THRES)

    # for cloud cover replace nan with 0 → needed for horizontal aggregation
    ice["cloud_cover"] = ice.cloud_cover.fillna(0)

    return ice


def run_l2_preproc(ds, altmax, altmin, layer_thickness):
    one_dim_vars = [var for var in ds.data_vars if len(ds[var].dims) == 1]
    two_dim_vars = [var for var in ds.data_vars if len(ds[var].dims) == 2]

    one_dim_ds = ds[one_dim_vars]

    fe_ds = l2_feature_engineering(ds[two_dim_vars])
    ice = l2_filter_cirrus_regime(fe_ds, altmax, altmin)
    ds_coarse = l2_vertical_regrid(ice, layer_thickness)
    merge = xr.merge([ds_coarse, one_dim_ds])

    return merge
