from pprint import pprint
import copy

import pandas as pd
import numpy as np

from src.ml_pipeline.experiment import run_experiment
from src.preprocess.helpers.constants import TEMP_THRES

experiment_configs = [
    #     {"filters": ["clm == 1","lat_region=='lat_50'"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "iwc",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU"],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1",'nightday_flag == 0','land_water_mask==7'],
    #      "predictors": ["DU",'instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     }
    #     ,
    #     {"filters": ["clm == 1",'nightday_flag == 1','land_water_mask==7'],
    #      "predictors": ["DU",'instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     }
    #     ,
    #     {"filters": ["clm == 1","instrument_flag==1"],
    #      "predictors": ["DU",'nightday_flag',],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1","instrument_flag==3"],
    #      "predictors": ["DU",'nightday_flag',],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": False,
    #          "oh_encoding": True
    #      }
    #     }
    #     ,
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": False,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     }
    #     ,
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": False,
    #          "y_log_trans": False,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm_v2 in [1,2,9,10]"],
    #      "predictors": ["DU","clm_v2",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_100um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": False,
    #          "kickout_outliers": False,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "iwc",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "reffcli",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1","lat_region=='lat_60'"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "iwc",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1","lat_region=='lat_10'"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "iwc",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1", "dz_top <= 200"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": False,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1","icnc_5um_quantiles in {}".format([str(q) for q in quantiles[-2:]])],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": False,
    #          "oh_encoding": True
    #      }
    #     },
    #     {"filters": ["clm == 1","rh_ice >= 90"],
    #      "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag'],
    #      "predictand": "icnc_5um",
    #      "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #      }
    #     },
    #     {
    #         "filters": ["clm == 1"],
    #         "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag',"clm_v2"],
    #         "predictand": "icnc_5um",
    #         "preproc_steps": {
    #              "x_log_trans": True,
    #              "y_log_trans": True,
    #              "kickout_outliers": True,
    #              "oh_encoding": True
    #     }
    #     },
    #     {
    #         "filters": ["clm == 1"],
    #         "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag','clm_v2','cloud_thickness'],
    #         "predictand": "icnc_5um",
    #         "preproc_steps": {
    #              "x_log_trans": True,
    #              "y_log_trans": True,
    #              "kickout_outliers": True,
    #              "oh_encoding": True
    #     }
    #     },
    #     {
    #         "filters": ["clm == 1"],
    #         "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag',"clm_v2",'cloud_thickness'],
    #         "predictand": "iwc",
    #         "preproc_steps": {
    #              "x_log_trans": True,
    #              "y_log_trans": True,
    #              "kickout_outliers": True,
    #              "oh_encoding": True
    #         }
    #     },
    #     {
    #     "filters": ["clm == 1"],
    #     "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag',"clm_v2",'cloud_thickness'],
    #     "predictand": "icnc_100um",
    #     "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": False,
    #          "kickout_outliers": True,
    #          "oh_encoding": True
    #     }
    #     },
    # {
    #     "filters": ["clm == 1"],
    #     "predictors": ["DU", 'nightday_flag', 'land_water_mask', 'instrument_flag', "clm_v2", 'cloud_thickness'],
    #     "predictand": "reffcli",
    #     "preproc_steps": {
    #         "x_log_trans": True,
    #         "y_log_trans": False,
    #         "kickout_outliers": True,
    #         "oh_encoding": True
    #     }
    # },
    # {
    #     "filters": ["clm == 1", "rh_ice >= 65"],
    #     "predictors": ["DU", 'nightday_flag', 'land_water_mask', 'instrument_flag', 'clm_v2', 'cloud_thickness'],
    #     "predictand": "icnc_5um",
    #     "preproc_steps": {
    #         "x_log_trans": True,
    #         "y_log_trans": True,
    #         "kickout_outliers": True,
    #         "oh_encoding": True
    #     }
    # },
    # {
    #     "filters": ["clm == 1", "t <= 223.15"],
    #     "predictors": ["DU", 'nightday_flag', 'land_water_mask', 'instrument_flag', 'clm_v2', 'cloud_thickness'],
    #     "predictand": "icnc_5um",
    #     "preproc_steps": {
    #         "x_log_trans": True,
    #         "y_log_trans": True,
    #         "kickout_outliers": True,
    #         "oh_encoding": True
    #     }
    # },
    {
        "filters": ["clm == 1", "t <= 223.15", "clm_v2 != 0"],
        "predictors": ["DU", 'nightday_flag', 'land_water_mask', 'instrument_flag', 'clm_v2', 'cloud_thickness'],
        "predictand": "icnc_5um",
        "preproc_steps": {
            "x_log_trans": True,
            "y_log_trans": True,
            "kickout_outliers": True,
            "oh_encoding": True
        }
    },
    #     {
    #     "filters": ["clm == 1"],
    #     "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag',"clm_v2"'cloud_thickness'],
    #     "predictand": "icnc_5um",
    #     "preproc_steps": {
    #          "x_log_trans": True,
    #          "y_log_trans": True,
    #          "kickout_outliers": False,
    #          "oh_encoding": True
    #     }
    #     }
]

experiment_configs = []

base_config = {
        "filters": ["clm == 1"],
        "predictors": ["DU",'nightday_flag','land_water_mask','instrument_flag','clm_v2','cloud_thickness'],
        "predictand": "icnc_5um",
        "preproc_steps": {
             "x_log_trans": True,
             "y_log_trans": True,
             "kickout_outliers": True,
             "oh_encoding": True
    }
    }

filters = []
# create filters for temperature bands
step = 10
for lev in np.arange(180,240,step):
    temp_band = [lev, lev+step]
    filters.append("t.between({},{})".format(temp_band[0],temp_band[1]))

# create filters for cloud thickness bands
step = 1000
for lev in np.arange(0,13000,step):
    thickness_band = [lev, lev+step]
    filters.append("cloud_thickness.between({},{})".format(thickness_band[0],thickness_band[1]))

for f in filters:
    config = copy.deepcopy(base_config)
    config["filters"].append(f)
    experiment_configs.append(config)


xgboost_config = {"objective": "reg:squarederror", 'subsample': 0.4, "colsample_bytree": 0.8, 'learning_rate': 0.02,
                  'max_depth': 15, 'alpha': 38, 'lambda': 7, 'n_estimators': 500, "n_jobs": 50}


def load_dataframe():
    df = pd.read_pickle("/net/n2o/wolke/kjeggle/Notebooks/DataCube/df_pre_filtering.pickle")
    ### Drop NaN
    df = df.dropna()
    ### Filter for Temperature
    df = df.query("ta <= {}".format(TEMP_THRES))

    return df


def run_experiments():
    df = load_dataframe()

    for config in experiment_configs:
        pprint(config)
        run_experiment(df, xgboost_config, experiment_config=config, log_figures=False)


if __name__ == "__main__":
    run_experiments()
