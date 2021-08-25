import tempfile
import sys
import argparse
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess.helpers.constants import TEMP_THRES
from src.ml_pipeline.experiment import get_experiment_assets, load_experiment, get_asset_id
from src.ml_pipeline.ml_preprocess import create_dataset

def load_shap_values(experiment_name, project_name="icnc-xgboost"):
    """load shap values and shap idx from comet"""
    experiment = load_experiment(experiment_name, project_name)
    experiment_assets = experiment.get_asset_list()

    # get shap values
    asset_id = get_asset_id(experiment_assets, "type", "shap_values")
    rawdata = experiment.get_asset(asset_id)

    fo = tempfile.NamedTemporaryFile(suffix=".npy")
    with open(fo.name, "wb") as f:
        f.write(rawdata)

    shap_values = np.load(fo.name)
    fo.close()

    # get shap indices to create X
    asset_id = get_asset_id(experiment_assets, "type", "shap_idx")
    rawdata = experiment.get_asset(asset_id)

    fo = tempfile.NamedTemporaryFile(suffix=".npy")
    with open(fo.name, "wb") as f:
        f.write(rawdata)

    shap_idx = np.load(fo.name)
    fo.close()

    # create dataframe that was used to create shap values
    asset_id = get_asset_id(experiment_assets, "fileName", "config")
    experiment_config = experiment.get_asset(asset_id, return_type="json")

    # load and create dataset
    df = pd.read_pickle("/net/n2o/wolke/kjeggle/Notebooks/DataCube/df_pre_filtering.pickle")
    # Drop NaN
    df = df.dropna()
    # Filter for Temperature
    df = df.query("ta <= {}".format(TEMP_THRES))

    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(df, **experiment_config)
    shap_df = X_test[X_test.index.isin(shap_idx)]

    return shap_values, shap_df, shap_idx

def load_shap_df(experiment_name):
    """load dataframe that was used to create shap values"""
    experiment = load_experiment(experiment_name)
    experiment_assets = experiment.get_asset_list()

    asset_id = get_asset_id(experiment_assets, "fileName", "config")
    experiment_config = experiment.get_asset(asset_id, return_type="json")

    # load and create dataset
    df = pd.read_pickle("/net/n2o/wolke/kjeggle/Notebooks/DataCube/df_pre_filtering.pickle")
    # Drop NaN
    df = df.dropna()
    # Filter for Temperature
    df = df.query("ta <= {}".format(TEMP_THRES))

    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(df, **experiment_config)
    shap_df  = X_test[X_test.index.isin(shap_idx)]

    return shap_df

def get_col_locs(columns, df):
    return [df.columns.get_loc(col) for col in columns]

def log_shap_plots(experiment_name, project_name="icnc-xgboost", summary_plots=True, dependence_plots=False):
    shap_values, shap_df, shap_idx = load_shap_values(experiment_name, project_name)

    experiment = load_experiment(experiment_name, project_name)

    var_groups = {
        "met_vars": ["t", "w", "u", "v", "rh_ice"],
        "aerosol_vars": ["DU_log", "SO4_log"],  # "SO2_log"],
        "vertical_cloud_info": ["cloud_thickness", "dz_top"],
        "instrument": [col for col in shap_df.columns if "instrument" in col],
        "nightday": [col for col in shap_df.columns if "nightday" in col],
        "land_water": [col for col in shap_df.columns if "land_water" in col],
        "region": [col for col in shap_df.columns if "lat_region" in col],
        "season": [col for col in shap_df.columns if "season" in col],
        "ic_cir": [col for col in shap_df.columns if "IC_CIR" in col]
    }

    if summary_plots:

        # plot all features
        fo = tempfile.NamedTemporaryFile(suffix=".png")
        shap.summary_plot(shap_values, shap_df, max_display=45, show=False)
        plt.savefig(fo.name)
        experiment.log_image(fo.name)
        plt.close()
        fo.close()

        # plot per var group
        for var_group, col_names in var_groups.items():
            print("###### {} ######".format(var_group))
            fo = tempfile.NamedTemporaryFile(suffix=".png")
            col_locs = get_col_locs(col_names, shap_df)
            shap.summary_plot(shap_values[:, col_locs], shap_df[col_names], title="{}".format(var_group), show=False)
            plt.savefig(fo.name)
            experiment.log_image(fo.name)
            plt.close()
            fo.close()

def calculate_and_log_shap_values(experiment_name, project_name, sample_size=None, log=True, interaction_values=False,
                                  check_additivity=False):
    """return explainer object and shap values and index of predictions

    Args:
        experiment_name:
        project_name:
        sample_size (int|None): specifies how many rows are randomly selected to calculate shap values for
        log (bool): If True, log to comet
        interaction_values:

    Returns:

    """
    experiment_config, xg_reg = get_experiment_assets(experiment_name, project_name)
    experiment = load_experiment(experiment_name, project_name)

    # load and create dataset
    df = pd.read_pickle("/net/n2o/wolke/kjeggle/Notebooks/DataCube/df_pre_filtering.pickle")
    # Drop NaN
    df = df.dropna()
    # Filter for Temperature
    df = df.query("ta <= {}".format(TEMP_THRES))
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(df, **experiment_config)

    if (sample_size) and (sample_size < X_test.t.count()):
        X_test = X_test.sample(n=sample_size, axis="index", random_state=123).sort_index()
        print("Training set after sampling: {} Datapoints".format(X_test.t.count()))

    # calculate shap_values
    explainer = shap.TreeExplainer(xg_reg)
    print("created explainer")
    shap_values = explainer.shap_values(X_test, check_additivity=check_additivity)
    print("calculated {} shap values".format(shap_values.shape[0]))

    if log:
        # log shap values to experiment
        fo = tempfile.NamedTemporaryFile(suffix=".npy")

        with open(fo.name, "wb") as f:
            np.save(f, shap_values)

        experiment.log_asset(fo.name, ftype="shap_values", overwrite=True)
        fo.close()

        # log indices of predictions for which shap values were calculated
        fo = tempfile.NamedTemporaryFile(suffix=".npy")
        print("saved shap values")

        with open(fo.name, "wb") as f:
            shap_idx = np.array(X_test.index)
            np.save(f, shap_idx)

        experiment.log_asset(fo.name, ftype="shap_idx", overwrite=True)
        fo.close()
        print("saved shap indices")

    return explainer, shap_values, shap_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_names", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=None, help="specifies # of rows to be selected for shap value calc")
    parser.add_argument("--project_name", type=str, default="icnc-xgboost")

    try:
        arguments = parser.parse_args()
        experiment_names = arguments.exp_names
        project_name = arguments.project_name
        sample_size = arguments.n
    except BaseException as exc:
        print("correct usage: python src/ml_pipeline/shap.py --exp_names name1 name2 --n 1000")
        raise

    for experiment_name in experiment_names:
        print(experiment_name)
        explainer, shap_values, shap_idx = calculate_and_log_shap_values(experiment_name,
                                                                         project_name,
                                                                         sample_size=sample_size,
                                                                         log=True,
                                                                         interaction_values=False)
