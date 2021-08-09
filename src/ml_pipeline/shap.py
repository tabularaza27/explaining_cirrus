import tempfile
import sys
import argparse
import xgboost as xgb
import shap
import numpy as np
import pandas as pd

from src.preprocess.helpers.constants import TEMP_THRES
from src.ml_pipeline.experiment import get_experiment_assets, load_experiment
from src.ml_pipeline.ml_preprocess import create_dataset


def calculate_and_log_shap_values(experiment_name, project_name, sample_size=None, log=True, interaction_values=False,
                                  check_additivity=True):
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

    if sample_size:
        X_test = X_test.sample(n=sample_size, axis="index", random_state=123).sort_index()
        print("Training set after sampling: {} Datapoints".format(X_test.t.count()))

    # calculate shap_values
    explainer = shap.TreeExplainer(xg_reg)
    print("created explainer")
    shap_values = explainer.shap_values(X_test, check_additivity=check_additivity)
    print("calculated shap values")

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
        explainer, shap_values, shap_idx = calculate_and_log_shap_values(experiment_name,
                                                                         project_name,
                                                                         sample_size=sample_size,
                                                                         log=True,
                                                                         interaction_values=False)
