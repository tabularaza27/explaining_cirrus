import sys
import json
import tempfile
import comet_ml
import pandas as pd
import hvplot.pandas  # noqa
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
import holoviews as hv
from pprint import pprint

sys.path.append("/net/n2o/wolke/kjeggle/Repos/cirrus/src")
from ml_pipeline.ml_preprocess import create_dataset
from preprocess.helpers.constants import TEMP_THRES

COMET_API_KEY = "Rrwj142Sk080T0Qth3KNdPQg5"
COMET_WORKSPACE = "tabularaza27"


def create_tags(config):
    tags = []
    for key, val in config.items():
        tags.append("{}: {}".format(key, str(val)))
    return tags


def evaluate_model(model, X_test, y_test, experiment=None):
    # evaluate
    preds = model.predict(X_test)
    validate_df = pd.DataFrame([preds, y_test], ["predictions", "ground_truth"]).T
    validate_df["abs_diff"] = np.abs(validate_df.predictions - validate_df.ground_truth)
    validate_df["diff"] = validate_df.predictions - validate_df.ground_truth
    validate_df["diff_round"] = np.round(validate_df["diff"], 0)

    # test performance
    rmse = np.sqrt(mean_squared_error(validate_df.predictions, validate_df.ground_truth))
    print("rmse", rmse)

    # correlation coefficient
    r = validate_df.corr().values[0][1]
    print("R", r)

    # r2 score
    r2 = r2_score(validate_df["ground_truth"], validate_df["predictions"])
    print("R2", r2)

    # log to experiment if exists
    if experiment:
        experiment.log_metric("test_rmse", rmse)
        experiment.log_metric("R", r)
        experiment.log_metric("R2", r2)

    return validate_df


def log_figures_to_experiment(validate_df, experiment):
    figures = []

    # hex plot ground_truth vs. predictions
    axes_lims = (validate_df["ground_truth"].min(), validate_df["ground_truth"].max())
    figures.append(
        validate_df.hvplot.hexbin(x="predictions", y="ground_truth", xlim=axes_lims, ylim=axes_lims, width=750,
                                  height=500, title="ground_truth vs. predictions"))

    # distributions ground_truth vs. predictions
    figures.append(validate_df.hvplot.hist(y=["ground_truth", "predictions"], bins=100, alpha=0.5,
                                           title="ground_truth vs. predictions"))

    # distributions of prediction differences
    figures.append(validate_df.hvplot.hist(y=["abs_diff", "diff"], bins=100, alpha=0.5,
                                           title="diff between ground truth and prediction"))

    for fig in figures:
        fo = tempfile.NamedTemporaryFile(suffix=".png")
        hv.save(fig, fo.name, fmt="png")
        experiment.log_image(fo.name)
        fo.close()


def run_experiment(df, xgboost_config, experiment_config, comet_project_name="icnc-xgboost", log_figures=True):
    tags = create_tags(experiment_config)
    pprint(experiment_config)

    # create data set
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(df, **experiment_config)

    n_datapoints = X_train.t.count() + X_val.t.count() + X_test.t.count()
    tags.append("Datapoints: {}".format(n_datapoints))
    print("tags:", tags)

    # run experiment
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name=comet_project_name,
        workspace=COMET_WORKSPACE,
    )
    experiment.add_tags(tags)

    # build and train model
    xg_reg = xgb.XGBRegressor(**xgboost_config)
    xg_reg.fit(X_train, y_train,
               eval_set=[(X_train, y_train), (X_val, y_val)],
               eval_metric="rmse", early_stopping_rounds=10)

    # evaluate performance
    validate_df = evaluate_model(xg_reg, X_test, y_test, experiment)

    # save model to comet
    xg_reg.save_model("xgboost_model.json")
    experiment.log_model("XGBoost Model", "xgboost_model.json")
    experiment.log_asset_data(data=experiment_config, name="config")

    print(type(experiment))

    if log_figures:
        log_figures_to_experiment(validate_df, experiment)

    print(type(experiment))

    experiment.end()

    return xg_reg, validate_df, experiment


def load_experiment(experiment_name, project_name="icnc-xgboost"):
    """returns experiment object

    Args:
        experiment_name:
        project_name:

    Returns:

    """
    comet_api_endpoint = comet_ml.api.API(api_key=COMET_API_KEY)
    experiment = comet_api_endpoint.get("{}/{}/{}".format(COMET_WORKSPACE, project_name, experiment_name))

    return experiment


def get_asset_id(experiment_asset_dict, asset_key, asset_value):
    asset_id = next((asset["assetId"] for asset in experiment_asset_dict if asset[asset_key] == asset_value), None)
    return asset_id


def get_experiment_assets(experiment_name, project_name="icnc-xgboost"):
    """returns config and model of experiment

    Args:
        experiment_name:
        project_name:

    Returns:
        config (dict), ml_model
    """
    experiment = load_experiment(experiment_name, project_name)

    # download model json from comet
    experiment_assets = experiment.get_asset_list()

    # get experiment config
    asset_id = get_asset_id(experiment_assets, "fileName", "config")
    experiment_config = experiment.get_asset(asset_id, return_type="json")

    # get and load ml model
    asset_id = get_asset_id(experiment_assets, "type", "model-element")
    model_json = experiment.get_asset(asset_id, return_type="json")

    # create temporary file for json
    fo = tempfile.NamedTemporaryFile(suffix=".json")
    fo.name

    with open(fo.name, "w") as file:
        json.dump(model_json, file)

    # load model into xgboost
    xg_reg = xgb.Booster()
    xg_reg.load_model(fo.name)

    fo.close()

    return experiment_config, xg_reg


def calculate_and_log_shap_values(experiment_name, project_name, sample_size=None, log=True, interaction_values=False, check_additivity=True):
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
    df = pd.read_pickle("df_pre_filtering.pickle")
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
    shap_values = explainer.shap_values(X_test, check_additivity=check_additivity)

    if log:
        # log shap values to experiment
        fo = tempfile.NamedTemporaryFile(suffix=".npy")

        with open(fo.name, "wb") as f:
            np.save(f, shap_values)

        experiment.log_asset(fo.name, ftype="shap_values")
        fo.close()

        # log indices of predictions for which shap values were calculated
        fo = tempfile.NamedTemporaryFile(suffix=".npy")

        with open(fo.name, "wb") as f:
            shap_idx = np.array(X_test.index)
            np.save(f, shap_idx)

        experiment.log_asset(fo.name, ftype="shap_idx")
        fo.close()

    return explainer, shap_values, shap_idx
