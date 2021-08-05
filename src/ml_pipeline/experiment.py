import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import comet_ml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from pprint import pprint

sys.path.append("/net/n2o/wolke/kjeggle/Repos/cirrus/src")
from ml_pipeline.preprocess import create_dataset

COMET_API_KEY = "Rrwj142Sk080T0Qth3KNdPQg5"
COMET_WORKSPACE = "tabularaza27"

def create_tags(config):
    tags = []
    for key, val in config.items():
        tags.append("{}: {}".format(key, str(val)))
    return tags


def evaluate_model(model, experiment, X_test, y_test):
    # evaluate
    preds = model.predict(X_test)
    validate_df = pd.DataFrame([preds, y_test], ["predictions", "ground_truth"]).T

    # test performance
    rmse = np.sqrt(mean_squared_error(validate_df.predictions, validate_df.ground_truth))
    print("rmse", rmse)
    experiment.log_metric("test_rmse", rmse)

    # correlation coefficient
    r = validate_df.corr().values[0][1]
    print("R", r)
    experiment.log_metric("R", r)

    # r2 score
    r2 = r2_score(validate_df["ground_truth"], validate_df["predictions"])
    print("R2", r2)
    experiment.log_metric("R2", r2)

    return validate_df


def run_experiment(df, xgboost_config, experiment_config, comet_project_name="icnc-xgboost"):
    tags = create_tags(experiment_config)
    pprint(experiment_config)

    # create data set
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(df, **experiment_config)

    n_datapoints = X_train.t.count() + X_test.t.count()
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
    validate_df = evaluate_model(xg_reg, experiment, X_test, y_test)

    # save model to comet
    xg_reg.save_model("xgboost_model.json")
    experiment.log_model("XGBoost Model", "xgboost_model.json")
    experiment.log_asset_data(data=experiment_config, name="config")

    experiment.end()

    return xg_reg, validate_df, experiment
