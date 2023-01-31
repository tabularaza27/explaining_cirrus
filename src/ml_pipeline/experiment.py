import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from pprint import pprint
from src.ml_pipeline.ml_preprocess import create_dataset


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
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

    return validate_df


def run_experiment(df, xgboost_config, experiment_config):
    """run experiment

    Args:
        df (pd.Dataframe): data frame with each row representing one sample
        xgboost_config (dict): xgboost hyperparameter
        experiment_config (dict): experiment configuration

    Examples:
        xgboost_config = {"objective": "reg:squarederror", 'subsample': 0.4, "colsample_bytree": 0.8, 'learning_rate': 0.02,
                  'max_depth': 15, 'alpha': 38, 'lambda': 7, 'n_estimators': 250, "n_jobs": 32}

      experiment_config =    {
        "filters": ["nightday_flag ==1"],
        "predictors": ["t",
                       "w",
                       "wind_speed",
                       "DU_sup",
                       "DU_sub",
                       "SO4",
                       "dz_top_v2",
                       "cloud_thickness_v2",
                       "surface_height",
                       "season",
                       "land_water_mask",
                       "lat_region"
                       ],
        "predictand": "icnc_5um",
        "preproc_steps": {
            "x_log_trans": True,
            "y_log_trans": True,
            "kickout_outliers": False,
            "oh_encoding": True
        },
        "random_state": 53
    }

    """
    pprint(experiment_config)

    # create data set
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset(df, **experiment_config)

    n_datapoints = X_train.count()[0] + X_val.count()[0] + X_test.count()[0]
    print(f"Number of samples: {n_datapoints}")

    # build and train model
    xg_reg = xgb.XGBRegressor(random_state=experiment_config["random_state"], **xgboost_config)
    xg_reg.fit(X_train, y_train,
               eval_set=[(X_train, y_train), (X_val, y_val)],
               eval_metric="rmse", early_stopping_rounds=10)

    # evaluate performance
    validate_df = evaluate_model(xg_reg, X_test, y_test)

    # save model to comet
    xg_reg.save_model("xgboost_model.json")

    return xg_reg, validate_df
