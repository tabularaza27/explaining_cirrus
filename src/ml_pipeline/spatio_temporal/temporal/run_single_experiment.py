import gc
import argparse
from pprint import pprint

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

import torch
import torch.nn as nn

from src.ml_pipeline.spatio_temporal.temporal.experiment_setup import experiment_setup

from src.preprocess.helpers.constants import *
from src.scaffolding.scaffolding import get_data_product_dir

# comet variables
COMET_API_KEY = "Rrwj142Sk080T0Qth3KNdPQg5"
COMET_WORKSPACE = "tabularaza27"
COMET_PROJECT = "cirrus-temporal"

# base features
features = ["p", "GPH", "T", "U", "V", "OMEGA", "o3", "RH_ice", 'DU001_traj', 'DU002_traj', 'DU003_traj', 'DU004_traj',
            'DU005_traj', 'SO2_traj', 'SO4_traj','z_traj']

# base hparams
hparams = dict(
    seq_len=1,
    batch_size=150,
    dense_layer_size=80,
    criterion=nn.MSELoss(),  # nn.HuberLoss(),
    scaler=StandardScaler(),
    max_epochs=60,
    features=features,
    n_features=len(features),
    hidden_size=64,
    num_layers=3,
    dropout=0.2,
    learning_rate=8.54e-4,
    attention=True,
    grad_clip=True,
    early_stopping=True
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="single experiment parser")
    parser.add_argument(
        "--n_month",
        default=2,
        type=int,
        help="use data up until which month, min:2, max: 13",
    )

    # use: --early-stopping | --no-early-stopping
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        help="use early stopping"
    )

    parser.add_argument(
        "--max_epochs",
        default=30,
        type=int,
        help="number of epochs model trains"
    )

    args = parser.parse_args()
    n_month=args.n_month
    hparams["early_stopping"]=args.early_stopping
    hparams["max_epochs"]=args.max_epochs

    pprint(hparams)

    # load only one month for now
    config_id = "larger_domain_high_res"
    year = 2008

    datacube_df_dir = get_data_product_dir(config_id, DATA_CUBE_DF_DIR)

    # read datacube merged with trajectories
    month_dfs = []
    for month in range(1, n_month):
        print(month)
        month_df = pd.read_pickle(datacube_df_dir + "/dardar_traj_traced_{}{:02d}.pickle".format(year, month))
        month_dfs.append(month_df)
        print(month_df.columns)

    df = pd.concat(month_dfs)
    print("loaded data frame")
    print(df.columns)

    # free up some memory
    del month_df
    gc.collect()

    # setup
    trainer, model, dm, comet_logger = experiment_setup(df, hparams)

    # train
    trainer.fit(model, dm)

    # evaluate
    trainer.test(model, datamodule=dm)
    torch.save(trainer.model.state_dict(), 'lstm_model')
    comet_logger.experiment.log_model('model_0', './lstm_model')
    comet_logger.log_graph(model=trainer.model)
    comet_logger.log_hyperparams(hparams)
    comet_logger.log_hyperparams({"start_time": dm.traj_df.date.min(), "end_time": dm.traj_df.date.max()})
    comet_logger.experiment.end()