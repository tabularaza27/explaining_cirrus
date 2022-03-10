from pprint import pprint
import gc
import argparse

import pandas as pd

import optuna

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
            'DU005_traj', 'SO2_traj', 'SO4_traj']

# base hparams
base_hparams = dict(
    seq_len=1,
    batch_size=500,
    dense_layer_size=128,
    criterion=nn.MSELoss(),  # nn.HuberLoss(),
    max_epochs=30,
    features=features,
    n_features=len(features),
    hidden_size=50,
    num_layers=5,
    dropout=0.2,
    learning_rate=0.001,
    attention=True,
    grad_clip=True,
    early_stopping=True
)


def objective(trial: optuna.trial.Trial, df: pd.DataFrame) -> float:
    num_layers = trial.suggest_int("num_layers", 1, 10)
    hidden_size = trial.suggest_int("hidden_size", 4, 512, log=True)
    dense_layer_size = trial.suggest_int("dense_layer_size", 64, 512, log=True)
    dropout = trial.suggest_uniform("dropout", 0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 64, 1024, log=True)
    attention = trial.suggest_categorical("attention", [True, False])
    cc_feature = trial.suggest_categorical("cc_feature", [True, False])
    grad_clip = trial.suggest_categorical("grad_clip", [True, False])

    hparams = dict(num_layers=num_layers,
                   hidden_size=hidden_size,
                   dense_layer_size=dense_layer_size,
                   dropout=dropout,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   attention=attention,
                   cc_feature=cc_feature,
                   grad_clip=grad_clip)

    # set hparams
    all_hparams = base_hparams.copy()
    for param in hparams:
        if param in all_hparams:
            all_hparams[param] = hparams[param]
            print(param)

    # other
    if hparams["cc_feature"]:
        all_hparams["features"] = features + ["cc_traj"]
    else:
        all_hparams["features"] = features

    all_hparams["n_features"] = len(all_hparams["features"])

    print("start experiment with hparams:")
    pprint(all_hparams)

    # setup
    trainer, model, dm, comet_logger = experiment_setup(df, all_hparams, optuna=True, trial=trial)

    # train
    trainer.fit(model, dm)

    # evaluate
    trainer.test(model, datamodule=dm)
    torch.save(trainer.model.state_dict(), 'lstm_model')
    comet_logger.experiment.log_model('model_0', './lstm_model')
    comet_logger.log_graph(model=trainer.model)

    comet_logger.experiment.end()
    gc.collect()  # garbae collector

    return trainer.callback_metrics["test_loss"].item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--n_trials",
        default=50,
        type=int,
        help="number of trials in the hparam tuning",
    )
    args = parser.parse_args()
    n_trials=args.n_trials

    # load only one month for now
    config_id = "larger_domain_high_res"
    year = 2008

    datacube_df_dir = get_data_product_dir(config_id, DATA_CUBE_DF_DIR)

    # read datacube merged with trajectories
    month_dfs = []
    for month in range(1, 2):
        print(month)
        month_df = pd.read_pickle(datacube_df_dir + "/dardar_traj_traced_{}{:02d}.pickle".format(year, month))
        month_dfs.append(month_df)

    df = pd.concat(month_dfs)
    print("loaded data frame")

    # free up some memory
    del month_df
    gc.collect()

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()
    sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler()
    # sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler()

    # Execute an optimization by using the above objective function wrapped by `lambda`.
    # https://optuna.readthedocs.io/en/stable/faq.html
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
