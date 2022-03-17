import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

from optuna.integration import PyTorchLightningPruningCallback

from src.ml_pipeline.spatio_temporal.temporal.data_module import BacktrajDataModule
from src.ml_pipeline.spatio_temporal.temporal.lstm_model import LSTMRegressor, LogCallback

COMET_API_KEY = "Rrwj142Sk080T0Qth3KNdPQg5"
COMET_WORKSPACE = "tabularaza27"
COMET_PROJECT = "cirrus-temporal"


def experiment_setup(df, p, optuna=False, trial=None):
    """Set up pytorch lightning experiment with parameter dict p"""

    comet_logger = pl_loggers.CometLogger(
        api_key=COMET_API_KEY,
        workspace=COMET_WORKSPACE,
        project_name=COMET_PROJECT,
        auto_histogram_gradient_logging=True
    )
    callbacks = [LogCallback()]

    if p["early_stopping"]:
        early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=15)
        callbacks.append(early_stop_callback)

    if optuna:
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_loss"))
        comet_logger.experiment.add_tag("optuna")

    # set grad clip
    if p["grad_clip"]:
        grad_clip = 0.5
    else:
        grad_clip = 0

    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=p['max_epochs'],
        logger=[comet_logger],
        gpus=1,
        log_every_n_steps=100,
        progress_bar_refresh_rate=1000,
        track_grad_norm=2,
        gradient_clip_val=grad_clip
        # fast_dev_run=1
    )

    model = LSTMRegressor(
        n_features=p['n_features'],
        hidden_size=p['hidden_size'],
        dense_layer_size=p['dense_layer_size'],
        seq_len=p['seq_len'],
        batch_size=p['batch_size'],
        criterion=p['criterion'],
        num_layers=p['num_layers'],
        dropout=p['dropout'],
        learning_rate=p['learning_rate'],
        attention=p['attention']
    )

    dm = BacktrajDataModule(
        traj_df=df,
        batch_size=p['batch_size'],
        features=p['features'],
        num_workers=16
    )

    return trainer, model, dm, comet_logger
