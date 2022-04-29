import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy
from scipy.special import exp10

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from src.ml_pipeline.spatio_temporal.temporal.custom_loss_functions import *


class LogCallback(pl.callbacks.Callback):

    def on_test_start(self, trainer, pl_module):
        # empty test results
        pl_module.test_results = {"y": [], "y_hat": []}

    def on_test_end(self, trainer, pl_module):
        # log distributions of prediction vs. target variable

        # get predictions for whole test dataset
        y = torch.concat(trainer.model.test_results["y"]).cpu().reshape(-1).numpy()
        y_hat = torch.concat(trainer.model.test_results["y_hat"]).cpu().reshape(-1).numpy()
        diff = y - y_hat  # residuals

        # inverse log10 transform transform back to original scale
        y_org = exp10(y)
        y_hat_org = exp10(y_hat)
        diff_org = y_org - y_hat_org

        # get comet logger
        comet_logger_idx = np.argmax([isinstance(logger, pl.loggers.comet.CometLogger) for logger in trainer.logger])
        comet_logger = trainer.logger[comet_logger_idx]

        # creating figure

        # get percentile for zooming in on axis
        percentile = 99
        y_org_percentile = np.percentile(y_org, percentile)
        diff_org_percentile = np.percentile(diff_org, percentile)
        diff_org_low_percentile = np.percentile(diff_org, 100 - percentile)

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))

        # predictand vs. ground truth # todo adapt to multitask
        axs[0, 0].hist([y_hat, y], bins=100, alpha=0.5, density=True)
        axs[0, 0].set_xlabel("log({})".format(trainer.datamodule.predictands))
        axs[0, 0].set_ylabel("density")
        axs[0, 0].legend(["predicted", "ground_truth"])
        axs[0, 0].set_title("predictand distribution (log scale)")

        axs[0, 1].hist([y_hat_org, y_org], bins=1000, alpha=0.5, density=True)
        axs[0, 1].set_xlabel(trainer.datamodule.predictands)
        axs[0, 1].set_ylabel("density")
        axs[0, 1].legend(["predicted", "ground_truth"])
        axs[0, 1].set_title("predictand distribution")
        axs[0, 1].set_xlim([0, y_org_percentile])

        # # predictand vs.ground truth scatter
        # axs[1, 0].scatter(y, y_hat, s=1, alpha=0.8)
        # axs[1, 0].plot(y, y, "r-")
        # axs[1, 0].set_xlabel("log({})".format(trainer.datamodule.predictand))
        # axs[1, 0]..set_xlabel("log({}) predicted".format(trainer.datamodule.predictand))
        # axs[1, 0].legend(["predicted", "ground_truth"])
        # axs[1, 0].set_title("scatter y vs y_hat (log scale)")
        #
        # axs[1, 1].scatter(y_org, y_hat_org, s=1, alpha=0.8)
        # axs[1, 1].plot(y_org, y_org, "r-")
        # axs[1, 1].set_xlabel(trainer.datamodule.predictand)
        # axs[1, 1].set_ylabel("{} predicted".format(trainer.datamodule.predictand)
        # axs[1, 1].legend(["predicted", "ground_truth"])
        # axs[1, 1].set_title("scatter y vs y_hat")
        # axs[1, 1].set_xlim([0, y_org_percentile])
        # axs[1, 1].set_ylim([0, y_org_percentile])

        # residuals
        axs[1, 0].hist(diff, bins=100, density=True)
        axs[1, 0].set_xlabel("residuals (log scale)")
        axs[1, 0].set_ylabel("density")
        axs[1, 0].set_title("residuals (log scale)")

        axs[1, 1].hist(diff_org, bins=1000, density=True)
        axs[1, 1].set_xlabel("residuals")
        axs[1, 1].set_ylabel("density")
        axs[1, 1].set_title("residuals")
        axs[1, 1].set_xlim([diff_org_low_percentile, diff_org_percentile])

        # # predicted vs. residuals scatter
        # axs[3, 0].scatter(diff, y_hat, alpha=0.8, s=1)
        # axs[3, 0].set_xlabel("residuals (log scale)")
        # axs[3, 0].set_ylabel("log(iwc) predicted")
        # axs[3, 0].set_title("prediction vs. residual (log scale)")
        #
        # axs[3, 1].scatter(diff_org, y_hat_org, alpha=0.8, s=1)
        # axs[3, 1].set_xlabel("residuals")
        # axs[3, 1].set_ylabel("iwc predicted")
        # axs[3, 1].set_title("prediction vs. residual")
        # axs[3, 1].set_xlim([0, diff_org_percentile])
        # axs[3, 1].set_ylim([0, y_org_percentile])

        plt.tight_layout()
        plt.show()

        comet_logger.experiment.log_figure(figure=fig, figure_name="test_set_prediction_distribution")


# class LSTMRegressor(pl.LightningModule):
#     '''
#     Standard PyTorch Lightning module:
#     https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
#     '''
#
#     def __init__(self,
#                  n_features,
#                  hidden_size,
#                  seq_len,
#                  batch_size,
#                  num_layers,
#                  dropout,
#                  learning_rate,
#                  criterion,
#                  dense_layer_size,
#                  grad_clip=False,
#                  attention=False):
#         super(LSTMRegressor, self).__init__()
#         self.n_features = n_features
#         self.hidden_size = hidden_size
#         self.dense_layer_size = dense_layer_size
#         self.seq_len = seq_len
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.criterion = criterion
#         self.learning_rate = learning_rate
#         self.test_results = {"y_hat": [], "y": []}
#         self.attention = attention
#         self.grad_clip = grad_clip
#
#         if self.attention:
#             # attention branch
#             # self.u_input = torch.ones(self.hidden_size,device=0,requires_grad=False) / self.hidden_size
#             # self.attention_dense = nn.Linear(hidden_size, hidden_size)
#
#             self.attention_layer = nn.Linear(hidden_size, hidden_size)
#             self.u_layer = nn.Linear(hidden_size, hidden_size, bias=False)
#
#         # lstm branch
#         self.lstm = nn.LSTM(input_size=n_features,
#                             hidden_size=hidden_size,
#                             num_layers=num_layers,
#                             dropout=dropout,
#                             batch_first=True)
#
#         self.linear = nn.Linear(hidden_size, dense_layer_size)
#         self.linear_out = nn.Linear(dense_layer_size, 1)
#         self.relu = nn.ReLU()
#
#     def simple_attention(self, lstm_out):
#         """
#         Following the implementation in:
#
#         1. Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
#         "Hierarchical Attention Networks for Document Classification"
#         accepted in NAACL 2016
#         2. Winata, et al. https://arxiv.org/abs/1805.12307
#         "Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision."
#         accepted in ICASSP 2018
#
#         implementation in TF:
#         https://github.com/gentaiscool/lstm-attention/blob/58adc7e345b5b3a79638483049704802a66aa1f4/layers.py#L50
#
#         follows these equations:
#
#         (1) u_t = tanh(W lstm_out + b)
#         (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
#         (3) z_t = \alpha_t * lstm_out, z in time t
#
#         # Input shape
#             3D tensor with shape: `(samples, steps, features)`.
#         # Output shape
#             z: 2D tensor with shape: `(samples, features)`.
#             alpha: 3D tensor with shape: `(samples, steps, features)`
#
#         """
#
#         # old: https://doi.org/10.1016/j.engappai.2020.103976
#
#         #         # attention branch
#         #         attention_parameter_vector = self.attention_dense(self.u_input)
#         #         # multiply lstm output with attention parameter vector
#         #         u_dot_y = torch.matmul(lstm_out, attention_parameter_vector) # N, T
#         #         alpha = F.softmax(u_dot_y, dim=1) # N, T
#
#         #         # combining lstm and attention
#         #         z = torch.matmul(lstm_out.transpose(1,2),alpha.unsqueeze(2)) # N, hidden_size, 1
#         #         z = z.squeeze() # N, hidden_size
#
#         u = torch.tanh(self.attention_layer(lstm_out))
#         alpha = self.u_layer(u)
#         alpha = F.softmax(alpha, dim=1)
#         z = torch.sum(lstm_out * alpha, dim=1)
#
#         return z, alpha
#
#     def forward(self, x):
#         # lstm_out = (batch_size, seq_len, hidden_size)
#         lstm_out, (hn, cn) = self.lstm(x)
#
#         if self.attention:
#             z, alpha = self.simple_attention(lstm_out)
#             out = self.relu(z)
#         else:
#             out = self.relu(lstm_out[:, -1])
#             # out = self.relu(hn[-1,:]) # it is the same as the line above
#
#         # dense layers
#         out = self.linear(out)
#         out = self.relu(out)
#         y_pred = self.linear_out(out)
#
#         return y_pred
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.reshape((-1, 1))
#         y_hat = self(x)
#         loss = self.criterion(y_hat, y)
#         # logs metrics for each training_step,
#         # and the average across the epoch, to the progress bar and logger
#         self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         #         self.additional_logging(y_hat, y, stage="train")
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.reshape((-1, 1))
#         y_hat = self(x)
#         loss = self.criterion(y_hat, y)
#         self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
#         self.additional_logging(y_hat, y, stage="val")
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y = y.reshape((-1, 1))
#         y_hat = self(x)
#
#         self.test_results["y_hat"].append(y_hat)
#         self.test_results["y"].append(y)
#
#         loss = self.criterion(y_hat, y)
#         self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         self.additional_logging(y_hat, y, stage="test")
#
#         return loss
#
#     def additional_logging(self, y_hat, y, stage):
#         # additional logging
#         y_hat = y_hat.cpu().reshape(-1).numpy()
#         y = y.cpu().reshape(-1).numpy()
#
#         # get original scale, i.e. inverse log10 transform
#         y_hat_org = exp10(y_hat)
#         y_org = exp10(y)
#
#         # calc metric on log scale
#         rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_hat, y))
#         spearmanr = scipy.stats.spearmanr(y_hat, y).correlation
#         r2 = sklearn.metrics.r2_score(y, y_hat)
#         mae = sklearn.metrics.mean_absolute_error(y_hat, y)
#         me = np.mean(y_hat - y)
#
#         # calc metrics on original scale
#         org_rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_hat_org, y_org))
#         org_spearmanr = scipy.stats.spearmanr(y_hat_org, y_org).correlation
#         org_r2 = sklearn.metrics.r2_score(y_org, y_hat_org)
#         org_mae = sklearn.metrics.mean_absolute_error(y_hat_org, y_org)
#         org_me = np.mean(y_hat_org - y_org)
#
#         self.log_dict(
#             {f"rmse_{stage}": rmse, f"mean_error_{stage}": me, f"mae_{stage}": mae, f"spearmanr_{stage}": spearmanr,
#              f"r2_{stage}": r2},
#             logger=True, on_epoch=True)
#
#         self.log_dict({f"org_rmse_{stage}": org_rmse, f"org_mean_error_{stage}": org_me, f"org_mae_{stage}": org_mae,
#                        f"org_spearmanr_{stage}": org_spearmanr, f"org_r2_{stage}": org_r2},
#                       logger=True, on_epoch=True)
#
#         # log gradients as histograms


class SimpleAttention(nn.Module):
    def __init__(self, layer_size, *args, **kwargs):
        super().__init__()
        self.input_size = layer_size
        self.attention_layer = nn.Linear(self.input_size, self.input_size)
        self.u_layer = nn.Linear(self.input_size, self.input_size, bias=False)

    def forward(self, x):
        u = torch.tanh(self.attention_layer(x))
        alpha = self.u_layer(u)
        alpha = F.softmax(alpha, dim=1)
        z = torch.sum(x * alpha, dim=1)

        return z, alpha


# lstm block
def lstm_block(input_size, hidden_size, num_layers, dropout, batch_first=True):
    lstm_cell = nn.LSTM(input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        batch_first=batch_first)

    return


def fc_layer(input_size, output_size, activation="relu", dropout=None, batchnorm=False):
    # todo add dropout, batch norm
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activations[activation]
    )


def multiple_fc_layers(layer_sizes, activation="relu", dropout=None, batchnorm=None):
    # todo doesnt work if len(layer_sizes)==0
    return nn.Sequential(
        *[fc_layer(in_f, out_f, activation, dropout, batchnorm) for in_f, out_f in zip(layer_sizes, layer_sizes[1:])])


class LSTMRegressor(pl.LightningModule):
    '''
    Standard PyTorch Lightning module:
    https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    '''

    def __init__(self,
                 predictands: list,
                 n_sequential_features: int,
                 n_static_features: int,
                 lstm_hparams: dict,
                 static_branch_hparams: dict,
                 final_fc_layers_hparams: dict,
                 batch_size: int,
                 learning_rate: float,
                 criterion,
                 grad_clip=False):
        super().__init__()

        ### assertions ###

        # multi task loss
        # keys in hparam dicts

        assert isinstance(criterion, MultiTaskLearningLoss) if len(predictands) > 1 else True, "criterion mast be of cla" \
                                                                                               "ss MultiTaskLearningLoss when training multiple predictands, is {}".format(type(criterion))

        ### init hparams ###

        # init general hparams
        self.predictands = predictands
        self.n_predictands = len(predictands)
        self.n_sequential_features = n_sequential_features
        self.n_static_features = n_static_features
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        # init hparams for lstm branch
        self.lstm_num_layers = lstm_hparams["num_layers"]
        self.lstm_hidden_size = lstm_hparams["hidden_size"]
        self.lstm_dropout = lstm_hparams["dropout"]
        self.attention = lstm_hparams["attention"]

        # init hparams for static branch
        self.static_branch_layer_sizes = static_branch_hparams["layer_sizes"]
        self.static_branch_layer_sizes.insert(0,
                                              self.n_static_features)  # input size of first layer is number of static features

        # init hparams for final fully connected layers
        self.final_fc_layer_sizes = final_fc_layers_hparams["layer_sizes"]
        fc_input_layer_size = self.lstm_hidden_size + self.static_branch_layer_sizes[
            -1] if self.n_static_features > 0 else self.lstm_hidden_size  # size of input layer is hidden_size of lstm + size of last layer of static branch (if exists)
        self.final_fc_layer_sizes.insert(0, fc_input_layer_size)

        # other
        self.test_results = {"y_hat": [], "y": []}  # todo adapt to Multi Task Learning

        ### define model ###

        # lstm branch
        self.lstm = nn.LSTM(input_size=self.n_sequential_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.lstm_dropout,
                            batch_first=True)

        # attention layer (part of lstm branch)
        if self.attention == True:
            self.attention_layer = SimpleAttention(
                layer_size=self.lstm_hidden_size)  # attention layer has same size as lstm hidden state
        else:
            self.attention_layer = None

        # static branch
        self.static_branch = multiple_fc_layers(layer_sizes=self.static_branch_layer_sizes)
        # todo potentially add other params dropout, batchnorm etc.

        # final fc branch: connecting temporal and static branch
        # if multi-task learning on final fc head per predictan
        self.final_layers_module_dict = nn.ModuleDict()
        for predictand in predictands:
            print(predictand)
            fc_layer_head = multiple_fc_layers(layer_sizes=self.final_fc_layer_sizes)
            last = nn.Linear(self.final_fc_layer_sizes[-1], 1)
            fc_layer_head = nn.Sequential(
                fc_layer_head,
                last
            )
            self.final_layers_module_dict[predictand] = fc_layer_head

    def forward(self, x_seq, x_static):
        ## sequential branch
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x_seq)
        if self.attention_layer:
            z, alpha = self.attention_layer(lstm_out)
            seq_out = F.relu(z)
        else:
            seq_out = F.relu(lstm_out[:, -1])
            # out = self.relu(hn[-1,:]) # it is the same as the line above

        ## static branch
        static_out = self.static_branch(x_static)

        ## concat outputs
        concat_out = torch.concat([seq_out, static_out], dim=1)

        ## fully connected layers
        preds = torch.concat([self.final_layers_module_dict[pred](concat_out) for pred in self.predictands],
                             -1)  # shape: n_samples, n_predictands

        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        X_seq, X_static, y, weights, coords = batch
        y_hat = self(X_seq, X_static)

        # if sample based weighted loss pass weights (i.e. imbalanced regression)
        if is_sample_based_weighted_loss(self.criterion):
            loss = self.criterion(y_hat, y, weights)
        else:
            loss = self.criterion(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #         self.additional_logging(y_hat, y, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        X_seq, X_static, y, weights, coords = batch
        y_hat = self(X_seq, X_static)

        # if sample based weighted loss pass weights (i.e. imbalanced regression)
        if is_sample_based_weighted_loss(self.criterion):
            loss = self.criterion(y_hat, y, weights)
        else:
            loss = self.criterion(y_hat, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.additional_logging(y_hat, y, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        X_seq, X_static, y, weights, coords = batch
        y_hat = self(X_seq, X_static)

        self.test_results["y_hat"].append(y_hat)
        self.test_results["y"].append(y)

        # if sample based weighted loss pass weights (i.e. imbalanced regression)
        if is_sample_based_weighted_loss(self.criterion):
            loss = self.criterion(y_hat, y, weights)
        else:
            loss = self.criterion(y_hat, y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.additional_logging(y_hat, y, stage="test")

        return loss

    def additional_logging(self, y_hat, y, stage):
        # additional logging
        y_hat = y_hat.cpu().reshape(-1).numpy()
        y = y.cpu().reshape(-1).numpy()

        # get original scale, i.e. inverse log10 transform
        y_hat_org = exp10(y_hat)
        y_org = exp10(y)

        # calc metric on log scale
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_hat, y))
        spearmanr = scipy.stats.spearmanr(y_hat, y).correlation
        r2 = sklearn.metrics.r2_score(y, y_hat)
        mae = sklearn.metrics.mean_absolute_error(y_hat, y)
        me = np.mean(y_hat - y)

        # calc metrics on original scale
        org_rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_hat_org, y_org))
        org_spearmanr = scipy.stats.spearmanr(y_hat_org, y_org).correlation
        org_r2 = sklearn.metrics.r2_score(y_org, y_hat_org)
        org_mae = sklearn.metrics.mean_absolute_error(y_hat_org, y_org)
        org_me = np.mean(y_hat_org - y_org)

        self.log_dict(
            {f"rmse_{stage}": rmse, f"mean_error_{stage}": me, f"mae_{stage}": mae, f"spearmanr_{stage}": spearmanr,
             f"r2_{stage}": r2},
            logger=True, on_epoch=True)

        self.log_dict({f"org_rmse_{stage}": org_rmse, f"org_mean_error_{stage}": org_me, f"org_mae_{stage}": org_mae,
                       f"org_spearmanr_{stage}": org_spearmanr, f"org_r2_{stage}": org_r2},
                      logger=True, on_epoch=True)

        # log gradients as histograms



