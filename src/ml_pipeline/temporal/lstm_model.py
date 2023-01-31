import numpy as np
import sklearn
import scipy
from scipy.special import exp10

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl


class LogCallback(pl.callbacks.Callback):

    def on_test_start(self, trainer, pl_module):
        # empty test results
        pl_module.test_results = {"y": [], "y_hat": [], "coords": []}

    def on_test_end(self, trainer, pl_module):
        """create distribution/residual figures for predictands"""

        # get comet logger
        if isinstance(trainer.logger, pl.loggers.comet.CometLogger):
            # only one logger exists
            comet_logger = trainer.logger
        else:
            # multiple logger exist
            comet_logger_idx = np.argmax(
                [isinstance(logger, pl.loggers.comet.CometLogger) for logger in trainer.logger])
            comet_logger = trainer.logger[comet_logger_idx]

        # get predictions for whole test dataset
        y = torch.concat(trainer.model.test_results["y"]).cpu().numpy()
        y_hat = torch.concat(trainer.model.test_results["y_hat"]).cpu().numpy()

        for pred_idx, predictand in enumerate(trainer.model.predictands):
            y_pred = y[:, pred_idx]
            y_hat_pred = y_hat[:, pred_idx]


class SimpleAttention(nn.Module):
    """
    Following the implementation in:

    1. Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    accepted in NAACL 2016
    2. Winata, et al. https://arxiv.org/abs/1805.12307
    "Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision."
    accepted in ICASSP 2018

    implementation in TF:
    https://github.com/gentaiscool/lstm-attention/blob/58adc7e345b5b3a79638483049704802a66aa1f4/layers.py#L50

    follows these equations:

    (1) u_t = tanh(W lstm_out + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) z_t = \sum(\alpha_t * lstm_out) # sum along time dimension

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        z: 2D tensor with shape: `(samples, features)`.
        alpha: 3D tensor with shape: `(samples, steps, features)`

    """

    def __init__(self, layer_size, *args, **kwargs):
        super().__init__()
        self.input_size = layer_size

        # context matrix: one weight per timestep per hidden size neuron
        # self.attention_layer = nn.Linear(self.input_size, self.input_size)
        # self.u_layer = nn.Linear(self.input_size, self.input_size, bias=False)

        # context vector: one weight per timestep
        self.attention_layer = nn.Linear(self.input_size, self.input_size)  # attention layer
        self.u = nn.Parameter(torch.rand(self.input_size), requires_grad=True)  # context vector

    def forward(self, x):
        # u = torch.tanh(self.attention_layer(x))
        # alpha = self.u_layer(u)
        # alpha = F.softmax(alpha, dim=1)
        # z = torch.sum(x * alpha, dim=1)

        # (1)
        u_it = torch.tanh(self.attention_layer(x))

        # (2)
        alpha = torch.matmul(u_it, self.u)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.unsqueeze(-1)  # add dim

        # (3)
        z = torch.sum(alpha * x, dim=1)  # sum along time axis

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
    """creates fc layer with optional batchnorm and dropout

    ordering: fc --> activation --> bn --> dropout
    because of: https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
    and: https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

    Args:
        input_size:
        output_size:
        activation:
        dropout:
        batchnorm:

    Returns:

    """
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU()]
    ])

    modules = [nn.Linear(input_size, output_size), activations[activation]]

    if batchnorm:
        modules.append(nn.BatchNorm1d(output_size))

    if dropout:
        modules.append(nn.Dropout(dropout))

    return nn.Sequential(*modules)


def multiple_fc_layers(layer_sizes, activation="relu", dropout=None, batchnorm=None):
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
                 criterion,
                 learning_rate: float,
                 lr_scheduler: bool = False,
                 grad_clip=False,
                 log_transform_predictands: list = ["iwc", "icnc_5um"]):
        super().__init__()

        ### assertions ###

        # multi task loss
        # keys in hparam dicts

        # todo uncomment
        # assert isinstance(criterion, MultiTaskLearningLoss) if len(
        #     predictands) > 1 else True, "criterion must be of cla" \
        #                                 "ss MultiTaskLearningLoss when training multiple predictands, is {}".format(
        #     type(criterion))

        ### init hparams ###

        # only for logging purposes
        self.log_transform_predictands = log_transform_predictands

        # init general hparams
        self.predictands = predictands
        self.n_predictands = len(predictands)
        self.n_sequential_features = n_sequential_features
        self.n_static_features = n_static_features
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
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
        self.static_branch_dropout = static_branch_hparams["dropout"]
        self.static_branch_batchnorm = static_branch_hparams["batchnorm"]

        # init hparams for final fully connected layers
        self.final_fc_layer_sizes = final_fc_layers_hparams["layer_sizes"]
        fc_input_layer_size = self.lstm_hidden_size + self.static_branch_layer_sizes[
            -1] if self.n_static_features > 0 else self.lstm_hidden_size  # size of input layer is hidden_size of lstm + size of last layer of static branch (if exists)
        self.final_fc_layer_sizes.insert(0, fc_input_layer_size)
        self.final_fc_layer_dropout = final_fc_layers_hparams["dropout"]
        self.final_fc_layer_batchnorm = final_fc_layers_hparams["batchnorm"]

        # other
        self.test_results = {"y_hat": [], "y": [], "coords": []}

        ### define model ###

        # lstm branch
        self.lstm = nn.LSTM(input_size=self.n_sequential_features,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_num_layers,
                            dropout=self.lstm_dropout,
                            batch_first=True)

        # attention layer (part of lstm branch)
        if self.attention == True:
            self.attention_module = SimpleAttention(
                layer_size=self.lstm_hidden_size)  # attention layer has same size as lstm hidden state
        else:
            self.attention_module = None

        # static branch
        self.static_branch = multiple_fc_layers(layer_sizes=self.static_branch_layer_sizes,
                                                dropout=self.static_branch_dropout,
                                                batchnorm=self.static_branch_batchnorm)

        # final fc branch: connecting temporal and static branch
        # if multi-task learning on final fc head per predictan
        self.final_layers_module_dict = nn.ModuleDict()
        for predictand in predictands:
            fc_layer_head = multiple_fc_layers(layer_sizes=self.final_fc_layer_sizes,
                                               dropout=self.final_fc_layer_dropout,
                                               batchnorm=self.final_fc_layer_batchnorm)
            last = nn.Linear(self.final_fc_layer_sizes[-1], 1)
            fc_layer_head = nn.Sequential(
                fc_layer_head,
                last
            )
            self.final_layers_module_dict[predictand] = fc_layer_head

        # fc_layer_head = multiple_fc_layers(layer_sizes=self.final_fc_layer_sizes, dropout=self.final_fc_layer_dropout,
        #                                    batchnorm=self.final_fc_layer_batchnorm)
        # last = nn.Linear(self.final_fc_layer_sizes[-1], len(self.predictands))
        # self.final_layers = nn.Sequential(
        #         fc_layer_head,
        #         last
        #     )

    def forward(self, x_seq, x_static):
        ## sequential branch
        lstm_out, (hn, cn) = self.lstm(x_seq)  # lstm_out[:,-1,:] == hn
        if self.attention_module:
            z, alpha = self.attention_module(lstm_out)
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

        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.lr_scheduler:
            # select lr_scheduler
            if self.lr_scheduler == "cosine_annealing":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=20)
            elif self.lr_scheduler == "cosine_annealing_wr":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim, T_0=3)
            elif self.lr_scheduler == "exponential":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.99)
            else:
                raise ValueError("lr_scheduler {} not implemented".format(self.lr_scheduler))
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}
        else:
            # no scheduler
            return optim

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
        # self.additional_logging(y_hat, y, stage="train")
        # → above line raises: y_hat = y_hat.cpu().numpy()
        # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
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
        self.test_results["coords"].append(coords)

        # if sample based weighted loss pass weights (i.e. imbalanced regression)
        if is_sample_based_weighted_loss(self.criterion):
            loss = self.criterion(y_hat, y, weights)
        else:
            loss = self.criterion(y_hat, y)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.additional_logging(y_hat, y, stage="test")

        return loss

    def additional_logging(self, y_hat: torch.Tensor, y: torch.Tensor, stage: str):
        # create numpy arrays from torch tensors
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()

        # log performance metrics for each predictand
        for idx, predictand in enumerate(self.predictands):
            y_hat_pred = y_hat[:, idx]
            y_pred = y[:, idx]
            self.log_performance_metrics_single_predictand(predictand, y_hat_pred, y_pred, stage)

            if predictand in self.log_transform_predictands:
                # get original scale, i.e. inverse log10 transform
                y_hat_pred_org = exp10(y_hat_pred)
                y_pred_org = exp10(y_pred)

                self.log_performance_metrics_single_predictand(f"{predictand}_org_scale", y_hat_pred_org, y_pred_org,
                                                               stage)

        # log weights for multi tasking loss
        if isinstance(self.criterion, MultiTaskLearningLoss) and self.criterion.mtl_weighting_type == "uncertainty":
            log_vars = self.criterion.log_vars.cpu().numpy()
            self.log_dict({f"{pred}_mtl_weight": log_var for pred, log_var in zip(self.predictands, log_vars)})

        # log gradients as histograms

        # log lr
        self.log("lr", self.learning_rate)

    def log_performance_metrics_single_predictand(self, predictand: str, y_hat: np.ndarray, y: np.ndarray, stage: str):
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_hat, y))
        spearmanr = scipy.stats.spearmanr(y_hat, y).correlation
        pearsonr, pearsonp = scipy.stats.pearsonr(y_hat, y)
        r2 = sklearn.metrics.r2_score(y, y_hat)
        mae = sklearn.metrics.mean_absolute_error(y_hat, y)
        me = np.mean(y_hat - y)

        self.log_dict(
            {f"rmse_{predictand}_{stage}": rmse, f"mean_error_{predictand}_{stage}": me,
             f"mae_{predictand}_{stage}": mae, f"spearmanr_{predictand}_{stage}": spearmanr,
             f"pearsonr_{predictand}_{stage}": pearsonr, f"pearsonp_{predictand}_{stage}": pearsonp,
             f"r2_{predictand}_{stage}": r2},
            logger=True, on_epoch=True)
