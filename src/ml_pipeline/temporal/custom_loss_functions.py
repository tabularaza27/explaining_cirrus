"""Losses implemented as pytorch modules"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=5, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=5, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_mse_loss(inputs, targets, weights=None):
    loss = F.mse_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduce=False)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


class ImbalancedRegressionLoss(nn.Module):
    """wrapper loss function for imbalanced regression losses, i.e. sample based weighting"""

    def __init__(self):
        super().__init__()

    def forward(self, yhat, y, weights):
        pass


class WeightedFocalMSELoss(ImbalancedRegressionLoss):
    def __init__(self, activate: str = 'sigmoid', beta: float = 5, gamma: int = 1):
        super().__init__()
        self.activate=activate
        self.beta=beta
        self.gamma=gamma

    def forward(self, yhat, y, weights):
        return weighted_focal_mse_loss(inputs=yhat,
                                       targets=y,
                                       weights=weights,
                                       activate=self.activate,
                                       beta=self.beta,
                                       gamma=self.gamma)


class WeightedFocalL1Loss(ImbalancedRegressionLoss):
    def __init__(self, activate: str = 'sigmoid', beta: float = 5, gamma: int = 1):
        super().__init__()
        self.activate = activate
        self.beta = beta
        self.gamma = gamma

    def forward(self, yhat, y, weights):
        return weighted_focal_l1_loss(inputs=yhat,
                                      targets=y,
                                      weights=weights,
                                      activate=self.activate,
                                      beta=self.beta,
                                      gamma=self.gamma)


class WeightedL1Loss(ImbalancedRegressionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y, weights):
        return weighted_l1_loss(yhat, y, weights)


class WeightedMSELoss(ImbalancedRegressionLoss):
    def __init__(self):
        super().__init__()

    def forward(self, yhat, y, weights):
        return weighted_mse_loss(yhat, y, weights)


class MultiTaskLearningLoss(nn.Module):
    """Multitask Learning Loss

    Weight losses of individual tasks and sum to total loss.

    Loss weighting strategies:
    - equal weights
    - uncertainty weighting [Kendall et al. 2018]
      implementations:
      https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
      https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb

    Can be combined with deep imbalanced regression, just pass a sample based criterion, e.g. weighted_focal_l1
    """

    def __init__(self, task_num: int, criterion: nn.Module, mtl_weighting_type: str = "equal"):
        """
        Args:
            task_num: number of predictands
            criterion: pytorch loss (nn.Module) to calculate individual losses
            mtl_weighting_type: weighting of individual losses
        """
        super().__init__()

        assert mtl_weighting_type in ["equal", "uncertainty",
                                      "sum"], 'mtl_weighting_type must be in ["equal","uncertainty"], ' \
                                              'is {}'.format(mtl_weighting_type)

        self.task_num = task_num
        self.criterion = criterion
        self.mtl_weighting_type = mtl_weighting_type  # "equal", "uncertainty"

        if self.mtl_weighting_type == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(task_num), requires_grad=True)

        print("initialized multitask loss with {} tasks, {} weighting and loss criterion {}".format(self.task_num,
                                                                                                    self.mtl_weighting_type,
                                                                                                    type(
                                                                                                        self.criterion)))

    def forward(self, yhat: list[torch.Tensor], y: list[torch.Tensor], weights: list[torch.Tensor] = None):
        # if deep imbalanced regression weighted loss pass weights, for test split all weights are 1

        # if isinstance(self.criterion, ImbalancedRegressionLoss):
        #     losses = torch.Tensor(
        #         [self.criterion(yhat[:, i], y[:, i], weights=weights[:, i]) for i in range(0, self.task_num)])
        # else:
        #     losses = torch.Tensor([self.criterion(yhat[:, i], y[:, i]) for i in range(0, self.task_num)])
        #
        # losses = losses.type_as(yhat) # transfer losses to gpu, see: https://pytorch-lightning.readthedocs.io/en/latest/accelerators/accelerator_prepare.html#init-tensors-using-type-as-and-register-buffer
        #
        # # equal weights
        # if self.mtl_weighting_type == "equal":
        #     weighted_losses = losses / self.task_num # n_predictands
        # elif self.mtl_weighting_type == "uncertainty":
        #     precisions = torch.exp(-self.log_vars)  # n_predictands
        #     weighted_losses = precisions * losses + self.log_vars  # n_predictands
        #
        # torch.sum(weighted_losses)

        # from https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
        # → don't fully understand though
        # loss = 0
        # for i in range(self.task_num):
        #     # sample based weighting unterscheidung
        #     # if isinstance(self.criterion, ImbalancedRegressionLoss):
        #     #     l = self.criterion(yhat[:, i], y[:, i], weights=weights[:, i])
        #     # else:
        #     #     l = self.criterion(yhat[:, i], y[:, i])
        #
        #     l = (yhat[:,i]-y[:,i])**2.
        #
        #     # loss weighting
        #     if self.mtl_weighting_type == "equal":
        #         loss += l
        #     elif self.mtl_weighting_type == "uncertainty":
        #         precision = torch.exp(-self.log_vars[i])
        #         loss += torch.sum(precision * l + self.log_vars[i], -1)  # n_predictands

        total_weighted_loss = 0
        for i in range(self.task_num):
            # sample based weighting unterscheidung
            if isinstance(self.criterion, ImbalancedRegressionLoss):
                if weights is None:
                    # only focal loss
                    predictand_loss = self.criterion(yhat[:, i], y[:, i], weights=None)
                else:
                    # weighted loss + evtl. focal
                    predictand_loss = self.criterion(yhat[:, i], y[:, i], weights=weights[:, i])
            else:
                predictand_loss = self.criterion(yhat[:, i], y[:, i])

            if self.mtl_weighting_type == "uncertainty":
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * predictand_loss + self.log_vars[i]
            elif self.mtl_weighting_type == "equal":
                weighted_loss = predictand_loss / self.task_num
            elif self.mtl_weighting_type == "sum":
                weighted_loss = predictand_loss

            total_weighted_loss += weighted_loss

        return total_weighted_loss


def is_sample_based_weighted_loss(criterion: nn.Module) -> bool:
    """checks wether criterion is sample based weighted loss for deep imbalanced regression, i.e. if forward method
    expects weights as argument
    """
    if isinstance(criterion, ImbalancedRegressionLoss):
        return True
    if isinstance(criterion, MultiTaskLearningLoss) and isinstance(criterion.criterion, ImbalancedRegressionLoss):
        return True
    else:
        return False

#
# def get_loss_module(loss, *args, **kwargs):
#     """returns instance of Loss """
#     losses = nn.ModuleDict([
#         ["rmse", RMSELoss(*args, **kwargs)],
#         ["mse", nn.MSELoss(*args, **kwargs)],
#         ["l1", nn.L1Loss(*args, **kwargs)],
#         ["weighted_focal_mse", WeightedFocalMSELoss(*args, **kwargs)],
#         ["weighted_focal_l1", WeightedFocalL1Loss(*args, **kwargs)],
#         ["mtl", MultiTaskLearningLoss(*args, **kwargs)]
#     ])
#
#     return losses[loss]
