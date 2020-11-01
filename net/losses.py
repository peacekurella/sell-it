import torch
import torch.nn as nn


def reconstruction_l1(predictions, targets, model_params, lmd):
    """
    Defines a reconstruction loss with L1 regularization loss
    :param predictions: predcitions from the model
    :param targets: ground truth targets
    :param model_params: model params to calculate l1 loss over
    :param lmd: smoothing parameter
    :return: total loss for the predictions
    """

    # set the criterion objects
    criterion1 = nn.MSELoss(reduction='mean')

    # calculate losses
    mse = criterion1(predictions, targets)
    l1_loss = 0
    for param in model_params:
        l1_loss += torch.norm(param, 1)

    return mse + (lmd * l1_loss), mse, lmd * l1_loss


def meanJointPoseError(predictions, targets):
    """
    Defines an MSE loss
    :param predictions: predictions from the model
    :param targets: ground truths
    :return: average loss for the predicition
    """

    loss = nn.MSELoss(reduction='mean')
    return loss(predictions, targets)


def reconstruction_VAE(predictions, target, model_params, lmd):
    """
    Defines the reconstruction and KL divergence loss for VAE
    :param predictions: prediction from model including mean and log_var
    :param target: ground truths
    :return: mean loss for the entire batch
    """
    # delete model params and lmd
    del model_params, lmd

    predictions, mu, log_var = predictions

    # set the criterion objects for mse
    criterion1 = nn.MSELoss(reduction='mean')

    # calculate the reconstruction loss
    loss_mse = criterion1(predictions, target)

    # calculate the KL Divergence loss
    loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    return loss_mse + lmd*loss_kld, loss_mse, loss_kld
