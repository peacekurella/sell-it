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
        l1_loss +=  torch.norm(param, 1)

    return mse + (lmd * l1_loss)

def meanJointPoseError(predictions, targets):
    """
    Defines an MSE loss
    :param predictions: predictions from the model
    :param targets: ground truths
    :return: average loss for the predicition
    """

    loss = nn.MSELoss(reduction='mean')
    return  loss(predictions, targets)