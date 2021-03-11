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
    pose_pred = predictions['pose_pred']
    pose_gt = targets['trainy']
    loss = nn.MSELoss(reduction='mean')
    return loss(pose_pred, pose_gt)


def reconstruction_VAE(predictions, target, model_params, lmd):
    """
    Defines the reconstruction and KL divergence loss for VAE
    :param predictions: prediction from model including mean and log_var
    :param target: ground truths
    :return: mean loss for the entire batch
    """
    del model_params
    predictions, mu, log_var, z, z_star = predictions

    # set the criterion objects for mse
    criterion1 = nn.MSELoss(reduction='mean')

    # calculate the reconstruction loss
    loss_mse = criterion1(predictions, target)

    # calculate the KL Divergence loss
    loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    # calculate cycle consistentcy loss
    loss_cycle = criterion1(z, z_star)

    return loss_mse + lmd*loss_kld + 0.1*loss_cycle, loss_mse, loss_kld

def sequential_reconstruction_VAE(predictions, target, model_params, FLAGS):
    """
    Returns the reconstruction and KL divergence loss
    :param predictions: predictions from the network
    :param target: target output
    :param model_parametres
    :param lmd: beta value for KLD
    :return: mean loss for the entire batch
    """

    del model_params

    # unpack the predictions
    pose_pred = predictions['pose_pred']
    mu = predictions['mus']
    log_var = predictions['log_vars']
    if FLAGS.speak:
        speech_pred = predictions['speech_pred']

    lmd = FLAGS.lmd

    lmd2 = FLAGS.lmd2

    loss_speech = torch.zeros(1).cuda()
    # set the criterion objects for mse
    criterion1 = nn.SmoothL1Loss(reduction='mean')

    # calculate the reconstruction loss
    loss_mse = criterion1(pose_pred, target['trainy'])

    # calculate the KL Divergence loss
    loss_kld = torch.mean(torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1), dim=0)

    if FLAGS.speak:
        criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
        loss_speech = criterion2(speech_pred, target['speaky'])

    losses = {
        'total_loss': loss_mse + (lmd * loss_kld) + (lmd2 * loss_speech),
        'loss_kld': loss_kld,
        'loss_speech': loss_speech,
        'loss_mse': loss_mse,
    }

    return losses