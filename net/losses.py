import torch
import torch.nn as nn


def meanJointPoseError(predictions, targets):
    """
    Defines an MSE loss
    :param predictions: predictions from the model
    :param targets: ground truths
    :return: average loss for the predicition
    """
    pose_pred = predictions['pose']
    pose_gt = targets['pose']
    loss = nn.MSELoss(reduction='mean')
    return loss(pose_pred, pose_gt)


def reconstruction_l1(predictions, targets, model_params, FLAGS):
    """
    Defines a reconstruction loss with L1 regularization loss
    :param predictions: predcitions from the model
    :param targets: ground truth targets
    :param model_params: model params to calculate l1 loss over
    :param lmd: smoothing parameter
    :return: total loss for the predictions
    """

    lmd = FLAGS.lmd

    # set the criterion objects
    criterion1 = nn.MSELoss(reduction='mean')

    # calculate losses
    mse = criterion1(predictions['pose'], targets['pose'])
    l1_loss = 0
    for param in model_params:
        l1_loss += torch.norm(param, 1)

    loss = {
        "Total_Loss": mse + (lmd * l1_loss),
        "Reconstruction_Loss": mse,
        "Regularization_Loss": lmd * l1_loss,
        "CrossEntropy_Loss": torch.zeros(1)
    }

    return loss


def reconstruction_VAE(predictions, targets, model_params, FLAGS):
    """
    Defines the reconstruction and KL divergence loss for VAE
    :param predictions: prediction from model including mean and log_var
    :param target: ground truths
    :return: mean loss for the entire batch
    """
    del model_params
    prediction = predictions['pose']
    mu = predictions['mu'].reshape(-1, FLAGS.latent_dim)
    log_var = predictions['log_var'].reshape(-1, FLAGS.latent_dim)
    z = predictions['z'].reshape(-1, FLAGS.latent_dim)
    z_star = predictions['z_star'].reshape(-1, FLAGS.latent_dim)

    target = targets['pose']

    lmd = FLAGS.lmd
    lmd2 = FLAGS.lmd2

    # set the criterion objects for mse
    criterion1 = nn.MSELoss(reduction='mean')

    # calculate the reconstruction loss
    Reconstruction_Loss = criterion1(prediction, target)

    # calculate the KL Divergence loss
    loss_kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    # calculate cycle consistentcy loss
    loss_cycle = criterion1(z, z_star)

    loss = {
        "Total_Loss": Reconstruction_Loss + lmd * loss_kld + lmd2 * loss_cycle,
        "Reconstruction_Loss": Reconstruction_Loss,
        "Regularization_Loss": loss_kld,
        "CrossEntropy_Loss": torch.zeros(1)
    }

    return loss


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
    pose_pred = predictions['pose']
    mu = predictions['mus']
    log_var = predictions['log_vars']
    if FLAGS.speak:
        speech_pred = predictions['speech']

    lmd = FLAGS.lmd

    lmd2 = FLAGS.lmd2

    CrossEntropy_Loss = torch.zeros(1).cuda()
    # set the criterion objects for mse
    criterion1 = nn.SmoothL1Loss(reduction='mean')
    criterion3 = nn.L1Loss(reduction='mean')

    # calculate the reconstruction loss
    Reconstruction_Loss = criterion1(pose_pred, target['pose'][:, :pose_pred.shape[1]])

    # calculate the KL Divergence loss
    loss_kld = torch.mean(torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1), dim=0)

    # add motion regularization
    loss_vel = criterion3(pose_pred[:, :, 69:132], target['pose'][:, :, 69:132]) + criterion3(pose_pred[:, :, :2],
                                                                                              target['pose'][:, :, :2])

    if FLAGS.speak:
        criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
        CrossEntropy_Loss = criterion2(speech_pred, target['speech'][:, :pose_pred.shape[1]])

    losses = {
        'Total_Loss': Reconstruction_Loss + (lmd * loss_kld) + (lmd2 * CrossEntropy_Loss) + (lmd2 * loss_vel),
        'Regularization_Loss': loss_kld,
        'CrossEntropy_Loss': CrossEntropy_Loss,
        'Reconstruction_Loss': Reconstruction_Loss,
    }

    return losses
