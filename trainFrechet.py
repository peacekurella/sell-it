import time
import sys
import os
import torch
import torch.nn as nn
import random
import numpy as np

sys.path.append("net")
sys.path.append("net/modelzoo")
sys.path.append("net/basemodel")

from DebugVisualizer import DebugVisualizer
from JointsDataset import JointsDataset

[sys.path.append(i) for i in ['.', '..']]

from torch import optim
from torch.utils.data import DataLoader
import matplotlib

from EmbeddingNet import EmbeddingNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('train', 'FrechetData/train/', 'Directory containing train files')
flags.DEFINE_string('test', 'FrechetData/test/', 'Directory containing train files')
flags.DEFINE_string('ckpt_dir', 'ckpt/', 'Directory to store checkpoints')
flags.DEFINE_string('model', 'frechet', 'model type')
flags.DEFINE_integer('batch_size', 48, 'Training set mini batch size')
flags.DEFINE_integer('epochs', 400, 'Training epochs')
flags.DEFINE_integer('nframes', 120, 'Window size in number of frames')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate')
flags.DEFINE_integer('ckpt', 10, 'Number of epochs to checkpoint')


def get_inputs(batch):
    b = batch['buyer']['directions']
    l = batch['leftSeller']['directions']
    r = batch['rightSeller']['directions']

    trainx = torch.cat((b, l, r), dim=0).float().cuda()

    return {'data':trainx, 'target': trainx}

def reconstruction(predictions, targets):
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

    return mse


def get_loss_fn():
    """
    Returns the appropriate loss function for training
    :return: loss function
    """
    return reconstruction

def reconstruct(predictions, batch, idx, num):
    vis = DebugVisualizer()
    skeleton = vis.humanSkeleton

    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1], -1, 3))

    keys = ['buyer', 'leftSeller', 'rightSeller']

    skels_pred = []

    skels_target = []

    for i in range(3):
        pred = predictions[i*FLAGS.batch_size + idx, :, :, :]
        positions = batch[keys[i]]['positions'][idx, :, :, :]
        new_skeleton = positions.copy()

        for idx, bone in enumerate(skeleton):
            new_skeleton[:, bone[1], :] = pred[:, idx, :] + new_skeleton[:, bone[0], :]

        skels_pred.append(vis.conv_debug_visual_form(new_skeleton))

        skels_target.append(vis.conv_debug_visual_form(positions))

    skels = skels_target + skels_pred

    vis.create_animation(skels, 'FrechetTest/'+str(num))


def main(args):

    # initialize the dataset and the data loader
    train_dataset = JointsDataset(FLAGS.train, FLAGS)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=10)
    test_dataset = JointsDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=10)

    # train
    pose_dim = 42  # 9 x 3

    # init model and optimizer
    model = EmbeddingNet(pose_dim, FLAGS.nframes).cuda()
    criterion = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, betas=(0.5, 0.999))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(FLAGS.epochs):

        total_train_loss = 0.0
        total_val_loss = 0.0

        model.train()

        for iter_idx, batch in enumerate(train_dataloader):
            inputs = get_inputs(batch)

            predictions, poses_feat, poses_mu, poses_logvar = model(inputs['data'])

            loss = criterion(predictions, inputs['target'])

            total_train_loss += loss.detatch().item()

            # calculate gradients
            loss.backward()
            optimizer.step()

        # set the model to evaluation mode
        model.eval()

        # calculate validation loss
        with torch.no_grad():
            count = 0
            for i_batch, batch in enumerate(test_dataloader):
                # get train input and labels
                inputs = get_inputs(batch)

                # forward pass through the network
                predictions, poses_feat, poses_mu, poses_logvar = model(inputs['data'])

                # calculate loss
                total_val_loss += reconstruction(predictions, inputs).detach().item()

                if epoch % 100 == 0:
                    if count < 5:
                        rand = random.randrange(0, FLAGS.batch_size)
                        reconstruct(predictions, batch, rand, epoch+count)
                        count += 1


        if epoch % FLAGS.ckpt == 0:
            ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.model + '/')
            model.save_model(ckpt, epoch)

if __name__ == '__main__':
    app.run(main)
