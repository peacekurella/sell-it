import os
import json
import numpy as np
import torch

from absl import app
from absl import flags

from torch.utils.data import Dataset, DataLoader
from DebugVisualizer import DebugVisualizer
from preprocess import SkeletonHandler
import Quaternions as Quaternions

from HagglingDataset import HagglingDataset
from BodyAE import BodyAE

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('input', 'Data/train/', 'Directory containing input files')

flags.DEFINE_integer('batch_size', 64, 'Training set mini batch size')
flags.DEFINE_bool('pretrain', True, 'pretrain the auto encoder')

def get_inputs(pretrain, batch):
    """
    Generates the inputs to the network
    :param pretrain: boolean variable set to True if same input and output is needed
    :param batch: batch for training
    :return: train_x, train_y tensors
    """
    b = batch['buyer']['joints21']
    l = batch['leftSeller']['joints21']
    r = batch['rightSeller']['joints21']

    if pretrain:
        train_x = torch.cat((b, l, r), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((b, l, r), dim=0).permute(0, 2, 1).float().cuda()
        return train_x, train_y
    else:
        set_x_a = torch.cat((b, l), dim=2)
        set_x_b = torch.cat((b, r), dim=2)
        train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
        return train_x, train_y


def get_model():
    """
    Returns the appropriate model for training
    :return: PyTorch model that extends nn.Module
    """

    return BodyAE(FLAGS).cuda()


def main(args):
    # initialize the dataset and the data loader
    normalized_dataset = HagglingDataset(FLAGS)
    dataloader = DataLoader(normalized_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    # initialize the model
    model = get_model()

    for i_batch, batch in enumerate(dataloader):
        train_x, train_y = get_inputs(True, batch)
        train_y_hat = model(train_x)

        print((train_y - train_y_hat))

        break


if __name__ == '__main__':
    app.run(main)
