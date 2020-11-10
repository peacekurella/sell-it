import torch
import torch.nn as nn

from LinearModel import LinearModel

class MVAEencoder(nn.Module):
    """Single layer frame by frame encoder"""

    def __init__(self, FLAGS):
        """
        Encoder layer for MVAE model
        :param FLAGS : abseil flags
        """
        super(MVAEencoder, self).__init__()

        # define the model
        self.network = LinearModel(FLAGS, FLAGS.input_dim*4, 256) # make this a FLAG later

        # two outputs layers for mean and std to sample from
        self.output_mean = nn.Linear(256, FLAGS.latent_dim)
        self.output_std = nn.Linear(256, FLAGS.latent_dim)

    def forward(self, x):
        """
        :param x: input 4 poses buyer, seller, previous pose seller2 and seller2 dimension (batch, 1, 292)
        """

        y = self.network(x)

        mean = self.output_mean(y)
        std = self.output_std(y)

        return mean, std
