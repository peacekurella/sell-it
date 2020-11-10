import torch
import torch.nn as nn


class MVAEdecoder(nn.Module):

    def __init__(self, FLAGS):
        """
        Decoder module for the char motion decoder
        :param FLAGS: abseil flags
        """

        super(MVAEdecoder, self).__init__()

        self.layers = nn.ModuleList([
            nn.Linear((FLAGS.input_dim * 3) + FLAGS.latent_dim, 256),
            nn.Linear(256 + FLAGS.latent_dim, 256),
            nn.Linear(256 + FLAGS.latent_dim, 256),
            nn.Linear(256, FLAGS.output_dim)
        ])

        self.activation = nn.ELU()

    def forward(self, x, z):
        """
        forward pass through the module
        :param x: input to the module
        :param z: random variable
        :return: output
        """

        for layer in self.layers[:-1]:
            x = layer(torch.cat([x, z], dim=1))
            x = self.activation(x)
            print(x.shape)

        return self.layers[-1](torch.cat([x, z], dim=1))