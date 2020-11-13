import torch
import torch.nn as nn


class MVAEdecoder(nn.Module):

    def __init__(self, FLAGS):
        """
        Decoder module for the char motion decoder
        :param FLAGS: abseil flags
        """

        super(MVAEdecoder, self).__init__()

        self.lys = nn.ModuleList([
            nn.Linear((FLAGS.input_dim * 3) + FLAGS.latent_dim, FLAGS.dec_hidden_units),
            nn.Linear(FLAGS.dec_hidden_units + FLAGS.latent_dim, FLAGS.dec_hidden_units),
            nn.Linear(FLAGS.dec_hidden_units + FLAGS.latent_dim, FLAGS.dec_hidden_units),
            nn.Linear(FLAGS.dec_hidden_units, FLAGS.output_dim)
        ])

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(FLAGS.dec_dropout)

    def forward(self, x, z):
        """
        forward pass through the module
        :param x: input to the module
        :param z: random variable
        :return: output
        """

        for layer in self.lys[:-1]:
            x = self.dropout(x)
            x = layer(torch.cat([x, z], dim=1))
            x = self.activation(x)

        return self.lys[-1](x)