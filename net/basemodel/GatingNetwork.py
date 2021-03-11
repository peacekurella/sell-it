import torch
import torch.nn as nn

from LinearModel import LinearModel


class GatingNetwork(nn.Module):

    def __init__(self, FLAGS):
        super(GatingNetwork, self).__init__()

        self.network = LinearModel(FLAGS, (FLAGS.input_dim * 3) + FLAGS.latent_dim + FLAGS.c_dim, FLAGS.num_experts, 'gat')
        self.activation = nn.Softmax(dim=1)

    def forward(self, x, z):
        x = self.network(torch.cat([x, z], dim=1))
        return self.activation(x)
