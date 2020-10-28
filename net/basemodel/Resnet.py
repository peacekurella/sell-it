import torch
import torch.nn as nn


class Resnet(nn.Module):

    def __init__(self, input_dimensions, latent_dimension):
        """define a resnet dense layer"""
        super(Resnet, self).__init__()
        self.mean = nn.Linear(input_dimensions, latent_dimension)
        self.std = nn.Linear(input_dimensions, latent_dimension)

    def forward(self, x):
        """forward pass to produce transformation into latent dimension"""
        x_mean = self.mean(x)
        x_std = torch.relu(self.std(x))

        return x_mean, x_std
