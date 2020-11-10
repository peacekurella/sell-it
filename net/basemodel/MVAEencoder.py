import torch
import torch.nn as nn


class Resnet(nn.Module):
    """Single layer frame by frame encoder"""

    def __init__(self, FLAGS):
        """
        Encoder layer for MVAE model
        :param FLAGS : abseil flags
        """
        super(Resnet, self).__init__()

        # define the model
        self.network = nn.Sequential(
            nn.Linear(146, 256),
            nn.ELU(),

            nn.Linear(146, 256),
            nn.ELU(),

            nn.Linear(146, 256),
            nn.ELU()
        )

        # two outputs layers for mean and std to sample from
        self.output_mean = nn.Linear(256, 32)
        self.output_std = nn.Linear(256, 32)
