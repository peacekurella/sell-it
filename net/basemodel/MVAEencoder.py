import torch
import torch.nn as nn


class Resnet(nn.Module):
    """Single layer frame by frame encoder"""

    def __init__(self, FLAGS):
        """
        :param FLAGS : abseil flags
        """
        super(Resnet, self).__init__()

        # define the model
        self.network = nn.Sequential(
            nn.Linear()

        )