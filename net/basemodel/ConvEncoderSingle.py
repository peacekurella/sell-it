import torch.nn as nn


class ConvEncoderSingle(nn.Module):
    """Single Skeleton Encoder"""

    def __init__(self, FLAGS):
        """
        :param FLAGS: abseil flags
        """
        super(ConvEncoderSingle, self).__init__()

        # define the network
        self.network = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(73, 256, 25, padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        Defines the foward pass through the network
        :param x: Input to the encoder (batch_size, f, 73)
        :return: Output of the encoder (batch_size, f/2, 256)
        """
        return self.network(x)