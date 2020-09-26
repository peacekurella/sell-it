import torch.nn as nn


class ConvDecoderSingle(nn.Module):
    """Convolutional Decoder for Single Skeleton Output"""

    def __init__(self, FLAGS):
        """
        :param FLAGS: abseil flags
        """

        super(ConvDecoderSingle, self).__init__()
        self.network = nn.Sequential(
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
        )

    def forward(self, x):
        """
        Defines the foward pass through the network
        :param x: Input to the decoder (batch_size, f/2, 256)
        :return: Output of the decoder (batch_size, f, 73)
        """
        return self.network(x)
