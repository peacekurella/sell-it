import torch.nn as nn
from ConvEncoderSingle import ConvEncoderSingle
from ConvDecoderSingle import ConvDecoderSingle

class BodyAE(nn.Module):
    """Defines the body auto encoder class"""

    def __init__(self, FLAGS):

        super(BodyAE, self).__init__()
        self.encoder = ConvEncoderSingle(FLAGS)
        self.decoder = ConvDecoderSingle(FLAGS)

    def forward(self, x):
        """
        Defines a foward pass through the Body Auto Encoder
        :param x: Input vector of shape (batch_size, f, 73)
        :return: output vector of shape (batch_size, f, 73)
        """

        latent = self.encoder(x)
        out = self.decoder(latent)

        return out
