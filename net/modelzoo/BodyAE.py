import os
import torch.nn as nn
import torch

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

    def save_model(self, path):
        """
        Saves the model parameters
        :param path: directory for saving the model
        :return: save successful or not in bool
        """

        # create the encoder and decoder paths
        enc_path = os.path.join(path, 'encoder/')
        dec_path = os.path.join(path, 'decoder/')

        # try to save the models
        # noinspection PyBroadException
        try:
            torch.save(self.encoder.state_dict(), enc_path)
            torch.save(self.decoder.state_dict(), dec_path)
            print("save successful!")
        except:
            print("save failed!")
            return False

        return True

    def load_model(self, path):
        """
        Loads the saved model parameters
        :param path: directory for saved model
        :return: load successful or not in bool
        """

        # create the encoder and decoder paths
        enc_path = os.path.join(path, 'encoder/')
        dec_path = os.path.join(path, 'decoder/')

        # try to load the models
        # noinspection PyBroadException
        try:
            self.encoder.load_state_dict(torch.load(enc_path))
            self.decoder.load_state_dict(torch.load(dec_path))
            print("Load successful!")
        except:
            print("Load failed!")
            return False

        return True

    def get_trainable_parameters(self):
        """
        Returns the trainable model parameters
        :return: Model parameters
        """

        params = [
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()},
        ]

        return params