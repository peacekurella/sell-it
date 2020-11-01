import os
import torch.nn as nn
import torch
import glob

from ConvEncoderDouble import ConvEncoderDouble
from ConvDecoderSingle import ConvDecoderSingle


class BodyMotionGenerator(nn.Module):
    """Defines the body auto encoder class"""

    def __init__(self, FLAGS):

        super(BodyMotionGenerator, self).__init__()
        self.encoder = ConvEncoderDouble(FLAGS)
        self.decoder = ConvDecoderSingle(FLAGS)

    def forward(self, x):
        """
        Defines a forward pass through the Body Auto Encoder
        :param x: Input vector of shape (batch_size, f, 146)
        :return: output vector of shape (batch_size, f, 73)
        """

        latent = self.encoder(x)
        out = self.decoder(latent)

        return out

    def save_model(self, path, epoch):
        """
        Saves the model parameters
        :param path: directory for saving the model
        :param epoch: epoch number
        :return: save successful or not in bool
        """

        # create the encoder and decoder paths
        enc_path = os.path.join(path, 'encoder/')
        dec_path = os.path.join(path, 'decoder/')

        # create a directory if not a directory
        if not os.path.isdir(enc_path):
            os.makedirs(enc_path)
        if not os.path.isdir(dec_path):
            os.makedirs(dec_path)

        # try to save the models
        # noinspection PyBroadException
        try:
            torch.save(self.encoder.state_dict(), enc_path + str(epoch) + '.ckpt')
            torch.save(self.decoder.state_dict(), dec_path + str(epoch) + '.ckpt')
            print("save successful!")
        except:
            print("save failed!")
            return False

        return True

    def load_model(self, path, epoch):
        """
        Loads the saved model parameters
        :param path: directory for saved model
        :param epoch: epoch number
        :return: load successful or not in bool
        """

        # create the encoder and decoder paths
        enc_path = os.path.join(path, 'encoder/')
        dec_path = os.path.join(path, 'decoder/')

        # check if epoch number is passed
        if epoch is not None:
            enc_path = enc_path + str(epoch) + '.ckpt'
            dec_path = dec_path + str(epoch) + '.ckpt'
        else:
            enc_path = max(glob.glob(enc_path + '*'), key=os.path.getctime)
            dec_path = max(glob.glob(dec_path + '*'), key=os.path.getctime)

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
            {'params': self.encoder.parameters()}
        ]

        return params

    def load_transfer_params(self, path, epoch):
        """
        loads transferable parameteres from other models
        :param path: directory for saved model
        :param epoch: epoch number
        :return: load succesful or not in bool
        """

        # create the encoder and decoder paths
        dec_path = os.path.join(path, 'decoder/')

        # check if epoch number is passed
        if epoch is not None:
            dec_path = dec_path + str(epoch) + '.ckpt'
        else:
            dec_path = max(glob.glob(dec_path + '*'), key=os.path.getctime)

        # try to load the models
        # noinspection PyBroadException
        try:
            self.decoder.load_state_dict(torch.load(dec_path))
            print("Transfer load successful!")
        except:
            print("Transfer load failed!")
            return False

        return True