import os
import torch
import torch.nn as nn
import torch.onnx
import glob

from ConvEncoderSingle import ConvEncoderSingle
from ConvDecoderSingle import ConvDecoderSingle


class BodyAE(nn.Module):
    """Defines the body auto encoder class"""

    def __init__(self, FLAGS):
        """
        Constructor for the class
        :param FLAGS: abseil flags
        """

        super(BodyAE, self).__init__()
        self.encoder = ConvEncoderSingle(FLAGS)
        self.decoder = ConvDecoderSingle(FLAGS)
        self.device = torch.device(FLAGS.device)

    def forward(self, x):
        """
        Defines a foward pass through the Body Auto Encoder
        :param x: Input vector of shape (batch_size, f, 73)
        :return: output vector of shape (batch_size, f, 73)
        """

        # store for exporting
        self.dummy_input = x

        # transform the input to the required shape and subjects
        x, y = self.transform_inputs(x)
        x = x.to(self.device)
        y = y.to(self.device)

        latent = self.encoder(x)
        out = self.decoder(latent)

        prediction = {
            "pose": out.permute(0, 2, 1)
        }

        target = {
            "pose": y.permute(0, 2, 1)
        }

        return prediction, target

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
            enc_path = max(glob.glob(enc_path+'*'), key=os.path.getctime)
            dec_path = max(glob.glob(dec_path + '*'), key=os.path.getctime)

        # try to load the models
        # noinspection PyBroadException
        print(enc_path)
        print(dec_path)
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

    def transform_inputs(self, batch):
        """
        Transforms the input dictionary to
        inputs for the model (batch , input_dim, sequence_length)
        """

        l = batch['leftSeller']['joints21']
        r = batch['rightSeller']['joints21']

        train_x = torch.cat((r, l), dim=0).permute(0, 2, 1).float()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float()
        return train_x, train_y
