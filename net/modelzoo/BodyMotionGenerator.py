import os
import torch.nn as nn
import torch
import glob

from net.basemodel.ConvDecoderSingle import ConvDecoderSingle
from net.basemodel.ConvEncoderDouble import ConvEncoderDouble


class BodyMotionGenerator(nn.Module):
    """Defines the body auto encoder class"""

    def __init__(self, FLAGS):

        super(BodyMotionGenerator, self).__init__()
        self.encoder = ConvEncoderDouble(FLAGS)
        self.decoder = ConvDecoderSingle(FLAGS)
        pretrained_ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.pretrainedModel + '/')

        # set device
        self.device = torch.device(FLAGS.device)

        if not self.load_transfer_params(pretrained_ckpt, FLAGS.pretrained_ckpt):
            raise Exception("bodyAE model needs to be trained")

    def forward(self, x):
        """
        Defines a forward pass through the Body Auto Encoder
        :param x: Input vector of shape (batch_size, input_dim*2, f)
        :return: output vector of shape (batch_size, output_dim, f)
        """
        x, y = self.transform_inputs(x)
        # set it to input device
        x = x.to(self.device)
        y = y.to(self.device)

        latent = self.encoder(x)
        out = self.decoder(latent)

        prediction = {
            "pose": out
        }

        target = {
            "pose": y
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

    def transform_inputs(self, batch):
        """
        Transforms the input dictionary to
        inputs for the model (batch , sequence_length, input_dim*2)
        """
        b = batch['buyer']['joints21']
        l = batch['leftSeller']['joints21']
        r = batch['rightSeller']['joints21']

        # Input splitting
        speaking_status = {
            'buyer': batch['buyer']['speakingStatus'],
            'leftSeller': batch['leftSeller']['speakingStatus'],
            'rightSeller': batch['rightSeller']['speakingStatus']
        }

        set_x_a = torch.cat((b, l), dim=2)
        set_x_b = torch.cat((b, r), dim=2)
        train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float()

        return train_x, train_y