import os
import torch
import random
import glob
import torch.nn as nn

from LstmEncoder import LstmEncoder
from LstmDecoder import LstmDecoder


class LstmBodyAE(nn.Module):
    """Defines the body motion Auto Encoder with LSTM"""

    def __init__(self, FLAGS):
        """
        Constructor for the class
        :param FLAGS: abseil FLAGS
        """

        super(LstmBodyAE, self).__init__()
        self.encoder = LstmEncoder(FLAGS)
        self.decoder = LstmDecoder(FLAGS)

        # set the teacher forcing ratio
        self.tf_ratio = FLAGS.tf_ratio

    def forward(self, input):
        """
        Defines a forward pass through the network
        :param input: input to the network of shape (batch_size, seq_length, 73)
        :return: output from the network of shape (batch_size, seq_length, 73)
        """

        # check to see whether teacher forcing needs to be enabled
        if self.training:
            teacher_forcing = True if random.random() < self.tf_ratio else False
        else:
            teacher_forcing = False

        # pass through encoder
        # discard the context vector and only use hidden state
        latent = self.encoder(input)
        h, _ = latent
        c = torch.zeros(h.shape).cuda()
        latent = (h, c)

        # if teacher forcing the network
        predictions = []
        if teacher_forcing:
            # pad with zeros as the first time step
            padding = torch.zeros((input.shape[0], 1, input.shape[2])).cuda()
            input = torch.cat([padding, input], dim=1)

            # run through the decoding
            for t in range(input.shape[1]):
                output, latent = self.decoder(torch.unsqueeze(input[:, t], 1), latent)
                predictions.append(output)

            # concatenate on time axis
            predictions = torch.cat(predictions, dim=1)[:, :-1]
        else:
            # first input is zero
            output = torch.zeros((input.shape[0], 1, input.shape[2])).cuda()

            # run through the decoding
            for t in range(input.shape[1]):
                output, latent = self.decoder(output, latent)
                predictions.append(output)

            # concatenate on time axis
            predictions = torch.cat(predictions, dim=1)

        return predictions

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
