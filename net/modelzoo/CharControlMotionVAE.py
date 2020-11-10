import os
import glob
import torch
import random
import torch.nn as nn

from MVAEencoder import MVAEencoder
from MVAEdecoder import MVAEdecoder

class CharControlMotionVAE(nn.Module):

    def __init__(self, FLAGS):
        """
        Defines the VAE class using Conv Encoder
        """
        super(CharControlMotionVAE, self).__init__()
        self.encoder = MVAEencoder(FLAGS)
        self.decoder = MVAEdecoder(FLAGS)

    def reparameterize(self, mu, log_var):
        """
        :param mu : mean from the encoder's latent space
        :param log_var: log variance from encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).cuda()
        sample = mu + (eps * std)
        return sample

    def forward(self, x, p):
        """
        Defines forward pass for the ConvMotionTransform VAE
        :param x : tuple containing (data, targets) of shape ((batch_size, f, 146),(batch, f, 73))
        :param p: probability of teacher forcing
        :output : prediction for seller 2 of shape (batch_size, f, 73)
        """

        # unpack into individual sequences
        x, y = x
        b = x[:, :, :73]
        s1 = x[:, :, 73:]
        s2 = y

        # keep track of outputs
        pred = [torch.unsqueeze(s2[:, 0, :], dim=1)]
        mus = []
        log_vars = []

        # scheduled teacher forcing
        if self.training:
            teacher_forcing = True if random.random() < p else False
        else:
            teacher_forcing = False

        # iterate through all time steps
        for t in range(1, b.shape[1]):
            # do the encoding
            if teacher_forcing:
                inp = torch.cat([b[:, t, :], s1[:, t, :], s2[:, t - 1, :], s2[:, t, :]], dim=1)
            else:
                inp = torch.cat([b[:, t, :], s1[:, t, :], pred[-1], s2[:, t, :]], dim=1)

            mu, log_var = self.encoder(inp)
            mus.append(torch.unsqueeze(mu, dim=1))
            log_vars.append(torch.unsqueeze(log_var, dim=1))

            z = self.reparameterize(mu, log_var)

            # do the decoding
            if teacher_forcing:
                inp = torch.cat([b[:, t, :], s1[:, t, :], s2[:, t - 1, :]], dim=1)
            else:
                inp = torch.cat([b[:, t, :], s1[:, t, :], pred[-1]], dim=1)

            pred.append(torch.unsqueeze(self.decoder(inp, z), dim=1))

        if self.training:
            return torch.cat(pred, dim=1), torch.cat(mus, dim=1), torch.cat(log_vars, dim=1)
        return torch.cat(pred, dim=1)

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
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ]

        return params

    def load_transfer_params(self, path, epoch):
        """
        loads transferable parameteres from other models
        :param path: directory for saved model
        :param epoch: epoch number
        :return: load succesful or not in bool
        """

        return True
