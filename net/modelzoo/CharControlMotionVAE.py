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
        self.latent_dim = FLAGS.latent_dim
        self.tf = 0.0
        self.device = torch.device(FLAGS.device)
        if FLAGS.speak:
            self.speak = nn.Linear(FLAGS.output_dim, 1)
            self.predict_speech = True
        else:
            self.predict_speech = False
        self.c_dim = FLAGS.c_dim

    def reparameterize(self, mu, log_var):
        """
        :param mu : mean from the encoder's latent space
        :param log_var: log variance from encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, inputs):
        """
        Defines forward pass for the ConvMotionTransform VAE
        :param inputs : dictionary containing inputs of shape (batch_size, F, input_dim*2),
        speaking_status (batch_size, F, 2)
        :param p: probability of teacher forcing
        :output : prediction for seller 2 of shape (batch_size, f, 73)
        """

        inputs = self.transform_inputs(inputs)

        # unpack into individual sequences
        x = inputs['trainx'].to(self.device)
        y = inputs['trainy'].to(self.device)
        speakx = inputs['speakx'].to(self.device)
        speaky = inputs['speaky'].to(self.device)

        # keep track of outputs
        pose_pred = [torch.unsqueeze(y[:, 0, :], dim=1)]
        speech_pred = [torch.unsqueeze(speaky[:, 0, :], dim=1)]
        mus = []
        log_vars = []
        output = {}
        inputs = {
            'pose': y,
            'speech': speaky
        }

        # scheduled t
        # eacher forcing
        if self.training:
            teacher_forcing = True if random.random() < self.tf else False
        else:
            teacher_forcing = False

        # iterate through all time steps
        for t in range(1, x.shape[1]):
            # do the encoding
            if teacher_forcing:
                inp = torch.cat([x[:, t, :], y[:, t - 1, :]], dim=1)
            else:
                inp = torch.cat([x[:, t, :], torch.squeeze(pose_pred[-1], dim=1)], dim=1)

            if self.c_dim == 2:
                inp = torch.cat((inp, speakx[:, t, :]), dim=-1)
            elif self.c_dim == 1:
                if teacher_forcing:
                    inp = torch.cat((inp, speaky[:, t - 1, :]), dim=-1)
                else:
                    inp = torch.cat((inp, torch.squeeze(speech_pred[-1], dim=1)), dim=-1)

            if self.training:
                enc_inp = torch.cat([y[:, t, :], inp], dim=1)
                mu, log_var = self.encoder(enc_inp)
                mus.append(torch.unsqueeze(mu, dim=1))
                log_vars.append(torch.unsqueeze(log_var, dim=1))

                z = self.reparameterize(mu, log_var)

            else:
                z = torch.randn((x.shape[0], self.latent_dim)).to(self.device)

            pose_pred.append(torch.unsqueeze(self.decoder(inp, z), dim=1))

            if self.predict_speech:
                speech_pred.append(self.speak(pose_pred[-1].squeeze(1)).unsqueeze(1))

        output['pose'] = torch.cat(pose_pred, dim=1)

        if self.training:
            output['mus'] = torch.cat(mus, dim=1)
            output['log_vars'] = torch.cat(log_vars, dim=1)

        if self.predict_speech:
            output['speech'] = torch.cat(speech_pred, dim=1)

        return output, inputs

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
        epoch = enc_path.split('/')[-1]
        epoch = int(epoch.split('.')[0])
        try:
            self.encoder.load_state_dict(torch.load(enc_path))
            self.decoder.load_state_dict(torch.load(dec_path))
            print("Load successful!")
        except:
            print("Load failed!")
            return 0

        return epoch

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
        :return: load successful or not in bool
        """
        pass

    def transform_inputs(self, batch):
        """
        Transforms the input dictionary to
        inputs for the model (batch , sequence_length, input_dim)
        """
        b = batch['buyer']['joints21']
        l = batch['leftSeller']['joints21']
        r = batch['rightSeller']['joints21']

        # get the speaking status
        speaking_status = {
            'buyer': batch['buyer']['speakingStatus'],
            'leftSeller': batch['leftSeller']['speakingStatus'],
            'rightSeller': batch['rightSeller']['speakingStatus']
        }

        set_x_a = torch.cat((b, l), dim=2)
        set_x_b = torch.cat((b, r), dim=2)
        train_x = torch.cat((set_x_a, set_x_b), dim=0).float()
        train_y = torch.cat((r, l), dim=0).float()
        speak_a = torch.cat((speaking_status['buyer'], speaking_status['leftSeller']), dim=2)
        speak_b = torch.cat((speaking_status['buyer'], speaking_status['rightSeller']), dim=2)
        speak_x = torch.cat((speak_a, speak_b), dim=0).float()
        speak_y = torch.cat((speaking_status['rightSeller'], speaking_status['leftSeller']), dim=0).float()
        input = {
            'trainx': train_x,
            'trainy': train_y,
            'speakx': speak_x,
            'speaky': speak_y,
        }
        return input

    def set_teacher_forcing(self, p):
        """
        Sets the degree of teacher forcing while training
        :param p: degree of teacher forcing
        """
        self.tf = p
