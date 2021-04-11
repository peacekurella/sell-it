import os
import torch
import glob
import torch.nn as nn

from net.basemodel.ConvDecoderSingle import ConvDecoderSingle
from net.basemodel.ConvEncoderSingle import ConvEncoderSingle
from net.basemodel.LatentParameterizer import LatentParameterizer


class ConvMotionTransformVAE(nn.Module):

    def __init__(self, FLAGS):
        """
        Defines the VAE class using Conv Encoder
        """
        super(ConvMotionTransformVAE, self).__init__()
        self.encoder_b = ConvEncoderSingle(FLAGS)
        self.encoder_s1 = ConvEncoderSingle(FLAGS)
        self.encoder_s2 = ConvEncoderSingle(FLAGS)

        self.encoder_output_dimensions = self.encoder_b.get_output_dimensions(FLAGS)
        self.resnet_input_dim = self.encoder_output_dimensions[1] * self.encoder_output_dimensions[2]
        self.resnet_enc = LatentParameterizer(self.resnet_input_dim, FLAGS.latent_dim)
        self.latent_dim = FLAGS.latent_dim

        self.decoder = ConvDecoderSingle(FLAGS)
        self.resenet_dec = nn.Linear(self.resnet_input_dim + FLAGS.latent_dim, self.resnet_input_dim)

        self.input_dim = FLAGS.input_dim
        self.device = torch.device(FLAGS.device)
        pretrained_ckpt = os.path.join(FLAGS.ckpt_dir, FLAGS.pretrainedModel + '/')
        if not self.load_transfer_params(pretrained_ckpt, FLAGS.pretrained_ckpt):
            raise Exception("AE model needs to be trained")

    def reparameterize(self, mu, log_var):
        """
        :param mu : mean from the encoder's latent space
        :param log_var: log variance from encoder's latent space
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).cuda()
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        """
        Defines forward pass for the ConvMotionTransform VAE
        :param x : batch containing dict of subjects
        :transform x : tuple containing (data, targets) of shape ((batch_size, input_dim*2, f),(batch, input_dim, f))
        :output : prediction for seller 2 of shape (batch_size, f, input_dim)
        """
        x, y = self.transform_inputs(x)
        x = x.to(self.device)
        y = y.to(self.device)
        b = x[:, :self.input_dim, :]
        s1 = x[:, :self.input_dim, :]
        s2 = y

        predictions = {}
        target = {}

        b = self.encoder_b(b)
        s1 = self.encoder_s1(s1)
        s2 = self.encoder_s2(s2)

        ea = s1 + b
        latent_dim_shape = (x.shape[0], self.resnet_input_dim)
        if self.training:
            eb = s2
            t = eb - ea
            t = torch.reshape(t, (t.shape[0], -1))
            # reparametrize for VAE
            mu, log_var = self.resnet_enc(t)
            z = self.reparameterize(mu, log_var)
        else:
            z = torch.randn((x.shape[0], self.latent_dim)).cuda()

        ea_flattened = torch.reshape(ea, latent_dim_shape)
        f = torch.cat((ea_flattened, z), dim=1)
        t_star = self.resenet_dec(f)
        t_star = torch.reshape(t_star, ea.shape)
        eb_star = t_star + ea

        if self.training:
            eb = eb_star
            t = eb - ea
            t = torch.reshape(t, (t.shape[0], -1))
            # reparametrize for VAE
            mu_s, log_var_s = self.resnet_enc(t)
            z_star = self.reparameterize(mu_s, log_var_s)

        output = self.decoder(eb_star)
        predictions['pose'] = output.permute(0, 2, 1)
        target['pose'] = y.permute(0, 2, 1)
        if self.training:
            predictions['mu'] = mu
            predictions['log_var'] = log_var
            predictions['z'] = z
            predictions['z_star'] = z_star

        return predictions, target

    def save_model(self, path, epoch):
        """
        Saves the model parameters
        :param path: directory for saving the model
        :param epoch: epoch number
        :return: save successful or not in bool
        """

        # create the encoder and decoder paths
        enc_b_path = os.path.join(path, 'encoder_b/')
        enc_s1_path = os.path.join(path, 'encoder_s1/')
        enc_s2_path = os.path.join(path, 'encoder_s2/')
        resnet_enc_path = os.path.join(path, 'resnet_enc/')
        resnet_dec_path = os.path.join(path, 'resnet_dec/')
        dec_path = os.path.join(path, 'decoder/')

        # create a directory if not a directory
        if not os.path.isdir(enc_b_path):
            os.makedirs(enc_b_path)
        if not os.path.isdir(enc_s1_path):
            os.makedirs(enc_s1_path)
        if not os.path.isdir(enc_s2_path):
            os.makedirs(enc_s2_path)
        if not os.path.isdir(resnet_enc_path):
            os.makedirs(resnet_enc_path)
        if not os.path.isdir(resnet_dec_path):
            os.makedirs(resnet_dec_path)
        if not os.path.isdir(dec_path):
            os.makedirs(dec_path)

        # try to save the models
        # noinspection PyBroadException
        try:
            torch.save(self.encoder_b.state_dict(), enc_b_path + str(epoch) + '.ckpt')
            torch.save(self.encoder_s1.state_dict(), enc_s1_path + str(epoch) + '.ckpt')
            torch.save(self.encoder_s2.state_dict(), enc_s2_path + str(epoch) + '.ckpt')
            torch.save(self.resnet_enc.state_dict(), resnet_enc_path + str(epoch) + '.ckpt')
            torch.save(self.resenet_dec.state_dict(), resnet_dec_path + str(epoch) + '.ckpt')
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
        enc_b_path = os.path.join(path, 'encoder_b/')
        enc_s1_path = os.path.join(path, 'encoder_s1/')
        enc_s2_path = os.path.join(path, 'encoder_s2/')
        resnet_enc_path = os.path.join(path, 'resnet_enc/')
        resnet_dec_path = os.path.join(path, 'resnet_dec/')
        dec_path = os.path.join(path, 'decoder/')

        # check if epoch number is passed
        if epoch is not None:
            enc_b_path = enc_b_path + str(epoch) + '.ckpt'
            enc_s1_path = enc_s1_path + str(epoch) + '.ckpt'
            enc_s2_path = enc_s2_path + str(epoch) + '.ckpt'
            resnet_enc_path = resnet_enc_path + str(epoch) + '.ckpt'
            resnet_dec_path = resnet_dec_path + str(epoch) + '.ckpt'
            dec_path = dec_path + str(epoch) + '.ckpt'
        else:
            enc_b_path = max(glob.glob(enc_b_path + '*'), key=os.path.getctime)
            enc_s1_path = max(glob.glob(enc_s1_path + '*'), key=os.path.getctime)
            enc_s2_path = max(glob.glob(enc_s2_path + '*'), key=os.path.getctime)
            resnet_enc_path = max(glob.glob(resnet_enc_path + '*'), key=os.path.getctime)
            resnet_dec_path = max(glob.glob(resnet_dec_path + '*'), key=os.path.getctime)
            dec_path = max(glob.glob(dec_path + '*'), key=os.path.getctime)

        # try to load the models
        # noinspection PyBroadException
        try:
            self.encoder_b.load_state_dict(torch.load(enc_b_path))
            self.encoder_s1.load_state_dict(torch.load(enc_s1_path))
            self.encoder_s2.load_state_dict(torch.load(enc_s2_path))
            self.resnet_enc.load_state_dict(torch.load(resnet_enc_path))
            self.resenet_dec.load_state_dict(torch.load(resnet_dec_path))
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
            {'params': self.encoder_b.parameters()},
            {'params': self.encoder_s1.parameters()},
            {'params': self.encoder_s2.parameters()},
            {'params': self.resnet_enc.parameters()},
            {'params': self.resenet_dec.parameters()}
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
        # enc_path = os.path.join(path, 'encoder/')

        # check if epoch number is passed
        if epoch is not None:
            dec_path = dec_path + str(epoch) + '.ckpt'
            # enc_path = enc_path + str(epoch) + '.ckpt'
        else:
            dec_path = max(glob.glob(dec_path + '*'), key=os.path.getctime)
            # enc_path = max(glob.glob(enc_path + '*'), key=os.path.getctime)

        # try to load the models
        # noinspection PyBroadException
        try:
            # self.encoder_b.load_state_dict(torch.load(enc_path))
            # self.encoder_s1.load_state_dict(torch.load(enc_path))
            # self.encoder_s2.load_state_dict(torch.load(enc_path))
            self.decoder.load_state_dict(torch.load(dec_path))
            print("Transfer load successful!")
        except:
            print("Transfer load failed!")
            return False

        return True

    def transform_inputs(self, batch):
        """
        Transforms the input dictionary to
        inputs for the model (batch , sequence_length, input_dim)
        """
        b = batch['buyer']['joints21']
        l = batch['leftSeller']['joints21']
        r = batch['rightSeller']['joints21']
        speaking_status = {'buyer': batch['buyer']['speakingStatus'],
                           'leftSeller': batch['leftSeller']['speakingStatus'],
                           'rightSeller': batch['rightSeller']['speakingStatus']}

        set_x_a = torch.cat((b, l), dim=2)
        set_x_b = torch.cat((b, r), dim=2)
        train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()

        return train_x, train_y
