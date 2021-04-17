import os
import glob
import torch
import torch.nn as nn

from PoseEncoderConv import PoseEncoderConv
from PoseDecoderConv import PoseDecoderConv


class EmbeddingNet(nn.Module):
    def __init__(self,pose_dim, n_frames):
        super().__init__()

        self.context_encoder = None
        self.encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)

    def forward(self, poses, variational_encoding=False):
        # poses
        poses_feat, pose_mu, pose_logvar = self.encoder(poses, variational_encoding)
        # decoder
        latent_feat = poses_feat

        out_poses = self.decoder(latent_feat, False)

        return out_poses, poses_feat, pose_mu, pose_logvar

    def freeze_pose_nets(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

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
        except Exception as e:
            print(e)
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
