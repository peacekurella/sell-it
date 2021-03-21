import torch
import torch.nn as nn

from PoseEncoderConv import PoseEncoderConv
from PoseDecoderConv import PoseDecoderConv


class EmbeddingNet(nn.Module):
    def __init__(self,pose_dim, n_frames):
        super().__init__()

        self.context_encoder = None
        self.pose_encoder = PoseEncoderConv(n_frames, pose_dim)
        self.decoder = PoseDecoderConv(n_frames, pose_dim)

    def forward(self, poses, variational_encoding=False):
        context_feat = context_mu = context_logvar = None

        # poses
        poses_feat, pose_mu, pose_logvar = self.pose_encoder(poses, variational_encoding)
        # decoder
        latent_feat = poses_feat

        out_poses = self.decoder(latent_feat, False)

        return out_poses, poses_feat, pose_mu, pose_logvar

    def freeze_pose_nets(self):
        for param in self.pose_encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
