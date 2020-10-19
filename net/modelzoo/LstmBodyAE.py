import torch
import random
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

        # check to see wether teacher forcing needs to be enabled
        teacher_forcing = True if random.random() < self.tf_ratio else False

        # pass through encoder
        # discard the context vector and only use hidden state
        latent = self.encoder(input)
        h, _ = latent
        c = torch.zeros(h.shape)
        latent = (h, c)
