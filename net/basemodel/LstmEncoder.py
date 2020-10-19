import torch.nn as nn
import torch


class LstmEncoder(nn.Module):
    """Sequence encoder based on LSTM"""

    def __init__(self, FLAGS):
        """
        Constructor for single sequence LSTM Encoder
        :param FLAGS: abseil Flags
        """

        super(LstmEncoder, self).__init__()

        # attributes
        self.hidden_units = FLAGS.dec_hidden_units
        self.input_dim = FLAGS.input_dim
        self.num_layers = FLAGS.enc_layers

        # Encoder LSTM
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.enc_dropout
        )

        # input dropout
        self.dropout = nn.Dropout(FLAGS.dropout)

    def init_hidden_state(self, batch_size):
        """
        creates the initial zero state
        :param batch_size: batch_size of the hidden state
        :return: initial zero state
        """

        # shape should be batch_size, num_layers, number of hidden units
        return torch.zeros((batch_size, self.num_layers, self.hidden_units))

    def forward(self, input):
        """
        Defines a forward pass through the network
        :param input: Input to the network of shape (batch_size, seq_length, input_dim)
        :return: Latent (h, c) output from the network ((batch_size, num_layers, hidden_units)*2)
        """

        # initialize the hidden state
        latent = (self.init_hidden_state(input.shape[0]), self.init_hidden_state(input.shape[0]))

        # dropout before input to make it robust to noise
        input = self.dropout(input)

        # pass through lstm
        _, latent = self.lstm(input, latent)

        return latent

