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
        self.hidden_units = FLAGS.enc_hidden_units
        self.input_dim = FLAGS.input_dim
        self.num_layers = FLAGS.enc_layers
        self.device = torch.device(FLAGS.device)

        # Encoder LSTM
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=FLAGS.enc_dropout
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
        return torch.zeros((self.num_layers, batch_size, self.hidden_units)).to(self.device)

    def forward(self, input):
        """
        Defines a forward pass through the network
        :param input: Input to the network of shape (batch_size, seq_length, input_dim)
        :return: Latent (h, c) output from the network ((num_layers, batch_size, hidden_units)*2)
        """

        # initialize the hidden state
        latent = (self.init_hidden_state(input.shape[0]), self.init_hidden_state(input.shape[0]))

        # dropout before input to make it robust to noise
        for t in range(input.shape[1]):
            x= self.dropout(input[:, t])
            _, latent = self.lstm(torch.unsqueeze(x, 1), latent)

        return latent

