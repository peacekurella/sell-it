import torch.nn.functional as F
import torch.nn as nn
import torch


class LstmDecoder(nn.Module):
    """Sequence encoder based on LSTM"""

    def __init__(self, FLAGS):
        """
        Constructor for single sequence LSTM Encoder
        :param FLAGS: abseil Flags
        """

        super(LstmDecoder, self).__init__()

        # attributes
        self.hidden_units = FLAGS.dec_hidden_units
        self.input_dim = FLAGS.input_dim
        self.output_dim = FLAGS.output_dim
        self.num_layers = FLAGS.dec_layers

        # Encoder LSTM
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=FLAGS.dec_dropout
        )

        # input dropout
        self.dropout = nn.Dropout(FLAGS.dropout)

        # output linear network
        self.linear = nn.Linear(self.hidden_units, self.output_dim)

    def init_hidden_state(self, batch_size):
        """
        creates the initial zero state
        :param batch_size: batch_size of the hidden state
        :return: initial zero state
        """

        # shape should be batch_size, 1, number of hidden units
        return torch.zeros((self.num_layers, batch_size, self.hidden_units)).cuda()

    def forward(self, input, latent):
        """
        Defines a forward pass through the network
        :param input: Input to the network of shape (batch_size, seq_length, input_dim)
            preferably only one time step at a time
        :return: output from the network(batch_size, seq_length, output_dim)
            Latent variable from the network ((batch_size, num_layers, hidden_units)*2)
        """

        # dropout before input to make it robust to noise
        for t in range(input.shape[1]):
            x = self.dropout(input[:, t])
            output, latent = self.lstm(torch.unsqueeze(x, 1), latent)
            output = self.linear(output)
            output = F.relu(output)

        return output, latent
