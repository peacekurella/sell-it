import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self, FLAGS, inp_size, op_size):

        super(LinearModel, self).__init__()

        self.input_size = inp_size
        self.output_size = op_size

        self.lys = [
            nn.Dropout(FLAGS.enc_dropout),
            nn.Linear(inp_size, FLAGS.enc_hidden_units),
            nn.ELU()
        ]

        for _ in range(FLAGS.enc_layers - 1):
            self.lys.extend([
                nn.Dropout(FLAGS.enc_dropout),
                nn.Linear(FLAGS.enc_hidden_units, FLAGS.enc_hidden_units),
                nn.ELU()
            ])

        self.lys.extend([
            nn.Dropout(FLAGS.enc_dropout),
            nn.Linear(FLAGS.enc_hidden_units, op_size),
            nn.ELU()
        ])

        self.network = nn.Sequential(*self.lys)

    def forward(self, x):
        """
        Forward pass through the model
        :param x: input to the network
        :return:
        """
        return self.network(x)