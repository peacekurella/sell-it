import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self, FLAGS, inp_size, op_size, type):

        super(LinearModel, self).__init__()

        self.input_size = inp_size
        self.output_size = op_size
        if type == 'enc':
            self.dp = FLAGS.enc_dropout
            self.hu = FLAGS.enc_hidden_units
        elif type == 'gat':
            self.dp = FLAGS.gat_dropout
            self.hu = FLAGS.gat_hidden_units

        self.lys = [
            nn.Dropout(self.dp),
            nn.Linear(inp_size, self.hu),
            nn.ELU()
        ]

        for _ in range(FLAGS.enc_layers - 1):
            self.lys.extend([
                nn.Dropout(self.dp),
                nn.Linear(self.hu, self.hu),
                nn.ELU()
            ])

        self.lys.extend([
            nn.Dropout(self.dp),
            nn.Linear(self.hu, op_size),
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