import torch.nn as nn


class LinearModel(nn.Module):

    def __init__(self, FLAGS, inp_size, op_size):

        super(LinearModel, self).__init__()

        self.input_size = inp_size
        self.output_size = op_size

        self.network = nn.Sequential(
            nn.Linear(inp_size, 256),
            nn.ELU(),

            nn.Linear(256, 256),
            nn.ELU(),

            nn.Linear(256, op_size),
            nn.ELU()
        )

    def forward(self, x):
        """
        Forward pass through the model
        :param x: input to the network
        :return:
        """
        return self.network(x)