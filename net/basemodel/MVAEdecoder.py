import torch.nn as nn
import torch

from GatingNetwork import GatingNetwork


class MVAEdecoder(nn.Module):

    def __init__(self, FLAGS):
        """
        Decoder module for the char motion decoder
        :param FLAGS: abseil flags
        """

        super(MVAEdecoder, self).__init__()

        self.experts = []
        self.num_experts = FLAGS.num_experts

        for _ in range(self.num_experts):
            self.experts.append(
                nn.ModuleList([
                    nn.Linear((FLAGS.input_dim * 3) + FLAGS.latent_dim + FLAGS.c_dim, FLAGS.dec_hidden_units),
                    nn.Linear(FLAGS.dec_hidden_units + FLAGS.latent_dim, FLAGS.dec_hidden_units),
                    nn.Linear(FLAGS.dec_hidden_units + FLAGS.latent_dim, FLAGS.dec_hidden_units),
                    nn.Linear(FLAGS.dec_hidden_units, FLAGS.output_dim)
                ])
            )

        self.experts = nn.ModuleList(self.experts)
        self.num_lys = len(self.experts[0])

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(FLAGS.dec_dropout)

        self.gatingNet = GatingNetwork(FLAGS)

    def forward(self, x, z):
        """
        forward pass through the module
        :param x: input to the module
        :param z: random variable
        :return: output
        """
        bc = self.gatingNet(x, z).T  # > num_experts * batch_size
        bc = torch.unsqueeze(torch.unsqueeze(bc, dim=-1), dim=-1)  # > num_experts * batch_size * 1 * 1

        x = torch.unsqueeze(x, dim=-1)  # > batch_size * input_dim * 1
        z = torch.unsqueeze(z, dim=-1)  # > batch_size * latent_dim * 1

        for j in range(self.num_lys):
            alpha = []
            beta = []

            for i in range(self.num_experts):
                # > 1 * 1 * out * in
                alpha.append(
                    torch.unsqueeze(torch.unsqueeze(self.experts[i][j].weight, dim=0), dim=0))
                # > 1 * 1 * out * 1
                beta.append(
                    torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.experts[i][j].bias, dim=-1), dim=0), dim=0))

            alpha = torch.cat(alpha, dim=0)  # > num_experts * 1 * out * in
            beta = torch.cat(beta, dim=0)  # > num_experts * 1 * out * 1

            weights = torch.sum(bc * alpha, dim=0)  # batch_size * out * in
            biases = torch.sum(bc * beta, dim=0)  # batch_size * out * 1

            del alpha, beta

            if j < self.num_lys - 1:
                x = self.dropout(x)
                x = torch.matmul(weights, torch.cat([x, z], dim=1)) + biases
                x = self.activation(x)
            else:
                x = torch.matmul(weights, x) + biases

        return torch.squeeze(x, dim=-1)
