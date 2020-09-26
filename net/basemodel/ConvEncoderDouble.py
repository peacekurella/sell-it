import torch.nn as nn


class ConvEncoderDouble(nn.Module):
    """Two skeleton input Encoder"""

    def __init__(self, FLAGS):
        """
        :param FLAGS: abseil flags
        """
        super(ConvEncoderDouble, self).__init__()
        self.network = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146, 256, 45, padding=22),  # 256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256, 256, 25, padding=12),  # 256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256, 256, 15, padding=7),  # 256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 256, 73, 120
        )

    def foward(self, x):
        """
        Defines the foward pass through the network
        :param x: Input to the encoder (batch_size, f, 146)
        :return: Output of the encoder (batch_size, f/2, 256)
        """
        return self.network(x)
