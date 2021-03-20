import os
import json
import pickle
import numpy as np

from torch.utils.data import Dataset


class JointsDataset(Dataset):
    """Joints Dataset"""

    def __init__(self, FLAGS):
        self.input = FLAGS.input
