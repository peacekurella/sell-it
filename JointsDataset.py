import os
import json
import pickle
import numpy as np

from torch.utils.data import Dataset


class JointsDataset(Dataset):
    """Joints Dataset"""

    def __init__(self, directory, FLAGS):
        self.input = directory

        self.meta = os.path.join(directory.split('/')[0], 'stats')

        # Load the mean and std filenames
        mean_file = os.path.join(self.meta, 'mean.pkl')
        std_file = os.path.join(self.meta, 'std.pkl')

        with open(mean_file, "rb") as f:
            self.mean = pickle.load(f, encoding='Latin-1')
            self.mean = np.array(self.mean['joints21'])
        with open(std_file, "rb") as f:
            self.std = pickle.load(f, encoding='Latin-1')
            self.std = np.array(self.std['joints21'])
            self.std[self.std == 0.0] = 1.0

    def __len__(self):
        """
        :return: Number of input data examples
        """
        return len(os.listdir(self.input))

    def __getitem__(self, idx):
        """
                Returns a training example
                :param idx: index of the training example
                :return:
                """

        # read the file
        filename = os.path.join(self.input, str(idx) + '.pkl')
        with open(filename, "rb") as f:
            data = pickle.load(f, encoding='Latin-1')

        # transformed data
        transformed_data = {}

        for subject in data.keys():
            # normalize the joint vectors
            joints21 = np.array(data[subject]['joints21'])
            positions = data[subject]['positions']
            joints21 = joints21 - self.mean
            joints21 = np.divide(joints21, self.std)

            transformed_data[subject] = {
                'directions': joints21,
                'positions': positions,
            }

        return transformed_data

    def denormalize_data(self, joints21):
        """
        Denormalizes the input data
        :param joints21: Input joints data of shape (F, output_shape)
        :return: denormalized joint data of shape (F, output_shape)
        """

        x = joints21 * self.std
        x = x + self.mean
        return x