import os
import json
import numpy as np

from torch.utils.data import Dataset


class HagglingDataset(Dataset):
    """Haggling Dataset"""

    def __init__(self, directory, FLAGS):
        """
        Constructor for the dataset
        :param FLAGS: required flags
        """

        # save the directory info
        self.input = directory
        self.meta = FLAGS.meta

        # Load the mean and std filenames
        mean_file = os.path.join(FLAGS.meta, 'mean.json')
        std_file = os.path.join(FLAGS.meta, 'std.json')

        # Check if the stats exists
        if not (os.path.exists(mean_file) and os.path.exists(std_file)):
            self.generate_stats(self.input, self.meta)

        with open(mean_file) as f:
            self.mean = json.load(f)
            self.mean = np.array(self.mean['joints21'])
        with open(std_file) as f:
            self.std = json.load(f)
            self.std = np.array(self.std['joints21'])

    def __len__(self):
        """
        :return: Number of input data examples
        """
        return len(os.listdir(self.input))

    @staticmethod
    def generate_stats(input, meta):
        """
        Generates the mean and std for the training data
        :param input: directory containing the input data
        :param meta: directory to store the stats files
        :return: None
        """

        # create the vectors
        all_frames = np.zeros((1, 73))
        # all_body_normals = np.zeros((1, 2))
        # all_face_normals = np.zeros((1, 2))

        # Read all files in directory
        for i, file in enumerate(os.listdir(input)):

            # progress report
            if i % 100 == 0:
                print('Read ' + str(i) + '.JSON')

            filename = os.path.join(input, file)
            with open(filename) as f:
                data = json.load(f)
                f.close()

            data = data['subjects']

            for subject in data:
                frames = np.array(data[subject]['frames']['joints21'])
                # body_norms = np.swapaxes(np.array(data[subject]['frames']['body_normal']), 0, 1)
                # face_norms = np.swapaxes(np.array(data[subject]['frames']['face_normal']), 0, 1)
                all_frames = np.concatenate([all_frames, frames], axis=0)
                # all_body_normals = np.concatenate([all_body_normals, body_norms], axis=0)
                # all_face_normals = np.concatenate([all_face_normals, face_norms], axis=0)

        # template for the stats files
        mean = {
            'joints21': np.mean(all_frames[1:], axis=0).reshape(1, -1).tolist(),
            # 'body_normal': np.mean(all_body_normals[1:], axis=0).reshape(1, -1).tolist(),
            # 'face_normal': np.mean(all_face_normals[1:], axis=0).reshape(1, -1).tolist()
        }

        std = {
            'joints21': np.std(all_frames[1:], axis=0).reshape(1, -1).tolist(),
            # 'body_normal': np.std(all_body_normals[1:], axis=0).reshape(1, -1).tolist(),
            # 'face_normal': np.std(all_face_normals[1:], axis=0).reshape(1, -1).tolist()
        }

        # write to files
        mean_file = os.path.join(meta, 'mean.json')
        std_file = os.path.join(meta, 'std.json')
        # noinspection PyBroadException
        try:
            with open(mean_file, 'w') as json_file:
                json.dump(mean, json_file)
                json_file.close()

            with open(std_file, 'w') as json_file:
                json.dump(std, json_file)
                json_file.close()
        except:
            print("error dumping stats to JSON")

    def __getitem__(self, idx):
        """
        Returns a training example
        :param idx: index of the training example
        :return: 
        """

        # read the file 
        filename = os.path.join(self.input, str(idx) + '.json')
        with open(filename) as f:
            data = json.load(f)

        # select only the subjects
        padLength = data['padding_length']
        data = data['subjects']

        # transformed data
        transformed_data = {}

        for subject in data:
            # get the initial translation and rotation values
            initRot = data[subject]['initRot']
            initTrans = data[subject]['initTrans']

            # normalize the joint vectors
            joints21 = np.array(data[subject]['frames']['joints21'])
            joints21 = joints21 - self.mean
            joints21 = np.divide(joints21, self.std, out=np.zeros_like(joints21), where=self.std != 0)

            transformed_data[subject] = {
                'initRot': np.array(initRot),
                'initTrans': np.array(initTrans),
                'joints21': joints21,
                'padLength': padLength
            }

        return transformed_data

    def denormalize_data(self, joints21):
        """
        Denormalizes the input data
        :param joints21: Input joints data of shape (F, 73)
        :return: denormalized joint data of shape (F, 73)
        """

        x = joints21 * self.std
        x = x + self.mean
        return x