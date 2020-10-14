import os
import torch
from absl import app
from absl import flags
import numpy as np
from joblib import dump, load
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from HagglingDataset import HagglingDataset
from Quaternions import Quaternions
from preprocess import SkeletonHandler

FLGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('test', '../Data/test/', 'Directory containing test files')
flags.DEFINE_string('output_dir', '../Data/output/', 'Folder to store final videos')
flags.DEFINE_string('ckpt', '../ckpt/AE', 'file containing the model weights')
flags.DEFINE_integer('k', 50, 'Number of clusters')
flags.DEFINE_string('train', '../Data/train/', 'Directory containing train files')


def get_input(batch):
    # concatenate along the 0 axis and returns a 360x73 matrix
    skels = []
    for key in batch.keys():
        joints = batch[key]['joints21'][0].cpu().numpy()
        skels.append(joints)
    skels = np.concatenate((skels[0], skels[1], skels[2]), axis=0)
    return skels


def main(arg):
    if os.path.exists(FLGS.meta + str(FLGS.k) + '.joblib'):
        # if model already exists, return that
        print("Pretained model present, returning that")
        kmeans = load(FLGS.meta + str(FLGS.k) + '.joblib')
    else:
        # need to train a new model, get poses from both test and train data
        print("Training new model")
        train_dataset = HagglingDataset(FLGS.train, FLGS)
        train_dataloader = DataLoader(train_dataset, num_workers=10)
        test_dataset = HagglingDataset(FLGS.test, FLGS)
        test_dataloader = DataLoader(test_dataset, num_workers=10)
        skeletons = []

        for i_batch, batch in enumerate(train_dataloader):
            skels = get_input(batch)
            skeletons.append(skels)
        for i_batch, batch in enumerate(test_dataloader):
            skels = get_input(batch)
            skeletons.append(skels)
        # concatenate all the elements in a list into a Nx73 matrix
        X = np.vstack(skeletons)

        print(X.shape)
        # train KMeans model
        kmeans = KMeans(n_clusters=FLGS.k, max_iter=1000).fit(X)
        # store the model
        dump(kmeans, FLGS.meta + str(FLGS.k) + '.joblib')

        print("Model write successful")

    return kmeans


if __name__ == "__main__":
    app.run(main)
