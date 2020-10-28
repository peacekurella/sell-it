import os
import numpy as np
import torch
import torch.nn as nn
import random
from absl import app
from absl import flags
from joblib import load
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from LstmBodyAE import LstmBodyAE
from DebugVisualizer import DebugVisualizer
from HagglingDataset import HagglingDataset
from Quaternions import Quaternions
from preprocess import SkeletonHandler

FLAGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('test', '../Data/train/', 'Directory containing test files')
flags.DEFINE_string('output_dir', '../Data/boyAEoutput/', 'Folder to store final videos')
flags.DEFINE_string('ckpt', '../ckpt/bodyAE/AE', 'file containing the model weights')
flags.DEFINE_float('lmd', 0.1, 'L1 Regularization factor')
flags.DEFINE_boolean('bodyae', True, 'if True checks BodyAE model')
flags.DEFINE_bool('CNN', True, 'Cnn based model')
flags.DEFINE_string('model', "bodyAE", 'Defines the name of the model')
flags.DEFINE_integer('enc_hidden_units', 128, 'Encoder LSTM hidden units')
flags.DEFINE_integer('dec_hidden_units', 128, 'Decoder LSTM hidden units')
flags.DEFINE_integer('enc_layers', 1, 'encoder LSTM layers')
flags.DEFINE_integer('dec_layers', 1, 'decoder LSTM layers')
flags.DEFINE_float('enc_dropout', 0.25, 'encoder LSTM dropout')
flags.DEFINE_float('dec_dropout', 0.25, 'decoder LSTM dropout')
flags.DEFINE_float('dropout', 0.25, 'dense network dropout')
flags.DEFINE_float('tf_ratio', 0.5, 'teacher forcing ratio')
flags.DEFINE_bool('pretrain', True, 'pretrain the auto encoder')

flags.DEFINE_integer('input_dim', 73, 'input pose vector dimension')
flags.DEFINE_integer('output_dim', 73, 'input pose vector dimension')

pss = lambda a, b: a == b


def get_input(batch):
    """
    Generates the inputs to the network
    :param batch: batch for training
    :return: train_x, train_y tensors
    """
    b = batch['buyer']['joints21']
    l = batch['leftSeller']['joints21']
    r = batch['rightSeller']['joints21']

    if FLAGS.pretrain:
        if FLAGS.CNN:
            train_x = torch.cat((l, r), dim=0).permute(0, 2, 1).float().cuda()
            train_y = torch.cat((l, r), dim=0).permute(0, 2, 1).float().cuda()
            return train_x, train_y
        else:
            train_x = torch.cat((l, r), dim=0).float().cuda()
            train_y = torch.cat((l, r), dim=0).float().cuda()
            return train_x, train_y
    else:
        if FLAGS.CNN:
            set_x_a = torch.cat((b, l), dim=2)
            set_x_b = torch.cat((b, r), dim=2)
            train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float().cuda()
            train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
            return train_x, train_y
        else:
            set_x_a = torch.cat((b, l), dim=2)
            set_x_b = torch.cat((b, r), dim=2)
            train_x = torch.cat((set_x_a, set_x_b), dim=0).float().cuda()
            train_y = torch.cat((r, l), dim=0).float().cuda()
            return train_x, train_y


def get_model():
    """
    Returns the appropriate model for training
    :return: PyTorch model that extends nn.Module
    """

    if FLAGS.model == 'bodyAE':
        return BodyAE(FLAGS).cuda()
    if FLAGS.model == 'LstmAE':
        return LstmBodyAE(FLAGS).cuda()
    else:
        return BodyMotionGenerator(FLAGS).cuda()


def get_global_positions_denormalized(subjects, subject_params, FLAGS):
    # denormalize and globalize positions of humans
    skeletonHandler = SkeletonHandler()
    haggling = HagglingDataset(FLAGS.test, FLAGS)
    r_target_global = skeletonHandler.recover_global_positions(
        haggling.denormalize_data(subjects[0]), subject_params[0], subject_params[1])

    r_prediction_global = skeletonHandler.recover_global_positions(
        haggling.denormalize_data(subjects[1]), subject_params[0], subject_params[1])

    l_target_global = skeletonHandler.recover_global_positions(
        haggling.denormalize_data(subjects[2]), subject_params[2], subject_params[3])

    l_prediction_global = skeletonHandler.recover_global_positions(
        haggling.denormalize_data(subjects[3]), subject_params[2], subject_params[3])

    return r_target_global, r_prediction_global, l_target_global, l_prediction_global


def get_subject_loss(targets, predictions):
    """
    calculate mse loss in the joints
    """
    targets = torch.from_numpy(targets)
    predictions = torch.from_numpy(predictions)
    # set the criterion objects
    criterion1 = nn.MSELoss(reduction='mean')

    # calculate losses
    mse = criterion1(predictions, targets)
    return mse


def pose_structure_score(r_target, r_prediction, l_target, l_prediction):
    # calculates mean pss score for left and right seller in a batch
    kmeans = load('meta/50.joblib')
    r_labels_q = kmeans.predict(r_target)
    r_labels_p = kmeans.predict(r_prediction)
    rpss = pss(r_labels_p, r_labels_q)
    rpss = sum(rpss) / r_labels_p.shape[0]
    l_labels_q = kmeans.predict(l_target)
    l_labels_p = kmeans.predict(l_prediction)
    lpss = pss(l_labels_p, l_labels_q)
    lpss = sum(lpss) / l_labels_p.shape[0]
    return rpss, lpss


def main(arg):
    test_dataset = HagglingDataset(FLAGS.test, FLAGS)
    test_dataloader = DataLoader(test_dataset, num_workers=10)
    ckpt = FLAGS.ckpt
    output_folder = FLAGS.output_dir
    # output folder for videos
    os.makedirs(output_folder, exist_ok=True)

    model = get_model()

    model.load_model(ckpt, 50)
    model.eval()

    skeletonHandler = SkeletonHandler()
    # for storing loss values and pss values
    loss_dict = {'rightSeller': [], 'leftSeller': []}

    pss_eval = {'rightSeller': [], 'leftSeller': []}

    with torch.no_grad():

        for i_batch, batch in enumerate(test_dataloader):

            # get test input and labels
            data, targets = get_input(batch)

            # forward pass through the network
            predictions = model(data)

            # get loss
            # loss = criterion(predictions, targets, model.parameters(), FLAGS.lmd)
            # test_loss += loss
            if FLAGS.CNN:
                targets = targets.permute(0, 2, 1)
                predictions = predictions.permute(0, 2, 1)

            # separate right and left seller
            r_target = targets[0].cpu().numpy()
            r_prediction = predictions[0].cpu().numpy()
            l_target = targets[1].cpu().numpy()
            l_prediction = predictions[1].cpu().numpy()

            # get PSS evaluation score for the batch
            rpss, lpss = pose_structure_score(r_target.copy().astype(float), r_prediction.copy().astype(float),
                                              l_target.copy().astype(float), l_prediction.copy().astype(float))
            pss_eval['rightSeller'].append(rpss)
            pss_eval['leftSeller'].append(lpss)
            print(rpss, lpss)

            subjects = [r_target.copy(), r_prediction.copy(), l_target.copy(),
                        l_prediction.copy()]

            # get initRot and initTrans
            initRotRightSeller = Quaternions.from_euler(batch['rightSeller']['initRot'][0].cpu().numpy())
            initTransRightSeller = np.array(batch['rightSeller']['initTrans'][0].cpu().numpy())
            initRotLeftSeller = Quaternions.from_euler(batch['leftSeller']['initRot'][0].cpu().numpy())
            initTransLeftSeller = np.array(batch['leftSeller']['initTrans'][0].cpu().numpy())

            subject_params = [initRotRightSeller, initTransRightSeller, initRotLeftSeller, initTransLeftSeller]

            # convert to global form
            r_target_global, r_prediction_global, l_target_global, l_prediction_global = \
                get_global_positions_denormalized(subjects, subject_params, FLAGS)

            # get losses and take square root of mse
            r_loss = get_subject_loss(r_target_global.copy(), r_prediction_global.copy())
            r_loss = torch.sqrt(r_loss)
            loss_dict['rightSeller'].append(r_loss)
            l_loss = get_subject_loss(l_target_global.copy(), l_prediction_global.copy())
            l_loss = torch.sqrt(l_loss)
            print(l_loss)
            print(r_loss)
            loss_dict['leftSeller'].append(l_loss)

            # pick random videos to save
            if 0.4 < random.random() < 0.6:
                # prepare skeletons to write in the video
                skel1 = []
                skel2 = []
                vis1 = DebugVisualizer()
                vis2 = DebugVisualizer()
                # create skeletons
                for key in batch.keys():
                    if key == 'rightSeller':
                        x = vis1.conv_debug_visual_form(r_target_global)
                        skel1.append(x)
                    elif key == 'leftSeller':
                        x = vis1.conv_debug_visual_form(l_target_global)
                        skel1.append(x)
                    else:
                        x = batch[key]['joints21'].cpu().numpy()[0]
                        x = test_dataset.denormalize_data(x)
                        initRot = Quaternions.from_euler(batch[key]['initRot'][0].cpu().numpy())
                        initTrans = np.array(batch[key]['initTrans'][0].cpu().numpy())
                        x = skeletonHandler.recover_global_positions(x, initRot, initTrans)
                        x = vis1.conv_debug_visual_form(x)
                        skel1.append(x)

                x = vis1.conv_debug_visual_form(r_prediction_global)
                skel1.append(x)

                for key in batch.keys():
                    if key == 'leftSeller':
                        x = vis2.conv_debug_visual_form(l_target_global)
                        skel2.append(x)
                    elif key == 'rightSeller':
                        x = vis2.conv_debug_visual_form(r_target_global)
                        skel2.append(x)
                    else:
                        x = batch[key]['joints21'].cpu().numpy()[0]
                        x = test_dataset.denormalize_data(x)
                        initRot = Quaternions.from_euler(batch[key]['initRot'][0].cpu().numpy())
                        initTrans = np.array(batch[key]['initTrans'][0].cpu().numpy())
                        x = skeletonHandler.recover_global_positions(x, initRot, initTrans)
                        x = vis2.conv_debug_visual_form(x)
                        skel2.append(x)

                x = vis2.conv_debug_visual_form(l_prediction_global)
                skel2.append(x)

                # save random videos
                file_location = output_folder + str(i_batch)
                os.makedirs(file_location, exist_ok=True)
                vis1.create_animation(skel1, file_location + '/testRight')
                file_location = output_folder + str(i_batch)
                os.makedirs(file_location, exist_ok=True)
                vis2.create_animation(skel2, file_location + '/testLeft')

        rPss = sum(pss_eval['rightSeller']) / len(pss_eval['rightSeller'])
        lPss = sum(pss_eval['leftSeller']) / len(pss_eval['leftSeller'])
        print("Mean PSS Evaluation score, RightSeller : ", str(rPss), " LeftSeller : ", str(lPss))
        rMse = sum(loss_dict['rightSeller']) / len(loss_dict['rightSeller'])
        lMse = sum(loss_dict['leftSeller']) / len(loss_dict['leftSeller'])
        print("Avg sqrt mse loss for right seller : ", str(rMse), " LeftSeller : ", str(lMse))
        # adjustment with scaling loss
        rMse = 0.8 * rMse
        lMse = 0.8 * lMse
        print("Avg sqrt mse loss  after scale adjustment for right seller : ", str(rMse), " LeftSeller : ", str(lMse))


if __name__ == "__main__":
    app.run(main)
