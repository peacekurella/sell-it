import os
import numpy as np
import torch
import torch.nn as nn
from absl import app
from absl import flags
from torch.utils.data import DataLoader

from BodyAE import BodyAE
from BodyMotionGenerator import BodyMotionGenerator
from DebugVisualizer import DebugVisualizer
from HagglingDataset import HagglingDataset
from Quaternions import Quaternions
from preprocess import SkeletonHandler

FLGS = flags.FLAGS

flags.DEFINE_string('meta', 'meta/', 'Directory containing metadata files')
flags.DEFINE_string('test', '../Data/test/', 'Directory containing test files')
flags.DEFINE_string('output_dir', '../Data/output/', 'Folder to store final videos')
flags.DEFINE_string('ckpt', '../ckpt/AE', 'file containing the model weights')
flags.DEFINE_float('lmd', 0.001, 'L1 Regularization factor')
flags.DEFINE_boolean('bodyae', True, 'if True checks BodyAE model')


def get_input(batch):
    b = batch['buyer']['joints21']
    l = batch['leftSeller']['joints21']
    r = batch['rightSeller']['joints21']

    if FLGS.bodyae:
        train_x = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
        return train_x, train_y
    else:
        set_x_a = torch.cat((b, l), dim=2)
        set_x_b = torch.cat((b, r), dim=2)
        train_x = torch.cat((set_x_a, set_x_b), dim=0).permute(0, 2, 1).float().cuda()
        train_y = torch.cat((r, l), dim=0).permute(0, 2, 1).float().cuda()
        return train_x, train_y


def get_model(FLAGS):
    if FLAGS.bodyae:
        return BodyAE(FLAGS).cuda()
    else:
        return BodyMotionGenerator(FLAGS).cuda()


def get_global_positions_denormalized(subjects, subject_params, FLAGS):
    print(subjects[0].shape)
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


def main(arg):
    test_dataset = HagglingDataset(FLGS.test, FLGS)
    test_dataloader = DataLoader(test_dataset, num_workers=10)
    ckpt = FLGS.ckpt
    output_folder = FLGS.output_dir

    os.makedirs(output_folder, exist_ok=True)

    model = get_model(FLGS)

    model.load_model(ckpt, 18)

    skeletonHandler = SkeletonHandler()

    loss_dict = {'rightSeller': [], 'leftSeller': []}

    with torch.no_grad():

        for i_batch, batch in enumerate(test_dataloader):
            model.eval()

            # get test input and labels
            data, targets = get_input(batch)

            # forward pass through the network
            predictions = model(data)

            # get loss
            # loss = criterion(predictions, targets, model.parameters(), FLAGS.lmd)
            # test_loss += loss

            targets = targets.permute(0, 2, 1)
            predictions = predictions.permute(0, 2, 1)

            # separate right and left seller
            r_target = targets[0].cpu().numpy()
            r_prediction = predictions[0].cpu().numpy()
            l_target = targets[1].cpu().numpy()
            l_prediction = predictions[1].cpu().numpy()

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
                get_global_positions_denormalized(subjects, subject_params, FLGS)

            # get losses
            r_loss = get_subject_loss(r_target_global.copy(), r_prediction_global.copy())
            loss_dict['rightSeller'].append(r_loss)
            l_loss = get_subject_loss(l_target_global.copy(), l_prediction_global.copy())
            print(l_loss)
            loss_dict['leftSeller'].append(l_loss)

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

            # save the videos
            file_location = output_folder + str(i_batch)
            os.makedirs(file_location, exist_ok=True)
            vis1.create_animation(skel1, file_location + '/test1')
            file_location = output_folder + str(i_batch)
            os.makedirs(file_location, exist_ok=True)
            vis2.create_animation(skel2, file_location + '/test2')


if __name__ == "__main__":
    app.run(main)