import numpy as np
import torch
import torch.nn as nn
import random
import os
from sklearn.metrics import accuracy_score
from preprocess import export_frechet_animation


from preprocess import MannDataFormat, HoldenDataFormat
from HagglingDataset import HagglingDataset
from Quaternions import Quaternions
from DebugVisualizer import DebugVisualizer
from FrechetEvaluation import FrechetEvaluation


class Metrics():

    def __init__(self, FLAGS):
        if FLAGS.fmt == 'mann':
            self.format = MannDataFormat()
        else:
            self.format = HoldenDataFormat()

        self.haggling = HagglingDataset(FLAGS.test, FLAGS)
        self.num_saves = FLAGS.num_saves
        self.frechet = FrechetEvaluation(FLAGS)
        print('got frechet model')
        self.skeleton = [
            [1, 5],
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [5, 6],
            [1, 2],
            [6, 7],
            [2, 3],
            [13, 17],
            [17, 18],
            [13, 14],
            [14, 15],
            [18, 19]
        ]
        print('got skeleton')
        self.output_folder = FLAGS.output_dir + FLAGS.model + '/'
        os.makedirs(self.output_folder, exist_ok=True)

        print('metrics successfully loaded')

    @staticmethod
    def get_mse_loss(predictions, targets):
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

    @staticmethod
    def get_npss_score(predictions, targets):
        """
        Calculate the power spectrum similiarity
        :param predictions:
        :param targets:
        :return:
        """

        targets = targets.reshape(-1, 63)
        predictions = predictions.reshape(-1, 63)

        target_fft = np.fft.fft(targets, axis=1)
        predictions_fft = np.fft.fft(predictions, axis=1)

        target_fft = np.square(np.absolute(target_fft))
        predictions_fft = np.square(np.absolute(predictions_fft))

        target_total_power = np.sum(target_fft, axis=0)[np.newaxis, :]
        target_total_power[target_total_power == 0] = 1.0

        predictions_total_power = np.sum(predictions_fft, axis=0)[np.newaxis, :]
        predictions_total_power[predictions_total_power == 0] = 1.0

        seq_feature_power = target_total_power

        target_fft = target_fft / target_total_power
        predictions_fft = predictions_fft / predictions_total_power

        target_fft = np.cumsum(target_fft, axis=1)
        predictions_fft = np.cumsum(predictions_fft, axis=1)

        emd = np.linalg.norm((predictions_fft - target_fft), ord=1, axis=0)[np.newaxis, :]

        return np.average(emd, weights=seq_feature_power)

    def get_frechet_distance(self, predictions, targets):
        """
        Calculate the frechet distance
        :param predictions:
        :param targets:
        :return:
        """

        prediction_feat, _ = export_frechet_animation(predictions, self.skeleton)

        prediction_feat = torch.from_numpy(prediction_feat).float().cuda()

        gt_feat, _ = export_frechet_animation(targets, self.skeleton)

        gt_feat = torch.from_numpy(gt_feat).float().cuda()

        frechet_dist = self.frechet.frechet_distance(prediction_feat, gt_feat)

        return frechet_dist

    def get_speech_accuracy(self, predictions, targets):
        """
        Calculate speech accuracy
        :param predictions: batch_size, seq_length, 1
        :param targets: batch_size, seq_legnth, 1
        :return : accuracy
        """
        predictions = (predictions > 0.5).astype(int).flatten()

        targets = targets.flatten()

        acc = accuracy_score(predictions, targets)

        return acc

    def get_speech_data(self, predictions, batch):

        batch_size = batch['buyer']['joints21'].shape[0]
        prediction_subjects = []
        target_subjects = []
        prediction_subjects.append(
            (predictions['speech'][:batch_size].cpu().numpy(),
             predictions['speech'][batch_size:].cpu().numpy()))
        target_subjects.append((batch['rightSeller']['speakingStatus'].cpu().numpy(),
                                batch['leftSeller']['speakingStatus'].cpu().numpy()))
        return prediction_subjects, target_subjects

    def get_global_positions(self, subjects):
        """
            denormalize and globalize positions of humans
        """
        dataformat = self.format
        r_subject = subjects[0]
        l_subject = subjects[1]
        buyer = subjects[2]

        r_global = dataformat.recover_global_positions(*r_subject)

        l_global = dataformat.recover_global_positions(*l_subject)

        b_global = dataformat.recover_global_positions(*buyer)

        global_subjects = [
            r_global,
            l_global,
            b_global
        ]

        return global_subjects

    def split_into_subjects(self, predictions, targets, batch):
        # separate right and left seller
        haggling = self.haggling
        batch_size = int(targets.shape[0] / 2)

        r_target = haggling.denormalize_data(targets[0:batch_size].cpu().numpy())
        r_prediction = haggling.denormalize_data(predictions[0:batch_size].cpu().numpy())
        l_target = haggling.denormalize_data(targets[batch_size:].cpu().numpy())
        l_prediction = haggling.denormalize_data(predictions[batch_size:].cpu().numpy())
        buyer = haggling.denormalize_data(batch['buyer']['joints21'].cpu().numpy())

        # get initRot and initTrans
        initRotRightSeller = Quaternions(batch['rightSeller']['initRot'].cpu().numpy())
        initTransRightSeller = batch['rightSeller']['initTrans'].cpu().numpy()
        initRotLeftSeller = Quaternions(batch['leftSeller']['initRot'].cpu().numpy())
        initTransLeftSeller = batch['leftSeller']['initTrans'].cpu().numpy()
        initRotBuyer = Quaternions(batch['buyer']['initRot'].cpu().numpy())
        initTransBuyer = batch['buyer']['initTrans'].cpu().numpy()

        prediction_subjects = [
            (r_prediction.copy(), initRotRightSeller, initTransRightSeller),
            (l_prediction.copy(), initRotLeftSeller, initTransLeftSeller),
            (buyer.copy(), initRotBuyer, initTransBuyer),
        ]

        target_subjects = [
            (r_target.copy(), initRotRightSeller, initTransRightSeller),
            (l_target.copy(), initRotLeftSeller, initTransLeftSeller),
            (buyer.copy(), initRotBuyer, initTransBuyer)
        ]

        return prediction_subjects, target_subjects

    def save_files(self, prediction, targets, role, i_batch, test_num, idx):
        # prepare skeletons to write in the video
        skel = []

        vis = DebugVisualizer()

        # create skeletons
        for target in targets:
            x = vis.conv_debug_visual_form(target[idx])
            skel.append(x)

        x = vis.conv_debug_visual_form(prediction[idx])
        skel.append(x)

        file_location = self.output_folder + str(test_num) + '_' + str(i_batch) + '_' + str(idx)
        os.makedirs(file_location, exist_ok=True)
        vis.create_animation(skel, file_location + '/' + role)
        del vis

    def compute_and_save(self, predictions, targets, batch, i_batch, test_num):
        prediction, targets = self.split_into_subjects(predictions['pose'], targets['pose'], batch)

        prediction = self.get_global_positions(prediction)
        targets = self.get_global_positions(targets)

        metrics = {
            "RightMSE": Metrics.get_mse_loss(prediction[0], targets[0][:, :prediction[0].shape[1]]).cpu().numpy().item(),
            "LeftMSE": Metrics.get_mse_loss(prediction[1], targets[1][:, :prediction[1].shape[1]]).cpu().numpy().item(),

            "RightNPSS": Metrics.get_npss_score(prediction[0], targets[0][:, :prediction[0].shape[1]]),
            "LeftNPSS": Metrics.get_npss_score(prediction[1], targets[1][:, :prediction[1].shape[1]]),

            "RightFrechet": Metrics.get_frechet_distance(self, prediction[0], targets[0][:, :prediction[0].shape[1]]),
            "LeftFrechet": Metrics.get_frechet_distance(self, prediction[1], targets[1][:, :prediction[1].shape[1]])
        }
        if 'speech' in predictions.keys():
            pred, tar = self.get_speech_data(predictions, batch)
            metrics['RightSpeech'] = self.get_speech_accuracy(pred[0][0], tar[0][0])
            metrics['LeftSpeech'] = self.get_speech_accuracy(pred[0][1], tar[0][1])
            metrics['Speech'] = (metrics['RightSpeech'] + metrics['LeftSpeech']) / 2

        idxs = random.sample(range(prediction[0].shape[0]), self.num_saves)

        for i in idxs:
            self.save_files(prediction[0], targets, 'right', i_batch, test_num, i)
            self.save_files(prediction[1], targets, 'left', i_batch, test_num, i)

        metrics['MSE'] = (metrics['RightMSE'] + metrics['LeftMSE']) / 2
        metrics['NPSS'] = (metrics['RightNPSS'] + metrics['LeftNPSS']) / 2
        metrics['Frechet'] = (metrics['RightFrechet'] + metrics['LeftFrechet']) / 2

        return metrics
