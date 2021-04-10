import numpy as np
import torch
import torch.nn as nn
import random
import os
from scipy import linalg
from preprocess import export_frechet_animation

from EmbeddingNet import EmbeddingNet
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

        self.output_folder = FLAGS.output_dir
        os.makedirs(self.output_folder, exist_ok=True)

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
        Calculate the power spectrum similiarity
        :param predictions:
        :param targets:
        :return:
        """
        vis = DebugVisualizer()

        prediction_feat, _ = export_frechet_animation(predictions, vis)

        prediction_feat = torch.from_numpy(prediction_feat).float().cuda()

        gt_feat, _ = export_frechet_animation(targets, vis)

        gt_feat = torch.from_numpy(gt_feat).float().cuda()

        frechet_dist = self.frechet.frechet_distance(prediction_feat, gt_feat)

        return frechet_dist

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
            (buyer.copy(), initRotBuyer, initTransBuyer)
        ]

        target_subjects = [
            (r_target.copy(), initRotRightSeller, initTransRightSeller),
            (l_target.copy(), initRotLeftSeller, initTransLeftSeller),
            (buyer.copy(), initRotBuyer, initTransBuyer)
        ]

        return prediction_subjects, target_subjects

    def save_files(self, prediction, targets, role, i_batch, test_num):
        # prepare skeletons to write in the video
        skel = []

        vis = DebugVisualizer()

        # create skeletons
        for target in targets:
            x = vis.conv_debug_visual_form(target)
            skel.append(x)

        x = vis.conv_debug_visual_form(prediction)
        skel.append(x)

        file_location = self.output_folder + str(i_batch) + '_' + str(test_num)
        os.makedirs(file_location, exist_ok=True)
        vis.create_animation(skel, file_location + '/' + role)

    def compute_and_save(self, predictions, targets, batch, i_batch, test_num):
        predictions, targets = self.split_into_subjects(predictions['pose'], targets['pose'], batch)

        predictions = self.get_global_positions(predictions)
        targets = self.get_global_positions(targets)

        metrics = {
            "mse_right_seller" : Metrics.get_mse_loss(predictions[0], targets[0]).cpu().numpy().item(),
            "mse_left_seller"  : Metrics.get_mse_loss(predictions[1], targets[1]).cpu().numpy().item(),

            "npss_right_seller" : Metrics.get_npss_score(predictions[0], targets[0]),
            "npss_left_seller" : Metrics.get_npss_score(predictions[1], targets[1]),

            "frechet_right_seller" : Metrics.get_frechet_distance(self, predictions[0], targets[0]),
            "frechet_left_seller" : Metrics.get_frechet_distance(self, predictions[1], targets[1])
        }

        # if 0.4 < random.random() < 0.5 and self.num_saves > 0:
        #     self.save_files(predictions[0], targets, 'right', i_batch, test_num)
        #     self.save_files(predictions[1], targets, 'left', i_batch, test_num)
        #     self.num_saves -= 1

        return metrics
