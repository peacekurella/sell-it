import sys
import os
import numpy as np
import pickle
import scipy.ndimage.filters as filters

import motion.BVH as BVH
import motion.Animation as Animation
from motion.Quaternions import Quaternions
from motion.InverseKinematics import BasicJacobianIK, JacobianInverseKinematics
from motion.Pivots import Pivots
from debugVisualizer import DataVisualizer


def visualize_points(positions):
    """
    Debug function for visualizing the rest skeleton
    :param positions: (F, J, 3) numpy array
    :return:
    """
    import matplotlib.pyplot as plt
    ax = plt.axes(projection='3d')

    points = positions[0]
    for i in range(points.shape[0]):
        ax.scatter(points[i][0], points[i][1], points[i][2])
        ax.text(points[i][0], points[i][1], points[i][2], str(i))

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def conv_debug_visual_form(rest_targets):  # skel: (F, J, 3)

    rest_targets = rest_targets.reshape(rest_targets.shape[0],
                                        rest_targets.shape[1] * rest_targets.shape[2])  # (F, 3J)
    rest_targets = np.swapaxes(rest_targets, 0, 1)  # (3J, F)

    return rest_targets


class DataHandler():

    def __init__(self):

        # selection indexes
        self.jointIdx = np.array([
            0,
            2, 3, 4, 5,
            7, 8, 9, 10,
            12, 13, 15, 16,
            18, 19, 20, 22,
            25, 26, 27, 29
        ])

        # Note: we flip Y axis for the Panoptic, so left-right are flipped
        self.mapping = {
            13: 0, 16: 1,
            2: 12, 3: 13, 4: 14,  # left hip,knee, ankle, footend
            7: 6, 8: 7, 9: 8,  # right hip,knee, ankle, footend
            17: 0, 18: 9, 19: 10, 20: 11,
            24: 0, 25: 3, 26: 4, 27: 5
        }

    def softmax(self, x, **kw):
        """
        Softmax function. Courtesy: @jhugestar
        :param x: input vector
        :param kw: input args
        :return: softmax output
        """
        softness = kw.pop('softness', 1.0)
        maxi, mini = np.max(x, **kw), np.min(x, **kw)
        return maxi + np.log(softness + np.exp(mini - maxi))

    def softmin(self, x, **kw):
        """
        Softmin function. Courtesy: @jhugestar
        :param x: input vector
        :param kw: arguments
        :return: softmin outputs
        """
        return -self.softmax(-x, **kw)

    @staticmethod
    def generate_rest_pose(in_path, out_path):
        """
        Generates a rest pose file with 21 joints
        :param in_path: Input path to the CMU mocap file containing T pose
        :param out_path: Output path to created rest file
        :return: None
        """
        # load the animation, first frame is T pose
        rest, names, _ = BVH.load(os.path.join(in_path, '01_01.bvh'))

        # get the first frame
        skel = rest.copy()
        skel.positions = rest.positions[0:1]
        skel.rotations = rest.rotations[0:1]

        # make sure it's at origin
        skel.positions[:, 0, 0] = 0
        skel.positions[:, 0, 2] = 0
        skel.offsets[0, 0] = 0
        skel.offsets[0, 2] = 0

        # scale the skeleton
        skel.offsets = skel.offsets * 6.25
        skel.positions = skel.positions * 6.25

        # save the rest skeleton and add new axis for frames
        rest.positions = skel.offsets[np.newaxis]
        rest.rotations.qs = skel.orients.qs[np.newaxis]

        # save the rest skeleton file
        BVH.save(os.path.join(out_path, 'rest.bvh'), rest, names)


    def retarget_skeleton(self, normalizedPose):
        """
        Retargets the Panoptic Skeleton onto CMU skeleton
        :param normalizedPose: Panoptic skeleton (57, F)
        :return: retargeted animation
        """

        # reshape
        normalizedPose = np.transpose(normalizedPose)  # (frames,57)
        normalizedPose = normalizedPose.reshape(normalizedPose.shape[0], 19, 3)  # (frames,19,3)

        # Flip Y axis
        normalizedPose[:, :, 1] = -normalizedPose[:, :, 1]

        # calculate panoptic height
        panopticThigh = normalizedPose[:, 6, :] - normalizedPose[:, 7, :]
        panopticThigh = panopticThigh ** 2
        panopticHeight = np.mean(np.sqrt(np.sum(panopticThigh, axis=1)))

        # load the rest skeleton
        rest, names, _ = BVH.load('rest.bvh')

        # create a mock animation for the required duration
        anim = rest.copy()
        anim.positions = anim.positions.repeat(normalizedPose.shape[0], axis=0)
        anim.rotations.qs = anim.rotations.qs.repeat(normalizedPose.shape[0], axis=0)

        # get the FK solved global positions
        cmuMocapJoints = Animation.positions_global(anim)

        # calculate CMU skeleton height
        cmuThigh = cmuMocapJoints[:, 2, :] - cmuMocapJoints[:, 3, :]
        cmuThigh = cmuThigh ** 2
        cmuMocapHeight = np.mean(np.sqrt(np.sum(cmuThigh, axis=1)))
        cmuMocapHeight = cmuMocapHeight * 0.9

        # scale the skelton appropriately
        scaleRatio = cmuMocapHeight / panopticHeight
        print("cmuMocapHeight: %f, panopticHeight %f, scaleRatio: %f " % (cmuMocapHeight, panopticHeight, scaleRatio))
        normalizedPose = normalizedPose * scaleRatio  # rescaling

        # compute mean across vector
        across1 = normalizedPose[:, 3] - normalizedPose[:, 9]  # Right -> left (3)  Shoulder
        across0 = normalizedPose[:, 6] - normalizedPose[:, 12]  # Right -> left (6) Hips
        across = across0 + across1  # frame x 3
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]  ##frame x 3. Unit vectors

        # compute forward direction
        forward = np.cross(across, np.array([[0, -1, 0]]))
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)

        # Set root's movement by hipCenter joints (idx=2)
        anim.positions[:, 0] = normalizedPose[:, 2] + np.array([0.0, 2.4, 0.0])
        anim.rotations[:, 0:1] = -Quaternions.between(forward, target)[:, np.newaxis]

        targetmap = {}
        for k in self.mapping:
            targetmap[k] = normalizedPose[:, self.mapping[k], :]

        # Retarget using JacobianIK
        ik = JacobianInverseKinematics(anim, targetmap, iterations=20, damping=10.0, silent=True)
        ik()

        # scale skeleton appropriately
        anim.positions = anim.positions * 6.25
        anim.offsets = anim.offsets * 6.25

        return anim

    def export_animation(self, anim):
        """
        Converts animations into Learning format
        :param anim: animation to be exported
        :return:
        """

        # do FK and get global positons
        positions = Animation.positions_global(anim)

        # select only required joints
        positions = positions[:, self.jointIdx]

        # floor the positions
        positions = self.floor_skelton(positions)

        # get root projection
        r_pjx = self.project_root(positions)
        positions = np.concatenate([r_pjx, positions], axis=1)

        # get foot contacts
        feet_l, feet_r = self.generate_foot_contacts(positions)

        # get root velocity
        velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()

        # remove global translation
        trans_x = positions[:, 0:1, 0].copy()
        trans_z = positions[:, 0:1, 2].copy()
        positions[:, :, 0] = positions[:, :, 0] - trans_x
        positions[:, :, 2] = positions[:, :, 2] - trans_z

        # get forward direction
        forward = self.get_forward(positions)

        # rotate by Y axis to make sure skeleton faces the forward direction
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        rotations = Quaternions.between(forward, target)[:, np.newaxis]
        positions = rotations * positions
        velocity = rotations[1:] * positions

        # get root velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

        # concatenate all the additional elements and reshape
        positions = positions[:-1]
        positions = positions.reshape(len(positions), -1)
        positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
        positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
        positions = np.concatenate([positions, rvelocity], axis=-1)
        positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

        return positions


    def floor_skelton(self, skeleton):
        """
        Puts the skeleton on the floor(x-z) plane
        :param skeleton: input global positions of skeleton (F, J, 3)
        :return: Floored skeleton (F, J, 3)
        """

        fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
        foot_heights = np.minimum(skeleton[:, fid_l, 1], skeleton[:, fid_r, 1]).min(axis=1)
        floor_height = self.softmin(foot_heights, softness=0.5, axis=0)
        skeleton[:,:,1] -= floor_height

        return skeleton

    def project_root(self, skeleton):
        """
        Returns the roots projection on the ground plane
        :param skeleton: input global positions of skeleton (F, J, 3)
        :return: projection of root on the ground (F, 1, 3)
        """
        trajectory_filterwidth = 3
        reference = skeleton[:, 0] * np.array([1, 0, 1])

        # smooth it out with gaussian filters
        reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')

        return reference[:, np.newaxis]

    def generate_foot_contacts(self, positions):
        """
        Generate foot contact signals
        :param positions: Input global positions (F, J+1, 3)
        :return: foot contact signals (F, 2)
        """

        velfactor, heightfactor = np.array([0.05, 0.05]), np.array([3.0, 2.0])
        fid_l, fid_r = np.array([4, 5]), np.array([8, 9])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)

        return feet_l, feet_r

    def get_forward(self, positions):
        """
        Returns the forward direction of the skeleton
        :param positions: Global position tensor (F, J+1, 3)
        :return: forward direction
        """

        sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
        across1 = positions[:, hip_l] - positions[:, hip_r]
        across0 = positions[:, sdr_l] - positions[:, sdr_r]
        across = across0 + across1
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        direction_filterwidth = 20
        forward = np.cross(across, np.array([[0, 1, 0]]))
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

        return forward



# motionData = pickle.load(open("170221_haggling_b1_group0.pkl", "rb"), encoding="Latin-1")
# datahandler = DataHandler()
#
# datahandler.generate_rest_pose('.', '.')
# skel = []
# for pid, subjectInfo in enumerate(motionData['subjects']):  # pid = 0,1, or 2. (Not humanId)
#
#     normalizedPose = subjectInfo['joints19']
#     anim = datahandler.retarget_skeleton(normalizedPose)
#     skel.append(conv_debug_visual_form(datahandler.export_animation(anim)))
#
# vis = DataVisualizer()
# vis.create_animation(skel, None)
