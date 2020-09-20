import os
import numpy as np
import pickle
import scipy.ndimage.filters as filters
import json
from absl import app
from absl import flags

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics
from Pivots import Pivots
from DebugVisualizer import DebugVisualizer

FLAGS = flags.FLAGS
flags.DEFINE_string('source', 'HagglingData/', 'Input folder containing source pickle files')
flags.DEFINE_string('output', 'Data/', 'Output folder to place new files')
flags.DEFINE_integer('window_size', 120, 'Number of frames in one window')
flags.DEFINE_integer('step_size', 10, 'window step size')


class SkeletonHandler:
    """ Class for Handling Skeleton Data """

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
        rest, names, _ = BVH.load('meta/rest.bvh')

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

    def export_animation(self, positions):
        """
        Converts animations into Learning format
        :param positions: animation to be exported
        :return:
        """

        # floor the positions
        positions = self.floor_skelton(positions)

        # get root projection
        r_pjx = self.project_root(positions)
        positions = np.concatenate([r_pjx, positions], axis=1)
        trans = positions[0, 0:1, :].copy()

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
        forward = self.get_body_normal(positions)

        # rotate by Y axis to make sure skeleton faces the forward direction
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        rotations = Quaternions.between(forward, target)[:, np.newaxis]
        positions = rotations * positions
        velocity = rotations[1:] * velocity

        # get root velocity
        rvelocity = Pivots.from_quaternions(rotations[1:] * -rotations[:-1]).ps

        # concatenate all the additional elements and reshape
        positions = positions[:-1]
        positions = positions.reshape(len(positions), -1)
        positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
        positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
        positions = np.concatenate([positions, rvelocity], axis=-1)
        positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

        return positions, -rotations[0], trans

    def floor_skelton(self, skeleton):
        """
        Puts the skeleton on the floor(x-z) plane
        :param skeleton: input global positions of skeleton (F, J, 3)
        :return: Floored skeleton (F, J, 3)
        """

        fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
        foot_heights = np.minimum(skeleton[:, fid_l, 1], skeleton[:, fid_r, 1]).min(axis=1)
        floor_height = self.softmin(foot_heights, softness=0.5, axis=0)
        skeleton[:, :, 1] -= floor_height

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

    def get_body_normal(self, positions):
        """
        Returns the forward direction of the skeleton
        :param positions: Global position tensor (F, J+1, 3)
        :return: Body normals paralell to x-z plane
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

    def recover_global_positions(self, processed, initRot, initTrans):
        """
        Rescovers the original global positions given the Holden form
        :param processed: Holden data format gestures (F, 73)
        :param initRot: intial rotation of the skeleton
        :param initTrans: initial translation of the skeleton
        :return:
        """

        # split into joints and root velocities
        joints = processed[:, :-7]
        root_x, root_z, root_r = processed[:, -7], processed[:, -6], processed[:, -5]

        # reshape into the right format
        joints = joints.reshape(len(joints), -1, 3)

        # set initial rotation and translation
        if initRot is None:
            rotation = Quaternions.id(1)
        else:
            rotation = initRot

        if initTrans is None:
            translation = np.array([[0, 0, 0]])
        else:
            translation = initTrans

        # maintain a list of the offsets
        offsets = []

        # integrate over time to recover original values
        for i in range(len(joints)):
            joints[i, :, :] = rotation * joints[i]
            joints[i, :, 0] = joints[i, :, 0] + translation[0, 0]
            joints[i, :, 2] = joints[i, :, 2] + translation[0, 2]
            rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
            offsets.append(rotation * np.array([0, 0, 1]))
            translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

        joints = self.floor_skelton(joints[:, 1:])

        return filters.gaussian_filter1d(joints, 1, axis=0, mode='nearest')


def main(argv):
    print('flag arguments')
    print('source folder', FLAGS.source)
    print('output folder', FLAGS.output)
    print('window_size', FLAGS.window_size)
    print('step_size', FLAGS.step_size)

    input_folder = FLAGS.source
    output_folder = FLAGS.output
    window_size = FLAGS.window_size
    step_size = FLAGS.step_size
    os.makedirs(output_folder, exist_ok=True)
    train_folder = output_folder + "/train/"
    test_folder = output_folder + "/test/"
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    f = open("meta/testing_files.json")
    test_list_file = json.load(f)
    test_list = test_list_file["file_names"]

    files = os.listdir(input_folder)
    test_count = 0
    train_count = 0
    for file in files:
        read_file = input_folder + file
        r = open(read_file, "rb")
        motionData = pickle.load(r, encoding="Latin-1")
        datahandler = SkeletonHandler()
        # The meta directory contains the rest poses
        # datahandler.generate_rest_pose('meta', 'meta')
        skel = []
        data_dict = {"winner_id": motionData['winnerId'], "subjects": {}}
        numFrames = motionData['subjects'][0]['joints19'].shape[1]
        padding_length = 0
        num_windows = int(numFrames / step_size)
        if numFrames % step_size != 0:
            padding_length = step_size - (numFrames % step_size)
            num_windows += 1

        bid = motionData['buyerId']
        lid = motionData['leftSellerId']
        rid = motionData['rightSellerId']
        sub = {bid: [], lid: [], rid: []}

        # Read the pkl file
        for pid, subjectInfo in enumerate(motionData['subjects']):  # pid = 0,1, or 2. (Not humanId)

            # get humanId
            humanId = subjectInfo['humanId']

            # get bodyNormal and faceNormal Info
            bodyNormal = subjectInfo['bodyNormal'][1:]
            faceNormal = subjectInfo['faceNormal'][1:]

            # read in the pose data from pkl file
            normalizedPose = subjectInfo['joints19']

            # retarget onto CMU skeleton
            anim = datahandler.retarget_skeleton(normalizedPose)

            # Do FK recover 3D joint positions, select required Joints only
            positions = Animation.positions_global(anim)
            positions = positions[:, datahandler.jointIdx]

            sub[humanId] = [positions, bodyNormal, faceNormal]

        subjects = {'buyer': {"human_Id": bid, "initRot": [],
                              "initTrans": [], "frames": [{}]},
                    'leftSeller': {"human_Id": lid, "initRot": [],
                                   "initTrans": [], "frames": [{}]},
                    "rightSeller": {"human_Id": rid, "initRot": [],
                                    "initTrans": [], "frames": [{}]}}

        name = file.split('.')[0]
        file_char = name.split('_')
        # to check if file should be in test or train folder
        file_group_name = '_'.join(file_char[0:-1])
        start_index = 0
        # if padding, then end_index needs to be adjusted
        if padding_length > 0:
            end_index = start_index + window_size - padding_length
        else:
            end_index = start_index + window_size
        while end_index <= numFrames:
            for key in sub.keys():
                positions = sub[bid][0][start_index:end_index, :, :]
                h_form, initRot, initTrans = datahandler.export_animation(positions)
                bodyNormal = sub[key][1][:, start_index:end_index]
                faceNormal = sub[key][2][:, start_index:end_index]
                # padding the arrays
                if start_index < padding_length:
                    h_form = np.pad(h_form, ((padding_length, 0), (0, 0)))
                    bodyNormal = np.pad(bodyNormal, ((0, 0), (padding_length, 0)))
                    faceNormal = np.pad(faceNormal, ((0, 0), (padding_length, 0)))
                # convert rotation quaternion to euler form list, also convert all np arrays to list
                initRotEuler = initRot.euler().tolist()
                initTrans = initTrans.tolist()
                h_form = h_form.tolist()
                bodyNormal = bodyNormal.tolist()
                faceNormal = faceNormal.tolist()
                # save to the appropriate dictionary
                if key == bid:
                    buyer = {"joints21": h_form,
                             "body_normal": bodyNormal,
                             "face_normal": faceNormal}
                    subjects["buyer"]["frames"] = [buyer]
                    subjects["buyer"]["initRot"] = initRotEuler
                    subjects["buyer"]["initTrans"] = initTrans
                elif key == lid:
                    leftSeller = {"joints21": h_form,
                                  "body_normal": bodyNormal,
                                  "face_normal": faceNormal}
                    subjects["leftSeller"]["frames"] = [leftSeller]
                    subjects["leftSeller"]["initRot"] = initRotEuler
                    subjects["leftSeller"]["initTrans"] = initTrans
                else:
                    rightSeller = {"joints21": h_form,
                                   "body_normal": bodyNormal,
                                   "face_normal": faceNormal}
                    subjects["rightSeller"]["frames"] = [rightSeller]
                    subjects["rightSeller"]["initRot"] = initRotEuler
                    subjects["rightSeller"]["initTrans"] = initTrans

            data_dict["subjects"] = [subjects]
            # save the file to the destined folder
            if file_group_name not in test_list:
                file_name = str(train_count) + '.json'
                x = open(train_folder + file_name, 'w')
                with x as outfile:
                    json.dump(data_dict, outfile, sort_keys=False, indent=2)
                train_count += 1
            else:
                file_name = str(test_count) + '.json'
                x = open(test_folder + file_name, 'w')
                with x as outfile:
                    json.dump(data_dict, outfile, sort_keys=False, indent=2)
                test_count += 1
            # if padding, start_index of next window needs to be adjusted to maintain proper overlap
            if start_index < padding_length:
                start_index += step_size - padding_length
                padding_length = 0
            else:
                start_index += step_size

            end_index = start_index + window_size

if __name__ == '__main__':
    app.run(main)
