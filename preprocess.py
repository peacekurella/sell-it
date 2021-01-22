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
from InverseKinematics import JacobianInverseKinematics
from Pivots import Pivots

FLAGS = flags.FLAGS
flags.DEFINE_string('bodyData', 'HagglingData/', 'Input folder containing source pickle files')
flags.DEFINE_string('faceData', None, 'Input folder containing source pickle files')
flags.DEFINE_string('speechData', 'HagglingSpeechData/', 'Input folder containing speech annotations')
flags.DEFINE_string('retData', 'Retargeted/', 'Intermediate output folder')
flags.DEFINE_string('output', 'MannData/', 'Output folder to place new files')
flags.DEFINE_string('format', 'mann', 'Data format to export')

flags.DEFINE_integer('window_size', 120, 'Number of frames in one window')
flags.DEFINE_integer('step_size', 10, 'window step size')


class MotionRetargeter:

    def __init__(self):
        # Note: we flip Y axis for the Panoptic, so left-right are flipped
        self.mapping = {
            13: 0, 16: 1,
            2: 12, 3: 13, 4: 14,  # left hip,knee, ankle, footend
            7: 6, 8: 7, 9: 8,  # right hip,knee, ankle, footend
            17: 0, 18: 9, 19: 10, 20: 11,
            24: 0, 25: 3, 26: 4, 27: 5
        }

        # selection indexes
        self.jointIdx = np.array([
            0,
            2, 3, 4, 5,
            7, 8, 9, 10,
            12, 13, 15, 16,
            18, 19, 20, 22,
            25, 26, 27, 29
        ])

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
        rest, names, _ = BVH.load('../meta/rest.bvh')  # temp

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

        # Do FK recover 3D joint positions, select required Joints only
        positions = Animation.positions_global(anim)
        positions = positions[:, self.jointIdx]

        return positions


def process_pkl_file(body_directory, face_directory, speech_directory, file):
    # get the motion data file
    motionData = os.path.join(body_directory, file)
    motionData = open(motionData, "rb")
    motionData = pickle.load(motionData, encoding="Latin-1")

    # get the role IDs
    lsId = motionData['leftSellerId']
    bId = motionData['buyerId']

    # get the motion data file
    speechData = os.path.join(speech_directory, file)
    speechData = open(speechData, "rb")
    speechData = pickle.load(speechData, encoding="Latin-1")

    # create an instance of the retargeter
    retargeter = MotionRetargeter()

    # final output dict
    sub = {}

    # go through all the subjects
    for pid, subjectInfo in enumerate(motionData['subjects']):  # pid = 0,1, or 2. (Not humanId)

        # get humanId
        humanId = subjectInfo['humanId']

        # get bodyNormal and faceNormal Info
        bodyNormal = subjectInfo['bodyNormal'][1:]
        faceNormal = subjectInfo['faceNormal'][1:]

        # read in the pose data from pkl file
        normalizedPose = subjectInfo['joints19']

        # retarget onto CMU skeleton
        positions = retargeter.retarget_skeleton(normalizedPose)

        # face data
        # replace with a function that returns the expression vector
        faceData = np.zeros((positions.shape[0], positions.shape[1] * positions.shape[2]))

        # read the speech data
        for speechInfo in speechData['speechData']:
            if speechInfo['humanId'] == humanId:
                speechInfo = speechInfo['indicator'][1:]
                break

        if humanId == bId:
            role = 'buyer'
        elif humanId == lsId:
            role = 'leftSeller'
        else:
            role = 'rightSeller'

        sub[role] = {
            'bodyData': [positions, bodyNormal, faceNormal],
            'faceData': faceData,
            'speechData': speechInfo
        }

    return sub


def export_retargeted(body_directory, face_directory, speech_directory, output_directory):
    """
    Unifies the body, face, and speech data
    :param body_directory: Body Data directory
    :param face_directory: Face Data directory
    :param speech_directory: Speech Data directory
    :param output_directory: Output Data directory
    :return: None
    """

    # load a list of all bad files
    bad_files = open("meta/bad_files.json")
    bad_files = json.load(bad_files)
    bad_files = bad_files["file_names"]

    # list of all files
    files = os.listdir(body_directory)

    # make sure ouput directory exists
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    # go through all the files in the dataset
    for file in files:

        # if not a bad file, retarget
        if file.split('.')[0] not in bad_files:
            # retargeted data with speech, face, body information
            retargeted_data = process_pkl_file(body_directory, face_directory, speech_directory, file)

            # dump the pickle file
            with open(os.path.join(output_directory, file), 'wb') as handle:
                pickle.dump(retargeted_data, handle)


class HoldenDataFormat:

    def __init__(self):
        pass

    @staticmethod
    def softmax(x, **kw):
        """
        Softmax function. Courtesy: @jhugestar
        :param x: input vector
        :param kw: input args
        :return: softmax output
        """
        softness = kw.pop('softness', 1.0)
        maxi, mini = np.max(x, **kw), np.min(x, **kw)
        return maxi + np.log(softness + np.exp(mini - maxi))

    @staticmethod
    def softmin(x, **kw):
        """
        Softmin function. Courtesy: @jhugestar
        :param x: input vector
        :param kw: arguments
        :return: softmin outputs
        """
        return abs(-HoldenDataFormat.softmax(-x, **kw))

    @staticmethod
    def export_animation(positions):
        """
        Converts animations into Learning format
        :param positions: animation to be exported
        :return:
        """

        # floor the positions
        positions = HoldenDataFormat.floor_skelton(positions)

        # get root projection
        r_pjx = HoldenDataFormat.project_root(positions)
        positions = np.concatenate([r_pjx, positions], axis=1)
        trans = positions[0, 0:1, :].copy()

        # get foot contacts
        feet_l, feet_r = HoldenDataFormat.generate_foot_contacts(positions)

        # get root velocity
        velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()

        # remove global translation
        trans_x = positions[:, 0:1, 0].copy()
        trans_z = positions[:, 0:1, 2].copy()
        positions[:, :, 0] = positions[:, :, 0] - trans_x
        positions[:, :, 2] = positions[:, :, 2] - trans_z

        # get forward direction
        forward = HoldenDataFormat.get_body_normal(positions)

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

    @staticmethod
    def floor_skelton(skeleton):
        """
        Puts the skeleton on the floor(x-z) plane
        :param skeleton: input global positions of skeleton (F, J, 3)
        :return: Floored skeleton (F, J, 3)
        """

        fid_l, fid_r = np.array([3, 4]), np.array([7, 8])
        foot_heights = np.minimum(skeleton[:, fid_l, 1], skeleton[:, fid_r, 1]).min(axis=1)
        floor_height = HoldenDataFormat.softmin(foot_heights, softness=0.5, axis=0)
        print(floor_height)
        skeleton[:, :, 1] -= floor_height

        return skeleton

    @staticmethod
    def project_root(skeleton):
        """
        Returns the roots projection on the ground plane
        :param skeleton: input global positions of skeleton (F, J, 3)
        :return: projection of root on the ground (F, 1, 3)
        """

        trajectory_filterwidth = 3

        # get the reference direction
        reference = skeleton[:, 0] * np.array([1, 0, 1])

        # smooth it out with gaussian filters
        reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')

        return reference[:, np.newaxis]

    @staticmethod
    def generate_foot_contacts(positions):
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

    @staticmethod
    def get_body_normal(positions):
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

    @staticmethod
    def recover_global_positions(processed, initRot, initTrans):
        """
        Recovers the original global positions given the Holden form
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

        joints = HoldenDataFormat.floor_skelton(joints[:, 1:])
        return filters.gaussian_filter1d(joints, 1, axis=0, mode='nearest')


def export_holden_data(input_directory, output_directory, window_length, stride):
    """
    exports the data in Holden format
    :param input_directory: Input directory of retargeted skeletons
    :param output_directory: Output directory
    :return:
    """

    # load a list of all testing_files
    test_list = open("meta/testing_files.json")
    test_list = json.load(test_list)
    test_list = test_list["file_names"]

    # train count and test count
    test_dir = os.path.join(output_directory, 'test')
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    train_dir = os.path.join(output_directory, 'train')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    stats_dir = os.path.join(output_directory, 'stats')
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)

    train, test = 0, 0

    joints_21 = []
    faces = []

    # go through all files
    for file in os.listdir(input_directory):

        print(file.split('.')[0])

        # read the pkl file
        with open(os.path.join(input_directory, file), "rb") as handle:
            retargeted_data = pickle.load(handle, encoding='Latin-1')

        # set num frames
        num_frames = len(retargeted_data['buyer']['bodyData'][0])

        # windowing through the sequence
        for idx in range(0, num_frames, stride):

            output_window = {}

            for role in retargeted_data:

                positions, bodyNormal, faceNormal = retargeted_data[role]['bodyData']
                faceData = retargeted_data[role]['faceData'][idx + 1: idx + window_length + 1]
                if len(faceData.shape) > 2:
                    faceData = faceData.reshape(faceData.shape[0], -1)
                speechData = retargeted_data[role]['speechData'][:, np.newaxis][idx + 1: idx + window_length + 1]

                # dont do anything if more than 95% of the window needs padding
                if positions[idx: idx + window_length + 1].shape[0] <= 0.95 * window_length:
                    continue

                anim, rot, trans = HoldenDataFormat.export_animation(positions[idx: idx + window_length + 1])

                # pad if needed
                if anim.shape[0] < window_length:
                    # set the padding length
                    pad_length = window_length - anim.shape[0]

                    # pad the outputs
                    anim = np.concatenate([anim, np.zeros((pad_length, anim.shape[1]))], axis=0)
                    faceData = np.concatenate([faceData, np.zeros((pad_length, faceData.shape[1]))], axis=0)
                    speechData = np.concatenate([speechData, np.zeros((pad_length, speechData.shape[1]))], axis=0)

                if file not in test_list:
                    joints_21.append(anim)
                    faces.append(faceData)

                output_window[role] = {
                    'joints21': anim,
                    'initRot': rot,
                    'initTrans': trans,
                    'faceData': faceData,
                    'speechData': speechData,
                    'bodyNormal': np.swapaxes(bodyNormal[:, idx + 1: idx + window_length + 1], 0, 1),
                    'faceNormal': np.swapaxes(faceNormal[:, idx + 1: idx + window_length + 1], 0, 1)
                }

            if '_'.join(file.split('.')[0].split('_')[:-1]) in test_list:
                with open(os.path.join(test_dir, str(test) + '.pkl'), 'wb') as handle:
                    pickle.dump(output_window, handle)
                test += 1
            else:
                with open(os.path.join(train_dir, str(train) + '.pkl'), 'wb') as handle:
                    pickle.dump(output_window, handle)
                train += 1

    # calculate the stats
    joints_21 = np.concatenate(joints_21, axis=0)
    faces = np.concatenate(faces, axis=0)

    bodyMean = np.mean(joints_21, axis=0)[np.newaxis, :]
    bodyStd = np.std(joints_21, axis=0)[np.newaxis, :]
    bodyStd[bodyStd == 0.0] = 1.0

    faceMean = np.mean(faces, axis=0)[np.newaxis, :]
    faceStd = np.std(faces, axis=0)[np.newaxis, :]
    faceStd[faceStd == 0.0] = 1.0

    means = {
        'joints21': bodyMean,
        'faceData': faceMean
    }

    stds = {
        'joints21': bodyStd,
        'faceData': faceStd
    }

    with open(os.path.join(stats_dir, 'mean.pkl'), 'wb') as handle:
        pickle.dump(means, handle)

    with open(os.path.join(stats_dir, 'std.pkl'), 'wb') as handle:
        pickle.dump(stds, handle)


class MannDataFormat(HoldenDataFormat):
    def __init__(self):
        pass

    @staticmethod
    def get_human_skeleton():
        # create the list of connections in the human body
        # 0 Root
        # 1 left hip 5 right hip
        # 2 left knee 6 right knee
        # 3 left ankle 7 right ankle
        # 4 left toe 8 right toe
        # 9 torso 10 chest
        # 11 neck 12 nose
        # 13 left shoulder 17 right shoulder
        # 14 left elbow 18 right elbow
        # 19 left wrist 15 right wrist
        # 16 left thumb 20 right thumb
        humanSkeleton = [
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
        return humanSkeleton

    @staticmethod
    def get_body_normal(positions):
        """
        Returns the forward direction of the skeleton
        :param positions: Global position tensor (F, J, 3)
        :return: Body normals paralell to x-z plane
        """

        sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
        across1 = positions[:, hip_l] - positions[:, hip_r]
        across0 = positions[:, sdr_l] - positions[:, sdr_r]
        across = across0 + across1
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

        direction_filterwidth = 20
        forward = np.cross(across, np.array([[0, 1, 0]]))
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

        return forward

    @staticmethod
    def export_animation(positions):
        """
        Defines the class to convert original data to MANN data format
        :param positions: original data (F, J, 3)
        :return: returns the exported dimensions
        """
        # floor the positions
        positions = HoldenDataFormat.floor_skelton(positions)

        # get root projection
        r_pjx = HoldenDataFormat.project_root(positions)

        # get forward direction and project it onto ground
        forward = MannDataFormat.get_body_normal(positions)

        # get root velocity
        r_vel = r_pjx[1:] - r_pjx[:-1]
        r_vel = np.reshape(r_vel[:, :, [0, 2]], (r_vel.shape[0], -1))

        # get angular velocity
        target = forward[:-1]
        a_vel = Quaternions.between(forward[1:], target)[:, np.newaxis].qs
        a_vel = np.reshape(a_vel, (a_vel.shape[0], -1))

        # calculate joint positions w.r.to root projection
        joint_positions = positions - r_pjx

        # calculate joint velocities in character space
        joint_velocities = joint_positions[1:] - joint_positions[:-1]
        joint_velocities = np.reshape(joint_velocities, (joint_velocities.shape[0], -1))

        # calculate joint orientations ( orientation of the bone )
        joint_orientation = []
        for bone in MannDataFormat.get_human_skeleton():
            start, end = bone[0], bone[1]
            orientation = joint_positions[:, end] - joint_positions[:, start]
            orientation = orientation / np.sqrt((orientation ** 2).sum(axis=-1))[..., np.newaxis]
            target = np.array([[0, 0, 1]]).repeat(len(orientation), axis=0)
            orientation_z = Quaternions.between(orientation, target)[:, np.newaxis].qs
            target = np.array([[0, 1, 0]]).repeat(len(orientation), axis=0)
            orientation_y = Quaternions.between(orientation, target)[:, np.newaxis].qs
            joint_orientation.append(np.concatenate([orientation_y, orientation_z], axis=-1))

        joint_orientation = np.concatenate(joint_orientation, axis=1)
        joint_orientation = np.reshape(joint_orientation, (joint_orientation.shape[0], -1))

        joint_positions = np.reshape(joint_positions, (joint_positions.shape[0], -1))

        pose = np.concatenate([r_vel, a_vel, joint_positions[:-1], joint_velocities, joint_orientation[:-1]], axis=-1)

        initTrans = r_pjx[0].copy()

        initRot = forward[0].copy()

        return pose, initRot, initTrans

    @staticmethod
    def recover_global_positions(processed, initRot, initTrans):
        """
        Recovers global positions from the given animation in Mann form
        :param processed: positions in Mann form in the character space
        :param initRot: initial normal direction of the body
        :param initTrans: initial position of the root projection
        """
        r_vel, joint_positions = processed[:, :2], processed[:, 6:69]

        joint_positions = np.reshape(joint_positions, (joint_positions.shape[0], -1, 3))

        initTrans = np.reshape(initTrans, (-1, 3))
        r_vel = np.insert(r_vel, 1, 0, axis=1)
        r_vel = np.reshape(r_vel, (r_vel.shape[0], 1, -1))

        r_pjx = r_vel.copy()
        r_pjx[0] += initTrans

        for i in range(1, r_pjx.shape[0]):
            r_pjx[i] += r_pjx[i - 1]

        joints = joint_positions + r_pjx

        for i in range(joints.shape[0]):
            print(joints[i, 4, :])

        joints = HoldenDataFormat.floor_skelton(joints)

        # return filters.gaussian_filter1d(joints, 1, axis=0, mode='nearest')
        return joints

def export_mann_data(input_directory, output_directory, window_length, stride):
    """
    exports the data in MANN format
    :param input_directory: Input directory of retargeted skeletons
    :param output_directory: Output directory
    :return:
    """

    # load a list of all testing_files
    test_list = open("meta/testing_files.json")
    test_list = json.load(test_list)
    test_list = test_list["file_names"]

    # train count and test count
    test_dir = os.path.join(output_directory, 'test')
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    train_dir = os.path.join(output_directory, 'train')
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)

    stats_dir = os.path.join(output_directory, 'stats')
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)

    train, test = 0, 0

    joints_21 = []
    faces = []

    # go through all files
    for file in os.listdir(input_directory):

        print(file.split('.')[0])

        # read the pkl file
        with open(os.path.join(input_directory, file), "rb") as handle:
            retargeted_data = pickle.load(handle, encoding='Latin-1')

        # set num frames
        num_frames = len(retargeted_data['buyer']['bodyData'][0])

        # windowing through the sequence
        for idx in range(0, num_frames, stride):

            output_window = {}

            for role in retargeted_data:

                positions, bodyNormal, faceNormal = retargeted_data[role]['bodyData']
                faceData = retargeted_data[role]['faceData'][idx + 1: idx + window_length + 1]
                if len(faceData.shape) > 2:
                    faceData = faceData.reshape(faceData.shape[0], -1)
                speechData = retargeted_data[role]['speechData'][:, np.newaxis][idx + 1: idx + window_length + 1]

                # dont do anything if more than 95% of the window needs padding
                if positions[idx: idx + window_length + 1].shape[0] <= 0.95 * window_length:
                    continue

                anim, rot, trans = MannDataFormat.export_animation(positions[idx: idx + window_length + 1])

                # pad if needed
                if anim.shape[0] < window_length:
                    # set the padding length
                    pad_length = window_length - anim.shape[0]

                    # pad the outputs
                    anim = np.concatenate([anim, np.zeros((pad_length, anim.shape[1]))], axis=0)
                    faceData = np.concatenate([faceData, np.zeros((pad_length, faceData.shape[1]))], axis=0)
                    speechData = np.concatenate([speechData, np.zeros((pad_length, speechData.shape[1]))], axis=0)

                if file not in test_list:
                    joints_21.append(anim)
                    faces.append(faceData)

                output_window[role] = {
                    'joints21': anim,
                    'initRot': rot,
                    'initTrans': trans,
                    'faceData': faceData,
                    'speechData': speechData,
                    'bodyNormal': np.swapaxes(bodyNormal[:, idx + 1: idx + window_length + 1], 0, 1),
                    'faceNormal': np.swapaxes(faceNormal[:, idx + 1: idx + window_length + 1], 0, 1)
                }

            if '_'.join(file.split('.')[0].split('_')[:-1]) in test_list:
                with open(os.path.join(test_dir, str(test) + '.pkl'), 'wb') as handle:
                    pickle.dump(output_window, handle)
                test += 1
            else:
                with open(os.path.join(train_dir, str(train) + '.pkl'), 'wb') as handle:
                    pickle.dump(output_window, handle)
                train += 1

    # calculate the stats
    joints_21 = np.concatenate(joints_21, axis=0)
    faces = np.concatenate(faces, axis=0)

    bodyMean = np.mean(joints_21, axis=0)[np.newaxis, :]
    bodyStd = np.std(joints_21, axis=0)[np.newaxis, :]
    bodyStd[bodyStd == 0.0] = 1.0

    faceMean = np.mean(faces, axis=0)[np.newaxis, :]
    faceStd = np.std(faces, axis=0)[np.newaxis, :]
    faceStd[faceStd == 0.0] = 1.0

    means = {
        'joints21': bodyMean,
        'faceData': faceMean
    }

    stds = {
        'joints21': bodyStd,
        'faceData': faceStd
    }

    with open(os.path.join(stats_dir, 'mean.pkl'), 'wb') as handle:
        pickle.dump(means, handle)

    with open(os.path.join(stats_dir, 'std.pkl'), 'wb') as handle:
        pickle.dump(stds, handle)


def main(args):
    # retarget skeletons if needed
    if not os.path.isdir(FLAGS.retData):
        export_retargeted(FLAGS.bodyData, FLAGS.faceData, FLAGS.speechData, FLAGS.retData)

    if FLAGS.format == 'holden':
        export_holden_data(FLAGS.retData, FLAGS.output, FLAGS.window_size, FLAGS.step_size)

    if FLAGS.format == 'mann':
        export_mann_data(FLAGS.retData, FLAGS.output, FLAGS.window_size, FLAGS.step_size)


if __name__ == '__main__':
    app.run(main)
