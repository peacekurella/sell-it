import argparse
import os
import json
import pickle
import numpy as np
import Motion.Animation as Animation
from preprocess import SkeletonHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='HagglingData/',
                        help='input pickle files folder')
    parser.add_argument('--output', type=str, default='Data/', help='output folder')
    parser.add_argument('--window_size', type=int, default=240, help='number of frames in one window')
    parser.add_argument('--step_size', type=int, default=120, help='window step size')

    args = parser.parse_args()
    input_folder = args.source
    output_folder = args.output
    window_size = args.window_size
    step_size = args.step_size
    os.makedirs(output_folder, exist_ok=True)
    train_folder = output_folder + "/train/"
    test_folder = output_folder + "/test/"
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    f = open("testing_files.json")
    test_list_file = json.load(f)
    test_list = test_list_file["file_names"]

    files = os.listdir(input_folder)
    for file in files:
        read_file = input_folder + file
        r = open(read_file, "rb")
        motionData = pickle.load(r, encoding="Latin-1")
        datahandler = SkeletonHandler()
        # The meta directory contains the rest poses
        datahandler.generate_rest_pose('meta', 'meta')
        skel = []
        data_dict = {"winner_id": motionData['winnerId'], "subjects": {}}
        numFrames = motionData['subjects'][0]['joints19'].shape[1]
        numWindows = int(numFrames / window_size)
        padding_length = window_size - (numFrames % window_size)
        if padding_length > 0:
            numWindows += 1

        bid = motionData['buyerId']
        lid = motionData['leftSellerId']
        rid = motionData['rightSellerId']
        sub = {bid: [], lid: [], rid: []}

        # Read the pkl file
        for pid, subjectInfo in enumerate(motionData['subjects']):  # pid = 0,1, or 2. (Not humanId)

            # read in the pose data from pkl file
            normalizedPose = subjectInfo['joints19']

            # retarget onto CMU skeleton
            anim = datahandler.retarget_skeleton(normalizedPose)

            # Do FK recover 3D joint positions, select required Joints only
            positions = Animation.positions_global(anim)
            positions = positions[:, datahandler.jointIdx]

            # convert to the Holden form and return initial rotation
            # and translation
            h_form, initRot, initTrans = datahandler.export_animation(positions)

            # convert rotation quaternion to euler form list
            initRotEuler = initRot.euler().tolist()
            initTrans = initTrans.tolist()

            # get bodyNormal and faceNormal Info
            bodyNormal = subjectInfo['bodyNormal'][1:]
            faceNormal = subjectInfo['faceNormal'][1:]

            # padding data if required
            if padding_length > 0:
                h_form = np.pad(h_form, ((padding_length, 0), (0, 0)))
                bodyNormal = np.pad(bodyNormal, ((0, 0), (padding_length, 0)))
                faceNormal = np.pad(faceNormal, ((0, 0), (padding_length, 0)))

            # get humanId
            humanId = subjectInfo['humanId']
            sub[humanId] = [h_form, bodyNormal, faceNormal, initTrans, initRotEuler]

        subjects = {'buyer': {"human_Id": bid, "initRot": sub[bid][-1],
                              "initTrans": sub[bid][-2], "frames": [{}]},
                    'leftSeller': {"human_Id": lid, "initRot": sub[lid][-1],
                                   "initTrans": sub[lid][-2], "frames": [{}]},
                    "rightSeller": {"human_Id": rid, "initRot": sub[rid][-1],
                                    "initTrans": sub[rid][-2], "frames": [{}]}}

        name = file.split('.')[0]
        file_char = name.split('_')
        file_group_name = '_'.join(file_char[0:-1])
        start_index = 0
        for num in range(numWindows):
            end_index = start_index + window_size
            buyer = {"joints21": sub[bid][0][start_index:end_index, :].tolist(),
                     "body_normal": sub[bid][1][:, start_index:end_index].tolist(),
                     "face_normal": sub[bid][2][:, start_index:end_index].tolist()}
            subjects["buyer"]["frames"] = [buyer]
            leftSeller = {"joints21": sub[lid][0][start_index:end_index, :].tolist(),
                          "body_normal": sub[lid][1][:, start_index:end_index].tolist(),
                          "face_normal": sub[lid][2][:, start_index:end_index].tolist()}
            subjects["leftSeller"]["frames"] = [leftSeller]
            rightSeller = {"joints21": sub[rid][0][start_index:end_index, :].tolist(),
                           "body_normal": sub[rid][1][:, start_index:end_index].tolist(),
                           "face_normal": sub[rid][2][:, start_index:end_index].tolist()}
            subjects["rightSeller"]["frames"] = [rightSeller]
            data_dict["subjects"] = subjects
            file_name = name + '_' + str(num) + '.json'
            if file_group_name not in test_list:
                x = open(train_folder + file_name, 'w')
                with x as outfile:
                    json.dump(data_dict, outfile, sort_keys=False, indent=2)
            else:
                x = open(test_folder + file_name, 'w')
                with x as outfile:
                    json.dump(data_dict, outfile, sort_keys=False, indent=2)
            start_index += step_size
