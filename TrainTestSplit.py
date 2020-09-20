import argparse
import os
import json
import pickle
import numpy as np
import Animation as Animation
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

    f = open("meta/testing_files.json")
    test_list_file = json.load(f)
    test_list = test_list_file["file_names"]

    files = os.listdir(input_folder)
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
        window_num = 0
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
            file_name = name + '_' + str(window_num) + '.json'
            # save the file to the destined folder
            if file_group_name not in test_list:
                x = open(train_folder + file_name, 'w')
                with x as outfile:
                    json.dump(data_dict, outfile, sort_keys=False, indent=2)
            else:
                x = open(test_folder + file_name, 'w')
                with x as outfile:
                    json.dump(data_dict, outfile, sort_keys=False, indent=2)

            # if padding, start_index of next window needs to be adjusted to maintain proper overlap
            if start_index < padding_length:
                start_index += step_size - padding_length
                padding_length = 0
            else:
                start_index += step_size

            end_index = start_index + window_size
            window_num += 1
