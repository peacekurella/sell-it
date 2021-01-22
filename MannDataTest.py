from DebugVisualizer import DebugVisualizer
from preprocess import MannDataFormat
import pickle

# f = open('MannData/train/2.pkl')
vis = DebugVisualizer()
with open('MannData/train/0.pkl', "rb") as handle:
    retargeted_data = pickle.load(handle, encoding='Latin-1')

skeletons = []

for subject in retargeted_data.keys():
    joints = MannDataFormat.recover_global_positions(retargeted_data[subject]['joints21'],
                                                     retargeted_data[subject]['initRot']
                                                     , retargeted_data[subject]['initTrans'])

    skeletons.append(vis.conv_debug_visual_form(joints))

vis.create_animation(skeletons, 'test')
