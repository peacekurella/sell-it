import numpy as np
import json

def vis(session):
    """
    session.shape = (num_skeletons, 63, num_frames)
    """
    # Create a json object of session data
    videos = []
    num_skeletons = data.shape[0]
    for skeleton_idx in range(num_skeletons):
        skeleton = data[skeleton_idx]
        num_frames = skeleton.shape[1]
        video = []
        for frame_idx in range(num_frames):
            frame = {
                'joints21': skeleton[:, frame_idx].tolist()
            }
            video.append(frame)
        video = {
            'id': skeleton_idx,
            'frames': video,
            }
        videos.append(video)
    # Write to file in /tmp
    filename = '/tmp/sell-it-vis'
    f = open(filename, 'w')
    f.write(json.dumps(videos))
    f.close()
    # Build and execute rust vis using the file as argument
    import os
    os.system('cargo +nightly run --features bevy/dynamic --release \'{}\''.format(filename))

data = np.load('example.npy')
vis(data)
