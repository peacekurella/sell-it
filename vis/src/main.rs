mod game;
mod skeleton_video;

use bevy::prelude::*;
use game::{
    camera_plugin::CameraPlugin,
    ground_plugin::GroundPlugin,
    pause_plugin::Pause,
    pause_plugin::PausePlugin,
    skeleton_videos_plugin::{Bone, SkeletonVideosPlugin},
    status_bar_plugin::{FrameNumber, StatusBarPlugin},
};
use skeleton_video::SkeletonVideo;
use std::env;
use std::fs::File;
use std::io::BufReader;

fn main() {
    // Load from file
    let args = env::args().collect::<Vec<String>>();
    let file = File::open(&args[1]).unwrap();
    let reader = BufReader::new(file);
    let skeletons: Vec<SkeletonVideo> = serde_json::from_reader(reader).unwrap();
    // Some assertions
    assert!(skeletons.len() > 0);
    let video_len0 = skeletons[0].num_frames();
    assert!(video_len0 > 0);
    let all_skeletons_have_equal_len = skeletons
        .iter()
        .all(|video| video.num_frames() == video_len0);
    assert!(all_skeletons_have_equal_len);

    App::build()
        .insert_resource(ClearColor(Color::rgb(0.0, 0.0, 0.0)))
        .insert_resource(WindowDescriptor {
            width: 1000.0,
            height: 600.0,
            ..Default::default()
        })
        .add_plugins(DefaultPlugins)
        .add_plugin(CameraPlugin)
        .add_plugin(StatusBarPlugin)
        .add_plugin(PausePlugin)
        .add_system(bevy::input::system::exit_on_esc_system.system())
        .add_plugin(GroundPlugin)
        .add_plugin(SkeletonVideosPlugin::new(skeletons))
        .add_system(control.system())
        .run();
}

fn control(
    pause: Res<Pause>,
    mut frame_number: ResMut<FrameNumber>,
    skeletons: Res<Vec<SkeletonVideo>>,
    mut root_query: Query<(&Bone, &mut Transform)>,
) {
    // Pause => pause everything
    if pause.0 {
        return;
    }
    for (idx, mut transform) in root_query.iter_mut() {
        let skeleton_idx = idx.0;
        let bone_idx = idx.1;
        let (_, trans, rot) = skeletons[skeleton_idx].bone(bone_idx, frame_number.0);
        transform.translation = trans;
        transform.rotation = rot;
    }
    if frame_number.0 + 1 < skeletons[0].num_frames() {
        frame_number.0 += 1;
    } else {
        frame_number.0 = 0;
    }
}
