use crate::skeleton_video::SkeletonVideo;
use bevy::prelude::*;

pub struct Bone(pub usize, pub usize);
pub struct SkeletonVideosPlugin {
    skeletons: Vec<SkeletonVideo>,
}

impl SkeletonVideosPlugin {
    const COLOR_MAP: [Color; 6] = [
        Color::rgb(1.0, 0.0, 0.0),
        Color::rgb(0.0, 1.0, 0.0),
        Color::rgb(0.0, 0.0, 1.0),
        Color::rgb(1.0, 1.0, 0.0),
        Color::rgb(1.0, 0.0, 1.0),
        Color::rgb(0.0, 1.0, 1.0),
    ];
    pub fn new(skeletons: Vec<SkeletonVideo>) -> SkeletonVideosPlugin {
        SkeletonVideosPlugin { skeletons }
    }
}

impl Plugin for SkeletonVideosPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(self.skeletons.clone())
            .add_startup_system(init_vis.system());
    }
}

fn init_vis(
    mut commands: Commands,
    skeletons: Res<Vec<SkeletonVideo>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (skeleton_idx, skeleton) in skeletons.iter().enumerate() {
        let bones = skeleton.bones(0);
        let color =
            SkeletonVideosPlugin::COLOR_MAP[skeleton_idx % SkeletonVideosPlugin::COLOR_MAP.len()];
        for (bone_idx, (len, trans, rot)) in bones.iter().enumerate() {
            commands
                .spawn_bundle(PbrBundle {
                    mesh: meshes.add(Mesh::from(shape::Box::new(
                        *len,
                        SkeletonVideo::THICKNESS,
                        SkeletonVideo::THICKNESS,
                    ))),
                    material: materials.add(color.into()),
                    transform: Transform {
                        translation: *trans,
                        rotation: *rot,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .insert(Bone(skeleton_idx, bone_idx));
        }
    }
}
