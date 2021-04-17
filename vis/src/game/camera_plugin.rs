use bevy::prelude::*;
use bevy_fly_camera::{FlyCamera, FlyCameraPlugin};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_plugin(FlyCameraPlugin)
            .add_startup_system(init_camera.system());
    }
}

pub fn init_camera(mut commands: Commands) {
    commands.spawn_bundle(UiCameraBundle::default());
    commands
        .spawn_bundle(PerspectiveCameraBundle {
            transform: Transform::from_translation(Vec3::new(0.0, 20.0, 15.0)),
            ..Default::default()
        })
        .insert(FlyCamera {
            key_up: KeyCode::E,
            key_down: KeyCode::Q,
            key_left: KeyCode::A,
            key_right: KeyCode::D,
            key_forward: KeyCode::W,
            key_backward: KeyCode::S,
            sensitivity: 3.0,
            ..Default::default()
        });
}
