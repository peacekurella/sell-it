use bevy::prelude::*;

pub struct GroundPlugin;

impl Plugin for GroundPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.add_startup_system(init_vis.system());
    }
}

fn init_vis(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let ground_size = 40.0;
    let xyz_small_thickness = 0.3;
    let xyz_large_thickness = 5.0;
    // Lights
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_xyz(0.0, 15.0, 0.0),
        ..Default::default()
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_xyz(15.0, 15.0, 15.0),
        ..Default::default()
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_xyz(-15.0, 15.0, -15.0),
        ..Default::default()
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_xyz(15.0, 15.0, -15.0),
        ..Default::default()
    });
    commands.spawn_bundle(LightBundle {
        transform: Transform::from_xyz(-15.0, 15.0, 15.0),
        ..Default::default()
    });
    // Ground
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Plane { size: ground_size })),
        material: materials.add(Color::rgb(0.5, 0.5, 0.5).into()),
        ..Default::default()
    });
    // XYZ
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Box::new(
            xyz_large_thickness,
            xyz_small_thickness,
            xyz_small_thickness,
        ))),
        material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Box::new(
            xyz_small_thickness,
            xyz_large_thickness,
            xyz_small_thickness,
        ))),
        material: materials.add(Color::rgb(0.0, 1.0, 0.0).into()),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(Mesh::from(shape::Box::new(
            xyz_small_thickness,
            xyz_small_thickness,
            xyz_large_thickness,
        ))),
        material: materials.add(Color::rgb(0.0, 0.0, 1.0).into()),
        ..Default::default()
    });
}
