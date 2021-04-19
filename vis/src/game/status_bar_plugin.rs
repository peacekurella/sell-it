use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*};

pub struct FrameNumber(pub usize);

pub struct StatusBarPlugin;

impl Plugin for StatusBarPlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(FrameNumber(0))
            .add_plugin(FrameTimeDiagnosticsPlugin)
            .add_startup_system(init_fps_vis.system())
            .add_system(fps_update_system.system());
    }
}

fn init_fps_vis(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn_bundle(TextBundle {
        text: Text {
            sections: vec![TextSection {
                value: "Frame Number: ".to_string(),
                style: TextStyle {
                    font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                    font_size: 30.0,
                    color: Color::rgb(0.5, 0.5, 1.0),
                    ..Default::default()
                },
            }],
            ..Default::default()
        },
        style: Style {
            position_type: PositionType::Absolute,
            position: Rect {
                top: Val::Px(5.0),
                left: Val::Px(5.0),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    });
}

fn fps_update_system(frame_number: Res<FrameNumber>, mut query: Query<&mut Text>) {
    let mut text = query.single_mut().unwrap();
    text.sections[0].value = format!("Frame Number: {}", frame_number.0);
}
