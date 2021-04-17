use bevy::prelude::*;

pub struct Pause(pub bool);

pub struct PausePlugin;

impl Plugin for PausePlugin {
    fn build(&self, app: &mut AppBuilder) {
        app.insert_resource(Pause(false))
            .add_system(toggle_pause.system());
    }
}

fn toggle_pause(keyboard_input: Res<Input<KeyCode>>, mut pause: ResMut<Pause>) {
    if keyboard_input.just_pressed(KeyCode::P) {
        pause.0 = !pause.0;
    }
}
