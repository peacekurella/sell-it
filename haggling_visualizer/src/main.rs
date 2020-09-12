use amethyst::{
    assets::{PrefabLoader, PrefabLoaderSystemDesc, RonFormat},
    controls::{FlyControlBundle, HideCursor},
    core::math::{Point3, Vector3},
    ecs::{Entity, WorldExt},
    input::{is_key_down, is_mouse_button_down, StringBindings, VirtualKeyCode},
    prelude::*,
    renderer::debug_drawing::{DebugLines, DebugLinesComponent},
    renderer::{
        palette::Srgba,
        rendy::mesh::{Normal, Position, TexCoord},
        types::DefaultBackend,
    },
    ui::UiGlyphsSystemDesc,
    ui::{Anchor, UiButtonBuilder, UiImage, UiText},
    utils::fps_counter::{FpsCounter, FpsCounterBundle},
    utils::scene::BasicScenePrefab,
    winit::MouseButton,
    Error,
};
use amethyst_precompile::{start_game, PrecompiledDefaultsBundle, PrecompiledRenderBundle};

mod haggling_session;

use haggling_session::HagglingSession;

type MyPrefabData = BasicScenePrefab<(Vec<Position>, Vec<Normal>, Vec<TexCoord>)>;

struct HagglingSessionRenderState {
    session: HagglingSession,
    frame: usize,
    is_paused: bool,
}

struct Dashboard {
    fps: Entity,
    progress: Entity,
}

struct OnlyState;

impl SimpleState for OnlyState {
    fn on_start(&mut self, data: StateData<'_, GameData<'_, '_>>) {
        let StateData { world, .. } = data;
        // Debug lines resource
        world.insert(DebugLines::default());
        // Debug lines component are automatically rendered by the debug lines rendering plugin
        let mut debug_lines_component = DebugLinesComponent::with_capacity(100);
        // XYZ axes
        debug_lines_component.add_direction(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.2, 0.0, 0.0),
            Srgba::new(1.0, 0.0, 0.0, 1.0),
        );
        debug_lines_component.add_direction(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
            Srgba::new(0.0, 1.0, 0.0, 1.0),
        );
        debug_lines_component.add_direction(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.2),
            Srgba::new(0.0, 0.0, 1.0, 1.0),
        );
        // Ground grid
        let width: u32 = 3;
        let depth: u32 = 3;
        let main_color = Srgba::new(0.1, 0.1, 0.1, 1.0);
        // Grid lines in X-axis
        for x in 0..=width {
            let (x, width, depth) = (x as f32, width as f32, depth as f32);
            let position = Point3::new(x - width / 2.0, 0.0, -depth / 2.0);
            let direction = Vector3::new(0.0, 0.0, depth);
            debug_lines_component.add_direction(position, direction, main_color);
        }
        // Grid lines in Z-axis
        for z in 0..=depth {
            let (z, width, depth) = (z as f32, width as f32, depth as f32);
            let position = Point3::new(-width / 2.0, 0.0, z - depth / 2.0);
            let direction = Vector3::new(width, 0.0, 0.0);
            debug_lines_component.add_direction(position, direction, main_color);
        }
        world.register::<DebugLinesComponent>();
        world.create_entity().with(debug_lines_component).build();
        // Camera
        let prefab_handle = world.exec(|loader: PrefabLoader<'_, MyPrefabData>| {
            loader.load("prefab/fly_camera.ron", RonFormat, ())
        });
        world.create_entity().with(prefab_handle).build();
        // FPS text
        let (_, fps_btn) = UiButtonBuilder::<(), u32>::new("fps".to_string())
            .with_anchor(Anchor::TopLeft)
            .with_position(32.0 * 2.0, -32.0)
            .with_font_size(32.0)
            .with_image(UiImage::SolidColor([0.2, 0.2, 0.2, 1.0]))
            .with_hover_image(UiImage::SolidColor([0.1, 0.1, 0.1, 0.5]))
            .build_from_world(&world);
        let (_, progress_btn) = UiButtonBuilder::<(), u32>::new("progress".to_string())
            .with_anchor(Anchor::TopRight)
            .with_position(-32.0 * 2.0, -32.0)
            .with_font_size(32.0)
            .with_image(UiImage::SolidColor([0.2, 0.2, 0.2, 1.0]))
            .with_hover_image(UiImage::SolidColor([0.1, 0.1, 0.1, 0.5]))
            .build_from_world(&world);
        world.insert(Dashboard {
            fps: fps_btn.text_entity,
            progress: progress_btn.text_entity,
        });
        // Load haggling session from file
        use std::env;
        use std::fs::File;
        use std::io::BufReader;
        let args = env::args().collect::<Vec<String>>();
        let file = File::open(&args[1]).unwrap();
        let reader = BufReader::new(file);
        let session: HagglingSession = serde_json::from_reader(reader).unwrap();
        world.insert(HagglingSessionRenderState {
            session,
            frame: 0,
            is_paused: false,
        });
    }

    fn handle_event(
        &mut self,
        data: StateData<'_, GameData<'_, '_>>,
        event: StateEvent,
    ) -> SimpleTrans {
        let StateData { world, .. } = data;
        // Check for input events
        if let StateEvent::Window(event) = &event {
            if is_key_down(&event, VirtualKeyCode::Escape) {
                return Trans::Quit;
            }
            if is_key_down(&event, VirtualKeyCode::P) {
                let mut render_state = world.write_resource::<HagglingSessionRenderState>();
                render_state.is_paused = !render_state.is_paused;
            } else if is_key_down(&event, VirtualKeyCode::R) {
                let mut render_state = world.write_resource::<HagglingSessionRenderState>();
                render_state.frame = 0;
            } else if is_mouse_button_down(&event, MouseButton::Left) {
                let mut hide_cursor = world.write_resource::<HideCursor>();
                hide_cursor.hide = !hide_cursor.hide;
            }
        }
        Trans::None
    }

    fn update(&mut self, data: &mut StateData<'_, GameData<'_, '_>>) -> SimpleTrans {
        let StateData { world, .. } = data;
        // Display the session subjects
        let mut render_state = world.write_resource::<HagglingSessionRenderState>();
        let mut debug_lines = world.write_resource::<DebugLines>();
        let endpoint_size = Vector3::new(0.01, 0.01, 0.01);
        let (left_seller_id, right_seller_id) = render_state.session.left_right_ids();
        for subject in render_state.session.subjects() {
            let human_id = subject.human_id();
            let color = if subject.human_id() == left_seller_id {
                Srgba::new(1.0, 0.0, 0.0, 1.0)
            } else if human_id == right_seller_id {
                Srgba::new(0.0, 1.0, 0.0, 1.0)
            } else {
                Srgba::new(0.0, 0.0, 1.0, 1.0)
            };
            for (e1, e2) in subject.pose_at(render_state.frame).unwrap().iter() {
                debug_lines.draw_line(e1 / 100.0, e2 / 100.0, color);
                debug_lines.draw_box(
                    e1 / 100.0 - endpoint_size,
                    e1 / 100.0 + endpoint_size,
                    color,
                );
                debug_lines.draw_box(
                    e2 / 100.0 - endpoint_size,
                    e2 / 100.0 + endpoint_size,
                    color,
                );
            }
        }
        // Update frame
        if !render_state.is_paused
            && render_state.frame < render_state.session.subjects()[0].num_frames() - 1
        {
            render_state.frame += 1;
        }
        // Update FPS in dashboard
        let mut ui_text = world.write_storage::<UiText>();
        let fps_text = ui_text
            .get_mut(world.read_resource::<Dashboard>().fps)
            .unwrap();
        fps_text.text = format!(
            "FPS: {:.0}",
            world.read_resource::<FpsCounter>().sampled_fps()
        );
        // Update progress in dashboard
        let progress_text = ui_text
            .get_mut(world.read_resource::<Dashboard>().progress)
            .unwrap();
        progress_text.text = format!(
            "{}/{}",
            render_state.frame,
            render_state.session.subjects()[0].num_frames() - 1
        );
        Trans::None
    }
}

fn main() -> Result<(), Error> {
    use std::env;
    let args = env::args().collect::<Vec<String>>();
    if args.len() != 2 {
        return Err(Error::from_string(
            "Usage: cargo run <relative-path-to-json>",
        ));
    }

    amethyst::start_logger(Default::default());

    let game_data = GameDataBuilder::default()
        .with_system_desc(PrefabLoaderSystemDesc::<MyPrefabData>::default(), "", &[])
        .with_bundle(
            FlyControlBundle::<StringBindings>::new(
                Some(String::from("move_x")),
                Some(String::from("move_y")),
                Some(String::from("move_z")),
            )
            .with_sensitivity(0.1, 0.1)
            .with_speed(4.0),
        )?
        .with_bundle(FpsCounterBundle::default())?
        .with_system_desc(
            UiGlyphsSystemDesc::<DefaultBackend>::default(),
            "ui_glyph_system",
            &[],
        )
        .with_bundle(PrecompiledDefaultsBundle {
            key_bindings_path: &"config/input.ron",
            display_config_path: &"config/display.ron",
        })?
        .with_bundle(PrecompiledRenderBundle)?;

    start_game(&"assets", game_data, Some(Box::new(OnlyState)));
    Ok(())
}
