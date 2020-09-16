use amethyst::{
    core::{frame_limiter::FrameRateLimitStrategy, transform::TransformBundle},
    ecs::prelude::{Entity, WorldExt},
    input::{InputBundle, StringBindings},
    prelude::*,
    renderer::{plugins::RenderToWindow, types::DefaultBackend, RenderingBundle},
    ui::{Anchor, RenderUi, UiBundle, UiButtonBuilder, UiImage, UiText},
    utils::{
        application_root_dir,
        fps_counter::{FpsCounter, FpsCounterBundle},
    },
};

#[derive(Default)]
struct Example {
    btn_text: Option<Entity>,
}

impl SimpleState for Example {
    fn on_start(&mut self, data: StateData<'_, GameData<'_, '_>>) {
        let StateData { world, .. } = data;

        let (_, label) = UiButtonBuilder::<(), u32>::new("Made with UiButtonBuilder".to_string())
            .with_anchor(Anchor::TopLeft)
            .with_position(32.0 * 2.0, -32.0)
            .with_font_size(32.0)
            .with_image(UiImage::SolidColor([0.8, 0.6, 0.3, 1.0]))
            .with_hover_image(UiImage::SolidColor([0.1, 0.1, 0.1, 0.5]))
            .build_from_world(&world);
        self.btn_text = Some(label.text_entity);
    }

    fn update(&mut self, state_data: &mut StateData<'_, GameData<'_, '_>>) -> SimpleTrans {
        let StateData { world, .. } = state_data;

        let mut ui_text = world.write_storage::<UiText>();
        if let Some(btn_text) = self.btn_text.and_then(|entity| ui_text.get_mut(entity)) {
            let fps = world.read_resource::<FpsCounter>().sampled_fps();
            btn_text.text = format!("FPS: {:.*}", 2, fps);
        }
        Trans::None
    }
}

fn main() -> amethyst::Result<()> {
    amethyst::start_logger(Default::default());

    let app_root = application_root_dir()?;

    let display_config_path = app_root.join("examples/ui/config/display.ron");
    let assets_dir = app_root.join("examples/ui/assets");

    let game_data = GameDataBuilder::default()
        .with_bundle(TransformBundle::new())?
        .with_bundle(InputBundle::<StringBindings>::new())?
        .with_bundle(UiBundle::<StringBindings>::new())?
        .with_bundle(FpsCounterBundle::default())?
        .with_bundle(
            RenderingBundle::<DefaultBackend>::new()
                .with_plugin(
                    RenderToWindow::from_config_path(display_config_path)?
                        .with_clear([0.34, 0.36, 0.52, 1.0]),
                )
                .with_plugin(RenderUi::default()),
        )?;

    let mut game = Application::build(assets_dir, Example::default())?
        // Unlimited FPS
        .with_frame_limit(FrameRateLimitStrategy::Unlimited, 9999)
        .build(game_data)?;
    game.run();
    Ok(())
}
