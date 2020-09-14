use amethyst::{
    assets::Processor,
    core::{transform::TransformBundle, SystemBundle},
    ecs::DispatcherBuilder,
    error::Error,
    input::{InputBundle, StringBindings},
    prelude::*,
    renderer::{
        mtl::Material, types::DefaultBackend, visibility::VisibilitySortingSystem,
        MeshProcessorSystem, RenderingSystem, SpriteSheet, TextureProcessorSystem,
    },
    ui::UiBundle,
    window::WindowBundle,
};

mod render_graph;

use render_graph::ExampleGraph;

pub struct MainState {
    real_state: Option<Box<dyn SimpleState>>,
}

impl SimpleState for MainState {
    fn on_start(&mut self, data: StateData<GameData>) {
        if let Some(ref mut state) = self.real_state {
            state.on_start(data);
        }
    }

    fn handle_event(
        &mut self,
        data: StateData<'_, GameData<'_, '_>>,
        event: StateEvent,
    ) -> SimpleTrans {
        if let Some(ref mut state) = self.real_state {
            state.handle_event(data, event)
        } else {
            Trans::None
        }
    }

    fn update(&mut self, data: &mut StateData<GameData>) -> SimpleTrans {
        if let Some(ref mut state) = self.real_state {
            state.update(data)
        } else {
            Trans::None
        }
    }
}

// saves ~2 seconds
pub fn start_game<'a>(
    assets_dir: &'a str,
    game_data_builder: GameDataBuilder<'static, 'static>,
    state: Option<Box<dyn SimpleState>>,
) {
    let mut game = Application::new(
        assets_dir,
        MainState { real_state: state },
        game_data_builder,
    )
    .unwrap();
    game.run();
}

pub struct PrecompiledDefaultsBundle<'a> {
    pub key_bindings_path: &'a str,
    pub display_config_path: &'a str,
}

impl<'a, 'b, 'c> SystemBundle<'a, 'b> for PrecompiledDefaultsBundle<'c> {
    fn build(
        self,
        world: &mut World,
        builder: &mut DispatcherBuilder<'a, 'b>,
    ) -> Result<(), Error> {
        // saves ~ 1 second
        InputBundle::<StringBindings>::new()
            .with_bindings_from_file(self.key_bindings_path)?
            .build(world, builder)?;
        // this set saves ~ 2 seconds
        TransformBundle::new()
            .with_dep(&["fly_movement"])
            .build(world, builder)?;
        UiBundle::<StringBindings>::new().build(world, builder)?;
        WindowBundle::from_config_path(self.display_config_path)?.build(world, builder)?;
        Ok(())
    }
}

pub struct PrecompiledRenderBundle;

// saves ~13 seconds
impl<'a, 'b> SystemBundle<'a, 'b> for PrecompiledRenderBundle {
    fn build(self, _: &mut World, builder: &mut DispatcherBuilder<'a, 'b>) -> Result<(), Error> {
        builder.add(
            Processor::<SpriteSheet>::new(),
            "sprite_sheet_processor",
            &[],
        );
        builder.add(
            VisibilitySortingSystem::new(),
            "visibility_sorting_system",
            &[],
        );
        builder.add(
            MeshProcessorSystem::<DefaultBackend>::default(),
            "mesh_processor",
            &[],
        );
        builder.add(
            TextureProcessorSystem::<DefaultBackend>::default(),
            "texture_processor",
            &[],
        );
        builder.add(Processor::<Material>::new(), "material_processor", &[]);
        builder.add_thread_local(RenderingSystem::<DefaultBackend, _>::new(
            ExampleGraph::default(),
        ));
        Ok(())
    }
}
