use amethyst::{
    ecs::{ReadExpect, SystemData, World},
    renderer::{
        pass::{DrawDebugLinesDesc, DrawShadedDesc},
        types::DefaultBackend,
        Factory, Format, GraphBuilder, GraphCreator, Kind, RenderGroupDesc, SubpassBuilder,
    },
    ui::DrawUiDesc,
    window::{ScreenDimensions, Window},
};

// This graph structure is used for creating a proper `RenderGraph` for rendering.
// A renderGraph can be thought of as the stages during a render pass. In our case,
// we are only executing one subpass. This graph
// also needs to be rebuilt whenever the window is resized, so the boilerplate code
// for that operation is also here.
#[derive(Default)]
pub struct ExampleGraph {
    dimensions: Option<ScreenDimensions>,
    dirty: bool,
}

#[allow(clippy::map_clone)]
impl GraphCreator<DefaultBackend> for ExampleGraph {
    // This trait method reports to the renderer if the graph must be rebuilt, usually because
    // the window has been resized. This implementation checks the screen size and returns true
    // if it has changed.
    fn rebuild(&mut self, world: &World) -> bool {
        // Rebuild when dimensions change, but wait until at least two frames have the same.
        let new_dimensions = world.try_fetch::<ScreenDimensions>();
        use std::ops::Deref;
        if self.dimensions.as_ref() != new_dimensions.as_deref() {
            self.dirty = true;
            self.dimensions = new_dimensions.map(|d| d.deref().clone());
            return false;
        }
        self.dirty
    }

    // This is the core of a RenderGraph, which is building the actual graph with subpasses and target
    // images.
    fn builder(
        &mut self,
        factory: &mut Factory<DefaultBackend>,
        world: &World,
    ) -> GraphBuilder<DefaultBackend, World> {
        use amethyst::renderer::rendy::{
            graph::present::PresentNode,
            hal::command::{ClearDepthStencil, ClearValue},
        };

        self.dirty = false;

        // Retrieve a reference to the target window, which is created by the WindowBundle
        let window = <ReadExpect<'_, Window>>::fetch(world);
        let dimensions = self.dimensions.as_ref().unwrap();
        let window_kind = Kind::D2(dimensions.width() as u32, dimensions.height() as u32, 1, 1);

        // Create a new drawing surface in our window
        let surface = factory.create_surface(&window);
        let surface_format = factory.get_surface_format(&surface);

        // Begin building our RenderGraph
        let mut graph_builder = GraphBuilder::new();
        let color = graph_builder.create_image(
            window_kind,
            1,
            surface_format,
            // clear screen to black
            Some(ClearValue::Color([0.34, 0.36, 0.52, 1.0].into())),
        );

        let depth = graph_builder.create_image(
            window_kind,
            1,
            Format::D32Sfloat,
            Some(ClearValue::DepthStencil(ClearDepthStencil(0.0, 0))),
        );

        // Create our first `Subpass`, which contains the DrawShaded and DrawUi render groups.
        // We pass the subpass builder a description of our groups for construction
        let pass = graph_builder.add_node(
            SubpassBuilder::new()
                .with_group(DrawShadedDesc::default().builder())
                .with_group(DrawDebugLinesDesc::default().builder())
                .with_group(DrawUiDesc::default().builder()) // Draws UI components
                .with_color(color)
                .with_depth_stencil(depth)
                .into_pass(),
        );

        // Finally, add the pass to the graph
        let _present = graph_builder
            .add_node(PresentNode::builder(factory, surface, color).with_dependency(pass));

        graph_builder
    }
}
