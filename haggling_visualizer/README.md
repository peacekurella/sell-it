# haggling\_visualizer

## description
- An interactive visualizer for haggling session data at <https://github.com/CMU-Perceptual-Computing-Lab/ssp>.

## roadmap
- [x] Pickle to JSON.
- Deserialization
    - [x] Use serde.
    - [x] Strongly typed class.
    - [x] Deny unknown fields.
    - [x] Frames instead of loose vecs.
    - [ ] Constraint on joints 19.
- [x] Ground.
- [x] XYZ axes.
- [x] FPS cam.
- [x] FPS display.
- [x] Get filename from cli.
- [x] Pause/Play and reset session.
- [x] Skeleton pose using debug lines and boxes.
- [x] Remove unnecessary HagglingSessionRenderSystem.
- [x] unique color for id, color id map legend.
- [x] current frame / total frames display.

## code
- The code is written in stable `rust`.
- `amethyst` library is used for rendering.
- `config` and `assets` contain amethyst config and assets.
- `pkl_to_json` contains a python script to convert data format.

## documentation
- The documentation for the code is itself.

## usage
- Use `rustup` to install the latest stable rust compiler `rustc` and package manager `cargo`.

### compile and run
- The original data is in pickle format, the visualizer requires it in json format.
- Follow instructions in `pkl_to_json/README.md` for the conversion.
- After that, use `cargo run <rel-path-to-json-file>` to visualize the haggling session.

### controls
- `w`, `s`, `a`, `d` for forward, backward, left, right respectively.
- `q`, `e` for vertical down and up respectively.
- `mouse` orient camera.
- `left mouse click` to toggle mouse control.
- `p` pause session.
- `r` restart session.

## demonstration
The following gif demonstrates the usage.

