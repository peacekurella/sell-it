# vis

## description
Visualization for sell-it.

## roadmap
- [x] Ground.
- [x] XYZ axes.
- [x] FPS cam.
- [x] Current frame display.
- [x] Get filename from cli.
- [x] Pause/Play.
- Deserialization
    - [x] Use serde.
    - [x] Strongly typed class.
    - [x] Deny unknown fields.
    - [x] Frames instead of loose vecs.
    - [x] Constraint on joints 21.
- [x] Skeleton pose.
- [x] Unique color for id, color id map legend.

## code
- The code is written in stable `rust`.
- `bevy` library is used for rendering.

## documentation
- The documentation for the code is itself.

## usage
- Use `rustup` to install the latest stable rust compiler `rustc` and package manager `cargo`.

### compile and run
- To install rust compiler and cargo,
    - First install `rustup`.
    - Then use it to install `stable` and `nightly` channels of rust.
    - Also install `lld`.
    - For extensive guidelines follow `https://bevyengine.org/learn/book/getting-started/setup/`.
- To build and run use `cargo +nightly run --features bevy/dynamic --release <path-to-file>`.

### controls
- `w`, `s`, `a`, `d` for forward, backward, left, right respectively.
- `q`, `e` for vertical down and up respectively.
- `mouse` orient camera.
- `p` pause session.

## demonstration
