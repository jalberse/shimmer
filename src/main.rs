// TODO We can likely get rid of this nightly requirement by using
//  interior mutability instead.
#![feature(get_mut_unchecked)]

use std::{
    fs::{self},
    time::Instant,
};

use clap::Parser;
use shimmer::{
    loading::{
        parser,
        scene::{BasicScene, BasicSceneBuilder},
    },
    options::Options,
    render::{self},
};
use string_interner::StringInterner;

#[derive(clap::Parser, Debug)]
#[command(author, version, about)]
struct Args {
    scene_file: String,
}

fn main() {
    let cli = Args::parse();

    // TODO Output time to render. Get consistent with timer in pbrt.

    let mut string_interner = StringInterner::new();
    let mut cached_spectra = std::collections::HashMap::new();
    let mut options = Options::default();
    let file = fs::read_to_string(cli.scene_file).unwrap();
    let scene = Box::new(BasicScene::default());
    let mut scene_builder = BasicSceneBuilder::new(scene, &mut string_interner);
    parser::parse_str(
        &file,
        &mut scene_builder,
        &mut options,
        &mut string_interner,
        &mut cached_spectra,
    );
    let scene = scene_builder.done();

    let start_time = Instant::now();
    render::render_cpu(scene, &options, &mut string_interner, &mut cached_spectra);
    let elapsed = start_time.elapsed();
    println!(
        "Render time: {}.{:03} seconds",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
}
