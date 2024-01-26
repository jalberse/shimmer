// TODO We can likely get rid of this nightly requirement by using
//  interior mutability instead.
#![feature(get_mut_unchecked)]

use std::{
    fs::{self},
    time::Instant,
};

use clap::Parser;
use shimmer::{
    bounding_box::Bounds2f,
    loading::{
        parser,
        scene::{BasicScene, BasicSceneBuilder},
    },
    options::Options,
    render::{self},
    vecmath::{Point2f, Tuple2},
    Float,
};
use string_interner::StringInterner;

#[derive(clap::Parser, Debug)]
#[command(author, version, about)]
struct Args {
    scene_file: String,

    /// Specify an image crop window w.r.t. [0,1]^2. <x0 x1 y0 y1>
    #[arg(short, long, num_args = 4, default_values = vec!["0.0", "1.0", "0.0", "1.0"])]
    crop_window: Vec<Float>,
}

fn main() {
    let cli = Args::parse();

    let mut options = Options::default();
    options.crop_window = Some(Bounds2f::new(
        Point2f::new(cli.crop_window[0], cli.crop_window[2]),
        Point2f::new(cli.crop_window[1], cli.crop_window[3]),
    ));

    let mut string_interner = StringInterner::new();
    let mut cached_spectra = std::collections::HashMap::new();
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
