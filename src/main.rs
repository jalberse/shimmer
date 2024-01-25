// TODO We can likely get rid of this nightly requirement by using
//  interior mutability instead.
#![feature(get_mut_unchecked)]

use std::fs::{self};

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

    render::render_cpu(scene, &options, &mut string_interner, &mut cached_spectra);
}
