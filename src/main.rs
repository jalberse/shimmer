// TODO We can likely get rid of this nightly requirement by using
//  interior mutability instead.
#![feature(get_mut_unchecked)]

use std::fs::{self};

use shimmer::{
    loading::{
        parser,
        scene::{BasicScene, BasicSceneBuilder},
    },
    options::Options,
    render::{self},
};
use string_interner::StringInterner;

fn main() {
    // TODO Parse from command line.

    let mut string_interner = StringInterner::new();
    let mut cached_spectra = std::collections::HashMap::new();
    let mut options = Options::default();
    let file = fs::read_to_string("scenes/cornell.pbrt").unwrap();
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
