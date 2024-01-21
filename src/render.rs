use std::collections::HashMap;

use log::info;
use string_interner::StringInterner;

use crate::{loading::scene::BasicScene, options::Options};

pub fn render_cpu(scene: &mut BasicScene, options: &Options) {
    let mut cached_spectra = HashMap::new();
    let mut string_interner = StringInterner::new();

    info!("Creating textures...");
    let textures = scene.create_textures(&mut cached_spectra, &mut string_interner);
    info!("Done creating textures.");

    info!("Creating lights...");
    let (lights, shape_index_to_area_lights) =
        scene.create_lights(&textures, &string_interner, options);
    info!("Done creating lights.");

    // TODO Create materials
    info!("Creating materials...");
    let (named_materials, materials) = scene.create_materials(&textures, &string_interner, options);
    info!("Done creating materials.");

    // TODO Create aggregate

    // TODO Get camera
    // TODO Get film
    // TODO Get sampler

    // TODO Create integrator

    // TODO Render

    todo!("render_cpu")
}
