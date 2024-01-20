use std::collections::HashMap;

use log::info;
use string_interner::StringInterner;

use crate::loading::scene::BasicScene;

pub fn render_cpu(scene: &mut BasicScene) {
    let mut cached_spectra = HashMap::new();
    let mut string_interner = StringInterner::new();

    info!("Creating textures...");
    let textures = scene.create_textures(&mut cached_spectra, &mut string_interner);
    info!("Done creating textures.");

    // TODO Create lights

    // TODO Create materials

    // TODO Create aggregate

    // TODO Get camera
    // TODO Get film
    // TODO Get sampler

    // TODO Create integrator

    // TODO Render

    todo!("render_cpu")
}
