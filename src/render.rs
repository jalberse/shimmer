use std::{collections::HashMap, sync::Arc};

use log::info;
use string_interner::StringInterner;

use crate::{loading::scene::BasicScene, options::Options, spectra::Spectrum};

pub fn render_cpu(
    mut scene: Box<BasicScene>,
    options: &Options,
    string_interner: &mut StringInterner,
    cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
) {
    // TODO Create media from scene; use an empty map for now.
    let media = HashMap::new();

    info!("Creating textures...");
    let textures = scene.create_textures(cached_spectra, string_interner);
    info!("Done creating textures.");

    info!("Creating lights...");
    let (lights, shape_index_to_area_lights) =
        scene.create_lights(&textures, &string_interner, options);
    info!("Done creating lights.");

    info!("Creating materials...");
    let (named_materials, materials) = scene.create_materials(&textures, &string_interner, options);
    info!("Done creating materials.");

    let accelerator = scene.create_aggregate(
        &textures,
        &shape_index_to_area_lights,
        &media,
        &named_materials,
        &materials,
        &string_interner,
    );

    let camera = scene.get_camera().unwrap();
    // let film = scene.get_film();
    let sampler = scene.get_sampler().unwrap();

    // TODO Can check various options here, give warnings, etc.

    let mut integrator =
        scene.create_integrator(camera, sampler, accelerator, lights, &string_interner);

    integrator.render(options);

    todo!("render_cpu")
}
