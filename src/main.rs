// TODO We can likely get rid of this nightly requirement by using
//  interior mutability instead.
#![feature(get_mut_unchecked)]

use std::{
    fs::{self},
    path::PathBuf,
    time::Instant,
};

use clap::Parser;
use shimmer::{
    bounding_box::{Bounds2f, Bounds2i},
    loading::{
        parser,
        scene::{BasicScene, BasicSceneBuilder},
    },
    options::{Options, RenderingCoordinateSystem},
    render::{self},
    vecmath::{Point2f, Point2i, Tuple2},
    Float,
};
use string_interner::StringInterner;

#[derive(clap::Parser, Debug)]
#[command(author, version, about)]
struct Args {
    scene_file: String,

    /// Write the final image to the given filename.
    #[arg(short, long)]
    outfile: Option<String>,

    /// Specify an image crop window w.r.t. [0,1]^2. <x0 x1 y0 y1>
    #[arg(long, num_args = 4)]
    crop_window: Option<Vec<Float>>,

    /// Specify image crop window w.r.t. pixel coordinates. <x0 x1 y0 y1>
    #[arg(long, num_args = 4)]
    pixel_bounds: Option<Vec<i32>>,

    /// Convert all materials to be diffuse
    #[arg(long, default_value = "false")]
    force_diffuse: bool,

    /// Set random number generator seed
    #[arg(long, default_value = "0")]
    seed: i32,

    /// Override samples per pixel in scene description file.
    #[arg(short, long)]
    spp: Option<i32>,

    /// Automaticall refuce a number of quality settings to render more quickly
    #[arg(short, long)]
    quick: bool,

    /// Always sample the same wavelengths of light.
    #[arg(long)]
    disable_wavelength_jitter: bool,

    /// Always sample pixels at their centers.
    #[arg(long)]
    disable_pixel_jitter: bool,

    /// Point-sample all textures.
    #[arg(long)]
    disable_texture_filtering: bool,

    /// Set the rendering coordinate system.
    #[arg(long, default_value = "camera-world")]
    render_coord_system: RenderingCoordinateSystem,

    /// Fullscreen
    #[arg(long, default_value = "false")]
    fullscreen: bool,

    /// Scale target triangle edge length by given value.
    #[arg(long, default_value = "1.0")]
    displacement_edge_scale: Float,

    /// Reference image for MSE calculations.
    #[arg(long)]
    mse_reference_image: Option<String>,

    /// Output MSE statistics to given filename (implies -mseReferenceImage).
    #[arg(long)]
    mse_reference_output: Option<String>,

    /// Record pixel statistics
    #[arg(long, default_value = "false")]
    record_pixel_statistics: bool,

    /// Use wavefront volumetric path integrator.
    #[arg(short, long, default_value = "false")]
    wavefront: bool,

    /// Directory to search for files.
    #[arg(long)]
    search_directory: Option<String>,
}

fn main() {
    let cli = Args::parse();

    let mut options = Options::default();
    if let Some(crop_window) = cli.crop_window {
        options.crop_window = Some(Bounds2f::new(
            Point2f::new(crop_window[0], crop_window[2]),
            Point2f::new(crop_window[1], crop_window[3]),
        ));
    }

    options.force_diffuse = cli.force_diffuse;
    options.seed = cli.seed;
    if let Some(pixel_bounds) = cli.pixel_bounds {
        options.pixel_bounds = Some(Bounds2i::new(
            Point2i::new(pixel_bounds[0], pixel_bounds[2]),
            Point2i::new(pixel_bounds[1], pixel_bounds[3]),
        ));
    }

    options.image_file = cli.outfile;
    if let Some(spp) = cli.spp {
        options.pixel_samples = Some(spp);
    } else {
        options.pixel_samples = None;
    }
    options.quick_render = cli.quick;
    if options.quick_render {
        todo!("Quick render is not yet implemented.")
    }
    options.disable_wavelength_jitter = cli.disable_wavelength_jitter;
    options.disable_pixel_jitter = cli.disable_pixel_jitter;
    options.disable_texture_filtering = cli.disable_texture_filtering;
    options.rendering_coord_system = cli.render_coord_system;
    options.fullscreen = cli.fullscreen;
    if options.fullscreen {
        todo!("Fullscreen is not yet implemented.")
    }
    options.displacement_edge_scale = cli.displacement_edge_scale;
    if cli.displacement_edge_scale != 1.0 {
        todo!("Displacement edge scale is not yet implemented.")
    }
    options.mse_reference_image = cli.mse_reference_image;
    if options.mse_reference_image.is_some() {
        todo!("MSE reference image is not yet implemented.")
    }
    options.mse_reference_output = cli.mse_reference_output;
    if options.mse_reference_output.is_some() {
        todo!("MSE reference output is not yet implemented.")
    }
    options.record_pixel_statistics = cli.record_pixel_statistics;
    if options.record_pixel_statistics {
        todo!("Record pixel statistics is not yet implemented.")
    }
    options.wavefront = cli.wavefront;
    if options.wavefront {
        todo!("Wavefront is not yet implemented.")
    }
    if let Some(search_directory) = cli.search_directory {
        options.search_directory = Some(PathBuf::from(search_directory));
    } else {
        options.search_directory = None;
    }
    if options.search_directory.is_some() {
        todo!("Search directory is not yet implemented.")
    }

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
