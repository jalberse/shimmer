use shimmer::bvh::Bvh;
use shimmer::camera::Camera;
use shimmer::geometry::moving_sphere::MovingSphere;
use shimmer::geometry::sphere::Sphere;
use shimmer::hittable::HittableList;
use shimmer::materials::{
    dialectric::Dialectric,
    lambertian::Lambertian,
    material::Material,
    metal::Metal,
    utils::{random_color, random_color_range},
};
use shimmer::renderer::Renderer;
use shimmer::textures::checker::Checker;
use shimmer::textures::image_texture::ImageTexture;

use clap::{Parser, ValueEnum};
use glam::{dvec3, DVec3};

use rand::random;
use shimmer::textures::marble::Marble;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[derive(ValueEnum, Clone)]
enum Scene {
    RandomSpheres,
    RandomMovingSpheres,
    TwoSpheres,
    Marble,
    Earth,
}

#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
    #[clap(value_enum)]
    scene: Scene,
    /// Image width; image height is determined by this value and the aspect ratio.
    #[arg(short = 'w', long, default_value = "1080")]
    image_width: usize,
    #[arg(short, long, num_args = 2, default_values = vec!["16.0", "9.0"])]
    /// Aspect ratio (horizontal, vertical).
    aspect_ratio: Vec<f64>,
    /// Number of ray samples per pixel.
    #[arg(short, long, default_value = "500")]
    samples_per_pixel: u32,
    /// Maximum number of bounces for each ray.
    #[arg(short, long, default_value = "50")]
    depth: u32,
    /// Width of each render tile, in pixels.
    #[arg(long, default_value = "8")]
    tile_width: usize,
    /// Height of each render tile, in pixels.
    #[arg(long, default_value = "8")]
    tile_height: usize,
    /// x, y, z
    /// Origin of the camera.
    #[arg(long, num_args = 3, default_values = vec!["13.0", "2.0", "3.0"])]
    cam_look_from: Vec<f64>,
    /// x, y, z
    /// Determines direction of camera.
    #[arg(long, num_args = 3, default_values = vec!["0.0", "0.0", "0.0"])]
    cam_look_at: Vec<f64>,
    /// x, y, z
    /// Determines roll of the camera along the vector from cam_look_from to cam_look_at.
    /// Useful for dutch angle shots.
    /// Typically "world up" (0.0, 1.0, 0.0).
    #[arg(long, num_args = 3, default_values = vec!["0.0", "1.0", "0.0"])]
    cam_view_up: Vec<f64>,
    /// Vertical field of view. This also dictates the horizontal FOV according to the aspect ratio.
    #[arg(long, default_value = "20.0")]
    cam_vertical_fov: f64,
    /// Camera aperture; twice the lens radius.
    #[arg(long, default_value = "0.1")]
    cam_aperture: f64,
    /// Distance to the focal plane from the camera.
    #[arg(long, default_value = "10.0")]
    cam_focus_dist: f64,
    /// Camera shutter open time.
    #[arg(long, default_value = "0.0")]
    cam_start_time: f64,
    /// Camera shutter close time.
    #[arg(long, default_value = "0.0")]
    cam_end_time: f64,
}

fn main() {
    let cli = Cli::parse();

    let aspect_ratio = cli.aspect_ratio;
    let aspect_ratio = aspect_ratio[0] / aspect_ratio[1];
    let look_from = dvec3(
        cli.cam_look_from[0],
        cli.cam_look_from[1],
        cli.cam_look_from[2],
    );
    let look_at = dvec3(cli.cam_look_at[0], cli.cam_look_at[1], cli.cam_look_at[2]);
    let view_up = dvec3(cli.cam_view_up[0], cli.cam_view_up[1], cli.cam_view_up[2]);
    let vfov = cli.cam_vertical_fov;
    let aperture = cli.cam_aperture;
    let focus_dist = cli.cam_focus_dist;
    let cam_start_time = cli.cam_start_time;
    let cam_end_time = cli.cam_end_time;

    let camera = Camera::new(
        look_from,
        look_at,
        view_up,
        vfov,
        aspect_ratio,
        aperture,
        focus_dist,
        cam_start_time,
        cam_end_time,
    );

    let image_width = cli.image_width;
    let renderer = Renderer::from_aspect_ratio(image_width, aspect_ratio);

    let start = Instant::now();

    let world = match cli.scene {
        Scene::RandomSpheres => random_spheres(),
        Scene::RandomMovingSpheres => random_moving_spheres(),
        Scene::TwoSpheres => two_spheres(),
        Scene::Marble => two_marble_spheres(),
        Scene::Earth => earth(),
    };

    let samples_per_pixel = cli.samples_per_pixel;
    let max_depth = cli.depth;
    renderer
        .render(
            &camera,
            &world,
            samples_per_pixel,
            max_depth,
            cli.tile_width,
            cli.tile_height,
        )
        .unwrap();

    let duration = start.elapsed();
    eprintln!("Render time: {:?}", duration);
}

fn random_spheres() -> HittableList {
    let mut world = HittableList::new();

    let material_ground = Arc::new(Lambertian::new(Arc::new(Checker::from_color(
        10.0,
        dvec3(0.2, 0.3, 0.1),
        dvec3(0.9, 0.9, 0.9),
    ))));
    world.add(Arc::new(Sphere::new(
        DVec3::new(0.0, -1000.0, 0.0),
        1000.0,
        material_ground,
    )));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random::<f32>();
            let center = dvec3(
                a as f64 + 0.9 * random::<f64>(),
                0.2,
                b as f64 + 0.9 * random::<f64>(),
            );

            if (center - dvec3(4.0, 0.2, 0.0)).length() > 0.9 {
                let material: Arc<dyn Material> = if choose_mat < 0.8 {
                    let albedo = random_color() * random_color();
                    Arc::new(Lambertian::from_color(albedo))
                } else if choose_mat < 0.95 {
                    let albedo = random_color_range(0.5, 1.0);
                    let fuzz = random::<f64>() * 0.5;
                    Arc::new(Metal::new(albedo, fuzz))
                } else {
                    Arc::new(Dialectric::new(1.5))
                };
                world.add(Arc::new(Sphere::new(center, 0.2, material)));
            }
        }
    }

    let large_sphere_radius = 1.0;
    let glass_material = Arc::new(Dialectric::new(1.5));
    world.add(Arc::new(Sphere::new(
        dvec3(0.0, 1.0, 0.0),
        large_sphere_radius,
        glass_material,
    )));

    let diffuse_material = Arc::new(Lambertian::from_color(dvec3(0.4, 0.2, 0.1)));
    world.add(Arc::new(Sphere::new(
        dvec3(-4.0, 1.0, 0.0),
        large_sphere_radius,
        diffuse_material,
    )));

    let metal_material = Arc::new(Metal::new(dvec3(0.7, 0.6, 0.5), 0.0));
    world.add(Arc::new(Sphere::new(
        dvec3(4.0, 1.0, 0.0),
        large_sphere_radius,
        metal_material,
    )));

    let bvh = Arc::new(Bvh::new(world, 0.0, 1.0));
    let mut world = HittableList::new();
    world.add(bvh);
    world
}

fn random_moving_spheres() -> HittableList {
    let mut world = HittableList::new();

    let material_ground = Arc::new(Lambertian::new(Arc::new(Checker::from_color(
        10.0,
        dvec3(0.2, 0.3, 0.1),
        dvec3(0.9, 0.9, 0.9),
    ))));
    world.add(Arc::new(Sphere::new(
        DVec3::new(0.0, -1000.0, 0.0),
        1000.0,
        material_ground,
    )));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random::<f32>();
            let center = dvec3(
                a as f64 + 0.9 * random::<f64>(),
                0.2,
                b as f64 + 0.9 * random::<f64>(),
            );

            if (center - dvec3(4.0, 0.2, 0.0)).length() > 0.9 {
                let material: Arc<dyn Material> = if choose_mat < 0.8 {
                    let albedo = random_color() * random_color();
                    Arc::new(Lambertian::from_color(albedo))
                } else if choose_mat < 0.95 {
                    let albedo = random_color_range(0.5, 1.0);
                    let fuzz = random::<f64>() * 0.5;
                    Arc::new(Metal::new(albedo, fuzz))
                } else {
                    Arc::new(Dialectric::new(1.5))
                };
                let center_end = center + dvec3(0.0, random::<f64>() * 0.5, 0.0);
                world.add(Arc::new(MovingSphere::new(
                    center, center_end, 0.0, 1.0, 0.2, material,
                )));
            }
        }
    }

    let large_sphere_radius = 1.0;
    let glass_material = Arc::new(Dialectric::new(1.5));
    world.add(Arc::new(Sphere::new(
        dvec3(0.0, 1.0, 0.0),
        large_sphere_radius,
        glass_material,
    )));

    let diffuse_material = Arc::new(Lambertian::from_color(dvec3(0.4, 0.2, 0.1)));
    world.add(Arc::new(Sphere::new(
        dvec3(-4.0, 1.0, 0.0),
        large_sphere_radius,
        diffuse_material,
    )));

    let metal_material = Arc::new(Metal::new(dvec3(0.7, 0.6, 0.5), 0.0));
    world.add(Arc::new(Sphere::new(
        dvec3(4.0, 1.0, 0.0),
        large_sphere_radius,
        metal_material,
    )));

    let bvh = Arc::new(Bvh::new(world, 0.0, 1.0));
    let mut world = HittableList::new();
    world.add(bvh);
    world
}

fn two_spheres() -> HittableList {
    let mut world = HittableList::new();
    let checkerboard = Arc::new(Lambertian::new(Arc::new(Checker::from_color(
        10.0,
        dvec3(0.2, 0.3, 0.1),
        dvec3(0.9, 0.9, 0.9),
    ))));

    world.add(Arc::new(Sphere::new(
        dvec3(0.0, -10.0, 0.0),
        10.0,
        checkerboard.clone(),
    )));
    world.add(Arc::new(Sphere::new(
        dvec3(0.0, 10.0, 0.0),
        10.0,
        checkerboard.clone(),
    )));

    world
}

fn two_marble_spheres() -> HittableList {
    let mut world = HittableList::new();

    let marble_texture = Arc::new(Marble::new(4.0));
    world.add(Arc::new(Sphere::new(
        dvec3(0.0, -1000.0, 0.0),
        1000.0,
        Arc::new(Lambertian::new(marble_texture.clone())),
    )));
    world.add(Arc::new(Sphere::new(
        dvec3(0.0, 2.0, 0.0),
        2.0,
        Arc::new(Lambertian::new(marble_texture)),
    )));
    world
}

// The relative filepath of the image texture means this works if running from the top level of the git repository,
// but not from other working directories (such as if the built app is run elsewhere).
// This is sufficient for now as this executable is just to demo the library for developers.
// Ideally, the image file (and other file resources) would be specified by a scene defined in some file (in JSON, maybe)
// and we wouldn't be defining sample scenes via code like this at all (we would provide sample scenes as separate files
// and would just use Shimmer to parse and render the provided scene).
fn earth() -> HittableList {
    let earth_texture = Arc::new(ImageTexture::new(Path::new("images/earthmap.jpg")));
    let earth_surface = Arc::new(Lambertian::new(earth_texture));
    let globe = Arc::new(Sphere::new(dvec3(0.0, 0.0, 0.0), 2.0, earth_surface));
    let mut world = HittableList::new();
    world.add(globe);
    world
}
