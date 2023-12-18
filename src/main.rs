use std::{process::id, rc::Rc, sync::Arc};

use itertools::Itertools;
use rand::Rng;
use shimmer::{
    aggregate::BvhAggregate,
    bounding_box::{Bounds2f, Bounds2i},
    camera::{Camera, CameraTransform, OrthographicCamera},
    colorspace::RgbColorSpace,
    film::{Film, PixelSensor, RgbFilm},
    filter::{BoxFilter, Filter},
    float::PI_F,
    integrator::{ImageTileIntegrator, Integrator},
    light::{DiffuseAreaLight, Light},
    material::{DiffuseMaterial, Material},
    options::Options,
    primitive::{GeometricPrimitive, Primitive},
    sampler::{IndependentSampler, Sampler},
    shape::{sphere::Sphere, Shape, Triangle, TriangleMesh},
    spectra::{spectrum::spectrum_to_photometric, ConstantSpectrum, Spectrum},
    texture::{SpectrumConstantTexture, SpectrumTexture},
    transform::Transform,
    vecmath::{
        spherical::spherical_direction, Point2f, Point2i, Point3f, Tuple2, Tuple3, Vector2f,
        Vector3f,
    },
    Float,
};

fn main() {
    let (prims, lights) = get_triangular_mesh_test_scene();

    let bvh = Primitive::BvhAggregate(BvhAggregate::new(
        prims,
        1,
        shimmer::aggregate::SplitMethod::Middle,
    ));

    let sampler = Sampler::Independent(IndependentSampler::new(0, 100));
    let full_resolution = Point2i::new(500, 500);
    let filter = Filter::BoxFilter(BoxFilter::new(Vector2f::new(0.5, 0.5)));
    let film = RgbFilm::new(
        full_resolution,
        Bounds2i::new(Point2i::new(0, 0), full_resolution),
        filter,
        1.0,
        PixelSensor::default(),
        "test_tiled_rayon.pfm",
        RgbColorSpace::get_named(shimmer::colorspace::NamedColorSpace::SRGB).clone(),
        Float::INFINITY,
        false,
    );
    let options = Options::default();
    // TODO Image is inverted it seems? Based on locations of some objects. Is it an issue when writing the image or in transform stuff?
    let camera_transform = Transform::look_at(
        &Point3f::new(0.0, 0.0, -5.0),
        &Point3f::new(0.0, 0.0, 0.0),
        &Vector3f::Y,
    );
    let camera = Camera::Orthographic(OrthographicCamera::new(
        CameraTransform::new(&camera_transform, &options),
        0.0,
        1.0,
        Film::RgbFilm(film),
        None,
        0.0,
        5.0,
        Bounds2f::new(Point2f::new(-1.0, -1.0), Point2f::new(1.0, 1.0)),
    ));

    let mut integrator = ImageTileIntegrator::new(bvh, lights, camera, sampler, 8);

    // Note this is just going to stdout right now.
    integrator.render(&options);
}

// Two triangular meshes representing rough spheres side-by-side.
// The vertices also have slight offsets from the radius.
fn get_triangular_mesh_test_scene() -> (Vec<Arc<Primitive>>, Vec<Light>) {
    // TODO This triangulated mesh has a bug where the endcap is missing
    // But it's really not that important, I don't think it's just a bug with this
    // mesh generation code.

    let mut rng = rand::thread_rng();
    // Make a triangular mesh for a triangulated sphere (with vertices randomly
    // offset along their normal), centered at the origin.
    let n_theta = 16;
    let n_phi = 16;
    let n_vertices = n_theta * n_phi;
    let mut vertices = Vec::new();
    for t in 0..n_theta {
        let theta = PI_F * t as Float / (n_theta - 1) as Float;
        let cos_theta = Float::cos(theta);
        let sin_theta = Float::sin(theta);
        for p in 0..n_phi {
            let phi = 2.0 * PI_F * p as Float / (n_phi - 1) as Float;
            let radius = 0.75;
            // Make sure all the top and bottom vertices are coincident
            if t == 0 {
                vertices.push(Point3f::new(0.0, 0.0, radius));
            } else if t == n_theta - 1 {
                vertices.push(Point3f::new(0.0, 0.0, -radius));
            } else if p == n_phi - 1 {
                // Close it up exactly at the end
                vertices.push(vertices[vertices.len() - (n_phi - 1)]);
            } else {
                let radius = radius + rng.gen_range(0.0..0.1);
                vertices
                    .push(Point3f::ZERO + radius * spherical_direction(sin_theta, cos_theta, phi));
            }
        }
    }
    assert_eq!(n_vertices, vertices.len());

    let mut indices = Vec::new();
    // fan at top
    let get_offset = |t: usize, p: usize| -> usize { t * n_phi + p };
    for p in 0..(n_phi - 1) {
        indices.push(get_offset(0, 0));
        indices.push(get_offset(1, p));
        indices.push(get_offset(1, p + 1));
    }

    // "Quads" (bisected) in the middle rows
    for t in 1..(n_theta - 2) {
        for p in 0..(n_phi - 1) {
            indices.push(get_offset(t, p));
            indices.push(get_offset(t + 1, p));
            indices.push(get_offset(t + 1, p + 1));

            indices.push(get_offset(t, p));
            indices.push(get_offset(t + 1, p + 1));
            indices.push(get_offset(t, p + 1));
        }
    }

    // Fan at the bottom
    for p in 0..(n_phi - 1) {
        indices.push(get_offset(n_theta - 1, 0));
        indices.push(get_offset(n_theta - 2, p));
        indices.push(get_offset(n_theta - 2, p + 1));
    }

    let diffuse_location = Transform::translate(Vector3f {
        x: -0.75,
        y: 0.0,
        z: 0.0,
    });
    let mesh = Arc::new(TriangleMesh::new(
        &diffuse_location,
        false,
        indices.clone(),
        vertices.clone(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ));

    let cs = Spectrum::Constant(ConstantSpectrum::new(0.7));
    let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

    let tris = Triangle::create_triangles(mesh);
    let mut prims = tris
        .into_iter()
        .map(|t| {
            Arc::new(Primitive::Geometric(GeometricPrimitive::new(
                t,
                material.clone(),
                None,
            )))
        })
        .collect_vec();

    // Test the same shape but as an emitter at a different location.
    let light_location = Transform::translate(Vector3f {
        x: 0.75,
        y: 0.0,
        z: 0.0,
    }) * Transform::rotate_x(1.0);
    let light_mesh = Arc::new(TriangleMesh::new(
        &light_location,
        false,
        indices,
        vertices,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ));
    let light_tris_shapes = Triangle::create_triangles(light_mesh);

    let le = Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)));
    let scale = 1.0 / spectrum_to_photometric(&le);
    let mut light_prims = light_tris_shapes
        .into_iter()
        .map(|t| {
            // TODO Is this the right transform for thise?
            let area_light = Some(Arc::new(Light::DiffuseAreaLight(DiffuseAreaLight::new(
                light_location.inverse(),
                le.clone(),
                scale,
                t.clone(),
                false,
            ))));
            Arc::new(Primitive::Geometric(GeometricPrimitive::new(
                t,
                material.clone(),
                area_light,
            )))
        })
        .collect_vec();
    prims.append(&mut light_prims);

    // Our lights array can actually be empty; the area light contributions will come from the area_light on
    // the primitives. We could add infinite light sources to this (or point lights, but those won't be hit
    // by a random walk).
    let lights = Vec::new();
    (prims, lights)
}

fn get_random_sphere_scene() -> (Vec<Arc<Primitive>>, Vec<Light>) {
    let mut rng = rand::thread_rng();
    // Create some random lights
    let (mut light_prims, lights) = {
        let mut light_prims = Vec::new();
        // Our lights array can actually be empty; the area light contributions will come from the area_light on
        // the primitives. We could add infinite light sources to this (or point lights, but those won't be hit
        // by a random walk).
        let lights = Vec::new();
        for _ in 0..10 {
            let object_from_render = Transform::translate(Vector3f {
                x: rng.gen_range(-0.8..0.8),
                y: rng.gen_range(-0.8..0.8),
                z: rng.gen_range(-0.8..0.8),
            });
            let light_radius = rng.gen_range(0.2..0.35);
            let sphere = Shape::Sphere(Sphere::new(
                object_from_render.inverse(),
                object_from_render,
                false,
                light_radius,
                -light_radius,
                light_radius,
                360.0,
            ));

            let le = Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)));
            let scale = 1.0 / spectrum_to_photometric(&le);
            let area_light = Light::DiffuseAreaLight(DiffuseAreaLight::new(
                Transform::default(),
                le,
                scale,
                sphere.clone(),
                false,
            ));

            let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
            let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
            let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

            let sphere_light_primitive =
                GeometricPrimitive::new(sphere, material, Some(Arc::new(area_light)));
            light_prims.push(Arc::new(Primitive::Geometric(sphere_light_primitive)));
        }
        (light_prims, lights)
    };

    // A diffuse sphere to be lit by the light.
    let mut sphere_prims = {
        let mut sphere_prims = Vec::new();
        for _ in 0..10 {
            let object_from_render = Transform::translate(Vector3f {
                x: rng.gen_range(-0.8..0.8),
                y: rng.gen_range(-0.8..0.8),
                z: rng.gen_range(-0.8..0.8),
            });
            let radius = rng.gen_range(0.1..0.3);
            let sphere = Shape::Sphere(Sphere::new(
                object_from_render.inverse(),
                object_from_render,
                false,
                radius,
                -radius,
                radius,
                360.0,
            ));

            let cs = Spectrum::Constant(ConstantSpectrum::new(0.6));
            let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
            let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));
            let sphere_primitive =
                Primitive::Geometric(GeometricPrimitive::new(sphere, material, None));
            sphere_prims.push(Arc::new(sphere_primitive));
        }
        sphere_prims
    };

    let mut prims = Vec::new();
    prims.append(&mut light_prims);
    prims.append(&mut sphere_prims);
    (prims, lights)
}
