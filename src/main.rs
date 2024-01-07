#![feature(get_mut_unchecked)]

use std::sync::Arc;

use itertools::Itertools;
use rand::Rng;
use shimmer::{
    aggregate::BvhAggregate,
    bounding_box::{Bounds2f, Bounds2i},
    camera::{Camera, CameraTransform, PerspectiveCamera},
    colorspace::RgbColorSpace,
    film::{Film, PixelSensor, RgbFilm},
    filter::{BoxFilter, Filter},
    float::PI_F,
    integrator::{
        ImageTileIntegrator, Integrator, PixelSampleEvaluator, RandomWalkIntegrator,
        SimplePathIntegrator,
    },
    light::{DiffuseAreaLight, Light, UniformInfiniteLight},
    light_sampler::UniformLightSampler,
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
    let (prims, lights) = get_tri_mesh_inf_light_scene();

    let bvh = Primitive::BvhAggregate(BvhAggregate::new(
        prims,
        1,
        shimmer::aggregate::SplitMethod::Middle,
    ));

    let sampler = Sampler::Independent(IndependentSampler::new(0, 1000));
    let full_resolution = Point2i::new(500, 500);
    let filter = Filter::BoxFilter(BoxFilter::new(Vector2f::new(0.5, 0.5)));
    let film = RgbFilm::new(
        full_resolution,
        Bounds2i::new(Point2i::new(0, 0), full_resolution),
        filter,
        1.0,
        PixelSensor::default(),
        "the_dark_side_of_the_moon_tris_random_walk_2000spp.pfm",
        RgbColorSpace::get_named(shimmer::colorspace::NamedColorSpace::SRGB).clone(),
        Float::INFINITY,
        false,
    );
    let options = Options::default();
    let camera_from_world = Transform::look_at(
        &Point3f::new(0.0, 0.0, 0.0),
        &Point3f::new(0.0, 0.0, -1.0),
        &Vector3f::Y,
    );

    let camera = Camera::Perspective(PerspectiveCamera::new(
        CameraTransform::new(&camera_from_world.inverse(), &options),
        0.0,
        1.0,
        Film::RgbFilm(film),
        None,
        90.0,
        Bounds2f::new(Point2f::new(-1.0, -1.0), Point2f::new(1.0, 1.0)),
        0.0,
        1e6,
    ));

    let light_sampler = UniformLightSampler {
        lights: lights.clone(),
    };

    let simple_path_pixel_evaluator = PixelSampleEvaluator::SimplePath(SimplePathIntegrator {
        max_depth: 8,
        sample_lights: true,
        sample_bsdf: true,
        light_sampler,
    });

    let random_walk_pixel_evaluator =
        PixelSampleEvaluator::RandomWalk(RandomWalkIntegrator { max_depth: 8 });

    let mut integrator =
        ImageTileIntegrator::new(bvh, lights, camera, sampler, random_walk_pixel_evaluator);

    // TODO - Maybe we can make a scene with one diffuse tri, and rotate it around. We can isolate axes rotations and see what's up.
    // TODO I'm a  bit confused how dpdus is getting used in the BSDF shading frame if it can be 0.
    integrator.render(&options);
}

fn one_sphere_inf_light_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
    // Create some random lights
    let render_from_object = Transform::translate(Vector3f {
        x: 0.0,
        y: 0.0,
        z: -2.0,
    });
    let radius = 1.0;
    let sphere = Shape::Sphere(Sphere::new(
        render_from_object,
        render_from_object.inverse(),
        false,
        radius,
        -radius,
        radius,
        360.0,
    ));

    let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
    let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

    let diffuse_sphere_primitive = GeometricPrimitive::new(sphere, material, None);
    let mut prims = Vec::new();
    prims.push(Arc::new(Primitive::Geometric(diffuse_sphere_primitive)));

    let inf_light = Arc::new(Light::UniformInfinite(UniformInfiniteLight::new(
        Transform::default(),
        Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.005))),
        1.0,
    )));
    let lights = vec![inf_light];

    (prims, lights)
}

// Triangulated sphere lit by a uniform infinite light.
fn get_tri_mesh_inf_light_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
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
            let radius = 1.0;
            // Make sure all the top and bottom vertices are coincident
            if t == 0 {
                vertices.push(Point3f::new(0.0, 0.0, radius));
            } else if t == n_theta - 1 {
                vertices.push(Point3f::new(0.0, 0.0, -radius));
            } else if p == n_phi - 1 {
                // Close it up exactly at the end
                vertices.push(vertices[vertices.len() - (n_phi - 1)]);
            } else {
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

    let diffuse_render_from_object = Transform::translate(Vector3f {
        x: 0.0,
        y: 0.0,
        z: -2.0,
    });
    let mesh = Arc::new(TriangleMesh::new(
        &diffuse_render_from_object,
        false,
        indices.clone(),
        vertices.clone(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ));

    let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
    let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

    let tris = Triangle::create_triangles(mesh);
    let prims = tris
        .into_iter()
        .map(|t| {
            Arc::new(Primitive::Geometric(GeometricPrimitive::new(
                t,
                material.clone(),
                None,
            )))
        })
        .collect_vec();

    let inf_light = Arc::new(Light::UniformInfinite(UniformInfiniteLight::new(
        Transform::default(),
        Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.005))),
        1.0,
    )));
    let lights = vec![inf_light];

    // We could also add infinite or point lights that don't have associated primitives to the lights vector
    (prims, lights)
}

// Two triangular meshes representing rough spheres side-by-side.
// The vertices also have slight offsets from the radius.
fn get_triangular_mesh_test_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
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

    let diffuse_render_from_object = Transform::translate(Vector3f {
        x: -0.75,
        y: 0.0,
        z: 2.0,
    });
    let mesh = Arc::new(TriangleMesh::new(
        &diffuse_render_from_object,
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
    let light_render_from_object = Transform::translate(Vector3f {
        x: 0.75,
        y: 0.0,
        z: 2.0,
    }) * Transform::rotate_x(1.0);
    let light_mesh = Arc::new(TriangleMesh::new(
        &light_render_from_object,
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

    // If we add the DiffuseAreaLights to the lights vector only, then the other objects are illuminated,
    // but we wouldn't see the light itself. If we add them to the primitives vector, then we see the light itself.
    let mut light_prims = light_tris_shapes
        .into_iter()
        .map(|t| -> Arc<Primitive> {
            let area_light = Some(Arc::new(Light::DiffuseAreaLight(DiffuseAreaLight::new(
                light_render_from_object.inverse(),
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

    // Get pointers to the lights data from the primitives, so we can add them to the lights vector.
    let lights = light_prims
        .iter()
        .map(|p| -> Arc<Light> {
            match p.as_ref() {
                Primitive::Geometric(p) => p.area_light.clone().expect("Expected area light"),
                _ => panic!("Expected GeometricPrimitive"),
            }
        })
        .collect_vec();

    prims.append(&mut light_prims);

    // We could also add infinite or point lights that don't have associated primitives to the lights vector
    (prims, lights)
}

fn get_random_sphere_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
    let mut rng = rand::thread_rng();
    // Create some random lights
    let (mut light_prims, lights) = {
        let mut light_prims = Vec::new();
        let mut lights = Vec::new();
        for _ in 0..10 {
            let object_from_render = Transform::translate(Vector3f {
                x: rng.gen_range(-0.8..0.8),
                y: rng.gen_range(-0.8..0.8),
                z: 2.0 + rng.gen_range(-0.8..0.8),
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
            let area_light = Arc::new(Light::DiffuseAreaLight(DiffuseAreaLight::new(
                Transform::default(),
                le,
                scale,
                sphere.clone(),
                false,
            )));

            let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
            let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
            let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

            let sphere_light_primitive =
                GeometricPrimitive::new(sphere, material, Some(area_light.clone()));
            light_prims.push(Arc::new(Primitive::Geometric(sphere_light_primitive)));
            lights.push(area_light)
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
                z: 2.0 + rng.gen_range(-0.8..0.8),
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

fn two_spheres_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
    // Create some random lights
    let (mut light_prims, lights) = {
        let mut light_prims = Vec::new();
        let mut lights = Vec::new();
        let object_from_render = Transform::translate(Vector3f {
            x: 0.4,
            y: 0.0,
            z: 2.0,
        });
        let light_radius = 0.33;
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
        let area_light = Arc::new(Light::DiffuseAreaLight(DiffuseAreaLight::new(
            Transform::translate(Vector3f {
                x: 0.0,
                y: 0.0,
                z: 2.0,
            }),
            le,
            scale,
            sphere.clone(),
            false,
        )));

        let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
        let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
        let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

        let sphere_light_primitive =
            GeometricPrimitive::new(sphere, material, Some(area_light.clone()));
        light_prims.push(Arc::new(Primitive::Geometric(sphere_light_primitive)));
        lights.push(area_light);
        (light_prims, lights)
    };

    // A diffuse sphere to be lit by the light.
    let mut sphere_prims = {
        let mut sphere_prims = Vec::new();
        let object_from_render = Transform::translate(Vector3f {
            x: -0.4,
            y: 0.0,
            z: 2.0,
        });
        let radius = 0.33;
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
        sphere_prims
    };

    let mut prims = Vec::new();
    prims.append(&mut light_prims);
    prims.append(&mut sphere_prims);
    (prims, lights)
}

// Two triangles side-by-side, angled towards each other + the camera at a 45 degree angle.
fn two_triangles_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
    // Construct the light
    // Clockwise winding order
    let mut light_verts = Vec::new();
    light_verts.push(Point3f::new(0.0, 0.0, 0.0));
    light_verts.push(Point3f::new(0.8, 0.8, -0.4));
    light_verts.push(Point3f::new(0.8, 0.1, -0.4));
    let light_indices = vec![0, 1, 2];

    let mut diff_verts = Vec::new();
    diff_verts.push(Point3f::new(-0.8, 0.8, -0.4));
    diff_verts.push(Point3f::new(0.0, 0.1, 0.0));
    diff_verts.push(Point3f::new(-0.8, 0.0, -0.4));
    let diff_indices = vec![0, 1, 2];

    let light_mesh = Arc::new(TriangleMesh::new(
        &Transform::translate(Vector3f {
            x: 0.0,
            y: 0.0,
            z: 2.0,
        }),
        false,
        light_indices.clone(),
        light_verts.clone(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ));

    let diff_mesh = Arc::new(TriangleMesh::new(
        &Transform::translate(Vector3f {
            x: 0.0,
            y: 0.0,
            z: 2.0,
        }),
        false,
        diff_indices.clone(),
        diff_verts.clone(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    ));

    let cs = Spectrum::Constant(ConstantSpectrum::new(0.7));
    let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

    let light_tris = Triangle::create_triangles(light_mesh);

    let le = Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)));
    let scale = 1.0 / spectrum_to_photometric(&le);

    let mut light_prims = light_tris
        .into_iter()
        .map(|t| -> Arc<Primitive> {
            let area_light = Some(Arc::new(Light::DiffuseAreaLight(DiffuseAreaLight::new(
                Transform::translate(Vector3f {
                    x: 0.0,
                    y: 0.0,
                    z: 2.0,
                }),
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

    // Get pointers to the lights data from the primitives, so we can add them to the lights vector.
    let lights = light_prims
        .iter()
        .map(|p| -> Arc<Light> {
            match p.as_ref() {
                Primitive::Geometric(p) => p.area_light.clone().expect("Expected area light"),
                _ => panic!("Expected GeometricPrimitive"),
            }
        })
        .collect_vec();

    let diff_tris = Triangle::create_triangles(diff_mesh);
    let mut diff_prims = diff_tris
        .into_iter()
        .map(|t| -> Arc<Primitive> {
            Arc::new(Primitive::Geometric(GeometricPrimitive::new(
                t,
                material.clone(),
                None,
            )))
        })
        .collect_vec();

    let mut prims = Vec::new();
    prims.append(&mut diff_prims);
    prims.append(&mut light_prims);

    // We could also add infinite or point lights that don't have associated primitives to the lights vector
    (prims, lights)
}

fn get_random_sphere_inf_light_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
    let mut rng = rand::thread_rng();

    let inf_light = Light::UniformInfinite(UniformInfiniteLight::new(
        Transform::default(),
        Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.1))),
        1.0,
    ));
    let lights = vec![Arc::new(inf_light)];

    // Diffuse spheres to be lit by the light.
    let mut sphere_prims = {
        let mut sphere_prims = Vec::new();
        for _ in 0..10 {
            let object_from_render = Transform::translate(Vector3f {
                x: rng.gen_range(-0.8..0.8),
                y: rng.gen_range(-0.8..0.8),
                z: 1.0 + rng.gen_range(-0.8..0.8),
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
    prims.append(&mut sphere_prims);
    (prims, lights)
}
