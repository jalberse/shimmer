// TODO We can likely get rid of this nightly requirement by using
//  interior mutability instead.
#![feature(get_mut_unchecked)]

use std::{
    fs::{self},
    sync::Arc,
};

use itertools::Itertools;
use rand::Rng;
use rayon::string;
use shimmer::{
    float::PI_F,
    light::{DiffuseAreaLight, Light, UniformInfiniteLight},
    loading::{
        parser,
        scene::{BasicScene, BasicSceneBuilder},
    },
    material::{DiffuseMaterial, Material},
    options::Options,
    primitive::{GeometricPrimitive, Primitive},
    render::{self},
    shape::{sphere::Sphere, Shape, Triangle, TriangleMesh},
    spectra::{spectrum::spectrum_to_photometric, ConstantSpectrum, Spectrum},
    texture::{SpectrumConstantTexture, SpectrumTexture},
    transform::Transform,
    vecmath::{spherical::spherical_direction, Point3f, Tuple3, Vector3f},
    Float,
};
use string_interner::StringInterner;

fn main() {
    // TODO Parse from command line.
    let mut string_interner = StringInterner::new();
    let mut cached_spectra = std::collections::HashMap::new();
    let mut options = Options::default();
    let file = fs::read_to_string("scenes/test.pbrt").unwrap();
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

// TODO Delete these test scenes; we will define our test scenes in .pbrt files from now on, thankfully.

fn one_sphere_inf_light_scene() -> (Vec<Arc<Primitive>>, Vec<Arc<Light>>) {
    // Create some random lights
    let render_from_object = Transform::translate(Vector3f {
        x: 0.0,
        y: 0.0,
        z: -2.0,
    });
    let radius = 1.0;
    let sphere = Arc::new(Shape::Sphere(Sphere::new(
        render_from_object,
        render_from_object.inverse(),
        false,
        radius,
        -radius,
        radius,
        360.0,
    )));

    let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5)));
    let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
        value: cs,
    }));
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));

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

    let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5)));
    let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
        value: cs,
    }));
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));

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

    let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.7)));
    let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
        value: cs,
    }));
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));

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

    // TODO Hmm, should create_triangles return Arc<Triangle> instead of Triangle? Does PBRT store Tris as a pointer in reality?
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
            let sphere = Arc::new(Shape::Sphere(Sphere::new(
                object_from_render.inverse(),
                object_from_render,
                false,
                light_radius,
                -light_radius,
                light_radius,
                360.0,
            )));

            let le = Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.0)));
            let scale = 1.0 / spectrum_to_photometric(&le);
            let area_light = Arc::new(Light::DiffuseAreaLight(DiffuseAreaLight::new(
                Transform::default(),
                le,
                scale,
                sphere.clone(),
                false,
            )));

            let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5)));
            let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
                value: cs,
            }));
            let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));

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
            let sphere = Arc::new(Shape::Sphere(Sphere::new(
                object_from_render.inverse(),
                object_from_render,
                false,
                radius,
                -radius,
                radius,
                360.0,
            )));

            let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.6)));
            let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
                value: cs,
            }));
            let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));
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
        let sphere = Arc::new(Shape::Sphere(Sphere::new(
            object_from_render.inverse(),
            object_from_render,
            false,
            light_radius,
            -light_radius,
            light_radius,
            360.0,
        )));

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

        let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5)));
        let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
            value: cs,
        }));
        let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));

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
        let sphere = Arc::new(Shape::Sphere(Sphere::new(
            object_from_render.inverse(),
            object_from_render,
            false,
            radius,
            -radius,
            radius,
            360.0,
        )));

        let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.6)));
        let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
            value: cs,
        }));
        let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));
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

    let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.7)));
    let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
        value: cs,
    }));
    let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));

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
            let sphere = Arc::new(Shape::Sphere(Sphere::new(
                object_from_render.inverse(),
                object_from_render,
                false,
                radius,
                -radius,
                radius,
                360.0,
            )));

            let cs = Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.6)));
            let kd = Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture {
                value: cs,
            }));
            let material = Arc::new(Material::Diffuse(DiffuseMaterial::new(kd, None, None)));
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
