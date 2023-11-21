use std::{rc::Rc, sync::Arc};

use shimmer::{
    aggregate::BvhAggregate,
    bounding_box::{Bounds2f, Bounds2i},
    camera::{Camera, CameraTransform, OrthographicCamera},
    colorspace::RgbColorSpace,
    film::{Film, PixelSensor, RgbFilm},
    filter::{BoxFilter, Filter},
    integrator::{IntegratorI, RandomWalkIntegrator},
    light::{DiffuseAreaLight, Light},
    material::{DiffuseMaterial, Material},
    options::Options,
    primitive::{GeometricPrimitive, Primitive},
    sampler::{IndependentSampler, Sampler},
    shape::{sphere::Sphere, Shape},
    spectra::{spectrum::spectrum_to_photometric, ConstantSpectrum, Spectrum},
    texture::{SpectrumConstantTexture, SpectrumTexture},
    transform::Transform,
    vecmath::{Point2f, Point2i, Point3f, Tuple2, Tuple3, Vector2f, Vector3f},
    Float,
};

fn main() {
    // Let's create a simple scene with a single sphere at the origin emitting light, with nothing else.
    let (light_prim, sphere_area_light) = {
        let object_from_render = Transform::translate(Vector3f {
            x: 0.0,
            y: 0.4,
            z: 0.0,
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
        let area_light = Light::DiffuseAreaLight(DiffuseAreaLight::new(
            Transform::default(),
            le,
            scale,
            sphere.clone(),
            false,
        ));

        let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
        let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
        let material = Rc::new(Material::Diffuse(DiffuseMaterial::new(kd)));

        let sphere_light_primitive =
            GeometricPrimitive::new(sphere, material, Some(Rc::new(area_light.clone())));
        (Primitive::Geometric(sphere_light_primitive), area_light)
    };

    // A diffuse sphere to be lit by the light.
    let sphere_prim = {
        let object_from_render = Transform::translate(Vector3f {
            x: 0.0,
            y: -0.4,
            z: -0.5,
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

        let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
        let kd = SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
        let material = Rc::new(Material::Diffuse(DiffuseMaterial::new(kd)));
        let sphere_primitive =
            Primitive::Geometric(GeometricPrimitive::new(sphere, material, None));
        sphere_primitive
    };

    // Assemble primitives, lights for the scene
    let lights = vec![sphere_area_light];
    let prims = vec![Rc::new(light_prim), Rc::new(sphere_prim)];
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
        "test_diffuse_neg_half_z.pfm",
        RgbColorSpace::get_named(shimmer::colorspace::NamedColorSpace::SRGB).clone(),
        Float::INFINITY,
        false,
    );
    let options = Options::default();
    let camera_transform = Transform::look_at(
        &Point3f::new(0.0, 0.0, -5.0),
        &Point3f::new(0.0, 0.0, 0.5),
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

    let mut integrator = RandomWalkIntegrator::new(bvh, lights, camera, sampler, 8);

    // Note this is just going to stdout right now.
    integrator.render(&options);
}
