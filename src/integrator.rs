use std::rc::Rc;

use bumpalo::Bump;
use itertools::Itertools;

use crate::{
    camera::{Camera, CameraI},
    film::{FilmI, VisibleSurface},
    image::Image,
    image_metadata::ImageMetadata,
    light::{Light, LightI, LightType},
    light_sampler::UniformLightSampler,
    options::Options,
    primitive::{Primitive, PrimitiveI},
    ray::{Ray, RayDifferential},
    sampler::{Sampler, SamplerI},
    shape::ShapeIntersection,
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    vecmath::{Point2i, Tuple2, Vector3f},
    Float,
};

// TODO In places where PBRT uses a ScatchBuffer, I think that an Arena Allocator is a good Rust alternative.

pub trait IntegratorI {
    fn render(&mut self, options: &Options);
}

/// Integrators can have an IntegratorBase that provides shared data and functionality.
pub struct IntegratorBase {
    aggregate: Primitive,
    /// Contains both finite and infinite lights
    lights: Vec<Rc<Light>>,
    /// Stores an additional copy of any infinite lights in their own vector.
    infinite_lights: Vec<Rc<Light>>,
}

impl IntegratorBase {
    /// Creates an IntegratorBase from the given aggregate and lights;
    /// Also does any necessary preprocessing for the lights.
    pub fn new(aggregate: Primitive, mut lights: Vec<Light>) -> IntegratorBase {
        let scene_bounds = aggregate.bounds();
        let mut infinite_lights = Vec::new();
        for light in &mut lights {
            light.preprocess(&scene_bounds);
        }
        let lights = lights.into_iter().map(|l| Rc::new(l)).collect_vec();
        for light in &lights {
            if light.light_type() == LightType::Infinite {
                infinite_lights.push(light.clone());
            }
        }
        IntegratorBase {
            aggregate,
            lights,
            infinite_lights,
        }
    }

    /// Traces the given ray into the scene and returns the closest ShapeIntersection if any.
    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        debug_assert!(ray.d != Vector3f::ZERO);
        self.aggregate.intersect(ray, t_max)
    }

    /// Like intersect(), but returns only a boolean regarding the existence of an
    /// intersection, rather than information about the intersection. Potentially
    /// more efficient if only the existence of an intersection is needed.
    /// Useful for shadow rays.
    pub fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        debug_assert!(ray.d != Vector3f::ZERO);
        self.aggregate.intersect_predicate(ray, t_max)
    }
}

// TODO I'm implementing just RandomWalkIntegrator first as a concrete class,
// The inheritance scheme from PBRT really doesn't work well in Rust, so we
// should find another way to share behavior. That's easier *asfter* writing
// out stuff concretely, so let's do that.
pub struct RandomWalkIntegrator {
    integrator_base: IntegratorBase,
    camera: Camera,
    // TODO note this is the sampler_prototype because we'll clone into individual
    //   samplers for each thread.
    sampler_prototype: Sampler,

    max_depth: i32,
}

impl IntegratorI for RandomWalkIntegrator {
    fn render(&mut self, options: &Options) {
        // Declare common variables for rendering image in tiles

        // TODO Be wary of allocating anything on this that uses the heap.
        // TODO When we implement multi-threading, we'll want this to be one per thread.
        let mut scratch_buffer = Bump::with_capacity(256);

        // TODO Render image in waves
        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler_prototype.samples_per_pixel();
        // TODO init a progress reporter. Reference my old ray tracer.

        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        // TODO record pixel statistics option and other things PBRT handles here.
        // Not necessary for just getting a rendered image though.

        // Render in waves until the samples per pixel limit is reached.
        while wave_start < spp {
            // TODO We won't divide into tiles right now, but we should later.
            //  We'll just go pixel by pixel to start, in waves.
            // TODO
            for x in pixel_bounds.min.x..pixel_bounds.max.x {
                for y in pixel_bounds.min.y..pixel_bounds.max.y {
                    let p_pixel = Point2i::new(x, y);
                    for sample_index in wave_start..wave_end {
                        self.sampler_prototype
                            .start_pixel_sample(p_pixel, sample_index, 0);
                        self.evaluate_pixel_sample(
                            p_pixel,
                            sample_index,
                            &mut scratch_buffer,
                            options,
                        );
                        // Note that this does not call drop() on anything allocated in
                        // the scratch buffer. If we allocate anything on the heap, we gotta clean
                        // that ourselves. This is where memory leaks can happen!
                        scratch_buffer.reset();
                    }
                }
            }

            wave_start = wave_end;
            wave_end = i32::min(spp, wave_end + next_wave_size);
            next_wave_size = i32::min(2 * next_wave_size, 64);

            // If we've reached the samples per pixel limit, write out the image.
            // TODO optionally write the current image to the disk as well
            if wave_start == spp {
                let metadata = ImageMetadata::new();
                // TODO populate metadata here!
                self.camera
                    .get_film()
                    .write_image(&metadata, 1.0 / wave_start as Float);
            }
        }
    }
}

impl RandomWalkIntegrator {
    pub fn new(
        aggregate: Primitive,
        lights: Vec<Light>,
        camera: Camera,
        sampler: Sampler,
        max_depth: i32,
    ) -> RandomWalkIntegrator {
        RandomWalkIntegrator {
            integrator_base: IntegratorBase::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            max_depth,
        }
    }

    // TODO This will also take a Sampler, when we aren't just using the sampler_protoype directly
    //  but rather using one per thread.
    pub fn evaluate_pixel_sample(
        &mut self,
        p_pixel: Point2i,
        sample_index: i32,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) {
        // TODO In PBRT, this is in RayIntegrator (which RandomWalk inherits from)
        // Sample wavelengths for the ray
        let lu = if options.disable_wavelength_jitter {
            0.5
        } else {
            self.sampler_prototype.get_1d()
        };
        let lambda = self.camera.get_film().sample_wavelengths(lu);

        // Initialize camera_sample for the current sample
        let filter = self.camera.get_film().get_filter();
        // TODO need to use get_camera_sample(); implement it.

        todo!()
    }

    // TODO should take buffer
    pub fn li(
        &self,
        ray: &RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &Sampler,
        _visible_surface: &VisibleSurface,
        scratch_buffer: &mut Bump,
    ) -> SampledSpectrum {
        self.li_random_walk(ray, lambda, sampler, 0, scratch_buffer)
    }

    /// Depth is the *current depth* of the recursive call to this.
    /// It's not to be confused with the max depth of the walk.
    fn li_random_walk(
        &self,
        ray: &RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &Sampler,
        depth: i32,
        scratch_buffer: &mut Bump,
    ) -> SampledSpectrum {
        // TODO and, finally, this is in RandomWalk.
        todo!()
    }
}
