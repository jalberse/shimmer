use std::cell::RefCell;
use std::sync::Arc;

use bumpalo::Bump;
use indicatif::ParallelProgressIterator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use thread_local::ThreadLocal;

use crate::{
    bounding_box::Bounds2i,
    bxdf::BxDFReflTransFlags,
    camera::{Camera, CameraI},
    colorspace::RgbColorSpace,
    film::{FilmI, VisibleSurface},
    float::PI_F,
    image::ImageMetadata,
    interaction::Interaction,
    light::{Light, LightI, LightSampleContext, LightType},
    light_sampler::{LightSamplerI, UniformLightSampler},
    loading::paramdict::ParameterDictionary,
    options::Options,
    primitive::{Primitive, PrimitiveI},
    ray::{Ray, RayDifferential},
    sampler::{Sampler, SamplerI},
    sampling::{
        get_camera_sample, sample_uniform_hemisphere, sample_uniform_sphere,
        uniform_hemisphere_pdf, uniform_sphere_pdf,
    },
    shape::ShapeIntersection,
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    vecmath::{vector::Vector3, HasNan, Length, Point2i, Tuple2, Vector3f},
    Float,
};

pub fn create_integrator(
    name: &str,
    parameters: &mut ParameterDictionary,
    camera: Camera,
    sampler: Sampler,
    aggregate: Arc<Primitive>,
    lights: Arc<Vec<Arc<Light>>>,
    color_space: Arc<RgbColorSpace>,
) -> Box<dyn IntegratorI> {
    let integrator = match name {
        "simplepath" => Box::new(ImageTileIntegrator::create_simple_path_integrator(
            parameters, camera, sampler, aggregate, lights,
        )),
        "randomwalk" => Box::new(ImageTileIntegrator::create_random_walk_integrator(
            parameters, camera, sampler, aggregate, lights,
        )),
        _ => {
            panic!("Unknown integrator {}", name);
        }
    };

    // TODO report unused params
    integrator
}

struct FilmSample {
    p_film: Point2i,
    l: SampledSpectrum,
    lambda: SampledWavelengths,
    visible_surface: Option<VisibleSurface>,
    weight: Float,
}

pub trait IntegratorI {
    fn render(&mut self, options: &Options);
}

/// Integrators can have an IntegratorBase that provides shared data and functionality.
pub struct IntegratorBase {
    aggregate: Arc<Primitive>,
    /// Contains both finite and infinite lights
    lights: Arc<Vec<Arc<Light>>>,
    /// Stores an additional copy of any infinite lights in their own vector.
    infinite_lights: Vec<Arc<Light>>,
}

impl IntegratorBase {
    const SHADOW_EPISLON: Float = 0.0001;

    /// Creates an IntegratorBase from the given aggregate and lights.
    /// Also does pre-processing for the lights.
    pub fn new(aggregate: Arc<Primitive>, mut lights: Arc<Vec<Arc<Light>>>) -> IntegratorBase {
        let scene_bounds = aggregate.bounds();

        // TODO Is there a better alternative here that does not require get_mut_unchecked?
        // That is nightly-only and unsafe, and I'd like to avoid that if possible.
        // But this is a quick way forward, so we'll do it for now.
        // The alternatives could include:
        // 1. Use Vec<Arc<Mutex<Light>>>, but we don't ever need the Mutex after this (read-only after this write),
        //    and I don't want to pay that cost. We could try to convert afterwards, but that's awkward, touching a lot of structs.
        // 2. Defer wrapping the Lights in Arc until after this pre-processing step. But that would mean only
        //    keeping one copy - maybe we keep one copy of the area_light Arc in the aggregate (we could use get_mut() for that,
        //    which is stable), and then keep one copy of all other lights (point, infinite) in the lights vector.
        //    Plus the lights in the light sampler.
        //    Then we can pre-process those while there's only one reference, and then copy the lights from the aggregate
        //    into the lights list (thereby making the second reference for those).
        // We almost certainly want to take advantage of interior mutability for this
        //  instead; https://ricardomartins.cc/2016/06/08/interior-mutability

        // Unsafe get_mut_unchecked() - If any other Arc or Weak pointers to the same allocation exist,
        // then they must not be dereferenced or have active borrows for the duration.
        // That should be okay here, because the only other references to this same light
        // should be in the aggregate and in the light sampler, which shouldn't
        // be processed in parallel with this.
        unsafe {
            for light in Arc::get_mut_unchecked(&mut lights).iter_mut() {
                Arc::get_mut_unchecked(light).preprocess(&scene_bounds);
            }
        }

        let mut infinite_lights = Vec::new();

        for light in lights.as_ref() {
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

    pub fn unoccluded(&self, p0: &Interaction, p1: &Interaction) -> bool {
        !self.intersect_predicate(&p0.spawn_ray_to_interaction(p1), 1.0 - Self::SHADOW_EPISLON)
    }
}

pub struct ImageTileIntegrator {
    base: IntegratorBase,
    camera: Camera,
    sampler_prototype: Sampler,

    pixel_sample_evaluator: PixelSampleEvaluator,
}

impl ImageTileIntegrator {
    pub fn create_simple_path_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<Vec<Arc<Light>>>,
    ) -> ImageTileIntegrator {
        let max_depth = parameters.get_one_int("maxdepth", 5);
        let sample_lights = parameters.get_one_bool("samplelights", true);
        let sample_bsdf = parameters.get_one_bool("samplebsdf", true);
        let light_sampler = UniformLightSampler {
            lights: lights.clone(),
        };

        let pixel_sample_evaluator = PixelSampleEvaluator::SimplePath(SimplePathIntegrator {
            max_depth,
            sample_lights,
            sample_bsdf,
            light_sampler,
        });

        // TODO I dislike having to clone the sampler and camera.
        ImageTileIntegrator::new(
            aggregate,
            lights,
            camera.clone(),
            sampler.clone(),
            pixel_sample_evaluator,
        )
    }

    pub fn create_random_walk_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<Vec<Arc<Light>>>,
    ) -> ImageTileIntegrator {
        let max_depth = parameters.get_one_int("maxdepth", 5);
        let pixel_sample_evaluator =
            PixelSampleEvaluator::RandomWalk(RandomWalkIntegrator { max_depth });

        // TODO I dislike having to clone the sampler and camera.
        ImageTileIntegrator::new(
            aggregate,
            lights,
            camera.clone(),
            sampler.clone(),
            pixel_sample_evaluator,
        )
    }

    pub fn new(
        aggregate: Arc<Primitive>,
        lights: Arc<Vec<Arc<Light>>>,
        camera: Camera,
        sampler: Sampler,
        pixel_sample_evaluator: PixelSampleEvaluator,
    ) -> ImageTileIntegrator {
        ImageTileIntegrator {
            base: IntegratorBase::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            pixel_sample_evaluator,
        }
    }
}

impl IntegratorI for ImageTileIntegrator {
    fn render(&mut self, options: &Options) {
        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler_prototype.samples_per_pixel();

        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        let tiles = Tile::tile(pixel_bounds, 8, 8);

        let scratch_buffer_tl = ThreadLocal::new();
        let sampler_tl = ThreadLocal::new();
        // Render in waves until the samples per pixel limit is reached.
        while wave_start < spp {
            let film_samples: Vec<Vec<FilmSample>> = tiles
                .par_iter()
                .progress()
                .map(|tile| -> Vec<FilmSample> {
                    // TODO Be wary of allocating anything on the scratchbuffer that uses the
                    // heap to avoid memory leaks; it doesn't call their drop().

                    let mut samples = Vec::with_capacity(
                        (tile.bounds.width() * tile.bounds.height() * wave_end - wave_start)
                            as usize,
                    );

                    // Initialize or get thread-local objects.
                    let scratch_buffer =
                        scratch_buffer_tl.get_or(|| RefCell::new(Bump::with_capacity(256)));
                    let sampler =
                        sampler_tl.get_or(|| RefCell::new(self.sampler_prototype.clone()));

                    for x in tile.bounds.min.x..tile.bounds.max.x {
                        for y in tile.bounds.min.y..tile.bounds.max.y {
                            let p_pixel = Point2i::new(x, y);
                            for sample_index in wave_start..wave_end {
                                sampler
                                    .borrow_mut()
                                    .start_pixel_sample(p_pixel, sample_index, 0);
                                let film_sample =
                                    self.pixel_sample_evaluator.evaluate_pixel_sample(
                                        &self.base,
                                        &self.camera,
                                        p_pixel,
                                        sample_index,
                                        &mut sampler.borrow_mut(),
                                        &mut scratch_buffer.borrow_mut(),
                                        options,
                                    );
                                samples.push(film_sample);
                                // Note that this does not call drop() on anything allocated in
                                // the scratch buffer. If we allocate anything on the heap, we gotta clean
                                // that ourselves. This is where memory leaks can happen!
                                scratch_buffer.borrow_mut().reset();
                            }
                        }
                    }
                    samples
                })
                .collect();
            let samples: Vec<FilmSample> = film_samples.into_iter().flatten().collect();
            samples.into_iter().for_each(|s| {
                self.camera.get_film().add_sample(
                    &s.p_film,
                    &s.l,
                    &s.lambda,
                    &s.visible_surface,
                    s.weight,
                )
            });

            wave_start = wave_end;
            wave_end = i32::min(spp, wave_end + next_wave_size);
            next_wave_size = i32::min(2 * next_wave_size, 64);

            // If we've reached the samples per pixel limit, write out the image.
            // TODO optionally write the current image to the disk as well
            if wave_start == spp {
                let mut metadata = ImageMetadata::default();
                // TODO populate metadata here!
                self.camera
                    .get_film()
                    .write_image(&mut metadata, 1.0 / wave_start as Float)
                    .unwrap();
            }
        }
    }
}

// TODO We don't need this to be a trait necessarily.
//   If we're just matching inside the PixelSampleEvaluator enum's impl,
//   then we can have each one take their own parameters for evaluate_pixel_sample().
//   We don't actually need the trait, just the match.
// We also want to share the evaluate_pixel_sample implementation, thought.
trait PixelSampleEvaluatorI {
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> FilmSample;
}

pub enum PixelSampleEvaluator {
    RandomWalk(RandomWalkIntegrator),
    SimplePath(SimplePathIntegrator),
}

impl PixelSampleEvaluatorI for PixelSampleEvaluator {
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> FilmSample {
        match self {
            PixelSampleEvaluator::RandomWalk(integrator) => integrator.evaluate_pixel_sample(
                base,
                camera,
                p_pixel,
                sample_index,
                sampler,
                scratch_buffer,
                options,
            ),
            PixelSampleEvaluator::SimplePath(integrator) => integrator.evaluate_pixel_sample(
                base,
                camera,
                p_pixel,
                sample_index,
                sampler,
                scratch_buffer,
                options,
            ),
        }
    }
}

pub struct RandomWalkIntegrator {
    pub max_depth: i32,
}

impl PixelSampleEvaluatorI for RandomWalkIntegrator {
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> FilmSample {
        // Sample wavelengths for the ray
        let lu = if options.disable_wavelength_jitter {
            0.5
        } else {
            sampler.get_1d()
        };
        let mut lambda = camera.get_film_const().sample_wavelengths(lu);

        // Initialize camera_sample for the current sample
        let filter = camera.get_film_const().get_filter();
        let camera_sample = get_camera_sample(sampler, p_pixel, filter, options);

        let camera_ray = camera.generate_ray_differential(&camera_sample, &lambda);

        let (l, visible_surface) = if let Some(mut camera_ray) = camera_ray {
            debug_assert!(camera_ray.ray.ray.d.length() > 0.999);
            debug_assert!(camera_ray.ray.ray.d.length() < 1.001);

            // TODO Scale camera ray differentials based on image sampling rate.
            let ray_diff_scale = Float::max(
                0.125,
                1.0 / Float::sqrt(sampler.samples_per_pixel() as Float),
            );
            if !options.disable_pixel_jitter {
                camera_ray.ray.scale_differentials(ray_diff_scale);
            }

            // Evaluate radiance along the camera ray
            let visible_surface = if camera.get_film_const().uses_visible_surface() {
                Some(VisibleSurface::default())
            } else {
                None
            };

            let l = camera_ray.weight
                * self.li(
                    base,
                    camera,
                    &camera_ray.ray,
                    &mut lambda,
                    sampler,
                    &visible_surface,
                    scratch_buffer,
                    options,
                );

            if l.has_nan() {
                // TODO log error, set SampledSpectrum l to 0.
                // Use env_logger.
            } else if l.y(&lambda).is_infinite() {
                // TODO log error, set SampledSpectrum l to 0
            }

            (l, visible_surface)
        } else {
            (
                SampledSpectrum::from_const(0.0),
                Some(VisibleSurface::default()),
            )
        };

        FilmSample {
            p_film: p_pixel,
            l,
            lambda,
            visible_surface,
            weight: camera_sample.filter_weight,
        }
    }
}

impl RandomWalkIntegrator {
    pub fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _visible_surface: &Option<VisibleSurface>,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> SampledSpectrum {
        self.li_random_walk(
            base,
            camera,
            ray,
            lambda,
            sampler,
            0,
            scratch_buffer,
            options,
        )
    }

    /// Depth is the *current depth* of the recursive call to this.
    /// It's not to be confused with the max depth of the walk.
    fn li_random_walk(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        depth: i32,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> SampledSpectrum {
        let si = base.intersect(&ray.ray, Float::INFINITY);

        if si.is_none() {
            // Return emitted light from infinite light sources
            let mut le = SampledSpectrum::from_const(0.0);
            for light in base.infinite_lights.iter() {
                le += light.le(&ray.ray, lambda);
            }
            return le;
        }
        // Note that we declare the interaction as mutable; this is because
        // we must calculate differentials stored within the surface interaction.
        let mut si = si.unwrap();
        let isect = &mut si.intr;

        // Get emitted radiance at surface intersection
        let wo = -ray.ray.d;
        let le = isect.le(wo, lambda);

        // Terminate the walk if the maximum depth has been reached
        if depth == self.max_depth {
            return le;
        }

        // Compute BSDF at random walk intersection point.
        let bsdf = isect.get_bsdf(ray, lambda, camera, sampler, &options);
        if bsdf.is_none() {
            return le;
        }
        let bsdf = bsdf.unwrap();

        // Randomnly sample direction leaving surface for random walk
        let u = sampler.get_2d();
        let wp = sample_uniform_sphere(u);

        // Evaluate bsdf at surface for sampled direction
        let f = bsdf.f(wo, wp, crate::bxdf::TransportMode::Radiance);
        if f.is_none() {
            return le;
        }
        let f = f.unwrap();
        let fcos = f * wp.abs_dot_normal(isect.shading.n);

        // Recursively trace ray to estimate incident radiance at surface
        let ray = isect.interaction.spawn_ray(wp);

        le + fcos
            * self.li_random_walk(
                base,
                camera,
                &ray,
                lambda,
                sampler,
                depth + 1,
                scratch_buffer,
                options,
            )
            / (1.0 / (4.0 * PI_F))
    }
}

/// A step up from a RandomWalkIntegrator.
/// The PathIntegrator is better when efficiency is important.
/// This is useful for debugging and for validating the implementation of sampling algorithms.
/// For example, it can be configured to use BSDFs’ sampling methods or to use uniform directional sampling;
/// given a sufficient number of samples, both approaches should converge to the same result
/// (assuming that the BSDF is not perfect specular).
/// If they do not, the error is presumably in the BSDF sampling code.
/// Light sampling techniques can be tested in a similar fashion.
pub struct SimplePathIntegrator {
    pub max_depth: i32,
    /// Determines whether lights’ SampleLi() methods should be used to sample direct illumination or
    /// whether illumination should only be found by rays randomly intersecting emissive surfaces,
    /// as was done in the RandomWalkIntegrator.
    pub sample_lights: bool,
    /// Determines whether BSDFs’ Sample_f() methods should be used to sample directions or whether
    /// uniform directional sampling should be used.
    pub sample_bsdf: bool,
    pub light_sampler: UniformLightSampler,
}

// TODO This should be shared between SimplePathIntegrator and RandomWalkIntegrator.
impl PixelSampleEvaluatorI for SimplePathIntegrator {
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> FilmSample {
        // Sample wavelengths for the ray
        let lu = if options.disable_wavelength_jitter {
            0.5
        } else {
            sampler.get_1d()
        };
        let mut lambda = camera.get_film_const().sample_wavelengths(lu);

        // Initialize camera_sample for the current sample
        let filter = camera.get_film_const().get_filter();
        let camera_sample = get_camera_sample(sampler, p_pixel, filter, options);

        let camera_ray = camera.generate_ray_differential(&camera_sample, &lambda);

        let (l, visible_surface) = if let Some(mut camera_ray) = camera_ray {
            debug_assert!(camera_ray.ray.ray.d.length() > 0.999);
            debug_assert!(camera_ray.ray.ray.d.length() < 1.001);

            // TODO Scale camera ray differentials absed on image sampling rate.
            let ray_diff_scale = Float::max(
                0.125,
                1.0 / Float::sqrt(sampler.samples_per_pixel() as Float),
            );
            if !options.disable_pixel_jitter {
                camera_ray.ray.scale_differentials(ray_diff_scale);
            }

            // Evaluate radiance along the camera ray
            let visible_surface = if camera.get_film_const().uses_visible_surface() {
                Some(VisibleSurface::default())
            } else {
                None
            };

            let l = camera_ray.weight
                * self.li(
                    base,
                    camera,
                    &mut camera_ray.ray,
                    &mut lambda,
                    sampler,
                    &visible_surface,
                    scratch_buffer,
                    options,
                );

            if l.has_nan() {
                // TODO log error, set SampledSpectrum l to 0.
                // Use env_logger.
            } else if l.y(&lambda).is_infinite() {
                // TODO log error, set SampledSpectrum l to 0
            }

            (l, visible_surface)
        } else {
            (
                SampledSpectrum::from_const(0.0),
                Some(VisibleSurface::default()),
            )
        };

        FilmSample {
            p_film: p_pixel,
            l,
            lambda,
            visible_surface,
            weight: camera_sample.filter_weight,
        }
    }
}

impl SimplePathIntegrator {
    /// Returns an estimate of the radiance along the provided ray.
    pub fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &mut RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        _visible_surface: &Option<VisibleSurface>,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> SampledSpectrum {
        // The current estimated scattered radiance
        let mut l = SampledSpectrum::from_const(0.0);
        // Tracks if the last outgoing path direction sampled was due to specular reflection.
        let mut specular_bounce = true;
        // The "path throughput weight". That is, the product of the BSDF values and cosine terms for the vertices
        // generated so far, divided by their respective sampling PDFs.
        let mut beta = SampledSpectrum::from_const(1.0);
        let mut depth = 0;

        while !beta.is_zero() {
            // Find next SimplePathIntegrator vertex and accumulate contribution
            let si = base.intersect(&ray.ray, Float::INFINITY);

            // Account for infinite lights if ray has no intersections
            if si.is_none() {
                if !self.sample_lights || specular_bounce {
                    for light in base.infinite_lights.iter() {
                        l += beta * light.le(&ray.ray, lambda);
                    }
                }
                break;
            }
            let si = si.unwrap();

            // Account for emissive surface if light was not sampled
            let mut isect = si.intr;
            if !self.sample_lights || specular_bounce {
                l += beta * isect.le(-ray.ray.d, lambda);
            }

            // End path if maximum depth reached
            if depth == self.max_depth {
                break;
            }
            depth += 1;

            // Get BSDF and skip over medium boundaries
            let bsdf = isect.get_bsdf(ray, lambda, &camera, sampler, options);
            if bsdf.is_none() {
                specular_bounce = true;
                isect.skip_intersection(ray, si.t_hit);
                continue;
            }
            let bsdf = bsdf.unwrap();

            let wo = -ray.ray.d;
            // Sample direct illumination if specified
            if self.sample_lights {
                let sampled_light = self.light_sampler.sample_light(sampler.get_1d());
                if let Some(sampled_light) = sampled_light {
                    // Sample point on sampled_light to estimate direct illumination
                    let u_light = sampler.get_2d();
                    let light_sample_ctx = LightSampleContext::from(&isect);
                    let ls =
                        sampled_light
                            .light
                            .sample_li(&light_sample_ctx, u_light, lambda, false);
                    if let Some(ls) = ls {
                        if !ls.l.is_zero() && ls.pdf > 0.0 {
                            // Evaluate BSDF for light and possibly scattered radiance
                            let wi = ls.wi;
                            let f = bsdf.f(wo, wi, crate::bxdf::TransportMode::Radiance);
                            if f.is_some() && base.unoccluded(&isect.interaction, &ls.p_light) {
                                l += beta * f.unwrap() * ls.l / (sampled_light.p * ls.pdf);
                            }
                        }
                    }
                }
            }

            // Sample outgoing direction at intersection to continue path.
            if self.sample_bsdf {
                // Sample BSDF for new path direction
                let u = sampler.get_1d();
                let bs = bsdf.sample_f(
                    wo,
                    u,
                    sampler.get_2d(),
                    crate::bxdf::TransportMode::Radiance,
                    BxDFReflTransFlags::all(),
                );
                if bs.is_none() {
                    break;
                }
                let bs = bs.unwrap();
                beta *= bs.f * bs.wi.abs_dot_normal(isect.shading.n) / bs.pdf;
                specular_bounce = bs.is_specular();
                *ray = isect.interaction.spawn_ray(bs.wi);
            } else {
                // Uniformly sample sphere or hemisphere to get new path direction
                let flags = bsdf.flags();
                let (pdf, wi) = if flags.is_reflective() && flags.is_transmissive() {
                    let wi = sample_uniform_sphere(sampler.get_2d());
                    let pdf = uniform_sphere_pdf();
                    (pdf, wi)
                } else {
                    let wi = sample_uniform_hemisphere(sampler.get_2d());
                    let pdf = uniform_hemisphere_pdf();
                    let wi = if (flags.is_reflective()
                        && wo.dot_normal(isect.interaction.n) * wi.dot_normal(isect.interaction.n)
                            < 0.0)
                        || (flags.is_transmissive()
                            && wo.dot_normal(isect.interaction.n)
                                * wi.dot_normal(isect.interaction.n)
                                > 0.0)
                    {
                        -wi
                    } else {
                        wi
                    };
                    (pdf, wi)
                };
                beta *= bsdf
                    .f(wo, wi, crate::bxdf::TransportMode::Radiance)
                    .unwrap()
                    * wi.abs_dot_normal(isect.shading.n)
                    / pdf;
                specular_bounce = false;
                *ray = isect.interaction.spawn_ray(wi);
            }

            debug_assert!(beta.y(lambda) >= 0.0);
            debug_assert!(beta.y(lambda).is_finite());
        }
        debug_assert!(!l.values[0].is_nan());
        debug_assert!(!l.values[1].is_nan());
        debug_assert!(!l.values[2].is_nan());
        debug_assert!(!l.values[3].is_nan());
        l
    }
}

struct Tile {
    bounds: Bounds2i,
}

impl Tile {
    /// Returns a list of Tiles covering the image.
    ///
    /// The tiles are returned in a flattened Vec in row-major order.
    /// If the image cannot be perfectly divided by the tile width or height,
    /// then smaller tiles are created to fill the remainder of the image width or height.
    /// It's recommended to pick a tiling size that fits into the image resolution well.
    /// Note that 8x8 is a reasonable tile size and 8 evenly divides common resolution
    /// sizes like 1920, 1080, 720, etc.
    ///
    /// * `pixel_bounds` - The pixel bounds of the image.
    /// * `tile_width` - Width of each tile, in pixels.
    /// * `tile_height` - Height of each tile, in pixels.
    pub fn tile(pixel_bounds: Bounds2i, tile_width: i32, tile_height: i32) -> Vec<Tile> {
        let image_width = pixel_bounds.width();
        let image_height = pixel_bounds.height();
        let num_horizontal_tiles = image_width / tile_width;
        let remainder_horizontal_pixels = image_width % tile_width;
        let num_vertical_tiles = image_height / tile_height;
        let remainder_vertical_pixels = image_height % tile_height;

        let mut tiles = Vec::with_capacity((num_horizontal_tiles * num_vertical_tiles) as usize);

        for tile_y in 0..num_vertical_tiles {
            for tile_x in 0..num_horizontal_tiles {
                let tile_start_x = pixel_bounds.min.x + tile_x * tile_width;
                let tile_start_y = pixel_bounds.min.y + tile_y * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i {
                            x: tile_start_x,
                            y: tile_start_y,
                        },
                        Point2i {
                            x: tile_start_x + tile_width,
                            y: tile_start_y + tile_height,
                        },
                    ),
                });
            }
            // Add the rightmost row if necessary
            if remainder_horizontal_pixels > 0 {
                let tile_start_x = pixel_bounds.min.x + num_horizontal_tiles * tile_width;
                let tile_start_y = pixel_bounds.min.y + tile_y * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i {
                            x: tile_start_x,
                            y: tile_start_y,
                        },
                        Point2i {
                            x: tile_start_x + remainder_horizontal_pixels,
                            y: tile_start_y + tile_height,
                        },
                    ),
                });
            }
        }
        // Add the bottom row if necessary
        if remainder_vertical_pixels > 0 {
            for tile_x in 0..num_horizontal_tiles {
                let tile_start_x = pixel_bounds.min.x + tile_x * tile_width;
                let tile_start_y = pixel_bounds.min.y + num_vertical_tiles * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i {
                            x: tile_start_x,
                            y: tile_start_y,
                        },
                        Point2i {
                            x: tile_start_x + tile_width,
                            y: tile_start_y + remainder_vertical_pixels,
                        },
                    ),
                });
            }
        }
        // Add the bottom-most, right-most Tile if necessary
        if remainder_horizontal_pixels > 0 && remainder_vertical_pixels > 0 {
            let tile_start_x = pixel_bounds.min.x + num_horizontal_tiles * tile_width;
            let tile_start_y = pixel_bounds.min.y + num_vertical_tiles * tile_height;
            tiles.push(Tile {
                bounds: Bounds2i::new(
                    Point2i {
                        x: tile_start_x,
                        y: tile_start_y,
                    },
                    Point2i {
                        x: tile_start_x + remainder_horizontal_pixels,
                        y: tile_start_y + remainder_vertical_pixels,
                    },
                ),
            });
        }

        tiles
    }
}
