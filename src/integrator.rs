use std::cell::RefCell;
use std::sync::Arc;

use bumpalo::Bump;
use indicatif::ParallelProgressIterator;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use thread_local::ThreadLocal;

use crate::{
    bsdf::BSDF, bxdf::{BxDFReflTransFlags, TransportMode}, camera::{Camera, CameraI}, colorspace::RgbColorSpace, film::{FilmI, VisibleSurface}, float::PI_F, image::ImageMetadata, interaction::{Interaction, SurfaceInteraction}, light::{Light, LightI, LightSampleContext, LightType}, light_sampler::{self, LightSampler, LightSamplerI, UniformLightSampler}, loading::paramdict::ParameterDictionary, math::sqr, options::Options, primitive::{Primitive, PrimitiveI}, ray::{Ray, RayDifferential}, sampler::{Sampler, SamplerI}, sampling::{
        get_camera_sample, power_heuristic, sample_uniform_hemisphere, sample_uniform_sphere, uniform_hemisphere_pdf, uniform_sphere_pdf
    }, shape::ShapeIntersection, spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths}, tile::Tile, vecmath::{vector::Vector3, HasNan, Length, Point2f, Point2i, Tuple2, Vector3f}, Float
};

pub fn create_integrator(
    name: &str,
    parameters: &mut ParameterDictionary,
    camera: Camera,
    sampler: Sampler,
    aggregate: Arc<Primitive>,
    lights: Arc<Vec<Arc<Light>>>,
    color_space: Arc<RgbColorSpace>,
) -> Box<dyn Integrator> {
    let integrator = match name {
        "simplepath" => Box::new(ImageTileIntegrator::create_simple_path_integrator(
            parameters, camera, sampler, aggregate, lights,
        )),
        "randomwalk" => Box::new(ImageTileIntegrator::create_random_walk_integrator(
            parameters, camera, sampler, aggregate, lights,
        )),
        "path" => Box::new(ImageTileIntegrator::create_path_integrator(
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

pub trait Integrator {
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

    ray_path_li_evaluator: RayPathLiEvaluator,
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

        let pixel_sample_evaluator = RayPathLiEvaluator::SimplePath(SimplePathIntegrator {
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
        RayPathLiEvaluator::RandomWalk(RandomWalkIntegrator { max_depth });

        // TODO I dislike having to clone the sampler and camera.
        ImageTileIntegrator::new(
            aggregate,
            lights,
            camera.clone(),
            sampler.clone(),
            pixel_sample_evaluator,
        )
    }

    pub fn create_path_integrator(
        parameters: &mut ParameterDictionary,
        camera: Camera,
        sampler: Sampler,
        aggregate: Arc<Primitive>,
        lights: Arc<Vec<Arc<Light>>>,
    ) -> ImageTileIntegrator
    {
        let max_depth = parameters.get_one_int("maxdepth", 5);
        let regularize = parameters.get_one_bool("regularize", false);
        // TODO Change default to bvh
        let light_strategy = parameters.get_one_string("lightsampler", "uniform");
        let light_sampler = LightSampler::create(&light_strategy, lights.clone());

        let pixel_sample_evaluator = RayPathLiEvaluator::Path(PathIntegrator::new(
            max_depth,
            light_sampler,
            regularize,
        ));

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
        ray_path_li_evaluator: RayPathLiEvaluator,
    ) -> ImageTileIntegrator {
        ImageTileIntegrator {
            base: IntegratorBase::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            ray_path_li_evaluator,
        }
    }
}

impl Integrator for ImageTileIntegrator {
    fn render(&mut self, options: &Options) {
        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler_prototype.samples_per_pixel();

        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        let tiles = Tile::tile(pixel_bounds, 8, 8);

        let scratch_buffer_tl = ThreadLocal::new();
        let sampler_tl = ThreadLocal::new();
        let mut film = self.camera.get_film().clone();
        // Render in waves until the samples per pixel limit is reached.
        while wave_start < spp {
            tiles
                .par_iter()
                .progress()
                .for_each(|tile| {
                    // Initialize or get thread-local objects.
                    //
                    // Be wary of allocating anything on the scratchbuffer that uses the
                    // heap to avoid memory leaks; it doesn't call their drop().
                    let scratch_buffer =
                        scratch_buffer_tl.get_or(|| RefCell::new(Bump::with_capacity(256)));
                    let sampler =
                        sampler_tl.get_or(|| RefCell::new(self.sampler_prototype.clone()));

                    let mut rng = SmallRng::from_entropy();

                    for x in tile.bounds.min.x..tile.bounds.max.x {
                        for y in tile.bounds.min.y..tile.bounds.max.y {
                            let p_pixel = Point2i::new(x, y);
                            for sample_index in wave_start..wave_end {
                                sampler
                                    .borrow_mut()
                                    .start_pixel_sample(p_pixel, sample_index, 0);
                                let film_sample =
                                    self.evaluate_pixel_sample(
                                        &self.base,
                                        &self.camera,
                                        p_pixel,
                                        sample_index,
                                        &mut sampler.borrow_mut(),
                                        &mut scratch_buffer.borrow_mut(),
                                        options,
                                        &mut rng,
                                    );

                                // Add the sample to the film.
                                // unsafe: While multiple threads reference this film, the sample
                                // other threads should not be modifying the portion of the film's
                                // data that this thread is modifying, because each thread handles its
                                // own tile.
                                // The typical alternative would be to wrap the Film in a mutex and lock it,
                                // but this slows down the program as we wait on locks (x3 in one test).
                                // Another alternative would be to accumulate film samples and process them
                                // sequentially after joining threads. That was the original approach used,
                                // but that required more memory, which again slowed down rendering too much
                                // for large resolutions.
                                unsafe {
                                    Arc::get_mut_unchecked(&mut film.clone()).add_sample(
                                        &film_sample.p_film,
                                        &film_sample.l,
                                        &film_sample.lambda,
                                        &film_sample.visible_surface,
                                        film_sample.weight,
                                    );
                                }
                                    
                                // Note that this does not call drop() on anything allocated in
                                // the scratch buffer. If we allocate anything on the heap, we gotta clean
                                // that ourselves. This is where memory leaks can happen!
                                scratch_buffer.borrow_mut().reset();
                            }
                        }
                    }
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

impl ImageTileIntegrator
{
    fn evaluate_pixel_sample(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> FilmSample
    {
        // Sample wavelengths for the ray
        let lu = if options.disable_wavelength_jitter {
            0.5
        } else {
            sampler.get_1d()
        };
        let mut lambda = camera.get_film_const().sample_wavelengths(lu);

        // Initialize camera_sample for the current sample
        let camera_sample = get_camera_sample(sampler, p_pixel, camera.get_film_const().get_filter(), options);

        let camera_ray = camera.generate_ray_differential(&camera_sample, &lambda);

        let l = if let Some(mut camera_ray) = camera_ray {
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
            let l = camera_ray.weight
                * self.ray_path_li_evaluator.li(
                    base,
                    camera,
                    &mut camera_ray.ray,
                    &mut lambda,
                    sampler,
                    scratch_buffer,
                    options,
                    rng,
                );

            if l.has_nan() {
                // TODO log error, set SampledSpectrum l to 0.
                // Use env_logger.
            } else if l.y(&lambda).is_infinite() {
                // TODO log error, set SampledSpectrum l to 0
            }

            l
        } else {
            SampledSpectrum::from_const(0.0)            
        };

        FilmSample {
            p_film: p_pixel,
            l,
            lambda,
            visible_surface: None,
            weight: camera_sample.filter_weight,
        }
    }
}

pub enum RayPathLiEvaluator {
    SimplePath(SimplePathIntegrator),
    RandomWalk(RandomWalkIntegrator),
    Path(PathIntegrator),
}

impl RayPathLiEvaluator
{
    fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &mut RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> SampledSpectrum
    {
        match self {
            RayPathLiEvaluator::SimplePath(simple_path_integrator) => {
                simple_path_integrator.li(
                    base,
                    camera,
                    ray,
                    lambda,
                    sampler,
                    scratch_buffer,
                    options,
                    rng,
                )
            }
            RayPathLiEvaluator::RandomWalk(random_walk_integrator) => {
                random_walk_integrator.li(
                    base,
                    camera,
                    ray,
                    lambda,
                    sampler,
                    scratch_buffer,
                    options,
                    rng,
                )
            }
            RayPathLiEvaluator::Path(path_integrator) => {
                path_integrator.li(
                    base,
                    camera,
                    *ray,
                    lambda,
                    sampler,
                    scratch_buffer,
                    options,
                    rng,
                )
            }
        }
    }

}

pub struct RandomWalkIntegrator {
    pub max_depth: i32,
}

impl RandomWalkIntegrator {
    pub fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
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
            rng,
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
        rng: &mut SmallRng,
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
        let bsdf = isect.get_bsdf(ray, lambda, camera, sampler, &options, rng);
        if bsdf.is_none() {
            return le;
        }
        let bsdf = bsdf.unwrap();

        // Randomnly sample direction leaving surface for random walk
        let u = sampler.get_2d();
        let wp = sample_uniform_sphere(u);

        // Evaluate bsdf at surface for sampled direction
        let f = bsdf.f(wo, wp, crate::bxdf::TransportMode::Radiance);
        if f.is_zero() {
            return le;
        }
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
                rng,
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

impl SimplePathIntegrator {
    /// Returns an estimate of the radiance along the provided ray.
    pub fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        ray: &mut RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
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
            let bsdf = isect.get_bsdf(ray, lambda, &camera, sampler, options, rng);
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
                            let f = bsdf.f(wo, wi, TransportMode::Radiance) * wi.abs_dot_normal(isect.shading.n);
                            if !f.is_zero() && base.unoccluded(&isect.interaction, &ls.p_light)
                            {
                                l += beta * f * ls.l / (sampled_light.p * ls.pdf);
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
                    TransportMode::Radiance,
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

pub struct PathIntegrator
{
    max_depth: i32,
    light_sampler: LightSampler,
    regularize: bool,
}

impl PathIntegrator
{
    pub fn new(max_depth: i32, light_sampler: LightSampler, regularize: bool) -> PathIntegrator
    {
        PathIntegrator {
            max_depth,
            light_sampler,
            regularize,
        }
    }

    pub fn li(
        &self,
        base: &IntegratorBase,
        camera: &Camera,
        mut ray: RayDifferential,
        lambda: &mut SampledWavelengths,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
        rng: &mut SmallRng,
    ) -> SampledSpectrum
    {
        let mut l = SampledSpectrum::from_const(0.0);
        let mut beta = SampledSpectrum::from_const(1.0);
        let mut depth = 0;

        let mut p_b = 1.0;
        let mut eta_scale = 1.0;

        let mut specular_bounce = false;
        let mut any_non_specular_bounces = false;
        let mut prev_intr_ctx = LightSampleContext::default();

        // Sample path from camera and accumulate radiance
        loop {
            // Trace ray and find the closest path vertex and its BSDF
            let si = base.intersect(&ray.ray, Float::INFINITY);

            if si.is_none()
            {
                // If no intersection is found, add emitted light from the environment
                for light in base.infinite_lights.iter()
                {
                    let le = light.le(&ray.ray, lambda);
                    if depth == 0 || specular_bounce
                    {
                        l += beta * le;
                    } else {
                        // Compute MIS weight for infinite light
                        let p_l = self.light_sampler.pmf(&prev_intr_ctx, light) *
                            light.pdf_li(&prev_intr_ctx, ray.ray.d, true);
                        let w_b = power_heuristic(1, p_b, 1, p_l);
                        l += beta * w_b * le;
                    }
                }
                break;
            }
            let mut si = si.unwrap();

            // Incorporate emission from surface hit by ray
            let le = si.intr.le(-ray.ray.d, lambda);
            if !le.is_zero()
            {
                if depth == 0 || specular_bounce
                {
                    l += beta * le;
                } else if let Some(light) = &si.intr.area_light
                {
                    // (since le is nonzero, there should always be a light here...)
                    // Compute MIS weight for area light
                    let p_l = self.light_sampler.pmf(&prev_intr_ctx, light) *
                        light.pdf_li(&prev_intr_ctx, ray.ray.d, true);
                    let w_l = power_heuristic(1, p_b, 1, p_l);
                    l += beta * w_l * le;
                }
            }

            // Get BSDF and skip over medium boundaries
            let mut bsdf = si.intr.get_bsdf(&ray, lambda, &camera, sampler, options, rng);
            if bsdf.is_none()
            {
                specular_bounce = true;
                si.intr.skip_intersection(&mut ray, si.t_hit);
                continue;
            }
            let mut bsdf = bsdf.unwrap();

            if self.regularize && any_non_specular_bounces
            {
                bsdf.regularize();
            }

            if depth == self.max_depth
            {
                break;
            }
            depth += 1;

            // Sample direct illumination from the light sources
            if bsdf.flags().is_non_specular()
            {
                let ld = self.sample_ld(base, &si.intr, &bsdf, lambda, sampler);
                l += beta * ld;
            }

            // Sample BSDF to get new path direction
            let wo = -ray.ray.d;
            let u = sampler.get_1d();
            let bs = bsdf.sample_f(
                wo,
                u,
                sampler.get_2d(),
                TransportMode::Radiance,
                BxDFReflTransFlags::all(),
            );
            if bs.is_none()
            {
                break;
            }
            let bs = bs.unwrap();
            // Update path state variables after surface scattering
            beta *= bs.f * bs.wi.abs_dot_normal(si.intr.shading.n) / bs.pdf;
            p_b = if bs.pdf_is_proportional
            {
                bsdf.pdf(wo, bs.wi, TransportMode::Radiance, BxDFReflTransFlags::all())
            } else {
                bs.pdf
            };
            debug_assert!(beta.y(lambda).is_finite());
            specular_bounce = bs.is_specular();
            any_non_specular_bounces |= !specular_bounce;
            if bs.is_transmission()
            {
                eta_scale *= sqr(bs.eta);
            }
            prev_intr_ctx = LightSampleContext::from(&si.intr);

            ray = si.intr.spawn_ray_with_differentials(&ray, bs.wi, bs.flags, bs.eta);

            // Possibly terminate the path with Russian roulette
            if eta_scale.is_finite()
            {
                let rr_beta = beta * eta_scale;
                if rr_beta.max_component_value() < 1.0 && depth > 1
                {
                    let q = Float::max(0.00, 1.0 - rr_beta.max_component_value());
                    if sampler.get_1d() < q
                    {
                        break;
                    }
                    beta /= 1.0 - q;
                    debug_assert!(beta.y(lambda).is_finite());
                }
            }
        }

        l
    }

    fn sample_ld(
        &self,
        base: &IntegratorBase,
        intr: &SurfaceInteraction,
        bsdf: &BSDF,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
    ) -> SampledSpectrum
    {
        let mut ctx: LightSampleContext = intr.into();

        // Try to nudge the light sampling position to the correct side of the surface
        let flags = bsdf.flags();
        if flags.is_reflective() && !flags.is_transmissive()
        {
            ctx.pi = intr.interaction.offset_ray_origin(intr.interaction.wo).into();
        } else if flags.is_transmissive() && !flags.is_reflective()
        {
            ctx.pi = intr.interaction.offset_ray_origin(-intr.interaction.wo).into();
        }

        // Choose a light source for the direct lighting calculation
        let u = sampler.get_1d();
        let sampled_light = self.light_sampler.sample(&ctx, u);
        // Sample before checking for is_none() so that the sampling dimension is
        // consistent across the render.
        let u_light = sampler.get_2d();
        if sampled_light.is_none()
        {
            return SampledSpectrum::from_const(0.0);
        }
        let sampled_light = sampled_light.unwrap();
        debug_assert!(sampled_light.p > 0.0);

        // Sample a point on the light source for direct lighting
        let light = sampled_light.light;
        let ls = light.sample_li(&ctx, u_light, lambda, true);
        if ls.is_none()
        {
            return SampledSpectrum::from_const(0.0);
        }
        let ls = ls.unwrap();
        if ls.l.is_zero() || ls.pdf == 0.0
        {
            return SampledSpectrum::from_const(0.0);
        }

        // Evaluate BSDF for light sample and check light visibility
        let wo = intr.interaction.wo;
        let wi = ls.wi;
        let f = bsdf.f(wo, wi, TransportMode::Radiance) * wi.abs_dot_normal(intr.shading.n);
        if f.is_zero() || !base.unoccluded(&intr.interaction, &ls.p_light)
        {
            return SampledSpectrum::from_const(0.0);
        }

        // Return the light's contribution to reflected radiance
        let p_l = sampled_light.p * ls.pdf;
        if light.light_type().is_delta()
        {
            ls.l * f / p_l
        } else {
            let p_b = bsdf.pdf(wo, wi, TransportMode::Radiance, BxDFReflTransFlags::ALL);
            let w_l = power_heuristic(1, p_l, 1, p_b);
            w_l * ls.l * f / p_l
        }
    }
}