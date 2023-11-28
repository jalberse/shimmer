use std::rc::Rc;

use bumpalo::Bump;
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    bounding_box::Bounds2i,
    camera::{Camera, CameraI},
    film::{FilmI, VisibleSurface},
    float::PI_F,
    image::ImageMetadata,
    light::{Light, LightI, LightType},
    options::Options,
    primitive::{Primitive, PrimitiveI},
    ray::{Ray, RayDifferential},
    sampler::{Sampler, SamplerI},
    sampling::{get_camera_sample, sample_uniform_sphere},
    shape::ShapeIntersection,
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    vecmath::{vector::Vector3, HasNan, Length, Point2i, Tuple2, Vector3f},
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

// TODO When we write other integrators, look at how we can pull out common functionality.
//   I am writing a single working RandomWalkIntegrator first, though.

pub struct RandomWalkIntegrator {
    base: IntegratorBase,
    camera: Camera,
    // This is named sampler_prototype because we will clone individual samplers
    // for each thread.
    sampler_prototype: Sampler,

    max_depth: i32,
}

impl IntegratorI for RandomWalkIntegrator {
    fn render(&mut self, options: &Options) {
        // TODO Be wary of allocating anything on this that uses the heap top avoid memory leaks;
        //   it doesn't call their drop()!
        // TODO When we implement multi-threading, we'll want this to be one per thread.
        let mut scratch_buffer = Bump::with_capacity(256);

        let pixel_bounds = self.camera.get_film().pixel_bounds();
        let spp = self.sampler_prototype.samples_per_pixel();
        // TODO init a progress reporter. Reference my old ray tracer.

        let mut wave_start = 0;
        let mut wave_end = 1;
        let mut next_wave_size = 1;

        // TODO We'll need to have a sampler for each thread; as with a few other variables.
        let mut sampler = self.sampler_prototype.clone();

        // TODO record pixel statistics option and other things PBRT handles here.
        // Not necessary for just getting a rendered image though.

        // TODO We may need to inform the "start pixel" etc based on the pixel bounds
        let tiles = Tile::tile(pixel_bounds, 8, 8);

        // Render in waves until the samples per pixel limit is reached.
        while wave_start < spp {
            tiles.par_iter().for_each(|tile| {
                // TODO actually, Tile should just be a Bounds2i.
                // That encodes the width/height and start/end position and stuff just fine.
                // But yeah I think we will absically just take the below loop, and do it per tile.
                // And then we'll need to sort out all the types for concurrency and mutability and
                // captures and stuff. Hopefully that all goes smoothly...
                for x in tile.bounds.min.x..tile.bounds.max.x {
                    for y in tile.bounds.min.y..tile.bounds.max.y {
                        let p_pixel = Point2i::new(x, y);
                        for sample_index in wave_start..wave_end {
                            self.sampler_prototype
                                .start_pixel_sample(p_pixel, sample_index, 0);
                            self.evaluate_pixel_sample(
                                p_pixel,
                                sample_index,
                                &mut sampler,
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
            });

            // TODO We won't divide into tiles right now, but we should later.
            //  We'll just go pixel by pixel to start, in waves.
            for x in pixel_bounds.min.x..pixel_bounds.max.x {
                for y in pixel_bounds.min.y..pixel_bounds.max.y {
                    let p_pixel = Point2i::new(x, y);
                    for sample_index in wave_start..wave_end {
                        self.sampler_prototype
                            .start_pixel_sample(p_pixel, sample_index, 0);
                        self.evaluate_pixel_sample(
                            p_pixel,
                            sample_index,
                            &mut sampler,
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

impl RandomWalkIntegrator {
    pub fn new(
        aggregate: Primitive,
        lights: Vec<Light>,
        camera: Camera,
        sampler: Sampler,
        max_depth: i32,
    ) -> RandomWalkIntegrator {
        RandomWalkIntegrator {
            base: IntegratorBase::new(aggregate, lights),
            camera,
            sampler_prototype: sampler,
            max_depth,
        }
    }

    pub fn evaluate_pixel_sample(
        &mut self,
        p_pixel: Point2i,
        sample_index: i32,
        sampler: &mut Sampler,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) {
        // Sample wavelengths for the ray
        let lu = if options.disable_wavelength_jitter {
            0.5
        } else {
            self.sampler_prototype.get_1d()
        };
        let lambda = self.camera.get_film().sample_wavelengths(lu);

        // Initialize camera_sample for the current sample
        let filter = self.camera.get_film().get_filter();
        let camera_sample =
            get_camera_sample(&mut self.sampler_prototype, p_pixel, filter, options);

        let camera_ray = self
            .camera
            .generate_ray_differential(&camera_sample, &lambda);

        let (l, visible_surface) = if let Some(mut camera_ray) = camera_ray {
            debug_assert!(camera_ray.ray.ray.d.length() > 0.999);
            debug_assert!(camera_ray.ray.ray.d.length() < 1.001);

            // TODO Scale camera ray differentials absed on image sampling rate.
            let ray_diff_scale = Float::max(
                0.125,
                1.0 / Float::sqrt(self.sampler_prototype.samples_per_pixel() as Float),
            );
            if !options.disable_pixel_jitter {
                camera_ray.ray.scale_differentials(ray_diff_scale);
            }

            // TODO increment number of camera rays (I think that's just useful for logging though)

            // Evaluate radiance along the camera ray
            let visible_surface = if self.camera.get_film().uses_visible_surface() {
                Some(VisibleSurface::default())
            } else {
                None
            };

            // TODO RandomWalk won't use visible surface, but others would expect
            //   one they can initialize. Q: why not restructure so li() just creates it?
            let l = camera_ray.weight
                * self.li(
                    &camera_ray.ray,
                    &lambda,
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

        // Add the camera ray's contribution to the image.
        self.camera.get_film().add_sample(
            &p_pixel,
            &l,
            &lambda,
            &visible_surface,
            camera_sample.filter_weight,
        );
    }

    pub fn li(
        &self,
        ray: &RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
        _visible_surface: &Option<VisibleSurface>,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> SampledSpectrum {
        self.li_random_walk(ray, lambda, sampler, 0, scratch_buffer, options)
    }

    /// Depth is the *current depth* of the recursive call to this.
    /// It's not to be confused with the max depth of the walk.
    fn li_random_walk(
        &self,
        ray: &RayDifferential,
        lambda: &SampledWavelengths,
        sampler: &mut Sampler,
        depth: i32,
        scratch_buffer: &mut Bump,
        options: &Options,
    ) -> SampledSpectrum {
        let si = self.base.intersect(&ray.ray, Float::INFINITY);

        // TODO Since this is a random walk, it will never hit point lights.
        //  We probably want to implement an infinite light, then.

        if si.is_none() {
            // Return emitted light from infinite light sources
            let mut le = SampledSpectrum::from_const(0.0);
            for light in self.base.infinite_lights.iter() {
                le += light.le(&ray.ray, lambda);
            }
            return le;
        }
        // Note that we declare the interaction as mutable; this is because
        // we must calculate differentials stored within the surface interaction.
        // TODO Could we modify how we handle this to allow us to use const?
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
        let bsdf = isect.get_bsdf(ray, lambda, &self.camera, sampler, &options);
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
        let fcos = f * wp.abs_dot_normal(&isect.shading.n);

        // Recursively trace ray to estimate incident radiance at surface
        let ray = isect.interaction.spawn_ray(wp);

        le + fcos * self.li_random_walk(&ray, lambda, sampler, depth + 1, scratch_buffer, options)
            / (1.0 / (4.0 * PI_F))
    }
}
