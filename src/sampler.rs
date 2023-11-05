use crate::{
    vecmath::{Point2f, Point2i},
    Float,
};

use rand::{rngs::SmallRng, Rng, SeedableRng};

pub trait SamplerI {
    fn samples_per_pixel(&self) -> i32;

    // TODO We don't actually implement this deterministic property! We need to do that.
    /// Sets up the RNG so that samples within a pixel can be deterministic.
    /// This is important for debugging lengthy renders.
    /// p: The pixel coordinate.
    /// sample_index: the index of the sample within the pixel. Within 0..samples_per_pixel().
    /// dimension
    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: i32);

    fn get_1d(&mut self) -> Float;
    fn get_2d(&mut self) -> Point2f;
    fn get_pixel_2d(&mut self) -> Point2f;
}

pub enum Sampler {
    Independent(IndependentSampler),
}

impl SamplerI for Sampler {
    fn samples_per_pixel(&self) -> i32 {
        match self {
            Sampler::Independent(s) => s.samples_per_pixel(),
        }
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: i32) {
        match self {
            Sampler::Independent(s) => s.start_pixel_sample(p, sample_index, dimension),
        }
    }

    fn get_1d(&mut self) -> Float {
        match self {
            Sampler::Independent(s) => s.get_1d(),
        }
    }

    fn get_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_2d(),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        match self {
            Sampler::Independent(s) => s.get_pixel_2d(),
        }
    }
}

pub struct IndependentSampler {
    /// Store seed for determinism in start_pixel_sample().
    seed: u64,
    samples_per_pixel: i32,
    rng: SmallRng,
}

impl IndependentSampler {
    pub fn new(seed: u64, samples_per_pixel: i32) -> IndependentSampler {
        IndependentSampler {
            seed,
            samples_per_pixel,
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl SamplerI for IndependentSampler {
    fn samples_per_pixel(&self) -> i32 {
        self.samples_per_pixel
    }

    fn start_pixel_sample(&mut self, p: Point2i, sample_index: i32, dimension: i32) {
        // TODO For now, I think we can disregard getting things to be deterministic
        // and just let the rng continue to roll naturally. But we should come back to this
        // and make it deterministic as described on 469.
    }

    fn get_1d(&mut self) -> Float {
        self.rng.gen()
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f {
            x: self.rng.gen(),
            y: self.rng.gen(),
        }
    }

    fn get_pixel_2d(&mut self) -> Point2f {
        self.get_2d()
    }
}

pub struct CameraSample {
    p_film: Point2f,
    p_lens: Point2f,
    time: Float,
    filter_wieght: Float,
}

impl Default for CameraSample {
    fn default() -> Self {
        Self {
            p_film: Default::default(),
            p_lens: Default::default(),
            time: 0.0,
            filter_wieght: 1.0,
        }
    }
}
