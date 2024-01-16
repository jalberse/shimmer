use crate::{
    loading::paramdict::ParameterDictionary,
    loading::parser_target::FileLoc,
    options::Options,
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

#[derive(Debug, Clone)]
pub enum Sampler {
    Independent(IndependentSampler),
}

impl Sampler {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        full_res: Point2i,
        options: &Options,
        loc: &FileLoc,
    ) -> Sampler {
        match name {
            "independent" => {
                Sampler::Independent(IndependentSampler::create(parameters, options, loc))
            }
            _ => panic!("Unknown sampler type!"),
        }
    }
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

#[derive(Debug, Clone)]
pub struct IndependentSampler {
    /// Store seed for determinism in start_pixel_sample().
    seed: u64,
    samples_per_pixel: i32,
    rng: SmallRng,
}

impl IndependentSampler {
    pub fn create(
        parameters: &mut ParameterDictionary,
        options: &Options,
        file_loc: &FileLoc,
    ) -> IndependentSampler {
        let mut ns = parameters.get_one_int("pixelsamples", 4);
        if let Some(pixel_samples) = options.pixel_samples {
            ns = pixel_samples;
        }
        let seed = parameters.get_one_int("seed", options.seed) as u64;
        IndependentSampler::new(seed, ns)
    }

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
