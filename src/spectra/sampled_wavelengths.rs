use super::{
    spectrum::{LAMBDA_MAX, LAMBDA_MIN},
    NUM_SPECTRUM_SAMPLES,
};

use crate::{math::lerp, Float};

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct SampledWavelengths {
    lambda: [Float; NUM_SPECTRUM_SAMPLES],
    pdf: [Float; NUM_SPECTRUM_SAMPLES],
}

impl SampledWavelengths {
    // TODO make a sample_uniform with defaults

    pub fn sample_uniform(u: Float) -> SampledWavelengths {
        Self::sample_uniform_range(u, LAMBDA_MIN, LAMBDA_MAX)
    }

    pub fn sample_uniform_range(
        u: Float,
        lambda_min: Float,
        lambda_max: Float,
    ) -> SampledWavelengths {
        let mut lambda = [0.0; NUM_SPECTRUM_SAMPLES];

        // Sample first wavelength using u
        lambda[0] = lerp(u, &lambda_min, &lambda_max);

        // Initialzie lambda for remaining wavelengths
        let delta = (lambda_max - lambda_min) / NUM_SPECTRUM_SAMPLES;
        for i in 1..NUM_SPECTRUM_SAMPLES {
            lambda[i] = lambda[i - 1] + delta;
            if lambda[i] > lambda_max {
                // Wrap around if past lambda_max
                lambda[i] = lambda_min + (lambda[i] - lambda_max)
            }
        }

        // Compute PDF for sampled wavelengths - easy, since sampling is uniform.
        let mut pdf = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            pdf[i] = 1.0 / (lambda_max - lambda_min)
        }

        SampledWavelengths { lambda, pdf }
    }
}
