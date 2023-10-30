use std::rc::Rc;

use crate::{
    light::{Light, LightSampleContext},
    Float,
};

pub trait LightSamplerI {
    /// Uses sampler to get a light given a point/context.
    fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight>;

    /// Probability mass function for sampling a light; useful to compute MIS weighting term.
    fn pmf(&self, ctx: &LightSampleContext, light: &Light) -> Float;

    /// Samples a light without context; useful for e.g. bidirectional path tracing
    /// where the path starts at the light source
    fn sample_light(&self, u: Float) -> Option<SampledLight>;

    /// Probability mass function for sampling a light without context; useful for e.g. bidirectional path tracing
    /// where the path starts at the light source
    fn pmf_light(&self, light: &Light) -> Float;
}

pub struct SampledLight {
    pub light: Rc<Light>,
    /// Discrete probability for this light to be sampled
    p: Float,
}

pub enum LightSampler {
    Uniform(UniformLightSampler),
    // TODO PowerLightSampler, ExhaustiveLightSampler, BVHLightSampler
}

impl LightSamplerI for LightSampler {
    fn sample(&self, ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        match self {
            LightSampler::Uniform(s) => s.sample(ctx, u),
        }
    }

    fn pmf(&self, ctx: &LightSampleContext, light: &Light) -> Float {
        match self {
            LightSampler::Uniform(s) => s.pmf(ctx, light),
        }
    }

    fn sample_light(&self, u: Float) -> Option<SampledLight> {
        match self {
            LightSampler::Uniform(s) => s.sample_light(u),
        }
    }

    fn pmf_light(&self, light: &Light) -> Float {
        match self {
            LightSampler::Uniform(s) => s.pmf_light(light),
        }
    }
}

/// Simplest possible light sampler; samples all lights with uniform probability.
/// In practice, other light samplers should be used.
pub struct UniformLightSampler {
    lights: Vec<Rc<Light>>,
}

impl LightSamplerI for UniformLightSampler {
    fn sample(&self, _ctx: &LightSampleContext, u: Float) -> Option<SampledLight> {
        self.sample_light(u)
    }

    fn pmf(&self, _ctx: &LightSampleContext, light: &Light) -> Float {
        self.pmf_light(light)
    }

    fn sample_light(&self, u: Float) -> Option<SampledLight> {
        if self.lights.is_empty() {
            return None;
        }
        let light_index = usize::min(
            (u * self.lights.len() as Float) as usize,
            self.lights.len() - 1,
        );
        Some(SampledLight {
            light: self.lights[light_index].clone(),
            p: 1.0 / self.lights.len() as Float,
        })
    }

    fn pmf_light(&self, _light: &Light) -> Float {
        if self.lights.is_empty() {
            0.0
        } else {
            1.0 / self.lights.len() as Float
        }
    }
}
