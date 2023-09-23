use std::ops::{Deref, Index, IndexMut};

use auto_ops::{impl_op_ex, impl_op_ex_commutative};

use crate::{math::Sqrt, vecmath::HasNan, Float};

use super::NUM_SPECTRUM_SAMPLES;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SampledSpectrum {
    values: [Float; NUM_SPECTRUM_SAMPLES],
}

impl SampledSpectrum {
    pub fn new(values: [Float; NUM_SPECTRUM_SAMPLES]) -> SampledSpectrum {
        SampledSpectrum { values }
    }

    pub fn from_const(c: Float) -> SampledSpectrum {
        SampledSpectrum {
            values: [c; NUM_SPECTRUM_SAMPLES],
        }
    }

    pub fn is_zero(&self) -> bool {
        self.values.iter().all(|x: &Float| x == &0.0)
    }

    pub fn safe_div(&self, other: &SampledSpectrum) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = if other[i] == 0.0 {
                0.0
            } else {
                self[i] / other[i]
            }
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn clamp(&self, min: Float, max: Float) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = self.values[i].clamp(min, max);
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn clamp_zero(&self) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = Float::max(0.0, self.values[i])
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn powf(&self, e: Float) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = self.values[i].powf(e);
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn powi(&self, e: i32) -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = self.values[i].powi(e);
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn exp(&self) -> SampledSpectrum {
        // TODO consider a similar elementwise FastExp().
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = self.values[i].exp()
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }

    pub fn lerp(&self, other: &SampledSpectrum, t: Float) -> SampledSpectrum {
        (1.0 - t) * self + t * other
    }

    pub fn average(&self) -> Float {
        self.values.iter().sum::<Float>() / self.values.len() as Float
    }

    pub fn min_component_value(&self) -> Float {
        debug_assert!(!self.values.has_nan());
        let min = self.values.iter().fold(Float::NAN, |a, &b| a.min(b));
        debug_assert!(!min.is_nan());
        min
    }

    pub fn max_component_value(&self) -> Float {
        debug_assert!(!self.values.has_nan());
        let max = self.values.iter().fold(Float::NAN, |a, &b| a.max(b));
        debug_assert!(!max.is_nan());
        max
    }
}

impl Index<usize> for SampledSpectrum {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        self.values.index(index)
    }
}

impl IndexMut<usize> for SampledSpectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.values.index_mut(index)
    }
}

// We implement Deref so that we can use the array's iter().
impl Deref for SampledSpectrum {
    type Target = [Float; NUM_SPECTRUM_SAMPLES];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl_op_ex!(+|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum
{
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        result[i] = s1[i] + s2[i];
    }
    debug_assert!(!result.has_nan());
    SampledSpectrum::new(result)
});

impl_op_ex!(
    -|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = s1[i] - s2[i];
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }
);

impl_op_ex!(
    *|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = s1[i] * s2[i];
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }
);

impl_op_ex_commutative!(*|s1: &SampledSpectrum, v: &Float| -> SampledSpectrum {
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES {
        result[i] = s1[i] * v;
    }
    debug_assert!(!result.has_nan());
    SampledSpectrum::new(result)
});

impl_op_ex!(/|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum
{
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        result[i] = s1[i] / s2[i];
    }
    debug_assert!(!result.has_nan());
    SampledSpectrum::new(result)
});

impl_op_ex!(+=|s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] += s2[i];
    }
});

impl_op_ex!(-=|s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] -= s2[i];
    }
});

impl_op_ex!(*=|s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] *= s2[i];
    }
});

impl_op_ex!(/=|s1: &mut SampledSpectrum, s2: &SampledSpectrum|
{
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        s1[i] /= s2[i];
    }
});

impl HasNan for SampledSpectrum {
    fn has_nan(&self) -> bool {
        self.values.iter().any(|x| x.is_nan())
    }
}

impl HasNan for [Float; NUM_SPECTRUM_SAMPLES] {
    fn has_nan(&self) -> bool {
        self.iter().any(|x| x.is_nan())
    }
}

impl Sqrt for SampledSpectrum {
    fn sqrt(self) -> Self {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = self[i].sqrt();
        }
        debug_assert!(!result.has_nan());
        SampledSpectrum::new(result)
    }
}