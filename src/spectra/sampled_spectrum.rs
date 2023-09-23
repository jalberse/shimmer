use std::ops::{Deref, Index, IndexMut};

use auto_ops::impl_op_ex;

use crate::Float;

const NUM_SPECTRUM_SAMPLES: usize = 4;

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
        SampledSpectrum::new(result)
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
    SampledSpectrum::new(result)
});

impl_op_ex!(
    -|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = s1[i] - s2[i];
        }
        SampledSpectrum::new(result)
    }
);

impl_op_ex!(
    *|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum {
        let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            result[i] = s1[i] * s2[i];
        }
        SampledSpectrum::new(result)
    }
);

impl_op_ex!(/|s1: &SampledSpectrum, s2: &SampledSpectrum| -> SampledSpectrum
{
    let mut result = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES
    {
        result[i] = s1[i] / s2[i];
    }
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
