use crate::{math::lerp, Float};

use itertools::Itertools;

/// Nanometers. Minimum of visible range of light.
const LAMBDA_MIN: Float = 360.0;
/// Nanometers. Maximum of visible range of light.
const LAMBDA_MAX: Float = 830.0;

pub trait SpectrumI {
    fn get(&self, lambda: Float) -> Float;

    fn max_value(&self) -> Float;
}

pub enum Spectrum {
    Constant(Constant),
    DenselySampled(DenselySampled),
    PiecewiseLinear(PiecewiseLinear),
    Blackbody(Blackbody),
}

impl SpectrumI for Spectrum {
    /// Gets the value of the spectral distribution at wavelength lambda
    fn get(&self, lambda: Float) -> Float {
        // PAPERDOC - this will be a great example against PBRTv4's call with TaggedPointer
        // to achieve static dispatch. If this is too verbose for someone's taste, then
        // there are crates like enum_dispatch which could generate this automatically from
        // the SpectrumI trait.
        match self {
            Spectrum::Constant(c) => c.get(lambda),
            Spectrum::DenselySampled(s) => s.get(lambda),
            Spectrum::PiecewiseLinear(s) => s.get(lambda),
            Spectrum::Blackbody(s) => s.get(lambda),
        }
    }

    /// Returns a bound on the maximum value of the psectral distribution over its wavelength range.
    /// The primary utility here is computing bounds on the power emitted by light sources so that
    /// lights can be sampled according to their expected contriibution to illumination in the scene.
    fn max_value(&self) -> Float {
        match self {
            Spectrum::Constant(c) => c.max_value(),
            Spectrum::DenselySampled(s) => s.max_value(),
            Spectrum::PiecewiseLinear(s) => s.max_value(),
            Spectrum::Blackbody(s) => s.max_value(),
        }
    }
}

pub struct Constant {
    c: Float,
}

impl Constant {
    pub fn new(c: Float) -> Self {
        Self { c }
    }
}

impl SpectrumI for Constant {
    fn get(&self, _lambda: Float) -> Float {
        self.c
    }

    fn max_value(&self) -> Float {
        self.c
    }
}

pub struct DenselySampled {
    lambda_min: i32,
    lambda_max: i32,
    values: Vec<Float>,
}

impl DenselySampled {
    /// Samples from the provided spectrum to create a DenselySampled spectrum
    pub fn new(spectrum: &Spectrum, lambda_min: i32, lambda_max: i32) -> DenselySampled {
        // PAPERDOC This is a fun area where idiomatic rust code (map -> collect) is arguably cleaner
        // than similar C++ code (allowing e.g. const correctness).
        let values: Vec<Float> = (lambda_min..lambda_max)
            .map(|lambda: i32| spectrum.get(lambda as Float))
            .collect();
        DenselySampled {
            lambda_min,
            lambda_max,
            values,
        }
    }
}

impl SpectrumI for DenselySampled {
    fn get(&self, lambda: Float) -> Float {
        let offset = lambda as i32 - self.lambda_min;
        if offset < 0 || offset > self.values.len() as i32 {
            return 0.0;
        }
        self.values[offset as usize]
    }

    fn max_value(&self) -> Float {
        // PAPERDOC This is a vector of Floats and we must find the maximum.
        // But what about NaN? In C++, this is a very easy case to miss.
        // In Rust, the compiler makes you handle NaN or else it won't compile.
        // e.g. using values.iter().max() will complain that f32 does not satisfy
        // the Ord constraint (due to NaN). We can instead fold() and clearly handle the
        // NaN case, with a panic!() in this case.
        let max = self.values.iter().fold(Float::NAN, |a, &b| a.max(b));
        if max.is_nan() {
            panic!("Empty or NaN-filled Densely Sampled Spectrum!")
        }
        max
    }
}

pub struct PiecewiseLinear {
    lambdas: Vec<Float>,
    values: Vec<Float>,
}

impl PiecewiseLinear {
    /// Creates a piecewise linear spectrum from associated lambdas and values;
    /// these slices must be sorted.
    pub fn new<const N: usize>(lambdas: &[Float; N], values: &[Float; N]) -> PiecewiseLinear {
        // PAPERDOC I think this is a good way to ensure lambdas.len() == values.len() at compile-time,
        // rather than a runtime check as in PBRTv4. I'll need to see how it fairs in practice.
        // Note that we're taking a slice, which is read-only, so we perform a copy here.
        // This follows PBRT, which also makes a copy; could it be beneficial to take ownership instead?
        let mut l = vec![0.0; lambdas.len()];
        l.copy_from_slice(lambdas);
        let mut v = vec![0.0; values.len()];
        v.copy_from_slice(values);
        // Check that they are sorted
        assert!(l.windows(2).all(|p| p[0] <= p[1]));
        assert!(v.windows(2).all(|p| p[0] <= p[1]));
        PiecewiseLinear {
            lambdas: l,
            values: v,
        }
    }
}

impl SpectrumI for PiecewiseLinear {
    fn get(&self, lambda: Float) -> Float {
        if self.lambdas.is_empty()
            || lambda < *self.lambdas.first().unwrap()
            || lambda > *self.lambdas.last().unwrap()
        {
            return 0.0;
        }

        // PAPERDOC I would contend that this is a much cleaner and simpler approach
        // than PBRTv4's FindInterval() approach.
        let interval: Option<(usize, usize)> = (0..self.lambdas.len())
            .tuples()
            .find(|(a, b)| -> bool { self.lambdas[*a] <= lambda && lambda < self.lambdas[*b] });
        let interval = interval.expect("Interval not found; edge cases should be handled above.");

        let t = (lambda - self.lambdas[interval.0])
            / (self.lambdas[interval.1] - self.lambdas[interval.0]);
        lerp(t, &self.values[interval.0], &self.values[interval.1])
    }

    fn max_value(&self) -> Float {
        let max = self.values.iter().fold(Float::NAN, |a, &b| a.max(b));
        if max.is_nan() {
            panic!("Empty or NaN-filled Spectrum!")
        }
        max
    }
}

/// Normalized blackbody spectrum where the maximum value at any wavelength is 1.
pub struct Blackbody {
    /// Temperature K
    t: Float,
    /// Normalization factor s.t. the maximum value is 1.0.
    normalization_factor: Float,
}

impl Blackbody {
    pub fn new(t: Float) -> Blackbody {
        let lambda_max = 2.8977721e-3 / t; // Wien's displacement law
        let normalization_factor = 1.0 / Blackbody::blackbody(lambda_max * 1e9, t);
        Blackbody {
            t,
            normalization_factor,
        }
    }

    /// The emitted radiance for blackbody at wavelength lambda (nanometers) at temperature (kelvin).
    fn blackbody(lambda: Float, temperature: Float) -> Float {
        if temperature < 0.0 {
            return 0.0;
        }
        let c: Float = 299792458.0;
        let h: Float = 6.62606957e-34;
        let kb: Float = 1.3806488e-23;
        // Convert to meters
        let l = lambda * 1e-9;
        // TODO consider Exponentiation by squaring for powi call, and fastexp for exp().
        let le =
            (2.0 * h * c * c) / l.powi(5) * (Float::exp((h * c) / (l * kb * temperature)) - 1.0);
        debug_assert!(!le.is_nan());
        le
    }
}

impl SpectrumI for Blackbody {
    fn get(&self, lambda: Float) -> Float {
        Blackbody::blackbody(lambda, self.t) * self.normalization_factor
    }

    fn max_value(&self) -> Float {
        1.0
    }
}

mod tests {
    use crate::spectra::{spectrum::SpectrumI, Constant, Spectrum};

    #[test]
    fn get_constant() {
        let c = Constant::new(5.0);
        assert_eq!(5.0, c.get(999.0));
        let spectrum = Spectrum::Constant(c);
        assert_eq!(5.0, spectrum.get(999.0))
    }

    // TODO test piecewiselinear ctor and get(). It's a bit invovled so we should test it.
}
