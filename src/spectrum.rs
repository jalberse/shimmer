use crate::Float;

/// Nanometers. Minimum of visible range of light.
const LAMBDA_MIN: Float = 360.0;
/// Nanometers. Maximum of visible range of light.
const LAMBDA_MAX: Float = 830.0;

// TODO Embedded spectral data 4.5.3.

/// The emitted radiance for blackbody at wavelength lambda (nanometers) at temperature (kelvin).
pub fn blackbody(lambda: Float, temperature: Float) -> Float {
    if temperature < 0.0 {
        return 0.0;
    }
    let c: Float = 299792458.0;
    let h: Float = 6.62606957e-34;
    let kb: Float = 1.3806488e-23;
    // Convert to meters
    let l = lambda * 1e-9;
    // TODO consider Exponentiation by squaring for powi call, and fastexp for exp().
    let le = (2.0 * h * c * c) / l.powi(5) * (Float::exp((h * c) / (l * kb * temperature)) - 1.0);
    debug_assert!(!le.is_nan());
    le
}

trait SpectrumI {
    fn get(&self, lambda: Float) -> Float;

    fn max_value(&self) -> Float;
}

enum Spectrum {
    Constant(Constant),
    DenselySampled {
        lambda_min: i32,
        lambda_max: i32,
        values: Vec<Float>,
    },
    /// Normalized blackbody spectrum where the maximum value at any wavelength is 1.
    Blackbody {
        /// Temperature K
        t: Float,
        normalization_factor: Float,
    },
}

impl Spectrum {
    /// Samples from the provided spectrum to create a DenselySampled spectrum
    pub fn densley_sampled(spectrum: &Spectrum, lambda_min: i32, lambda_max: i32) -> Spectrum {
        // PAPERDOC This is a fun area where idiomatic rust code (map -> collect) is arguably cleaner
        // than similar C++ code (allowing e.g. const correctness).
        let values: Vec<Float> = (lambda_min..lambda_max)
            .map(|lambda: i32| spectrum.get(lambda as Float))
            .collect();
        Spectrum::DenselySampled {
            lambda_min,
            lambda_max,
            values,
        }
    }

    pub fn blackbody(t: Float) -> Spectrum {
        let lambda_max = 2.8977721e-3 / t; // Wien's displacement law
        let normalization_factor = 1.0 / blackbody(lambda_max * 1e9, t);
        Spectrum::Blackbody {
            t,
            normalization_factor,
        }
    }

    /// Gets the value of the spectral distribution at wavelength lambda
    pub fn get(&self, lambda: Float) -> Float {
        // PAPERDOC - this will be a great example against PBRTv4's call with TaggedPointer
        //  to achieve static dispatch.
        match self {
            Spectrum::Constant(c) => c.get(lambda),
            Spectrum::DenselySampled {
                lambda_min,
                lambda_max: _,
                values,
            } => {
                let offset = lambda as i32 - lambda_min;
                if offset < 0 || offset > values.len() as i32 {
                    return 0.0;
                }
                values[offset as usize]
            }
            Spectrum::Blackbody {
                t,
                normalization_factor,
            } => blackbody(lambda, *t) * normalization_factor,
        }
    }

    /// Returns a bound on the maximum value of the psectral distribution over its wavelength range.
    /// The primary utility here is computing bounds on the power emitted by light sources so that
    /// lights can be sampled according to their expected contriibution to illumination in the scene.
    pub fn max_value(&self) -> Float {
        match self {
            Spectrum::Constant(c) => c.max_value(),
            Spectrum::DenselySampled {
                lambda_min: _,
                lambda_max: _,
                values,
            } => {
                // PAPERDOC This is a vector of Floats and we must find the minimum.
                // But what about NaN? In C++, this is a very easy case to miss.
                // In Rust, the compiler makes you handle NaN or else it won't compile.
                // e.g. using values.iter().min() will complain that f32 does not satisfy
                // the Ord constraint (due to NaN). In this case, we opt to fold() instead,
                // and use f32::min() which itself handles the NaN case. This is a bit more verbose,
                // but that's a worthwhile trade-off to ensure NaNs are handled correctly at compile-time,
                // rather tracking down propagated NaNs at runtime.

                // TODO actually fix, maybe need neg_infinity, was using min() before.
                // Though a list of neg_infinity would technically have its max there...
                // Maybe we need an option instead.
                let max = values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                if max == Float::NEG_INFINITY {
                    panic!("Empty or NaN-filled Densely Sampled Spectrum!")
                }
                max
            }
            Spectrum::Blackbody {
                t: _,
                normalization_factor: _,
            } => 1.0,
        }
    }
}

struct Constant {
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

mod tests {
    use crate::spectrum::{Constant, SpectrumI};

    use super::Spectrum;

    #[test]
    fn get_constant() {
        let c = Constant::new(5.0);
        assert_eq!(5.0, c.get(999.0));
        let spectrum = Spectrum::Constant(c);
        assert_eq!(5.0, spectrum.get(999.0))
    }
}
