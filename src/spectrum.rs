use crate::Float;

/// Nanometers. Minimum of visible range of light.
const LAMBDA_MIN: Float = 360.0;
/// Nanometers. Maximum of visible range of light.
const LAMBDA_MAX: Float = 830.0;

// TODO blackbody function

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
    // PAPERDOC - Note that I'm using the Rust intrinsic functions for powi and exp here.
    // That differs from PBRT, which uses exponentiation by squaring and a fastexp() algorithm
    // (pages 1034, 1036).
    // Intrinsics in Rust leave the development to LLVM, and thus hardware optimization.
    // I typically wouldn't be against that - but exponentiation by squaring does take advantage
    // of the power being known at compile-time, perhaps allowing LLVM to do some magic.
    // This might be something I'd want to write 2 quick demos to test and compare the speeds.
    // Are these kinds of micro-optimizations worth it in Rust? In C++?
    // My money is on "neither, until you have the data to show it, and even then be skeptical of your data" 
    // TODO consider Exponentiation by squaring for powi call, and fastexp for exp().
    // 
    let Le = (2.0 * h * c * c) / l.powi(5) * (Float::exp((h * c) / (l * kb * temperature)) - 1.0)
    debug_assert!(!Le.is_nan());
    Le   
}

enum Spectrum {
    Constant {
        c: Float,
    },
    DenselySampled {
        lambda_min: i32,
        lambda_max: i32,
        values: Vec<Float>,
    },
}

impl Spectrum {
    pub fn constant(c: Float) -> Spectrum {
        Spectrum::Constant { c }
    }

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

    /// Gets the value of the spectral distribution at wavelength lambda
    pub fn get(&self, lambda: Float) -> Float {
        // PAPERDOC - this will be a great example against PBRTv4's call with TaggedPointer
        //  to achieve static dispatch.
        match self {
            Spectrum::Constant { c } => *c,
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
        }
    }

    /// Returns a bound on the maximum value of the psectral distribution over its wavelength range.
    /// The primary utility here is computing bounds on the power emitted by light sources so that
    /// lights can be sampled according to their expected contriibution to illumination in the scene.
    pub fn max_value(&self) -> Float {
        match self {
            Spectrum::Constant { c } => *c,
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
                let min = values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
                if min == Float::INFINITY {
                    panic!("Empty or NaN-filled Densely Sampled Spectrum!")
                }
                min
            }
        }
    }
}

mod tests {
    use super::Spectrum;

    #[test]
    fn get_constant() {
        let c = Spectrum::constant(5.0);
        assert_eq!(5.0, c.get(999.0))
    }
}
