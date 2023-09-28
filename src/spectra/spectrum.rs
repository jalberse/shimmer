use std::{rc::Rc, sync::Arc};

use crate::{
    color::{RgbSigmoidPolynomial, RGB},
    colorspace::RgbColorSpace,
    math::lerp,
    spectra::cie::{CIE, CIE_Y_INTEGRAL},
    Float,
};

use itertools::Itertools;
use once_cell::sync::Lazy;

use super::{
    sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths, NamedSpectrum,
    NUM_SPECTRUM_SAMPLES,
};

/// Nanometers. Minimum of visible range of light.
pub const LAMBDA_MIN: Float = 360.0;
/// Nanometers. Maximum of visible range of light.
pub const LAMBDA_MAX: Float = 830.0;

pub trait SpectrumI {
    fn get(&self, lambda: Float) -> Float;

    /// Returns a bound on the maximum value of the spectral distribution over its wavelength range.
    /// The primary utility here is computing bounds on the power emitted by light sources so that
    /// lights can be sampled according to their expected contriibution to illumination in the scene.
    fn max_value(&self) -> Float;

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum;
}

#[derive(Debug, PartialEq)]
pub enum Spectrum {
    Constant(Constant),
    DenselySampled(DenselySampled),
    PiecewiseLinear(PiecewiseLinear),
    Blackbody(Blackbody),
    RgbAlbedoSpectrum(RgbAlbedoSpectrum),
    RgbUnboundedSpectrum(RgbUnboundedSpectrum),
    RgbIlluminantSpectrum(RgbIlluminantSpectrum),
}

impl SpectrumI for Spectrum {
    // PAPERDOC - this will be a great example against PBRTv4's call with TaggedPointer
    // to achieve static dispatch. If this is too verbose for someone's taste, then
    // there are crates like enum_dispatch which could generate this automatically from
    // the SpectrumI trait.

    /// Gets the value of the spectral distribution at wavelength lambda
    fn get(&self, lambda: Float) -> Float {
        match self {
            Spectrum::Constant(c) => c.get(lambda),
            Spectrum::DenselySampled(s) => s.get(lambda),
            Spectrum::PiecewiseLinear(s) => s.get(lambda),
            Spectrum::Blackbody(s) => s.get(lambda),
            Spectrum::RgbAlbedoSpectrum(s) => s.get(lambda),
            Spectrum::RgbUnboundedSpectrum(s) => s.get(lambda),
            Spectrum::RgbIlluminantSpectrum(s) => s.get(lambda),
        }
    }

    fn max_value(&self) -> Float {
        match self {
            Spectrum::Constant(c) => c.max_value(),
            Spectrum::DenselySampled(s) => s.max_value(),
            Spectrum::PiecewiseLinear(s) => s.max_value(),
            Spectrum::Blackbody(s) => s.max_value(),
            Spectrum::RgbAlbedoSpectrum(s) => s.max_value(),
            Spectrum::RgbUnboundedSpectrum(s) => s.max_value(),
            Spectrum::RgbIlluminantSpectrum(s) => s.max_value(),
        }
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Spectrum::Constant(c) => c.sample(lambda),
            Spectrum::DenselySampled(s) => s.sample(lambda),
            Spectrum::PiecewiseLinear(s) => s.sample(lambda),
            Spectrum::Blackbody(s) => s.sample(lambda),
            Spectrum::RgbAlbedoSpectrum(s) => s.sample(lambda),
            Spectrum::RgbUnboundedSpectrum(s) => s.sample(lambda),
            Spectrum::RgbIlluminantSpectrum(s) => s.sample(lambda),
        }
    }
}

impl Spectrum {
    /// Gets a lazily-evaluated named spectrum.
    pub fn get_named_spectrum(spectrum: NamedSpectrum) -> &'static Spectrum {
        // TODO Maybe for this, rather than using once_cell, could we use static RwLock?
        // https://www.cs.brandeis.edu/~cs146a/rust/doc-02-21-2015/std/sync/struct.StaticRwLock.html
        // Since RwLock and Mutex are now const functions, we can define them as static during compile time,
        // as long as the code involved can also be const.
        // This way we don't need lazy evaluation, we can do it at compile time.
        // Plus, I think it might be more efficient than the match, but that's not so important.

        // PAPERDOC Embedded spectral data 4.5.3.
        // Instead of a map on string keys (which requires evaluating the hash),
        // we use an Enum with the names and match on it and return ref to static-lifetimed
        // spectra. These are thread-safe read-only single-instance objects.
        match spectrum {
            NamedSpectrum::StdIllumD65 => Lazy::force(&super::named_spectrum::STD_ILLUM_D65),
            NamedSpectrum::IllumAcesD60 => Lazy::force(&super::named_spectrum::ILLUM_ACES_D60),
            NamedSpectrum::GlassBk7 => Lazy::force(&super::named_spectrum::GLASS_BK7_ETA),
            NamedSpectrum::GlassBaf10 => Lazy::force(&super::named_spectrum::GLASS_BAF10_ETA),
        }
    }

    /// Gets a CIE spectrum
    pub fn get_cie(cie: CIE) -> &'static Spectrum {
        match cie {
            CIE::X => Lazy::force(&super::cie::X),
            CIE::Y => Lazy::force(&super::cie::Y),
            CIE::Z => Lazy::force(&super::cie::Z),
        }
    }
}

#[derive(Debug, PartialEq)]
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

    fn sample(&self, _lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::from_const(self.c)
    }
}

#[derive(Debug, PartialEq)]
pub struct DenselySampled {
    lambda_min: i32,
    lambda_max: i32,
    values: Vec<Float>,
}

impl DenselySampled {
    /// Samples from the provided spectrum to create a DenselySampled spectrum
    pub fn new(spectrum: &Spectrum) -> DenselySampled {
        Self::new_range(spectrum, LAMBDA_MIN as i32, LAMBDA_MAX as i32)
    }

    /// Samples from the provided spectrum to create a DenselySampled spectrum
    pub fn new_range(spectrum: &Spectrum, lambda_min: i32, lambda_max: i32) -> DenselySampled {
        // PAPERDOC This is a fun area where idiomatic rust code (map -> collect) is arguably cleaner
        // than similar C++ code (allowing e.g. const correctness). Of course, C++ can accomplish similar.
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

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            let offset: i32 = lambda[i].round() as i32 - self.lambda_min;
            s[i] = if offset < 0 || offset >= self.values.len() as i32 {
                0.0
            } else {
                self.values[offset as usize]
            }
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq)]
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
        // I think you can do this in C++ too though.

        // TODO This follows PBRT, which also makes a copy; could it be beneficial to take ownership instead?
        // Note that we're taking a slice, which is read-only, so we perform a copy here.
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

    /// Gets a PiecewiseLinear spectrum from an interleaved samples array;
    /// i.e. the wavelength and value for each sample i are at (i * 2) and (i * 2) + 1.
    ///
    /// * N: The length of the interleaved samples array. 2 * S.
    /// * S: The number of samples; 1/2 of N.
    pub fn from_interleaved<const N: usize, const S: usize>(
        samples: &[Float; N],
        normalize: bool,
    ) -> PiecewiseLinear {
        assert_eq!(N, S / 2);
        assert_eq!(0, samples.len() % 2);
        let n = samples.len() / 2;
        let mut lambda: Vec<Float> = Vec::new();
        let mut v: Vec<Float> = Vec::new();

        // Extend samples to cover range of visible wavelengths if needed.
        // Note since we're making a piecewise spectrum, we only need one more entry.
        if samples[0] > LAMBDA_MIN {
            lambda.push(LAMBDA_MIN - 1.0);
            v.push(samples[1]);
        }
        for i in 0..n {
            lambda.push(samples[2 * i]);
            v.push(samples[2 * i + 1]);
            if i > 0 {
                // Check ordering
                assert!(*lambda.last().unwrap() > lambda[lambda.len() - 2]);
            }
        }
        // Extend to cover maximum wavelength if necessary.
        if *lambda.last().unwrap() < LAMBDA_MAX {
            lambda.push(LAMBDA_MAX + 1.0);
            v.push(*v.last().unwrap());
        }

        // PAPERDOC Interesting callsite as we enforce the slices must have the same length N.
        // Note that N must propagate up through this function, then; but that should be okay for our case.
        // This is enabled by const generics, which were added recently (in 2022?) to Rust.
        // Note that the N, S arrangement is a bit awkward - but that could be avoided
        // by not interleaving the data, which would honestly be better. But for convenience,
        // let's do this for now.
        // TODO switch off of interleaved data structures so we can just use one value N.
        // This means changing the named spectra to have separate lambda and value arrays.
        let mut spectrum = PiecewiseLinear::new::<S>(
            lambda.as_slice().try_into().expect("Invalid length"),
            v.as_slice().try_into().expect("Invalid length"),
        );

        if normalize {
            spectrum.scale(
                CIE_Y_INTEGRAL
                    / inner_product::<PiecewiseLinear, Spectrum>(
                        &spectrum,
                        Spectrum::get_cie(CIE::Y),
                    ),
            );
        }

        spectrum
    }

    pub fn scale(&mut self, s: Float) {
        for v in &mut self.values {
            *v *= s;
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

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.get(lambda[i]);
        }
        SampledSpectrum::new(s)
    }
}

/// Normalized blackbody spectrum where the maximum value at any wavelength is 1.
#[derive(Debug, PartialEq)]
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

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = Blackbody::blackbody(lambda[i], self.t) * self.normalization_factor;
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq)]
pub struct RgbAlbedoSpectrum {
    rsp: RgbSigmoidPolynomial,
}

impl RgbAlbedoSpectrum {
    pub fn new(cs: &RgbColorSpace, rgb: &RGB) -> RgbAlbedoSpectrum {
        debug_assert!(Float::max(Float::max(rgb.r, rgb.g), rgb.b) <= 1.0);
        debug_assert!(Float::max(Float::min(rgb.r, rgb.g), rgb.b) >= 0.0);
        RgbAlbedoSpectrum {
            rsp: cs.to_rgb_coeffs(rgb),
        }
    }
}

impl SpectrumI for RgbAlbedoSpectrum {
    fn get(&self, lambda: Float) -> Float {
        self.rsp.get(lambda)
    }

    fn max_value(&self) -> Float {
        self.rsp.max_value()
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.rsp.get(lambda[i]);
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq)]
pub struct RgbUnboundedSpectrum {
    scale: Float,
    rsp: RgbSigmoidPolynomial,
}

impl RgbUnboundedSpectrum {
    pub fn new(cs: &RgbColorSpace, rgb: &RGB) -> RgbUnboundedSpectrum {
        let m = Float::max(Float::max(rgb.r, rgb.g), rgb.b);
        let scale = 2.0 * m;
        let rsp = if scale != 0.0 {
            cs.to_rgb_coeffs(&(rgb / scale))
        } else {
            cs.to_rgb_coeffs(&RGB::new(0.0, 0.0, 0.0))
        };
        RgbUnboundedSpectrum { scale, rsp }
    }
}

impl SpectrumI for RgbUnboundedSpectrum {
    fn get(&self, lambda: Float) -> Float {
        self.scale * self.rsp.get(lambda)
    }

    fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value()
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.scale * self.rsp.get(lambda[i]);
        }
        SampledSpectrum::new(s)
    }
}

#[derive(Debug, PartialEq)]
pub struct RgbIlluminantSpectrum {
    scale: Float,
    rsp: RgbSigmoidPolynomial,
    illuminant: Arc<DenselySampled>,
}

impl RgbIlluminantSpectrum {
    pub fn new(cs: &RgbColorSpace, rgb: &RGB) -> RgbIlluminantSpectrum {
        let m = Float::max(Float::max(rgb.r, rgb.g), rgb.b);
        let scale = 2.0 * m;
        let rsp = if scale != 0.0 {
            cs.to_rgb_coeffs(&(rgb / scale))
        } else {
            cs.to_rgb_coeffs(&RGB::new(0.0, 0.0, 0.0))
        };
        RgbIlluminantSpectrum {
            scale,
            rsp,
            illuminant: cs.illuminant.clone(),
        }
    }
}

impl SpectrumI for RgbIlluminantSpectrum {
    fn get(&self, lambda: Float) -> Float {
        self.scale * self.rsp.get(lambda) * self.illuminant.get(lambda)
    }

    fn max_value(&self) -> Float {
        self.scale * self.rsp.max_value() * self.illuminant.max_value()
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = self.scale * self.rsp.get(lambda[i]);
        }
        SampledSpectrum::new(s) * self.illuminant.sample(lambda)
    }
}

pub fn inner_product<T: SpectrumI, G: SpectrumI>(a: &T, b: &G) -> Float {
    let mut integral = 0.0;
    for lambda in (LAMBDA_MIN as i32)..(LAMBDA_MAX as i32) {
        integral += a.get(lambda as Float) * b.get(lambda as Float);
    }
    integral
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
