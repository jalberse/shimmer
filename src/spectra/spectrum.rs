use std::sync::Arc;

use crate::{
    color::{RgbSigmoidPolynomial, RGB},
    colorspace::RgbColorSpace,
    math::{self, lerp},
    spectra::cie::{CIE, CIE_Y_INTEGRAL},
    Float,
};

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
    Constant(ConstantSpectrum),
    DenselySampled(DenselySampledSpectrum),
    PiecewiseLinear(PiecewiseLinearSpectrum),
    Blackbody(BlackbodySpectrum),
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
pub struct ConstantSpectrum {
    c: Float,
}

impl ConstantSpectrum {
    pub fn new(c: Float) -> Self {
        Self { c }
    }
}

impl SpectrumI for ConstantSpectrum {
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

#[derive(Debug, Clone, PartialEq)]
pub struct DenselySampledSpectrum {
    lambda_min: i32,
    lambda_max: i32,
    values: Vec<Float>,
}

impl DenselySampledSpectrum {
    /// Samples from the provided spectrum to create a DenselySampled spectrum
    pub fn new(spectrum: &Spectrum) -> DenselySampledSpectrum {
        Self::new_range(spectrum, LAMBDA_MIN as i32, LAMBDA_MAX as i32)
    }

    /// Samples from the provided spectrum to create a DenselySampled spectrum
    pub fn new_range(
        spectrum: &Spectrum,
        lambda_min: i32,
        lambda_max: i32,
    ) -> DenselySampledSpectrum {
        // PAPERDOC This is a fun area where idiomatic rust code (map -> collect) is arguably cleaner
        // than similar C++ code (allowing e.g. const correctness). Of course, C++ can accomplish similar.
        let values: Vec<Float> = (lambda_min..=lambda_max)
            .map(|lambda: i32| spectrum.get(lambda as Float))
            .collect();
        DenselySampledSpectrum {
            lambda_min,
            lambda_max,
            values,
        }
    }

    pub fn sample_function(
        f: impl Fn(Float) -> Float,
        lambda_min: usize,
        lambda_max: usize,
    ) -> DenselySampledSpectrum {
        let mut values = vec![0.0; lambda_max - lambda_min + 1];
        for lambda in lambda_min..=lambda_max {
            values[lambda - lambda_min] = f(lambda as Float);
        }
        DenselySampledSpectrum {
            values,
            lambda_min: lambda_min as i32,
            lambda_max: lambda_max as i32,
        }
    }
}

impl SpectrumI for DenselySampledSpectrum {
    fn get(&self, lambda: Float) -> Float {
        let offset = lambda as i32 - self.lambda_min;
        if offset < 0 || offset >= self.values.len() as i32 {
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
pub struct PiecewiseLinearSpectrum {
    lambdas: Vec<Float>,
    values: Vec<Float>,
}

impl PiecewiseLinearSpectrum {
    /// Creates a piecewise linear spectrum from associated lambdas and values;
    /// these slices must be sorted.
    pub fn new<const N: usize>(
        lambdas: &[Float; N],
        values: &[Float; N],
    ) -> PiecewiseLinearSpectrum {
        // PAPERDOC I think this is a good way to ensure lambdas.len() == values.len() at compile-time,
        // rather than a runtime check as in PBRTv4. I'll need to see how it fairs in practice.
        // I think you can do this in C++ too though.

        // TODO This follows PBRT, which also makes a copy; could it be beneficial to take ownership instead?
        // Note that we're taking a slice, which is read-only, so we perform a copy here.
        let mut l = vec![0.0; lambdas.len()];
        l.copy_from_slice(lambdas);
        let mut v = vec![0.0; values.len()];
        v.copy_from_slice(values);
        // Check that lambdas are sorted
        assert!(l.windows(2).all(|p| p[0] <= p[1]));
        PiecewiseLinearSpectrum {
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
    ) -> PiecewiseLinearSpectrum {
        assert_eq!(N / 2, S);
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
        let mut spectrum = PiecewiseLinearSpectrum::new::<S>(
            lambda.as_slice().try_into().expect("Invalid length"),
            v.as_slice().try_into().expect("Invalid length"),
        );

        if normalize {
            spectrum.scale(
                CIE_Y_INTEGRAL
                    / inner_product::<PiecewiseLinearSpectrum, Spectrum>(
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

impl SpectrumI for PiecewiseLinearSpectrum {
    fn get(&self, lambda: Float) -> Float {
        if self.lambdas.is_empty()
            || lambda < *self.lambdas.first().unwrap()
            || lambda > *self.lambdas.last().unwrap()
        {
            return 0.0;
        }

        let o = math::find_interval(self.lambdas.len(), |i| -> bool {
            self.lambdas[i] <= lambda
        });
        debug_assert!(lambda >= self.lambdas[o] && lambda <= self.lambdas[o + 1]);

        let t = (lambda - self.lambdas[o]) / (self.lambdas[o + 1] - self.lambdas[o]);
        lerp(t, &self.values[o], &self.values[o + 1])
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
pub struct BlackbodySpectrum {
    /// Temperature K
    t: Float,
    /// Normalization factor s.t. the maximum value is 1.0.
    normalization_factor: Float,
}

impl BlackbodySpectrum {
    pub fn new(t: Float) -> BlackbodySpectrum {
        let lambda_max = 2.8977721e-3 / t; // Wien's displacement law
        let normalization_factor = 1.0 / BlackbodySpectrum::blackbody(lambda_max * 1e9, t);
        BlackbodySpectrum {
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
            (2.0 * h * c * c) / (l.powi(5) * (Float::exp((h * c) / (l * kb * temperature)) - 1.0));
        debug_assert!(!le.is_nan());
        le
    }
}

impl SpectrumI for BlackbodySpectrum {
    fn get(&self, lambda: Float) -> Float {
        BlackbodySpectrum::blackbody(lambda, self.t) * self.normalization_factor
    }

    fn max_value(&self) -> Float {
        1.0
    }

    fn sample(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
        for i in 0..NUM_SPECTRUM_SAMPLES {
            s[i] = BlackbodySpectrum::blackbody(lambda[i], self.t) * self.normalization_factor;
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
    illuminant: Arc<Spectrum>,
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
    for lambda in (LAMBDA_MIN as i32)..=(LAMBDA_MAX as i32) {
        integral += a.get(lambda as Float) * b.get(lambda as Float);
    }
    integral
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};

    use crate::{
        color::RGB,
        colorspace::{NamedColorSpace, RgbColorSpace},
        math::lerp,
        sampling::{sample_visible_wavelengths, visible_wavelengths_pdf},
        spectra::{
            sampled_spectrum::SampledSpectrum,
            sampled_wavelengths::SampledWavelengths,
            spectrum::{RgbAlbedoSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum, SpectrumI},
            BlackbodySpectrum, ConstantSpectrum, Spectrum, CIE, CIE_Y_INTEGRAL,
        },
        Float,
    };

    use super::{DenselySampledSpectrum, PiecewiseLinearSpectrum, LAMBDA_MAX, LAMBDA_MIN};

    #[test]
    fn get_constant() {
        let c = ConstantSpectrum::new(5.0);
        assert_eq!(5.0, c.get(999.0));
        let spectrum = Spectrum::Constant(c);
        assert_eq!(5.0, spectrum.get(999.0))
    }

    #[test]
    fn blackbody() {
        let err =
            |val: Float, reference: Float| -> Float { Float::abs(val - reference) / reference };

        // Planck's law.
        // A few values via
        // http://www.spectralcalc.com/blackbody_calculator/blackbody.php
        // lambda, T, expected radiance
        let v: [[Float; 3]; 4] = [
            [483.0, 6000.0, 3.1849e13],
            [600.0, 6000.0, 2.86772e13],
            [500.0, 3700.0, 1.59845e12],
            [600.0, 4500.0, 7.46497e12],
        ];
        for i in 0..4 {
            let lambda = v[i][0];
            let t = v[i][1];
            let le_expected = v[i][2];
            assert!(err(BlackbodySpectrum::blackbody(lambda, t), le_expected) < 0.001);
        }

        // Use Wien's displacement law to compute maximum wavelength for a few
        // temperatures, then confirm that the value returned by Blackbody() is
        // consistent with this.
        for t in [2700.0, 3000.0, 4500.0, 5600.0, 6000.0] {
            let lambda_max = 2.8977721e-3 / t * 1e9;
            let lambda: [Float; 3] = [0.99 * lambda_max, lambda_max, 1.01 * lambda_max];
            let result0 = BlackbodySpectrum::blackbody(lambda[0], t);
            let result1 = BlackbodySpectrum::blackbody(lambda[1], t);
            let result2 = BlackbodySpectrum::blackbody(lambda[2], t);
            assert!(result0 < result1);
            assert!(result1 > result2);
        }
    }

    #[test]
    fn xyz_integral() {
        // Make sure the integral of all matching function sample values is
        // basically one in x, y, and z.
        let mut xx: Float = 0.0;
        let mut yy: Float = 0.0;
        let mut zz: Float = 0.0;

        for lambda in (LAMBDA_MIN as i32)..=(LAMBDA_MAX as i32) {
            xx += Spectrum::get_cie(crate::spectra::CIE::X).get(lambda as Float);
            yy += Spectrum::get_cie(crate::spectra::CIE::Y).get(lambda as Float);
            zz += Spectrum::get_cie(crate::spectra::CIE::Z).get(lambda as Float);
        }
        let cie_y_integral = 106.856895;
        xx /= cie_y_integral;
        yy /= cie_y_integral;
        zz /= cie_y_integral;

        assert_approx_eq!(Float, 1.0, xx, epsilon = 0.005);
        assert_approx_eq!(Float, 1.0, yy, epsilon = 0.005);
        assert_approx_eq!(Float, 1.0, zz, epsilon = 0.005);
    }

    #[test]
    fn xyz_constant_spectrum() {
        let mut xyz_sum: [Float; 3] = [0.0; 3];
        let n = 100;
        let between = Uniform::from(0.0..1.0);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..n {
            let rand = between.sample(&mut rng);
            let lambda = &&SampledWavelengths::sample_uniform(rand);
            let xyz = SampledSpectrum::from_const(1.0).to_xyz(lambda);
            for c in 0..3 {
                xyz_sum[c] += xyz[c];
            }
        }
        for c in 0..3 {
            xyz_sum[c] /= n as Float;
        }

        // The epsilon is a bit high here because we're using a uniform sample
        // rather than a stratified sample, with kind-of-low sample size.
        // But this test DID catch a bug and I'm now reasonably certain this
        // is correct.
        assert_approx_eq!(Float, 1.0, xyz_sum[0], epsilon = 0.1);
        assert_approx_eq!(Float, 1.0, xyz_sum[1], epsilon = 0.1);
        assert_approx_eq!(Float, 1.0, xyz_sum[2], epsilon = 0.1);
    }

    #[test]
    fn piecewise_linear_ctor() {
        let lambdas = [0.0, 5.0, 10.0, 100.0];
        let values = [0.0, 10.0, 20.0, 200.0];
        let spectrum = PiecewiseLinearSpectrum::new(&lambdas, &values);
        assert_eq!(
            [0.0, 5.0, 10.0, 100.0].as_slice(),
            spectrum.lambdas.as_slice()
        );
        assert_eq!(
            [0.0, 10.0, 20.0, 200.0].as_slice(),
            spectrum.values.as_slice()
        );
    }

    #[test]
    fn piecewise_linear_get() {
        let lambdas = [0.0, 5.0, 10.0, 100.0];
        let values = [0.0, 10.0, 20.0, 200.0];
        let spectrum = PiecewiseLinearSpectrum::new(&lambdas, &values);
        assert_eq!(5.0, spectrum.get(2.5));
        assert_eq!(15.0, spectrum.get(7.5));
        assert_eq!(110.0, spectrum.get(55.0));
        assert_eq!(0.0, spectrum.get(99999.0));
        assert_eq!(0.0, spectrum.get(0.0));
    }

    #[test]
    fn densely_sampled_basic() {
        let lambdas = [360.0, 820.0];
        let values = [0.0, 100.0];
        let spectrum = PiecewiseLinearSpectrum::new(&lambdas, &values);
        let spectrum = DenselySampledSpectrum::new(&Spectrum::PiecewiseLinear(spectrum));

        assert_approx_eq!(Float, 0.0, spectrum.get(360.0));
        assert_approx_eq!(Float, 100.0, spectrum.get(820.0));
        assert_approx_eq!(Float, 50.0, spectrum.get(590.0));
    }

    #[test]
    fn spectrum_max_value() {
        assert_eq!(2.5, ConstantSpectrum::new(2.5).max_value());

        assert_eq!(
            10.1,
            PiecewiseLinearSpectrum::new(
                &[300.0, 380.0, 510.0, 620.0, 700.0],
                &[1.5, 2.6, 10.1, 5.3, 7.7]
            )
            .max_value()
        );

        assert_approx_eq!(
            Float,
            1.0,
            BlackbodySpectrum::new(5000.0).max_value(),
            epsilon = 0.0001
        );

        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::SRGB);
        for _ in 0..20 {
            let rgb = RGB::new(
                between.sample(&mut rng),
                between.sample(&mut rng),
                between.sample(&mut rng),
            );

            let spectrum = RgbAlbedoSpectrum::new(cs, &rgb);
            let max = spectrum.max_value() * 1.00001;
            for lambda in 360..=820 {
                assert!(spectrum.get(lambda as Float) < max)
            }

            let spectrum = RgbUnboundedSpectrum::new(cs, &(&rgb * 10.0));
            let max = spectrum.max_value() * 1.00001;
            for lambda in 360..=820 {
                assert!(spectrum.get(lambda as Float) < max)
            }

            let spectrum = RgbIlluminantSpectrum::new(cs, &rgb);
            let max = spectrum.max_value() * 1.00001;
            for lambda in 360..=820 {
                assert!(spectrum.get(lambda as Float) < max)
            }
        }
    }

    #[test]
    fn sampling_pdf_y() {
        let mut ysum = 0.0;
        let mut rng = StdRng::seed_from_u64(1);
        let between = Uniform::from(0.0..1.0);
        let n = 10000;
        for _ in 0..n {
            let u = between.sample(&mut rng);
            let lambda = sample_visible_wavelengths(u);
            let pdf = visible_wavelengths_pdf(lambda);
            if pdf > 0.0 {
                ysum += Spectrum::get_cie(crate::spectra::CIE::Y).get(lambda) / pdf;
            }
        }
        let y_integral = ysum / n as Float;
        // Allow a sort-of-large epsilon since we're  not using stratified sampling
        // and I don't want to sample 100,000 times.
        assert_approx_eq!(Float, y_integral, CIE_Y_INTEGRAL, epsilon = 0.2);
    }

    #[test]
    fn sampling_pdf_xyz() {
        let mut impsum = 0.0;
        let mut unifsum = 0.0;
        let mut rng = StdRng::seed_from_u64(29378409);
        let between = Uniform::from(0.0..1.0);
        let n = 900000;
        for _ in 0..n {
            let u = between.sample(&mut rng);
            let lambda = lerp::<Float>(u, &LAMBDA_MIN, &LAMBDA_MAX);
            let pdf = 1.0 / (LAMBDA_MAX - LAMBDA_MIN);
            unifsum += (Spectrum::get_cie(CIE::X).get(lambda)
                + Spectrum::get_cie(CIE::Z).get(lambda)
                + Spectrum::get_cie(CIE::Z).get(lambda))
                / pdf;

            let lambda = sample_visible_wavelengths(u);
            let pdf = visible_wavelengths_pdf(lambda);
            if pdf > 0.0 {
                impsum += (Spectrum::get_cie(CIE::X).get(lambda)
                    + Spectrum::get_cie(CIE::Z).get(lambda)
                    + Spectrum::get_cie(CIE::Z).get(lambda))
                    / pdf;
            }
        }
        let imp_int = impsum / n as Float;
        let unif_int = unifsum / n as Float;

        // Allow a relatively large epsilong because we're not using stratified sampling
        // and to get an exact value would require many samples.
        assert_approx_eq!(Float, imp_int, unif_int, epsilon = 1.0);
    }
}
