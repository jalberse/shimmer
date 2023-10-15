use std::ops::{Index, IndexMut};

use auto_ops::impl_op_ex;

use crate::{
    math::evaluate_polynomial,
    spectra::{
        inner_product,
        spectrum::{SpectrumI, LAMBDA_MAX, LAMBDA_MIN},
        Spectrum, CIE_Y_INTEGRAL,
    },
    square_matrix::SquareMatrix,
    vecmath::{HasNan, Point2f, Tuple2},
    Float,
};

#[derive(Debug, PartialEq, PartialOrd)]
pub struct XYZ {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl XYZ {
    pub fn new(x: Float, y: Float, z: Float) -> XYZ {
        XYZ { x, y, z }
    }

    pub fn from_spectrum<T: SpectrumI>(s: &T) -> XYZ {
        XYZ::new(
            inner_product::<Spectrum, T>(Spectrum::get_cie(crate::spectra::CIE::X), s),
            inner_product::<Spectrum, T>(Spectrum::get_cie(crate::spectra::CIE::Y), s),
            inner_product::<Spectrum, T>(Spectrum::get_cie(crate::spectra::CIE::Z), s),
        ) / CIE_Y_INTEGRAL
    }

    pub fn from_xy_y_default(xy: &Point2f) -> XYZ {
        Self::from_xy_y(xy, 1.0)
    }

    pub fn from_xy_y(xy: &Point2f, y: Float) -> XYZ {
        if xy.y == 0.0 {
            return XYZ::new(0.0, 0.0, 0.0);
        }
        XYZ::new(xy.x * y / xy.y, y, (1.0 - xy.x - xy.y) * y / xy.y)
    }

    pub fn xy(&self) -> Point2f {
        Point2f::new(
            self.x / (self.x + self.y + self.z),
            self.y / (self.x + self.y + self.z),
        )
    }
}

impl HasNan for XYZ {
    fn has_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl Default for XYZ {
    fn default() -> Self {
        Self {
            x: Default::default(),
            y: Default::default(),
            z: Default::default(),
        }
    }
}

impl Index<usize> for XYZ {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            &self.x
        } else if index == 1 {
            &self.y
        } else {
            &self.z
        }
    }
}

impl IndexMut<usize> for XYZ {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == 0 {
            &mut self.x
        } else if index == 1 {
            &mut self.y
        } else {
            &mut self.z
        }
    }
}

impl_op_ex!(+|a: &XYZ, b: &XYZ| -> XYZ
{
    debug_assert!(!a.has_nan() && !b.has_nan());
    XYZ { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
});

impl_op_ex!(-|a: &XYZ, b: &XYZ| -> XYZ {
    debug_assert!(!a.has_nan() && !b.has_nan());
    XYZ {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
});

impl_op_ex!(*|a: &XYZ, b: &XYZ| -> XYZ {
    debug_assert!(!a.has_nan() && !b.has_nan());
    XYZ {
        x: a.x * b.x,
        y: a.y * b.y,
        z: a.z * b.z,
    }
});

impl_op_ex!(/|a: &XYZ, b: &XYZ| -> XYZ {
    debug_assert!(!a.has_nan() && !b.has_nan());
    XYZ {
        x: a.x / b.x,
        y: a.y / b.y,
        z: a.z / b.z,
    }
});

impl_op_ex!(+|a: &XYZ, b: &Float| -> XYZ
{
    debug_assert!(!b.is_nan());
    XYZ { x: a.x + b, y: a.y + b, z: a.z + b }
});

impl_op_ex!(-|a: &XYZ, b: &Float| -> XYZ {
    debug_assert!(!b.is_nan());
    XYZ {
        x: a.x - b,
        y: a.y - b,
        z: a.z - b,
    }
});

impl_op_ex!(*|a: &XYZ, b: &Float| -> XYZ {
    debug_assert!(!b.is_nan());
    XYZ {
        x: a.x * b,
        y: a.y * b,
        z: a.z * b,
    }
});

impl_op_ex!(/|a: &XYZ, b: &Float| -> XYZ {
    debug_assert!(!b.is_nan());
    XYZ {
        x: a.x / b,
        y: a.y / b,
        z: a.z / b,
    }
});

impl_op_ex!(+=|a: &mut XYZ, b: &Float| {
    a.x += b;
    a.y += b;
    a.z += b;
});

impl_op_ex!(-=|a: &mut XYZ, b: &Float| {
    a.x -= b;
    a.y -= b;
    a.z -= b;
});

impl_op_ex!(*=|a: &mut XYZ, b: &Float| {
    a.x *= b;
    a.y *= b;
    a.z *= b;
});

impl_op_ex!(/=|a: &mut XYZ, b: &Float| {
    a.x /= b;
    a.y /= b;
    a.z /= b;
});

#[derive(Debug, Copy, Clone)]
pub struct RGB {
    pub r: Float,
    pub g: Float,
    pub b: Float,
}

impl RGB {
    pub fn new(r: Float, g: Float, b: Float) -> RGB {
        RGB { r, g, b }
    }

    pub fn clamp_zero(self) -> RGB {
        RGB::new(
            self.r.clamp(0.0, self.r),
            self.g.clamp(0.0, self.g),
            self.b.clamp(0.0, self.b),
        )
    }
}

impl From<&RGB> for [f32; 3] {
    fn from(value: &RGB) -> Self {
        [value.r as f32, value.g as f32, value.b as f32]
    }
}

impl_op_ex!(+|a: &RGB, b: &RGB| -> RGB {
    RGB::new(a.r + b.r, a.g + b.g, a.b + b.b)
});

impl_op_ex!(-|a: &RGB, b: &RGB| -> RGB { RGB::new(a.r - b.r, a.g - b.g, a.b - b.b) });

impl_op_ex!(*|a: &RGB, b: &RGB| -> RGB { RGB::new(a.r * b.r, a.g * b.g, a.b * b.b) });

impl_op_ex!(/|a: &RGB, b: &RGB| -> RGB { RGB::new(a.r / b.r, a.g / b.g, a.b / b.b) });

impl_op_ex!(+|a: &RGB, b: &Float| -> RGB {
    RGB::new(a.r + b, a.g + b, a.b + b)
});

impl_op_ex!(-|a: &RGB, b: &Float| -> RGB { RGB::new(a.r - b, a.g - b, a.b - b) });

impl_op_ex!(*|a: &RGB, b: &Float| -> RGB { RGB::new(a.r * b, a.g * b, a.b * b) });

impl_op_ex!(/|a: &RGB, b: &Float| -> RGB { RGB::new(a.r / b, a.g / b, a.b / b) });

impl_op_ex!(+=|a: &mut RGB, b: &RGB| {
    a.r += b.r;
    a.g += b.g;
    a.b += b.b;
});

impl_op_ex!(-=|a: &mut RGB, b: &RGB| {
    a.r -= b.r;
    a.g -= b.g;
    a.b -= b.b;
});

impl_op_ex!(*=|a: &mut RGB, b: &RGB| {
    a.r *= b.r;
    a.g *= b.g;
    a.b *= b.b;
});

impl_op_ex!(/=|a: &mut RGB, b: &RGB| {
    a.r /= b.r;
    a.g /= b.g;
    a.b /= b.b;
});

impl_op_ex!(+=|a: &mut RGB, b: &Float| {
    a.r += b;
    a.g += b;
    a.b += b;
});

impl_op_ex!(-=|a: &mut RGB, b: &Float| {
    a.r -= b;
    a.g -= b;
    a.b -= b;
});

impl_op_ex!(*=|a: &mut RGB, b: &Float| {
    a.r *= b;
    a.g *= b;
    a.b *= b;
});

impl_op_ex!(/=|a: &mut RGB, b: &Float| {
    a.r /= b;
    a.g /= b;
    a.b /= b;
});

impl HasNan for RGB {
    fn has_nan(&self) -> bool {
        self.r.is_nan() || self.g.is_nan() || self.b.is_nan()
    }
}

impl Default for RGB {
    fn default() -> Self {
        Self {
            r: Default::default(),
            g: Default::default(),
            b: Default::default(),
        }
    }
}

impl Index<usize> for RGB {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            &self.r
        } else if index == 1 {
            &self.g
        } else {
            &self.b
        }
    }
}

impl IndexMut<usize> for RGB {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == 0 {
            &mut self.r
        } else if index == 1 {
            &mut self.g
        } else {
            &mut self.b
        }
    }
}

/// Stores c_n for s(c_0 * lambda^2 + c_1 * lambda + c_2)
/// and evaluates that function, where s() is 4.25 in PBRTv4,
/// for converting an rgb value into a corresponding reflectance spectra.
#[derive(Debug, PartialEq)]
pub struct RgbSigmoidPolynomial {
    c0: Float,
    c1: Float,
    c2: Float,
}

impl RgbSigmoidPolynomial {
    pub fn new(c0: Float, c1: Float, c2: Float) -> RgbSigmoidPolynomial {
        RgbSigmoidPolynomial { c0, c1, c2 }
    }

    pub fn from_array(c: [Float; 3]) -> RgbSigmoidPolynomial {
        RgbSigmoidPolynomial {
            c0: c[0],
            c1: c[1],
            c2: c[2],
        }
    }

    pub fn get(&self, lambda: Float) -> Float {
        Self::s(evaluate_polynomial(lambda, &[self.c0, self.c1, self.c2]))
    }

    pub fn max_value(&self) -> Float {
        let result = Float::max(self.get(LAMBDA_MIN), self.get(LAMBDA_MAX));
        let lambda = -self.c1 / (2.0 * self.c0);
        if lambda >= LAMBDA_MIN && lambda <= LAMBDA_MAX {
            return Float::max(result, self.get(lambda));
        } else {
            result
        }
    }

    fn s(x: Float) -> Float {
        if x.is_infinite() {
            if x > 0.0 {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        0.5 + x / (2.0 * Float::sqrt(1.0 + x * x))
    }
}

impl HasNan for RgbSigmoidPolynomial {
    fn has_nan(&self) -> bool {
        self.c0.is_nan() || self.c1.is_nan() || self.c2.is_nan()
    }
}

// These are the Bradford transformation matrices.
const LMS_FROM_XYZ: SquareMatrix<3> = SquareMatrix::new([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
]);
const XYZ_FROM_LMS: SquareMatrix<3> = SquareMatrix::new([
    [0.986993, -0.147054, 0.159963],
    [0.432305, 0.51836, 0.0492912],
    [-0.00852866, 0.0400428, 0.968487],
]);

/// von Kries transform.
pub fn white_balance(src_white: &Point2f, target_white: &Point2f) -> SquareMatrix<3> {
    let src_xyz = XYZ::from_xy_y_default(src_white);
    let dst_xyz = XYZ::from_xy_y_default(target_white);
    let src_lms = LMS_FROM_XYZ * src_xyz;
    let dst_lms = LMS_FROM_XYZ * dst_xyz;

    let lms_correct = SquareMatrix::<3>::diag([
        dst_lms[0] / src_lms[0],
        dst_lms[1] / src_lms[1],
        dst_lms[2] / src_lms[2],
    ]);
    XYZ_FROM_LMS * lms_correct * LMS_FROM_XYZ
}

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;
    use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};

    use crate::{
        colorspace::{NamedColorSpace, RgbColorSpace},
        spectra::{
            spectrum::{
                RgbAlbedoSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum, SpectrumI,
                LAMBDA_MAX, LAMBDA_MIN,
            },
            DenselySampled,
        },
        vecmath::{Point2f, Tuple2},
        Float,
    };

    use super::{RGB, XYZ};

    #[test]
    fn from_xy_zero() {
        let point = Point2f::new(1.0, 0.0);
        let res = XYZ::from_xy_y(&point, 0.5);
        assert_eq!(XYZ::new(0.0, 0.0, 0.0), res);
    }

    #[test]
    fn rgb_xyz() {
        for cs in [
            NamedColorSpace::ACES2065_1,
            NamedColorSpace::REC2020,
            NamedColorSpace::SRGB,
        ] {
            let cs = RgbColorSpace::get_named(cs);
            let xyz = cs.to_xyz(&RGB::new(1.0, 1.0, 1.0));
            let rgb = cs.to_rgb(&xyz);
            assert_approx_eq!(Float, 1.0, rgb[0]);
            assert_approx_eq!(Float, 1.0, rgb[1]);
            assert_approx_eq!(Float, 1.0, rgb[2]);
        }
    }

    #[test]
    fn srgb() {
        // Make sure the matrix values are sensible by throwing the x, y, and z basis vectors
        // at it to pull out columns.
        let cs = RgbColorSpace::get_named(NamedColorSpace::SRGB);
        let rgb = cs.to_rgb(&XYZ::new(1.0, 0.0, 0.0));

        assert_approx_eq!(Float, 3.2406, rgb[0], epsilon = 0.001);
        assert_approx_eq!(Float, -0.9689, rgb[1], epsilon = 0.001);
        assert_approx_eq!(Float, 0.0557, rgb[2], epsilon = 0.001);

        let rgb = cs.to_rgb(&XYZ::new(0.0, 1.0, 0.0));
        assert_approx_eq!(Float, -1.5372, rgb[0], epsilon = 0.001);
        assert_approx_eq!(Float, 1.8758, rgb[1], epsilon = 0.001);
        assert_approx_eq!(Float, -0.2040, rgb[2], epsilon = 0.001);

        let rgb = cs.to_rgb(&XYZ::new(0.0, 0.0, 1.0));
        assert_approx_eq!(Float, -0.4986, rgb[0], epsilon = 0.001);
        assert_approx_eq!(Float, 0.0415, rgb[1], epsilon = 0.001);
        assert_approx_eq!(Float, 1.0570, rgb[2], epsilon = 0.001);
    }

    #[test]
    fn std_illum_whites_rgb() {
        let srgb = RgbColorSpace::get_named(NamedColorSpace::SRGB);
        let xyz = XYZ::from_spectrum(&*srgb.illuminant);
        let rgb = srgb.to_rgb(&xyz);
        assert!(rgb.r > 0.99);
        assert!(rgb.r < 1.01);
        assert!(rgb.g > 0.99);
        assert!(rgb.g < 1.01);
        assert!(rgb.b > 0.99);
        assert!(rgb.b < 1.01);
    }

    #[test]
    fn std_illum_whites_rec2020() {
        let cs = RgbColorSpace::get_named(NamedColorSpace::REC2020);
        let xyz = XYZ::from_spectrum(&*cs.illuminant);
        let rgb = cs.to_rgb(&xyz);
        assert!(rgb.r > 0.99);
        assert!(rgb.r < 1.01);
        assert!(rgb.g > 0.99);
        assert!(rgb.g < 1.01);
        assert!(rgb.b > 0.99);
        assert!(rgb.b < 1.01);
    }

    #[test]
    fn std_illum_whites_aces2065_1() {
        let cs = RgbColorSpace::get_named(NamedColorSpace::ACES2065_1);
        let xyz = XYZ::from_spectrum(&*cs.illuminant);
        let rgb = cs.to_rgb(&xyz);
        assert!(rgb.r > 0.99);
        assert!(rgb.r < 1.01);
        assert!(rgb.g > 0.99);
        assert!(rgb.g < 1.01);
        assert!(rgb.b > 0.99);
        assert!(rgb.b < 1.01);
    }

    #[test]
    fn rgb_unbounded_spectrum_max_value() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        for cs in [
            NamedColorSpace::SRGB,
            NamedColorSpace::REC2020,
            NamedColorSpace::ACES2065_1,
        ] {
            let cs = RgbColorSpace::get_named(cs);
            for _ in 0..100 {
                let rgb = RGB::new(
                    between.sample(&mut rng),
                    between.sample(&mut rng),
                    between.sample(&mut rng),
                ) * 10.0;
                let rs = RgbUnboundedSpectrum::new(&cs, &rgb);

                let m = rs.max_value();
                let mut sm = 0.0;
                let mut lambda = 360.0;
                while lambda <= 830.0 {
                    sm = Float::max(sm, rs.get(lambda));
                    lambda += 1.0 / 16.0;
                }
                assert!(Float::abs(sm - m) / sm < 1e-4);
            }
        }
    }

    #[test]
    fn rgb_albedo_spectrum_max_value() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        for cs in [
            NamedColorSpace::SRGB,
            NamedColorSpace::REC2020,
            NamedColorSpace::ACES2065_1,
        ] {
            let cs = RgbColorSpace::get_named(cs);
            for _ in 0..100 {
                let rgb = RGB::new(
                    between.sample(&mut rng),
                    between.sample(&mut rng),
                    between.sample(&mut rng),
                );
                let rs = RgbAlbedoSpectrum::new(&cs, &rgb);

                let m = rs.max_value();
                let mut sm = 0.0;
                let mut lambda = 360.0;
                while lambda <= 830.0 {
                    sm = Float::max(sm, rs.get(lambda));
                    lambda += 1.0 / 16.0;
                }
                assert!(Float::abs(sm - m) / sm < 1e-4);
            }
        }
    }

    #[test]
    fn rgb_albedo_spectrum_round_trip_rgb() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::SRGB);

        for _ in 0..100 {
            let rgb = RGB::new(
                between.sample(&mut rng),
                between.sample(&mut rng),
                between.sample(&mut rng),
            );
            let rs = RgbAlbedoSpectrum::new(cs, &rgb);

            let rs_illum = DenselySampled::sample_function(
                |lambda: Float| -> Float { rs.get(lambda) * cs.illuminant.get(lambda) },
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            let xyz = XYZ::from_spectrum(&rs_illum);
            let rgb2 = cs.to_rgb(&xyz);

            // Some error comes from the fact that piecewise linear (at 5nm)
            // CIE curves were used for the optimization while we use piecewise
            // linear at 1nm spacing converted to 1nm constant / densely
            // sampled.
            assert_approx_eq!(Float, rgb.r, rgb2.r, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.g, rgb2.g, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.b, rgb2.b, epsilon = 0.01);
        }
    }

    #[test]
    fn rgb_albedo_spectrum_round_trip_rec2020() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::REC2020);

        for _ in 0..100 {
            let rgb = RGB::new(
                0.1 + 0.7 * between.sample(&mut rng),
                0.1 + 0.7 * between.sample(&mut rng),
                0.1 + 0.7 * between.sample(&mut rng),
            );
            let rs = RgbAlbedoSpectrum::new(cs, &rgb);

            let rs_illum = DenselySampled::sample_function(
                |lambda: Float| -> Float { rs.get(lambda) * cs.illuminant.get(lambda) },
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            let xyz = XYZ::from_spectrum(&rs_illum);
            let rgb2 = cs.to_rgb(&xyz);

            // Some error comes from the fact that piecewise linear (at 5nm)
            // CIE curves were used for the optimization while we use piecewise
            // linear at 1nm spacing converted to 1nm constant / densely
            // sampled.
            assert_approx_eq!(Float, rgb.r, rgb2.r, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.g, rgb2.g, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.b, rgb2.b, epsilon = 0.01);
        }
    }

    #[test]
    fn rgb_albedo_spectrum_round_trip_aces() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::ACES2065_1);

        for _ in 0..100 {
            let rgb = RGB::new(
                0.3 + 0.4 * between.sample(&mut rng),
                0.3 + 0.4 * between.sample(&mut rng),
                0.3 + 0.4 * between.sample(&mut rng),
            );
            let rs = RgbAlbedoSpectrum::new(cs, &rgb);

            let rs_illum = DenselySampled::sample_function(
                |lambda: Float| -> Float { rs.get(lambda) * cs.illuminant.get(lambda) },
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            let xyz = XYZ::from_spectrum(&rs_illum);
            let rgb2 = cs.to_rgb(&xyz);

            // Some error comes from the fact that piecewise linear (at 5nm)
            // CIE curves were used for the optimization while we use piecewise
            // linear at 1nm spacing converted to 1nm constant / densely
            // sampled.
            assert_approx_eq!(Float, rgb.r, rgb2.r, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.g, rgb2.g, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.b, rgb2.b, epsilon = 0.01);
        }
    }

    #[test]
    fn rgb_illum_spectrum_round_trip_rgb() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::SRGB);

        for _ in 0..100 {
            let rgb = RGB::new(
                between.sample(&mut rng),
                between.sample(&mut rng),
                between.sample(&mut rng),
            );
            let rs = RgbIlluminantSpectrum::new(cs, &rgb);

            let rs_illum = DenselySampled::sample_function(
                |lambda: Float| -> Float { rs.get(lambda) },
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            let xyz = XYZ::from_spectrum(&rs_illum);
            let rgb2 = cs.to_rgb(&xyz);

            // Some error comes from the fact that piecewise linear (at 5nm)
            // CIE curves were used for the optimization while we use piecewise
            // linear at 1nm spacing converted to 1nm constant / densely
            // sampled.
            assert_approx_eq!(Float, rgb.r, rgb2.r, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.g, rgb2.g, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.b, rgb2.b, epsilon = 0.01);
        }
    }

    #[test]
    fn rgb_illum_spectrum_round_trip_rec2020() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::REC2020);

        for _ in 0..100 {
            let rgb = RGB::new(
                0.1 + 0.7 * between.sample(&mut rng),
                0.1 + 0.7 * between.sample(&mut rng),
                0.1 + 0.7 * between.sample(&mut rng),
            );
            let rs = RgbIlluminantSpectrum::new(cs, &rgb);

            let rs_illum = DenselySampled::sample_function(
                |lambda: Float| -> Float { rs.get(lambda) },
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            let xyz = XYZ::from_spectrum(&rs_illum);
            let rgb2 = cs.to_rgb(&xyz);

            // Some error comes from the fact that piecewise linear (at 5nm)
            // CIE curves were used for the optimization while we use piecewise
            // linear at 1nm spacing converted to 1nm constant / densely
            // sampled.
            assert_approx_eq!(Float, rgb.r, rgb2.r, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.g, rgb2.g, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.b, rgb2.b, epsilon = 0.01);
        }
    }

    #[test]
    fn rgb_illum_spectrum_round_trip_aces() {
        let mut rng = StdRng::seed_from_u64(0);
        let between = Uniform::from(0.0..1.0);
        let cs = RgbColorSpace::get_named(NamedColorSpace::ACES2065_1);

        for _ in 0..100 {
            let rgb = RGB::new(
                0.3 + 0.4 * between.sample(&mut rng),
                0.3 + 0.4 * between.sample(&mut rng),
                0.3 + 0.4 * between.sample(&mut rng),
            );
            let rs = RgbIlluminantSpectrum::new(cs, &rgb);

            let rs_illum = DenselySampled::sample_function(
                |lambda: Float| -> Float { rs.get(lambda) },
                LAMBDA_MIN as usize,
                LAMBDA_MAX as usize,
            );
            let xyz = XYZ::from_spectrum(&rs_illum);
            let rgb2 = cs.to_rgb(&xyz);

            // Some error comes from the fact that piecewise linear (at 5nm)
            // CIE curves were used for the optimization while we use piecewise
            // linear at 1nm spacing converted to 1nm constant / densely
            // sampled.
            assert_approx_eq!(Float, rgb.r, rgb2.r, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.g, rgb2.g, epsilon = 0.01);
            assert_approx_eq!(Float, rgb.b, rgb2.b, epsilon = 0.01);
        }
    }
}
