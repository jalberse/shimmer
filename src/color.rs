use std::ops::{Index, IndexMut};

use auto_ops::impl_op_ex;

use crate::{
    math::evaluate_polynomial,
    spectra::{
        inner_product,
        spectrum::{SpectrumI, LAMBDA_MAX, LAMBDA_MIN},
        Spectrum, CIE_Y_INTEGRAL,
    },
    vecmath::{HasNan, Point2f, Tuple2},
    Float,
};

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
            inner_product::<Spectrum, T>(Spectrum::get_cie(crate::spectra::CIE::Z), s)
                / CIE_Y_INTEGRAL,
        )
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

        0.5 + x / (2.0 * Float::sqrt(1.0 + Float::sqrt(x)))
    }
}

impl HasNan for RgbSigmoidPolynomial {
    fn has_nan(&self) -> bool {
        self.c0.is_nan() || self.c1.is_nan() || self.c2.is_nan()
    }
}

mod tests {
    use float_cmp::assert_approx_eq;

    use crate::{
        colorspace::{NamedColorSpace, RgbColorSpace},
        rgb_to_spectra::Gamut,
        Float,
    };

    use super::RGB;

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
}
