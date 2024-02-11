use std::ops::{Index, IndexMut};

use auto_ops::impl_op_ex;
use fast_polynomial::poly;

use crate::{
    math::safe_sqrt,
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
        // TODO This may be wrong? It looks like the coefficients were right,
        //  but I am getting a different value for R in the diffuse material
        //  compared to PBRT (our values were too small).
        // I know I had bugs with this function before, but I swear I
        //  have a test for this and it was working well??
        // Unless I'm getting confused, because I guess I'm seeing different coefficients coming in.
        Self::s(poly(lambda, &[self.c2, self.c1, self.c0]))
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

/// Color encoding is necessary for both reading/decoding color alues from non-floating-point image formats
/// like PNG, as well as encoding them before writing to such image formats. ColorEncoding handles this.
pub trait ColorEncodingI {
    /// Given a set of encoded 8-bit color values, converts to linear Floats.
    /// We use a slice to avoid the dispatch cost of invoking many times for each
    /// value individually.
    fn to_linear(&self, vin: &[u8], vout: &mut [Float]);

    /// The inverse of to_linear.
    fn from_linear(&self, vin: &[Float], vout: &mut [u8]);

    /// It is sometimes useful to decode values with greater than 8-bit precision
    /// (e.g., some image formats like PNG are able to store 16-bit color channels).
    /// Such cases are handled by to_float_linear(), which takes a single encoded
    /// value stored as a Float and decodes it.
    fn to_float_linear(&self, v: Float) -> Float;
}

#[derive(Debug, Clone)]
pub enum ColorEncoding {
    Linear(LinearColorEncoding),
    SRGB(SRgbColorEncoding),
    Gamma(GammaColorEncoding),
}

impl ColorEncodingI for ColorEncoding {
    fn to_linear(&self, vin: &[u8], vout: &mut [Float]) {
        match self {
            ColorEncoding::Linear(e) => e.to_linear(vin, vout),
            ColorEncoding::SRGB(e) => e.to_linear(vin, vout),
            ColorEncoding::Gamma(e) => e.to_linear(vin, vout),
        }
    }

    fn from_linear(&self, vin: &[Float], vout: &mut [u8]) {
        match self {
            ColorEncoding::Linear(e) => e.from_linear(vin, vout),
            ColorEncoding::SRGB(e) => e.from_linear(vin, vout),
            ColorEncoding::Gamma(e) => e.from_linear(vin, vout),
        }
    }

    fn to_float_linear(&self, v: Float) -> Float {
        match self {
            ColorEncoding::Linear(e) => e.to_float_linear(v),
            ColorEncoding::SRGB(e) => e.to_float_linear(v),
            ColorEncoding::Gamma(e) => e.to_float_linear(v),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearColorEncoding {}

impl ColorEncodingI for LinearColorEncoding {
    fn to_linear(&self, vin: &[u8], vout: &mut [Float]) {
        debug_assert!(vin.len() == vout.len());
        for i in 0..vin.len() {
            vout[i] = vin[i] as Float / 255.0
        }
    }

    fn from_linear(&self, vin: &[Float], vout: &mut [u8]) {
        debug_assert!(vin.len() == vout.len());
        for i in 0..vin.len() {
            vout[i] = (vin[i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8
        }
    }

    fn to_float_linear(&self, v: Float) -> Float {
        v
    }
}

#[derive(Debug, Clone)]
pub struct SRgbColorEncoding {}

impl ColorEncodingI for SRgbColorEncoding {
    fn to_linear(&self, vin: &[u8], vout: &mut [Float]) {
        debug_assert!(vin.len() == vout.len());
        for i in 0..vin.len() {
            vout[i] = srgb_8_to_linear(vin[i]);
        }
    }

    fn from_linear(&self, vin: &[Float], vout: &mut [u8]) {
        debug_assert!(vin.len() == vout.len());
        for i in 0..vin.len() {
            vout[i] = linear_to_srgb_8(vin[i], 0.0);
        }
    }

    fn to_float_linear(&self, v: Float) -> Float {
        srgb_to_linear(v)
    }
}

#[derive(Debug, Clone)]
pub struct GammaColorEncoding {
    gamma: Float,
    apply_lut: [Float; 256],
    inverse_lut: [Float; 1024],
}

impl GammaColorEncoding {
    pub fn new(gamma: Float) -> GammaColorEncoding {
        let mut apply_lut = [0.0; 256];
        for i in 0..256 {
            let v = i as Float / 255.0;
            apply_lut[i] = Float::powf(v, gamma);
        }

        let mut inverse_lut = [0.0; 1024];
        for i in 0..1024 {
            let v = i as Float / (inverse_lut.len() - 1) as Float;
            inverse_lut[i] = Float::clamp(255.0 * Float::powf(v, 1.0 / gamma) + 0.5, 0.0, 255.0);
        }
        GammaColorEncoding {
            gamma,
            apply_lut,
            inverse_lut,
        }
    }
}

impl ColorEncodingI for GammaColorEncoding {
    fn to_linear(&self, vin: &[u8], vout: &mut [Float]) {
        debug_assert!(vin.len() == vout.len());
        for i in 0..vin.len() {
            vout[i] = self.apply_lut[vin[i] as usize];
        }
    }

    fn from_linear(&self, vin: &[Float], vout: &mut [u8]) {
        debug_assert!(vin.len() == vout.len());
        for i in 0..vin.len() {
            vout[i] = self.inverse_lut[Float::clamp(
                vin[i] * (self.inverse_lut.len() - 1) as Float,
                0.0,
                (self.inverse_lut.len() - 1) as Float,
            ) as usize] as u8;
        }
    }

    fn to_float_linear(&self, v: Float) -> Float {
        Float::powf(v, self.gamma)
    }
}

fn linear_to_srgb(value: Float) -> Float {
    if value <= 0.0031308 {
        return 12.92 * value;
    }
    let sqrt_value = safe_sqrt(value);
    // Minimax polynomial approximation from enoki's color.h.
    let p = poly(
        sqrt_value,
        &[
            -0.0016829072605308378,
            0.03453868659826638,
            0.7642611304733891,
            2.0041169284241644,
            0.7551545191665577,
            -0.016202083165206348,
        ],
    );
    let q = poly(
        sqrt_value,
        &[
            4.178892964897981e-7,
            -0.00004375359692957097,
            0.03467195408529984,
            0.6085338522168684,
            1.8970238036421054,
            1.0,
        ],
    );
    p / q * value
}

fn srgb_to_linear(value: Float) -> Float {
    if value <= 0.04045 {
        return value * (1.0 / 12.92);
    }

    // Minimax polynomial approximation from enoki's color.h.
    let p = poly(
        value,
        &[
            -0.0163933279112946,
            -0.7386328024653209,
            -11.199318357635072,
            -47.46726633009393,
            -36.04572663838034,
        ],
    );
    let q = poly(
        value,
        &[
            -0.004261480793199332,
            -19.140923959601675,
            -59.096406619244426,
            -18.225745396846637,
            1.0,
        ],
    );
    p / q * value
}

fn linear_to_srgb_8(value: Float, dither: Float) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 1.0 {
        255
    } else {
        Float::round(255.0 * linear_to_srgb(value) + dither).clamp(0.0, 255.0) as u8
    }
}

fn srgb_8_to_linear(value: u8) -> Float {
    SRGB_TO_LINEAR_LUT[value as usize]
}

const SRGB_TO_LINEAR_LUT: [Float; 256] = [
    0.0000000000,
    0.0003035270,
    0.0006070540,
    0.0009105810,
    0.0012141080,
    0.0015176350,
    0.0018211619,
    0.0021246888,
    0.0024282159,
    0.0027317430,
    0.0030352699,
    0.0033465356,
    0.0036765069,
    0.0040247170,
    0.0043914421,
    0.0047769533,
    0.0051815170,
    0.0056053917,
    0.0060488326,
    0.0065120910,
    0.0069954102,
    0.0074990317,
    0.0080231922,
    0.0085681248,
    0.0091340570,
    0.0097212177,
    0.0103298230,
    0.0109600937,
    0.0116122449,
    0.0122864870,
    0.0129830306,
    0.0137020806,
    0.0144438436,
    0.0152085144,
    0.0159962922,
    0.0168073755,
    0.0176419523,
    0.0185002182,
    0.0193823613,
    0.0202885624,
    0.0212190095,
    0.0221738834,
    0.0231533647,
    0.0241576303,
    0.0251868572,
    0.0262412224,
    0.0273208916,
    0.0284260381,
    0.0295568332,
    0.0307134409,
    0.0318960287,
    0.0331047624,
    0.0343398079,
    0.0356013142,
    0.0368894450,
    0.0382043645,
    0.0395462364,
    0.0409151986,
    0.0423114114,
    0.0437350273,
    0.0451862030,
    0.0466650836,
    0.0481718220,
    0.0497065634,
    0.0512694679,
    0.0528606549,
    0.0544802807,
    0.0561284944,
    0.0578054339,
    0.0595112406,
    0.0612460710,
    0.0630100295,
    0.0648032799,
    0.0666259527,
    0.0684781820,
    0.0703601092,
    0.0722718611,
    0.0742135793,
    0.0761853904,
    0.0781874284,
    0.0802198276,
    0.0822827145,
    0.0843762159,
    0.0865004659,
    0.0886556059,
    0.0908417329,
    0.0930589810,
    0.0953074843,
    0.0975873619,
    0.0998987406,
    0.1022417471,
    0.1046164930,
    0.1070231125,
    0.1094617173,
    0.1119324341,
    0.1144353822,
    0.1169706732,
    0.1195384338,
    0.1221387982,
    0.1247718409,
    0.1274376959,
    0.1301364899,
    0.1328683347,
    0.1356333494,
    0.1384316236,
    0.1412633061,
    0.1441284865,
    0.1470272839,
    0.1499598026,
    0.1529261619,
    0.1559264660,
    0.1589608639,
    0.1620294005,
    0.1651322246,
    0.1682693958,
    0.1714410931,
    0.1746473908,
    0.1778884083,
    0.1811642349,
    0.1844749898,
    0.1878207624,
    0.1912016720,
    0.1946178079,
    0.1980693042,
    0.2015562356,
    0.2050787061,
    0.2086368501,
    0.2122307271,
    0.2158605307,
    0.2195262313,
    0.2232279778,
    0.2269658893,
    0.2307400703,
    0.2345506549,
    0.2383976579,
    0.2422811985,
    0.2462013960,
    0.2501583695,
    0.2541521788,
    0.2581829131,
    0.2622507215,
    0.2663556635,
    0.2704978585,
    0.2746773660,
    0.2788943350,
    0.2831487954,
    0.2874408960,
    0.2917706966,
    0.2961383164,
    0.3005438447,
    0.3049873710,
    0.3094689548,
    0.3139887452,
    0.3185468316,
    0.3231432438,
    0.3277781308,
    0.3324515820,
    0.3371636569,
    0.3419144452,
    0.3467040956,
    0.3515326977,
    0.3564002514,
    0.3613068759,
    0.3662526906,
    0.3712377846,
    0.3762622178,
    0.3813261092,
    0.3864295185,
    0.3915725648,
    0.3967553079,
    0.4019778669,
    0.4072403014,
    0.4125427008,
    0.4178851545,
    0.4232677519,
    0.4286905527,
    0.4341537058,
    0.4396572411,
    0.4452012479,
    0.4507858455,
    0.4564110637,
    0.4620770514,
    0.4677838385,
    0.4735315442,
    0.4793202281,
    0.4851499796,
    0.4910208881,
    0.4969330430,
    0.5028865933,
    0.5088814497,
    0.5149177909,
    0.5209956765,
    0.5271152258,
    0.5332764983,
    0.5394796133,
    0.5457245708,
    0.5520114899,
    0.5583404899,
    0.5647116303,
    0.5711249113,
    0.5775805116,
    0.5840784907,
    0.5906189084,
    0.5972018838,
    0.6038274169,
    0.6104956269,
    0.6172066331,
    0.6239604354,
    0.6307572126,
    0.6375969648,
    0.6444797516,
    0.6514056921,
    0.6583748460,
    0.6653873324,
    0.6724432111,
    0.6795425415,
    0.6866854429,
    0.6938719153,
    0.7011020184,
    0.7083759308,
    0.7156936526,
    0.7230552435,
    0.7304608822,
    0.7379105687,
    0.7454043627,
    0.7529423237,
    0.7605246305,
    0.7681512833,
    0.7758223414,
    0.7835379243,
    0.7912980318,
    0.7991028428,
    0.8069523573,
    0.8148466945,
    0.8227858543,
    0.8307699561,
    0.8387991190,
    0.8468732834,
    0.8549926877,
    0.8631572723,
    0.8713672161,
    0.8796223402,
    0.8879231811,
    0.8962693810,
    0.9046613574,
    0.9130986929,
    0.9215820432,
    0.9301108718,
    0.9386858940,
    0.9473065734,
    0.9559735060,
    0.9646862745,
    0.9734454751,
    0.9822505713,
    0.9911022186,
    1.0000000000,
];

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
            DenselySampledSpectrum,
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

            let rs_illum = DenselySampledSpectrum::sample_function(
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

            let rs_illum = DenselySampledSpectrum::sample_function(
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

            let rs_illum = DenselySampledSpectrum::sample_function(
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

            let rs_illum = DenselySampledSpectrum::sample_function(
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

            let rs_illum = DenselySampledSpectrum::sample_function(
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

            let rs_illum = DenselySampledSpectrum::sample_function(
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
