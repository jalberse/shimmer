use auto_ops::impl_op_ex;

use crate::{
    spectra::{inner_product, spectrum::SpectrumI, Spectrum, CIE_Y_INTEGRAL},
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
