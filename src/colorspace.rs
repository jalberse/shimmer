use std::rc::Rc;

use crate::{
    color::{RgbSigmoidPolynomial, RGB, XYZ},
    rgb_to_spectra::{self, Gamut},
    spectra::{DenselySampled, Spectrum},
    square_matrix::{mul_mat_vec, Invertible, SquareMatrix},
    vecmath::Point2f,
};

#[derive(Debug, PartialEq)]
pub struct RgbColorSpace {
    /// Red primary
    pub r: Point2f,
    /// Green primary
    pub g: Point2f,
    /// Blue primary
    pub b: Point2f,
    pub whitepoint: Point2f,
    pub illuminant: Rc<DenselySampled>,
    pub xyz_from_rgb: SquareMatrix<3>,
    pub rgb_from_xyz: SquareMatrix<3>,
    /// This is analogous to the RGBToSpectrumTable pointer used by PBRT;
    /// we use an Enum instead to match to get a lazily-loaded converter using the rgb2spectra
    /// crate. The crate is based on the same paper as PBRT's implementation, and is basically
    /// a port of it. We load it lazily, however,  rather than using an init(), and access
    /// via an enum and a match rather than a pointer.
    gamut: Gamut,
}

// TODO equality, inequality operators should compare everything but illuminant; the
// illuminant is actuaklly to expensive to compare (at least, it's not ocmpared in PBRT)
impl RgbColorSpace {
    pub fn new(
        r: Point2f,
        g: Point2f,
        b: Point2f,
        illuminant: DenselySampled,
        gamut: Gamut,
    ) -> RgbColorSpace {
        // Compute the whitepoint primaries and XYZ coordinates.
        let w: XYZ = XYZ::from_spectrum(&illuminant);
        let whitepoint = w.xy();
        let r_xyz = XYZ::from_xy_y_default(&r);
        let g_xyz = XYZ::from_xy_y_default(&g);
        let b_xyz = XYZ::from_xy_y_default(&b);

        // Initialize XYZ color space conversion matrices
        let rgb = SquareMatrix::<3>::new([
            [r_xyz.x, g_xyz.x, b_xyz.x],
            [r_xyz.y, g_xyz.y, b_xyz.y],
            [r_xyz.z, g_xyz.z, b_xyz.z],
        ]);
        let c = rgb.inverse().expect("Uninvertible!") * w;
        let xyz_from_rgb = rgb * SquareMatrix::<3>::diag([c[0], c[1], c[2]]);
        let rgb_from_xyz = xyz_from_rgb.inverse().expect("Uninvertible!");

        RgbColorSpace {
            r,
            g,
            b,
            whitepoint,
            illuminant: Rc::new(illuminant),
            xyz_from_rgb,
            rgb_from_xyz,
            gamut,
        }
    }

    pub fn to_rgb(&self, xyz: &XYZ) -> RGB {
        mul_mat_vec::<3, XYZ, RGB>(&self.rgb_from_xyz, &xyz)
    }

    pub fn to_xyz(&self, rgb: &RGB) -> XYZ {
        mul_mat_vec::<3, RGB, XYZ>(&self.xyz_from_rgb, &rgb)
    }

    pub fn to_rgb_coeffs(&self, rgb: &RGB) -> RgbSigmoidPolynomial {
        RgbSigmoidPolynomial::from_array(rgb_to_spectra::get_rgb_to_spec(&self.gamut, rgb))
    }

    pub fn convert_rgb_colorspace(&self, to: &RgbColorSpace) -> SquareMatrix<3> {
        if self == to {
            return SquareMatrix::<3>::default();
        }
        to.rgb_from_xyz * self.xyz_from_rgb
    }
}

// TODO lazy-initialized RGBColorSpaces that are of each of the standard ones.
//   stand-in for preinitialized pointers I suppose
// TODO pg 186, get_named and lookup?
