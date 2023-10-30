use std::sync::Arc;

use once_cell::sync::Lazy;

use crate::{
    color::{RgbSigmoidPolynomial, RGB, XYZ},
    rgb_to_spectra::{self, Gamut},
    spectra::{DenselySampledSpectrum, Spectrum},
    square_matrix::{mul_mat_vec, Invertible, SquareMatrix},
    vecmath::{Point2f, Tuple2},
};

// TODO I derived Clone for the construction of Film; but I think we might want to wrap the RgbColorSpace in an Rc instead?
// I'd rather have one copy than many, but it shouldn't be that expensive to clone either.
#[derive(Debug, PartialEq, Clone)]
pub struct RgbColorSpace {
    /// Red primary
    pub r: Point2f,
    /// Green primary
    pub g: Point2f,
    /// Blue primary
    pub b: Point2f,
    pub whitepoint: Point2f,
    pub illuminant: Arc<Spectrum>,
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
// illuminant is actually too expensive to compare (at least, it's not ocmpared in PBRT)
impl RgbColorSpace {
    pub fn new(
        r: Point2f,
        g: Point2f,
        b: Point2f,
        illuminant: &Spectrum,
        gamut: Gamut,
    ) -> RgbColorSpace {
        // Compute the whitepoint primaries and XYZ coordinates.
        let w: XYZ = XYZ::from_spectrum(illuminant);

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

        // TODO these matrices are also wrong. Was this fixed with spectra fixes?
        let xyz_from_rgb = rgb * SquareMatrix::<3>::diag([c[0], c[1], c[2]]);
        let rgb_from_xyz = xyz_from_rgb.inverse().expect("Uninvertible!");

        // Convert the spectrum given to a DenselySampled spectrum.
        let illuminant = DenselySampledSpectrum::new(illuminant);

        RgbColorSpace {
            r,
            g,
            b,
            whitepoint,
            illuminant: Arc::new(Spectrum::DenselySampled(illuminant)),
            xyz_from_rgb,
            rgb_from_xyz,
            gamut,
        }
    }

    pub fn get_named(cs: NamedColorSpace) -> &'static RgbColorSpace {
        match cs {
            NamedColorSpace::SRGB => Lazy::force(&SRGB),
            NamedColorSpace::REC2020 => Lazy::force(&REC_2020),
            NamedColorSpace::ACES2065_1 => Lazy::force(&ACES2065_1),
        }
    }

    pub fn to_rgb(&self, xyz: &XYZ) -> RGB {
        mul_mat_vec::<3, XYZ, RGB>(&self.rgb_from_xyz, &xyz)
    }

    pub fn to_xyz(&self, rgb: &RGB) -> XYZ {
        mul_mat_vec::<3, RGB, XYZ>(&self.xyz_from_rgb, &rgb)
    }

    pub fn to_rgb_coeffs(&self, rgb: &RGB) -> RgbSigmoidPolynomial {
        debug_assert!(rgb.r >= 0.0 && rgb.g >= 0.0 && rgb.b >= 0.0);
        RgbSigmoidPolynomial::from_array(rgb_to_spectra::get_rgb_to_spec(&self.gamut, rgb))
    }

    pub fn convert_rgb_colorspace(&self, to: &RgbColorSpace) -> SquareMatrix<3> {
        if self == to {
            return SquareMatrix::<3>::default();
        }
        to.rgb_from_xyz * self.xyz_from_rgb
    }
}

// TODO We currently don't support DCI_P3, but PBRTv4 does. The reason is because
// rgb2spec-rs, which we use for the rgb-to-spectra table generation, doesn't
// support DCI_P3. We should add support for it sometime in the future.

pub enum NamedColorSpace {
    SRGB,
    REC2020,
    ACES2065_1,
}

pub static SRGB: Lazy<RgbColorSpace> = Lazy::new(|| {
    RgbColorSpace::new(
        Point2f::new(0.64, 0.33),
        Point2f::new(0.3, 0.6),
        Point2f::new(0.15, 0.06),
        Spectrum::get_named_spectrum(crate::spectra::NamedSpectrum::StdIllumD65),
        Gamut::SRGB,
    )
});

pub static REC_2020: Lazy<RgbColorSpace> = Lazy::new(|| {
    RgbColorSpace::new(
        Point2f::new(0.708, 0.292),
        Point2f::new(0.170, 0.797),
        Point2f::new(0.131, 0.046),
        Spectrum::get_named_spectrum(crate::spectra::NamedSpectrum::StdIllumD65),
        Gamut::Rec2020,
    )
});

pub static ACES2065_1: Lazy<RgbColorSpace> = Lazy::new(|| {
    RgbColorSpace::new(
        Point2f::new(0.7347, 0.2653),
        Point2f::new(0.0, 1.0),
        Point2f::new(0.0001, -0.077),
        Spectrum::get_named_spectrum(crate::spectra::NamedSpectrum::IllumAcesD60),
        Gamut::Aces2065_1,
    )
});

// TODO pg 186, lookup color space fn?
