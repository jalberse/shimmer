use std::ops::{AddAssign, Index, IndexMut, MulAssign};

use once_cell::sync::Lazy;

use crate::{
    bounding_box::Bounds2f,
    color::{white_balance, RGB, XYZ},
    colorspace::RgbColorSpace,
    interaction::SurfaceInteraction,
    math::linear_least_squares_3,
    spectra::{
        inner_product,
        sampled_spectrum::SampledSpectrum,
        sampled_wavelengths::SampledWavelengths,
        spectrum::{SpectrumI, LAMBDA_MAX, LAMBDA_MIN},
        DenselySampled, PiecewiseLinear, Spectrum,
    },
    square_matrix::SquareMatrix,
    vecmath::{Normal3f, Point2f, Point2i, Point3f, Vector3f},
    Float,
};

pub trait FilmI {
    // PAPERDOC - can likely discuss this.
    // TODO - PBRT correctly assumes that multiple threads won't call add_sample() concurrently
    // with the same p_film location, so I think it plays loose with mutual exclusion in implementation.
    // We won't have that because we need to prove to the compiler that access will be OK.
    // I don't have the full architecture in my head yet so I'm not sure how we'll do this differently;
    // possibly with data structures independent per thread that are then collected later?
    fn add_sample(
        &mut self,
        p_film: &Point2f,
        l: &SampledSpectrum,
        lambda: &SampledWavelengths,
        visible_surface: &Option<VisibleSurface>,
        weight: Float,
    );

    fn full_resolution(&self) -> Point2i;

    /// The bounds of all the samples that may be generated;
    /// different from the image bounds in the common case that the pixel filter
    /// extents are wider than a pixel.
    fn sample_bounds(&self) -> Bounds2f;
}

#[derive(Debug, Copy, Clone)]
pub enum Film {
    RgbFilm,
}

impl Film {}

impl FilmI for Film {
    fn add_sample(
        &mut self,
        p_film: &Point2f,
        l: &SampledSpectrum,
        lambda: &SampledWavelengths,
        visible_surface: &Option<VisibleSurface>,
        weight: Float,
    ) {
        todo!()
    }

    fn full_resolution(&self) -> Point2i {
        todo!()
    }

    fn sample_bounds(&self) -> Bounds2f {
        todo!()
    }
}

pub struct PixelSensor {
    pub xyz_from_sensor_rgb: SquareMatrix<3>,
    /// The red RGB matching function
    r_bar: DenselySampled,
    /// The green RGB matching function
    g_bar: DenselySampled,
    /// The blue RGB matching function
    b_bar: DenselySampled,
    /// Shutter time and ISO setting are collected into this quantity
    imaging_ratio: Float,
}

impl PixelSensor {
    /// Uses the XYZ matching functions for the pixel sensor's spectral response curves.
    /// This is a reasonable default.
    /// sensor_illum: If None is provided, no white balancing is done (it can be done in post-processing).
    /// If a sensor_illum is provided to specify a color temperature, white balancing is handled via the
    /// xyz_from_sensor_rgb matrix.
    pub fn new(
        output_colorspace: &RgbColorSpace,
        sensor_illum: &Option<Spectrum>,
        imaging_ratio: Float,
    ) -> PixelSensor {
        let xyz_from_sensor_rgb = if let Some(illum) = sensor_illum {
            let source_white = XYZ::from_spectrum(illum).xy();
            let target_white = output_colorspace.whitepoint;
            white_balance(&source_white, &target_white)
        } else {
            SquareMatrix::<3>::default()
        };
        PixelSensor {
            xyz_from_sensor_rgb,
            r_bar: DenselySampled::new(Spectrum::get_cie(crate::spectra::CIE::X)),
            g_bar: DenselySampled::new(Spectrum::get_cie(crate::spectra::CIE::Y)),
            b_bar: DenselySampled::new(Spectrum::get_cie(crate::spectra::CIE::Z)),
            imaging_ratio,
        }
    }

    /// Creates a new sensor given the RGB response curves.
    pub fn new_with_rgb(
        r: &Spectrum,
        g: &Spectrum,
        b: &Spectrum,
        output_colorspace: &RgbColorSpace,
        sensor_illum: &Spectrum,
        imaging_ratio: Float,
    ) -> PixelSensor {
        // The RGB colorspace in which a pixel sensor records light is generally not
        // the same colorspace as the one to be displayed e.g. sRGB. It is instead
        // based on the physical responsiveness of a camera's sensors, which doesn't
        // match up well with colorspaces such as sRGB. So, we construct a matrix for
        // converting to XYZ so that it can later be converted to the display colorspace.

        // Compute rgb_camera values for training swatches
        let mut rgb_camera = [[0.0; 3]; NUM_SWATCH_REFLECTANCES];
        for i in 0..NUM_SWATCH_REFLECTANCES {
            let rgb = Self::project_reflectance::<RGB>(
                &SWATCH_REFLECTANCES[i],
                &sensor_illum,
                &r,
                &b,
                &g,
            );
            for c in 0..3 {
                rgb_camera[i][c] = rgb[c];
            }
        }

        // compute xyz_output values for training swatches
        let mut xyz_output = [[0.0; 3]; NUM_SWATCH_REFLECTANCES];
        let sensor_white_g = inner_product(sensor_illum, g);
        let sensor_white_y = inner_product(sensor_illum, Spectrum::get_cie(crate::spectra::CIE::Y));
        for i in 0..NUM_SWATCH_REFLECTANCES {
            let s = &SWATCH_REFLECTANCES[i];
            let xyz = Self::project_reflectance::<XYZ>(
                s,
                output_colorspace.illuminant.as_ref(),
                Spectrum::get_cie(crate::spectra::CIE::X),
                Spectrum::get_cie(crate::spectra::CIE::Y),
                Spectrum::get_cie(crate::spectra::CIE::Z),
            ) * (sensor_white_y / sensor_white_g);
            for c in 0..3 {
                xyz_output[i][c] = xyz[c];
            }
        }

        let m = linear_least_squares_3::<NUM_SWATCH_REFLECTANCES>(&rgb_camera, &xyz_output)
            .expect("Sensor XYZ from RGB matrix could not be solved");

        PixelSensor {
            xyz_from_sensor_rgb: m,
            r_bar: DenselySampled::new(&r),
            g_bar: DenselySampled::new(&g),
            b_bar: DenselySampled::new(&b),
            imaging_ratio: imaging_ratio,
        }
    }

    /// Converts a point-sampled spectrum distribution to RGB coefficients in the sensor's color space.
    pub fn to_sensor_rgb(&self, l: &SampledSpectrum, lambda: &SampledWavelengths) -> RGB {
        let l = l.safe_div(&lambda.pdf());
        RGB::new(
            (self.r_bar.sample(lambda) * l).average(),
            (self.g_bar.sample(lambda) * l).average(),
            (self.b_bar.sample(lambda) * l).average(),
        ) * self.imaging_ratio
    }

    // TODO we could further restrict to some "Triplet" T which only has 3 elements.
    /// TRIPLET is a color such as XYZ or RGB with 3 values that can be accessed and modified
    /// as the constraints specify.
    fn project_reflectance<TRIPLET>(
        ref1: &Spectrum,
        illum: &Spectrum,
        b1: &Spectrum,
        b2: &Spectrum,
        b3: &Spectrum,
    ) -> TRIPLET
    where
        TRIPLET:
            Default + IndexMut<usize> + MulAssign<Float> + std::ops::Div<f32, Output = TRIPLET>,
        <TRIPLET as Index<usize>>::Output: AddAssign<f32>,
    {
        let mut result = TRIPLET::default();
        let mut g_integral = 0.0;
        for lambda in (LAMBDA_MIN as i32)..=(LAMBDA_MAX as i32) {
            let lambda = lambda as Float;
            g_integral += b2.get(lambda) * illum.get(lambda);
            result[0] += b1.get(lambda) * ref1.get(lambda) * illum.get(lambda);
            result[1] += b2.get(lambda) * ref1.get(lambda) * illum.get(lambda);
            result[2] += b3.get(lambda) * ref1.get(lambda) * illum.get(lambda);
        }
        result / g_integral
    }
}

/// Information about a point on a surface which is visible from the camera.
pub struct VisibleSurface {
    /// Point in space
    p: Point3f,
    /// Normal
    n: Normal3f,
    /// Shading normal
    ns: Normal3f,
    uv: Point2f,
    time: Float,
    /// Partial derivative of depth at each pixel, where x and y are in raster space and
    /// z is in camera space.
    dpdx: Vector3f,
    /// Partial derivative of depth at each pixel, where x and y are in raster space and
    /// z is in camera space.
    dpdy: Vector3f,
    /// spectral distribution of reflected light under uniform illumination;
    /// useful for separating texture from illumination before denoising.
    albedo: SampledSpectrum,
    set: bool,
}

impl VisibleSurface {
    pub fn new(
        si: &SurfaceInteraction,
        albedo: &SampledSpectrum,
        lambda: &SampledWavelengths,
    ) -> VisibleSurface {
        let set = true;
        // TODO to set p from surface interaction, need to implement Point3fi -> Point3f, so we need to
        // implement Interval, and Point3fi, and that conversion... Let's do it.
        todo!()
    }
}

const NUM_SWATCH_REFLECTANCES: usize = 24;
const NUM_SWATCH_SPECTRUM_SAMPLES: usize = 72;
const HALF_NUM_SWATCH_SPECTRUM_SAMPLES: usize = 36;
// Swatch reflectances are taken from Danny Pascale's Macbeth chart measurements
// BabelColor ColorChecker data: Copyright (c) 2004-2012 Danny Pascale
// (www.babelcolor.com); used by permission.
// http://www.babelcolor.com/index_htm_files/ColorChecker_RGB_and_spectra.zip
static SWATCH_REFLECTANCES: Lazy<[Spectrum; NUM_SWATCH_REFLECTANCES]> = Lazy::new(|| {
    [
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.055, 390.0, 0.058, 400.0, 0.061, 410.0, 0.062, 420.0, 0.062, 430.0, 0.062,
                440.0, 0.062, 450.0, 0.062, 460.0, 0.062, 470.0, 0.062, 480.0, 0.062, 490.0, 0.063,
                500.0, 0.065, 510.0, 0.070, 520.0, 0.076, 530.0, 0.079, 540.0, 0.081, 550.0, 0.084,
                560.0, 0.091, 570.0, 0.103, 580.0, 0.119, 590.0, 0.134, 600.0, 0.143, 610.0, 0.147,
                620.0, 0.151, 630.0, 0.158, 640.0, 0.168, 650.0, 0.179, 660.0, 0.188, 670.0, 0.190,
                680.0, 0.186, 690.0, 0.181, 700.0, 0.182, 710.0, 0.187, 720.0, 0.196, 730.0, 0.209,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.117, 390.0, 0.143, 400.0, 0.175, 410.0, 0.191, 420.0, 0.196, 430.0, 0.199,
                440.0, 0.204, 450.0, 0.213, 460.0, 0.228, 470.0, 0.251, 480.0, 0.280, 490.0, 0.309,
                500.0, 0.329, 510.0, 0.333, 520.0, 0.315, 530.0, 0.286, 540.0, 0.273, 550.0, 0.276,
                560.0, 0.277, 570.0, 0.289, 580.0, 0.339, 590.0, 0.420, 600.0, 0.488, 610.0, 0.525,
                620.0, 0.546, 630.0, 0.562, 640.0, 0.578, 650.0, 0.595, 660.0, 0.612, 670.0, 0.625,
                680.0, 0.638, 690.0, 0.656, 700.0, 0.678, 710.0, 0.700, 720.0, 0.717, 730.0, 0.734,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.130, 390.0, 0.177, 400.0, 0.251, 410.0, 0.306, 420.0, 0.324, 430.0, 0.330,
                440.0, 0.333, 450.0, 0.331, 460.0, 0.323, 470.0, 0.311, 480.0, 0.298, 490.0, 0.285,
                500.0, 0.269, 510.0, 0.250, 520.0, 0.231, 530.0, 0.214, 540.0, 0.199, 550.0, 0.185,
                560.0, 0.169, 570.0, 0.157, 580.0, 0.149, 590.0, 0.145, 600.0, 0.142, 610.0, 0.141,
                620.0, 0.141, 630.0, 0.141, 640.0, 0.143, 650.0, 0.147, 660.0, 0.152, 670.0, 0.154,
                680.0, 0.150, 690.0, 0.144, 700.0, 0.136, 710.0, 0.132, 720.0, 0.135, 730.0, 0.147,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.051, 390.0, 0.054, 400.0, 0.056, 410.0, 0.057, 420.0, 0.058, 430.0, 0.059,
                440.0, 0.060, 450.0, 0.061, 460.0, 0.062, 470.0, 0.063, 480.0, 0.065, 490.0, 0.067,
                500.0, 0.075, 510.0, 0.101, 520.0, 0.145, 530.0, 0.178, 540.0, 0.184, 550.0, 0.170,
                560.0, 0.149, 570.0, 0.133, 580.0, 0.122, 590.0, 0.115, 600.0, 0.109, 610.0, 0.105,
                620.0, 0.104, 630.0, 0.106, 640.0, 0.109, 650.0, 0.112, 660.0, 0.114, 670.0, 0.114,
                680.0, 0.112, 690.0, 0.112, 700.0, 0.115, 710.0, 0.120, 720.0, 0.125, 730.0, 0.130,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.144, 390.0, 0.198, 400.0, 0.294, 410.0, 0.375, 420.0, 0.408, 430.0, 0.421,
                440.0, 0.426, 450.0, 0.426, 460.0, 0.419, 470.0, 0.403, 480.0, 0.379, 490.0, 0.346,
                500.0, 0.311, 510.0, 0.281, 520.0, 0.254, 530.0, 0.229, 540.0, 0.214, 550.0, 0.208,
                560.0, 0.202, 570.0, 0.194, 580.0, 0.193, 590.0, 0.200, 600.0, 0.214, 610.0, 0.230,
                620.0, 0.241, 630.0, 0.254, 640.0, 0.279, 650.0, 0.313, 660.0, 0.348, 670.0, 0.366,
                680.0, 0.366, 690.0, 0.359, 700.0, 0.358, 710.0, 0.365, 720.0, 0.377, 730.0, 0.398,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.136, 390.0, 0.179, 400.0, 0.247, 410.0, 0.297, 420.0, 0.320, 430.0, 0.337,
                440.0, 0.355, 450.0, 0.381, 460.0, 0.419, 470.0, 0.466, 480.0, 0.510, 490.0, 0.546,
                500.0, 0.567, 510.0, 0.574, 520.0, 0.569, 530.0, 0.551, 540.0, 0.524, 550.0, 0.488,
                560.0, 0.445, 570.0, 0.400, 580.0, 0.350, 590.0, 0.299, 600.0, 0.252, 610.0, 0.221,
                620.0, 0.204, 630.0, 0.196, 640.0, 0.191, 650.0, 0.188, 660.0, 0.191, 670.0, 0.199,
                680.0, 0.212, 690.0, 0.223, 700.0, 0.232, 710.0, 0.233, 720.0, 0.229, 730.0, 0.229,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.054, 390.0, 0.054, 400.0, 0.053, 410.0, 0.054, 420.0, 0.054, 430.0, 0.055,
                440.0, 0.055, 450.0, 0.055, 460.0, 0.056, 470.0, 0.057, 480.0, 0.058, 490.0, 0.061,
                500.0, 0.068, 510.0, 0.089, 520.0, 0.125, 530.0, 0.154, 540.0, 0.174, 550.0, 0.199,
                560.0, 0.248, 570.0, 0.335, 580.0, 0.444, 590.0, 0.538, 600.0, 0.587, 610.0, 0.595,
                620.0, 0.591, 630.0, 0.587, 640.0, 0.584, 650.0, 0.584, 660.0, 0.590, 670.0, 0.603,
                680.0, 0.620, 690.0, 0.639, 700.0, 0.655, 710.0, 0.663, 720.0, 0.663, 730.0, 0.667,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.122, 390.0, 0.164, 400.0, 0.229, 410.0, 0.286, 420.0, 0.327, 430.0, 0.361,
                440.0, 0.388, 450.0, 0.400, 460.0, 0.392, 470.0, 0.362, 480.0, 0.316, 490.0, 0.260,
                500.0, 0.209, 510.0, 0.168, 520.0, 0.138, 530.0, 0.117, 540.0, 0.104, 550.0, 0.096,
                560.0, 0.090, 570.0, 0.086, 580.0, 0.084, 590.0, 0.084, 600.0, 0.084, 610.0, 0.084,
                620.0, 0.084, 630.0, 0.085, 640.0, 0.090, 650.0, 0.098, 660.0, 0.109, 670.0, 0.123,
                680.0, 0.143, 690.0, 0.169, 700.0, 0.205, 710.0, 0.244, 720.0, 0.287, 730.0, 0.332,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.096, 390.0, 0.115, 400.0, 0.131, 410.0, 0.135, 420.0, 0.133, 430.0, 0.132,
                440.0, 0.130, 450.0, 0.128, 460.0, 0.125, 470.0, 0.120, 480.0, 0.115, 490.0, 0.110,
                500.0, 0.105, 510.0, 0.100, 520.0, 0.095, 530.0, 0.093, 540.0, 0.092, 550.0, 0.093,
                560.0, 0.096, 570.0, 0.108, 580.0, 0.156, 590.0, 0.265, 600.0, 0.399, 610.0, 0.500,
                620.0, 0.556, 630.0, 0.579, 640.0, 0.588, 650.0, 0.591, 660.0, 0.593, 670.0, 0.594,
                680.0, 0.598, 690.0, 0.602, 700.0, 0.607, 710.0, 0.609, 720.0, 0.609, 730.0, 0.610,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.092, 390.0, 0.116, 400.0, 0.146, 410.0, 0.169, 420.0, 0.178, 430.0, 0.173,
                440.0, 0.158, 450.0, 0.139, 460.0, 0.119, 470.0, 0.101, 480.0, 0.087, 490.0, 0.075,
                500.0, 0.066, 510.0, 0.060, 520.0, 0.056, 530.0, 0.053, 540.0, 0.051, 550.0, 0.051,
                560.0, 0.052, 570.0, 0.052, 580.0, 0.051, 590.0, 0.052, 600.0, 0.058, 610.0, 0.073,
                620.0, 0.096, 630.0, 0.119, 640.0, 0.141, 650.0, 0.166, 660.0, 0.194, 670.0, 0.227,
                680.0, 0.265, 690.0, 0.309, 700.0, 0.355, 710.0, 0.396, 720.0, 0.436, 730.0, 0.478,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.061, 390.0, 0.061, 400.0, 0.062, 410.0, 0.063, 420.0, 0.064, 430.0, 0.066,
                440.0, 0.069, 450.0, 0.075, 460.0, 0.085, 470.0, 0.105, 480.0, 0.139, 490.0, 0.192,
                500.0, 0.271, 510.0, 0.376, 520.0, 0.476, 530.0, 0.531, 540.0, 0.549, 550.0, 0.546,
                560.0, 0.528, 570.0, 0.504, 580.0, 0.471, 590.0, 0.428, 600.0, 0.381, 610.0, 0.347,
                620.0, 0.327, 630.0, 0.318, 640.0, 0.312, 650.0, 0.310, 660.0, 0.314, 670.0, 0.327,
                680.0, 0.345, 690.0, 0.363, 700.0, 0.376, 710.0, 0.381, 720.0, 0.378, 730.0, 0.379,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.063, 390.0, 0.063, 400.0, 0.063, 410.0, 0.064, 420.0, 0.064, 430.0, 0.064,
                440.0, 0.065, 450.0, 0.066, 460.0, 0.067, 470.0, 0.068, 480.0, 0.071, 490.0, 0.076,
                500.0, 0.087, 510.0, 0.125, 520.0, 0.206, 530.0, 0.305, 540.0, 0.383, 550.0, 0.431,
                560.0, 0.469, 570.0, 0.518, 580.0, 0.568, 590.0, 0.607, 600.0, 0.628, 610.0, 0.637,
                620.0, 0.640, 630.0, 0.642, 640.0, 0.645, 650.0, 0.648, 660.0, 0.651, 670.0, 0.653,
                680.0, 0.657, 690.0, 0.664, 700.0, 0.673, 710.0, 0.680, 720.0, 0.684, 730.0, 0.688,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.066, 390.0, 0.079, 400.0, 0.102, 410.0, 0.146, 420.0, 0.200, 430.0, 0.244,
                440.0, 0.282, 450.0, 0.309, 460.0, 0.308, 470.0, 0.278, 480.0, 0.231, 490.0, 0.178,
                500.0, 0.130, 510.0, 0.094, 520.0, 0.070, 530.0, 0.054, 540.0, 0.046, 550.0, 0.042,
                560.0, 0.039, 570.0, 0.038, 580.0, 0.038, 590.0, 0.038, 600.0, 0.038, 610.0, 0.039,
                620.0, 0.039, 630.0, 0.040, 640.0, 0.041, 650.0, 0.042, 660.0, 0.044, 670.0, 0.045,
                680.0, 0.046, 690.0, 0.046, 700.0, 0.048, 710.0, 0.052, 720.0, 0.057, 730.0, 0.065,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.052, 390.0, 0.053, 400.0, 0.054, 410.0, 0.055, 420.0, 0.057, 430.0, 0.059,
                440.0, 0.061, 450.0, 0.066, 460.0, 0.075, 470.0, 0.093, 480.0, 0.125, 490.0, 0.178,
                500.0, 0.246, 510.0, 0.307, 520.0, 0.337, 530.0, 0.334, 540.0, 0.317, 550.0, 0.293,
                560.0, 0.262, 570.0, 0.230, 580.0, 0.198, 590.0, 0.165, 600.0, 0.135, 610.0, 0.115,
                620.0, 0.104, 630.0, 0.098, 640.0, 0.094, 650.0, 0.092, 660.0, 0.093, 670.0, 0.097,
                680.0, 0.102, 690.0, 0.108, 700.0, 0.113, 710.0, 0.115, 720.0, 0.114, 730.0, 0.114,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.050, 390.0, 0.049, 400.0, 0.048, 410.0, 0.047, 420.0, 0.047, 430.0, 0.047,
                440.0, 0.047, 450.0, 0.047, 460.0, 0.046, 470.0, 0.045, 480.0, 0.044, 490.0, 0.044,
                500.0, 0.045, 510.0, 0.046, 520.0, 0.047, 530.0, 0.048, 540.0, 0.049, 550.0, 0.050,
                560.0, 0.054, 570.0, 0.060, 580.0, 0.072, 590.0, 0.104, 600.0, 0.178, 610.0, 0.312,
                620.0, 0.467, 630.0, 0.581, 640.0, 0.644, 650.0, 0.675, 660.0, 0.690, 670.0, 0.698,
                680.0, 0.706, 690.0, 0.715, 700.0, 0.724, 710.0, 0.730, 720.0, 0.734, 730.0, 0.738,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.058, 390.0, 0.054, 400.0, 0.052, 410.0, 0.052, 420.0, 0.053, 430.0, 0.054,
                440.0, 0.056, 450.0, 0.059, 460.0, 0.067, 470.0, 0.081, 480.0, 0.107, 490.0, 0.152,
                500.0, 0.225, 510.0, 0.336, 520.0, 0.462, 530.0, 0.559, 540.0, 0.616, 550.0, 0.650,
                560.0, 0.672, 570.0, 0.694, 580.0, 0.710, 590.0, 0.723, 600.0, 0.731, 610.0, 0.739,
                620.0, 0.746, 630.0, 0.752, 640.0, 0.758, 650.0, 0.764, 660.0, 0.769, 670.0, 0.771,
                680.0, 0.776, 690.0, 0.782, 700.0, 0.790, 710.0, 0.796, 720.0, 0.799, 730.0, 0.804,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.145, 390.0, 0.195, 400.0, 0.283, 410.0, 0.346, 420.0, 0.362, 430.0, 0.354,
                440.0, 0.334, 450.0, 0.306, 460.0, 0.276, 470.0, 0.248, 480.0, 0.218, 490.0, 0.190,
                500.0, 0.168, 510.0, 0.149, 520.0, 0.127, 530.0, 0.107, 540.0, 0.100, 550.0, 0.102,
                560.0, 0.104, 570.0, 0.109, 580.0, 0.137, 590.0, 0.200, 600.0, 0.290, 610.0, 0.400,
                620.0, 0.516, 630.0, 0.615, 640.0, 0.687, 650.0, 0.732, 660.0, 0.760, 670.0, 0.774,
                680.0, 0.783, 690.0, 0.793, 700.0, 0.803, 710.0, 0.812, 720.0, 0.817, 730.0, 0.825,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.108, 390.0, 0.141, 400.0, 0.192, 410.0, 0.236, 420.0, 0.261, 430.0, 0.286,
                440.0, 0.317, 450.0, 0.353, 460.0, 0.390, 470.0, 0.426, 480.0, 0.446, 490.0, 0.444,
                500.0, 0.423, 510.0, 0.385, 520.0, 0.337, 530.0, 0.283, 540.0, 0.231, 550.0, 0.185,
                560.0, 0.146, 570.0, 0.118, 580.0, 0.101, 590.0, 0.090, 600.0, 0.082, 610.0, 0.076,
                620.0, 0.074, 630.0, 0.073, 640.0, 0.073, 650.0, 0.074, 660.0, 0.076, 670.0, 0.077,
                680.0, 0.076, 690.0, 0.075, 700.0, 0.073, 710.0, 0.072, 720.0, 0.074, 730.0, 0.079,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.189, 390.0, 0.255, 400.0, 0.423, 410.0, 0.660, 420.0, 0.811, 430.0, 0.862,
                440.0, 0.877, 450.0, 0.884, 460.0, 0.891, 470.0, 0.896, 480.0, 0.899, 490.0, 0.904,
                500.0, 0.907, 510.0, 0.909, 520.0, 0.911, 530.0, 0.910, 540.0, 0.911, 550.0, 0.914,
                560.0, 0.913, 570.0, 0.916, 580.0, 0.915, 590.0, 0.916, 600.0, 0.914, 610.0, 0.915,
                620.0, 0.918, 630.0, 0.919, 640.0, 0.921, 650.0, 0.923, 660.0, 0.924, 670.0, 0.922,
                680.0, 0.922, 690.0, 0.925, 700.0, 0.927, 710.0, 0.930, 720.0, 0.930, 730.0, 0.933,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.171, 390.0, 0.232, 400.0, 0.365, 410.0, 0.507, 420.0, 0.567, 430.0, 0.583,
                440.0, 0.588, 450.0, 0.590, 460.0, 0.591, 470.0, 0.590, 480.0, 0.588, 490.0, 0.588,
                500.0, 0.589, 510.0, 0.589, 520.0, 0.591, 530.0, 0.590, 540.0, 0.590, 550.0, 0.590,
                560.0, 0.589, 570.0, 0.591, 580.0, 0.590, 590.0, 0.590, 600.0, 0.587, 610.0, 0.585,
                620.0, 0.583, 630.0, 0.580, 640.0, 0.578, 650.0, 0.576, 660.0, 0.574, 670.0, 0.572,
                680.0, 0.571, 690.0, 0.569, 700.0, 0.568, 710.0, 0.568, 720.0, 0.566, 730.0, 0.566,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.144, 390.0, 0.192, 400.0, 0.272, 410.0, 0.331, 420.0, 0.350, 430.0, 0.357,
                440.0, 0.361, 450.0, 0.363, 460.0, 0.363, 470.0, 0.361, 480.0, 0.359, 490.0, 0.358,
                500.0, 0.358, 510.0, 0.359, 520.0, 0.360, 530.0, 0.360, 540.0, 0.361, 550.0, 0.361,
                560.0, 0.360, 570.0, 0.362, 580.0, 0.362, 590.0, 0.361, 600.0, 0.359, 610.0, 0.358,
                620.0, 0.355, 630.0, 0.352, 640.0, 0.350, 650.0, 0.348, 660.0, 0.345, 670.0, 0.343,
                680.0, 0.340, 690.0, 0.338, 700.0, 0.335, 710.0, 0.334, 720.0, 0.332, 730.0, 0.331,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.105, 390.0, 0.131, 400.0, 0.163, 410.0, 0.180, 420.0, 0.186, 430.0, 0.190,
                440.0, 0.193, 450.0, 0.194, 460.0, 0.194, 470.0, 0.192, 480.0, 0.191, 490.0, 0.191,
                500.0, 0.191, 510.0, 0.192, 520.0, 0.192, 530.0, 0.192, 540.0, 0.192, 550.0, 0.192,
                560.0, 0.192, 570.0, 0.193, 580.0, 0.192, 590.0, 0.192, 600.0, 0.191, 610.0, 0.189,
                620.0, 0.188, 630.0, 0.186, 640.0, 0.184, 650.0, 0.182, 660.0, 0.181, 670.0, 0.179,
                680.0, 0.178, 690.0, 0.176, 700.0, 0.174, 710.0, 0.173, 720.0, 0.172, 730.0, 0.171,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.068, 390.0, 0.077, 400.0, 0.084, 410.0, 0.087, 420.0, 0.089, 430.0, 0.090,
                440.0, 0.092, 450.0, 0.092, 460.0, 0.091, 470.0, 0.090, 480.0, 0.090, 490.0, 0.090,
                500.0, 0.090, 510.0, 0.090, 520.0, 0.090, 530.0, 0.090, 540.0, 0.090, 550.0, 0.090,
                560.0, 0.090, 570.0, 0.090, 580.0, 0.090, 590.0, 0.089, 600.0, 0.089, 610.0, 0.088,
                620.0, 0.087, 630.0, 0.086, 640.0, 0.086, 650.0, 0.085, 660.0, 0.084, 670.0, 0.084,
                680.0, 0.083, 690.0, 0.083, 700.0, 0.082, 710.0, 0.081, 720.0, 0.081, 730.0, 0.081,
            ],
            false,
        )),
        Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<
            NUM_SWATCH_SPECTRUM_SAMPLES,
            HALF_NUM_SWATCH_SPECTRUM_SAMPLES,
        >(
            &[
                380.0, 0.031, 390.0, 0.032, 400.0, 0.032, 410.0, 0.033, 420.0, 0.033, 430.0, 0.033,
                440.0, 0.033, 450.0, 0.033, 460.0, 0.032, 470.0, 0.032, 480.0, 0.032, 490.0, 0.032,
                500.0, 0.032, 510.0, 0.032, 520.0, 0.032, 530.0, 0.032, 540.0, 0.032, 550.0, 0.032,
                560.0, 0.032, 570.0, 0.032, 580.0, 0.032, 590.0, 0.032, 600.0, 0.032, 610.0, 0.032,
                620.0, 0.032, 630.0, 0.032, 640.0, 0.032, 650.0, 0.032, 660.0, 0.032, 670.0, 0.032,
                680.0, 0.032, 690.0, 0.032, 700.0, 0.032, 710.0, 0.032, 720.0, 0.032, 730.0, 0.033,
            ],
            false,
        )),
    ]
});
