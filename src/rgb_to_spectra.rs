use once_cell::sync::Lazy;
use rgb2spec::{self, RGB2Spec};

use crate::color::RGB;

#[derive(Debug, PartialEq, Clone)]
pub enum Gamut {
    SRGB,
    XYZ,
    ERGB,
    Aces2065_1,
    ProPhotoRGB,
    Rec2020,
}

pub fn get_rgb_to_spec(gamut: &Gamut, rgb: &RGB) -> [f32; 3] {
    match gamut {
        Gamut::SRGB => Lazy::force(&SRGB_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::XYZ => Lazy::force(&XYZ_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::ERGB => Lazy::force(&ERGB_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::Aces2065_1 => Lazy::force(&ACES_2065_1_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::ProPhotoRGB => Lazy::force(&PROPHOTORGB_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
        Gamut::Rec2020 => Lazy::force(&REC2020_RGB2SPEC).fetch(<[f32; 3]>::from(rgb)),
    }
}

// Note that files in rgbtospec/ were generated with a resolution of 64; this matches the cmake
// command used by PBRT for generating their tables.
pub static SRGB_RGB2SPEC: Lazy<RGB2Spec> =
    Lazy::new(|| RGB2Spec::load("rgbtospec/srgb.spec").unwrap());

pub static XYZ_RGB2SPEC: Lazy<RGB2Spec> =
    Lazy::new(|| RGB2Spec::load("rgbtospec/xyz.spec").unwrap());

pub static ERGB_RGB2SPEC: Lazy<RGB2Spec> =
    Lazy::new(|| RGB2Spec::load("rgbtospec/ergb.spec").unwrap());

pub static ACES_2065_1_RGB2SPEC: Lazy<RGB2Spec> =
    Lazy::new(|| RGB2Spec::load("rgbtospec/aces2065_1.spec").unwrap());

pub static PROPHOTORGB_RGB2SPEC: Lazy<RGB2Spec> =
    Lazy::new(|| RGB2Spec::load("rgbtospec/prophotorgb.spec").unwrap());

pub static REC2020_RGB2SPEC: Lazy<RGB2Spec> =
    Lazy::new(|| RGB2Spec::load("rgbtospec/rec2020.spec").unwrap());
