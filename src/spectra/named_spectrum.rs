/// Provides named spectra.
/// Largely separated from the spectrum module to isolate large amounts of embedded
/// data in its own file; this and spectrum should be re-exported to the top of the
/// module anyways, so this just makes things more readable.
use once_cell::sync::Lazy;

use crate::Float;

use super::{PiecewiseLinearSpectrum, Spectrum};

// TODO add other named spectra. I'm happy to just have the system in place for
// now though, we can add them as we need them or when I have time to go through
// and copy/generate all the data.
pub enum NamedSpectrum {
    StdIllumD65,
    IllumAcesD60,
    GlassBk7,
    GlassBaf10,
}

impl NamedSpectrum {
    pub fn from_str(name: &str) -> Option<NamedSpectrum> {
        match name {
            "StdIllum-D65" => Some(NamedSpectrum::StdIllumD65),
            "illum-acesD60" => Some(NamedSpectrum::IllumAcesD60),
            "glass-BK7" => Some(NamedSpectrum::GlassBk7),
            "glass-baf10" => Some(NamedSpectrum::GlassBaf10),
            _ => None,
        }
    }
}

// NOTE: These intentionally use static, not const.
// const will compile, but will not work properly. See once_cell documentation.
pub static STD_ILLUM_D65: Lazy<Spectrum> = Lazy::new(|| {
    Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::from_interleaved::<214, 107>(
        &CIE_ILLUM_D6500,
        true,
    ))
});

pub static ILLUM_ACES_D60: Lazy<Spectrum> = Lazy::new(|| {
    Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::from_interleaved::<214, 107>(
        &ACES_ILLUM_D60,
        true,
    ))
});

pub static GLASS_BK7_ETA: Lazy<Spectrum> = Lazy::new(|| {
    Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::from_interleaved::<58, 24>(
        &GLASS_BK7_ETA_SAMPLES,
        false,
    ))
});

pub static GLASS_BAF10_ETA: Lazy<Spectrum> = Lazy::new(|| {
    Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::from_interleaved::<54, 27>(
        &GLASS_BAF10_ETA_SAMPLES,
        false,
    ))
});

const CIE_ILLUM_D6500: [Float; 214] = [
    300.000000, 0.034100, 305.000000, 1.664300, 310.000000, 3.294500, 315.000000, 11.765200,
    320.000000, 20.236000, 325.000000, 28.644699, 330.000000, 37.053501, 335.000000, 38.501099,
    340.000000, 39.948799, 345.000000, 42.430199, 350.000000, 44.911701, 355.000000, 45.775002,
    360.000000, 46.638302, 365.000000, 49.363701, 370.000000, 52.089100, 375.000000, 51.032299,
    380.000000, 49.975498, 385.000000, 52.311798, 390.000000, 54.648201, 395.000000, 68.701500,
    400.000000, 82.754898, 405.000000, 87.120399, 410.000000, 91.486000, 415.000000, 92.458900,
    420.000000, 93.431801, 425.000000, 90.056999, 430.000000, 86.682297, 435.000000, 95.773598,
    440.000000, 104.864998, 445.000000, 110.935997, 450.000000, 117.008003, 455.000000, 117.410004,
    460.000000, 117.811996, 465.000000, 116.335999, 470.000000, 114.861000, 475.000000, 115.391998,
    480.000000, 115.922997, 485.000000, 112.366997, 490.000000, 108.810997, 495.000000, 109.082001,
    500.000000, 109.353996, 505.000000, 108.578003, 510.000000, 107.802002, 515.000000, 106.295998,
    520.000000, 104.790001, 525.000000, 106.238998, 530.000000, 107.689003, 535.000000, 106.046997,
    540.000000, 104.404999, 545.000000, 104.224998, 550.000000, 104.045998, 555.000000, 102.023003,
    560.000000, 100.000000, 565.000000, 98.167099, 570.000000, 96.334198, 575.000000, 96.061096,
    580.000000, 95.788002, 585.000000, 92.236801, 590.000000, 88.685600, 595.000000, 89.345901,
    600.000000, 90.006203, 605.000000, 89.802597, 610.000000, 89.599098, 615.000000, 88.648903,
    620.000000, 87.698700, 625.000000, 85.493599, 630.000000, 83.288597, 635.000000, 83.493896,
    640.000000, 83.699203, 645.000000, 81.862999, 650.000000, 80.026802, 655.000000, 80.120697,
    660.000000, 80.214600, 665.000000, 81.246201, 670.000000, 82.277802, 675.000000, 80.280998,
    680.000000, 78.284203, 685.000000, 74.002701, 690.000000, 69.721298, 695.000000, 70.665199,
    700.000000, 71.609100, 705.000000, 72.978996, 710.000000, 74.348999, 715.000000, 67.976501,
    720.000000, 61.604000, 725.000000, 65.744797, 730.000000, 69.885597, 735.000000, 72.486298,
    740.000000, 75.086998, 745.000000, 69.339798, 750.000000, 63.592701, 755.000000, 55.005402,
    760.000000, 46.418201, 765.000000, 56.611801, 770.000000, 66.805397, 775.000000, 65.094101,
    780.000000, 63.382801, 785.000000, 63.843399, 790.000000, 64.304001, 795.000000, 61.877899,
    800.000000, 59.451900, 805.000000, 55.705399, 810.000000, 51.959000, 815.000000, 54.699799,
    820.000000, 57.440601, 825.000000, 58.876499, 830.000000, 60.312500,
];

const GLASS_BK7_ETA_SAMPLES: [Float; 58] = [
    300.0,
    1.5527702635739,
    322.0,
    1.5458699289209,
    344.0,
    1.5404466868331,
    366.0,
    1.536090527917,
    388.0,
    1.53252773217,
    410.0,
    1.529568767224,
    432.0,
    1.5270784291406,
    454.0,
    1.5249578457324,
    476.0,
    1.5231331738499,
    498.0,
    1.5215482528369,
    520.0,
    1.5201596882463,
    542.0,
    1.5189334783109,
    564.0,
    1.5178426478869,
    586.0,
    1.516865556749,
    608.0,
    1.5159846691816,
    630.0,
    1.5151856452759,
    652.0,
    1.5144566604975,
    674.0,
    1.513787889767,
    696.0,
    1.5131711117948,
    718.0,
    1.5125994024544,
    740.0,
    1.5120668948646,
    762.0,
    1.5115685899969,
    784.0,
    1.5111002059336,
    806.0,
    1.5106580569705,
    828.0,
    1.5102389559626,
    850.0,
    1.5098401349174,
    872.0,
    1.5094591800239,
    894.0,
    1.5090939781792,
    916.0,
    1.5087426727363,
];

// Via https://gist.github.com/aforsythe/4df4e5377853df76a5a83a3c001c7eeb
// with the critial bugfix:
// <    cct = 6000
// --
// >    cct = 6000.
const ACES_ILLUM_D60: [Float; 214] = [
    300.0, 0.02928, 305.0, 1.28964, 310.0, 2.55, 315.0, 9.0338, 320.0, 15.5176, 325.0, 21.94705,
    330.0, 28.3765, 335.0, 29.93335, 340.0, 31.4902, 345.0, 33.75765, 350.0, 36.0251, 355.0,
    37.2032, 360.0, 38.3813, 365.0, 40.6445, 370.0, 42.9077, 375.0, 42.05735, 380.0, 41.207, 385.0,
    43.8121, 390.0, 46.4172, 395.0, 59.26285, 400.0, 72.1085, 405.0, 76.1756, 410.0, 80.2427,
    415.0, 81.4878, 420.0, 82.7329, 425.0, 80.13505, 430.0, 77.5372, 435.0, 86.5577, 440.0,
    95.5782, 445.0, 101.72045, 450.0, 107.8627, 455.0, 108.67115, 460.0, 109.4796, 465.0, 108.5873,
    470.0, 107.695, 475.0, 108.6598, 480.0, 109.6246, 485.0, 106.6426, 490.0, 103.6606, 495.0,
    104.42795, 500.0, 105.1953, 505.0, 104.7974, 510.0, 104.3995, 515.0, 103.45635, 520.0,
    102.5132, 525.0, 104.2813, 530.0, 106.0494, 535.0, 104.67885, 540.0, 103.3083, 545.0, 103.4228,
    550.0, 103.5373, 555.0, 101.76865, 560.0, 100.0, 565.0, 98.3769, 570.0, 96.7538, 575.0,
    96.73515, 580.0, 96.7165, 585.0, 93.3013, 590.0, 89.8861, 595.0, 90.91705, 600.0, 91.948,
    605.0, 91.98965, 610.0, 92.0313, 615.0, 91.3008, 620.0, 90.5703, 625.0, 88.5077, 630.0,
    86.4451, 635.0, 86.9551, 640.0, 87.4651, 645.0, 85.6558, 650.0, 83.8465, 655.0, 84.20755,
    660.0, 84.5686, 665.0, 85.9432, 670.0, 87.3178, 675.0, 85.3068, 680.0, 83.2958, 685.0,
    78.66005, 690.0, 74.0243, 695.0, 75.23535, 700.0, 76.4464, 705.0, 77.67465, 710.0, 78.9029,
    715.0, 72.12575, 720.0, 65.3486, 725.0, 69.6609, 730.0, 73.9732, 735.0, 76.6802, 740.0,
    79.3872, 745.0, 73.28855, 750.0, 67.1899, 755.0, 58.18595, 760.0, 49.182, 765.0, 59.9723,
    770.0, 70.7626, 775.0, 68.9039, 780.0, 67.0452, 785.0, 67.5469, 790.0, 68.0486, 795.0, 65.4631,
    800.0, 62.8776, 805.0, 58.88595, 810.0, 54.8943, 815.0, 57.8066, 820.0, 60.7189, 825.0,
    62.2491, 830.0, 63.7793,
];

const GLASS_BAF10_ETA_SAMPLES: [Float; 54] = [
    350.0,
    1.7126880848268,
    371.0,
    1.7044510025682,
    393.0,
    1.6978539633931,
    414.0,
    1.6924597573902,
    436.0,
    1.6879747521657,
    457.0,
    1.6841935148947,
    479.0,
    1.6809676313681,
    500.0,
    1.6781870617363,
    522.0,
    1.6757684467878,
    543.0,
    1.6736474831891,
    565.0,
    1.6717737892968,
    586.0,
    1.6701073530462,
    608.0,
    1.6686160168249,
    629.0,
    1.6672736605352,
    651.0,
    1.6660588657981,
    672.0,
    1.6649539185393,
    694.0,
    1.6639440538738,
    715.0,
    1.6630168772865,
    737.0,
    1.6621619159417,
    758.0,
    1.6613702672977,
    780.0,
    1.6606343213443,
    801.0,
    1.6599475391478,
    823.0,
    1.6593042748862,
    844.0,
    1.6586996317841,
    866.0,
    1.6581293446924,
    887.0,
    1.6575896837763,
    909.0,
    1.6570773750475,
];
