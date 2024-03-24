use std::sync::Arc;

/// Provides named spectra.
/// Largely separated from the spectrum module to isolate large amounts of embedded
/// data in its own file; this and spectrum should be re-exported to the top of the
/// module anyways, so this just makes things more readable.
use once_cell::sync::Lazy;

use crate::Float;

use super::{cie::NUM_CIES_SAMPLES, PiecewiseLinearSpectrum, Spectrum};

pub enum NamedSpectrum {
    StdIllumD65,
    IllumAcesD60,
    GlassBk7,
    GlassBaf10,
    CuEta,
    CuK,
    AuEta,
    AuK,
}

impl NamedSpectrum {
    pub fn from_str(name: &str) -> Option<NamedSpectrum> {
        match name {
            "StdIllum-D65" => Some(NamedSpectrum::StdIllumD65),
            "illum-acesD60" => Some(NamedSpectrum::IllumAcesD60),
            "glass-BK7" => Some(NamedSpectrum::GlassBk7),
            "glass-baf10" => Some(NamedSpectrum::GlassBaf10),
            "metal-Cu-eta" => Some(NamedSpectrum::CuEta),
            "metal-Cu-k" => Some(NamedSpectrum::CuK),
            "metal-Au-eta" => Some(NamedSpectrum::AuEta),
            "metal-Au-k" => Some(NamedSpectrum::AuK),
            _ => None,
        }
    }
}

// NOTE: These intentionally use static, not const.
// const will compile, but will not work properly. See once_cell documentation.
pub static STD_ILLUM_D65: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<214, 107>(&CIE_ILLUM_D6500, true),
    ))
});

pub static ILLUM_ACES_D60: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<214, 107>(&ACES_ILLUM_D60, true),
    ))
});

pub static GLASS_BK7_ETA: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<58, 29>(&GLASS_BK7_ETA_SAMPLES, false),
    ))
});

pub static GLASS_BAF10_ETA: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<54, 27>(&GLASS_BAF10_ETA_SAMPLES, false),
    ))
});

pub static CU_ETA: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<112, 56>(&CU_ETA_SAMPLES, false),
    ))
});

pub static CU_K: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<112, 56>(&CU_K_SAMPLES, false),
    ))
});

pub static AU_ETA: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<112, 56>(&AU_ETA_SAMPLES, false),
    ))
});

pub static AU_K: Lazy<Arc<Spectrum>> = Lazy::new(|| {
    Arc::new(Spectrum::PiecewiseLinear(
        PiecewiseLinearSpectrum::from_interleaved::<112, 56>(&AU_K_SAMPLES, false),
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

// CIE Illuminant D S Basis Functions
pub const CIE_S_LAMBDA: [Float; NUM_CIES_SAMPLES] = [
    300.000000, 305.000000, 310.000000, 315.000000, 320.000000, 325.000000, 330.000000, 335.000000,
    340.000000, 345.000000, 350.000000, 355.000000, 360.000000, 365.000000, 370.000000, 375.000000,
    380.000000, 385.000000, 390.000000, 395.000000, 400.000000, 405.000000, 410.000000, 415.000000,
    420.000000, 425.000000, 430.000000, 435.000000, 440.000000, 445.000000, 450.000000, 455.000000,
    460.000000, 465.000000, 470.000000, 475.000000, 480.000000, 485.000000, 490.000000, 495.000000,
    500.000000, 505.000000, 510.000000, 515.000000, 520.000000, 525.000000, 530.000000, 535.000000,
    540.000000, 545.000000, 550.000000, 555.000000, 560.000000, 565.000000, 570.000000, 575.000000,
    580.000000, 585.000000, 590.000000, 595.000000, 600.000000, 605.000000, 610.000000, 615.000000,
    620.000000, 625.000000, 630.000000, 635.000000, 640.000000, 645.000000, 650.000000, 655.000000,
    660.000000, 665.000000, 670.000000, 675.000000, 680.000000, 685.000000, 690.000000, 695.000000,
    700.000000, 705.000000, 710.000000, 715.000000, 720.000000, 725.000000, 730.000000, 735.000000,
    740.000000, 745.000000, 750.000000, 755.000000, 760.000000, 765.000000, 770.000000, 775.000000,
    780.000000, 785.000000, 790.000000, 795.000000, 800.000000, 805.000000, 810.000000, 815.000000,
    820.000000, 825.000000, 830.000000,
];

pub const CIE_S0: [Float; NUM_CIES_SAMPLES] = [
    0.040000, 3.020000, 6.000000, 17.800000, 29.600000, 42.450000, 55.300000, 56.300000, 57.300000,
    59.550000, 61.800000, 61.650000, 61.500000, 65.150000, 68.800000, 66.100000, 63.400000,
    64.600000, 65.800000, 80.300000, 94.800000, 99.800000, 104.800000, 105.350000, 105.900000,
    101.350000, 96.800000, 105.350000, 113.900000, 119.750000, 125.600000, 125.550000, 125.500000,
    123.400000, 121.300000, 121.300000, 121.300000, 117.400000, 113.500000, 113.300000, 113.100000,
    111.950000, 110.800000, 108.650000, 106.500000, 107.650000, 108.800000, 107.050000, 105.300000,
    104.850000, 104.400000, 102.200000, 100.000000, 98.000000, 96.000000, 95.550000, 95.100000,
    92.100000, 89.100000, 89.800000, 90.500000, 90.400000, 90.300000, 89.350000, 88.400000,
    86.200000, 84.000000, 84.550000, 85.100000, 83.500000, 81.900000, 82.250000, 82.600000,
    83.750000, 84.900000, 83.100000, 81.300000, 76.600000, 71.900000, 73.100000, 74.300000,
    75.350000, 76.400000, 69.850000, 63.300000, 67.500000, 71.700000, 74.350000, 77.000000,
    71.100000, 65.200000, 56.450000, 47.700000, 58.150000, 68.600000, 66.800000, 65.000000,
    65.500000, 66.000000, 63.500000, 61.000000, 57.150000, 53.300000, 56.100000, 58.900000,
    60.400000, 61.900000,
];

pub const CIE_S1: [Float; NUM_CIES_SAMPLES] = [
    0.020000, 2.260000, 4.500000, 13.450000, 22.400000, 32.200000, 42.000000, 41.300000, 40.600000,
    41.100000, 41.600000, 39.800000, 38.000000, 40.200000, 42.400000, 40.450000, 38.500000,
    36.750000, 35.000000, 39.200000, 43.400000, 44.850000, 46.300000, 45.100000, 43.900000,
    40.500000, 37.100000, 36.900000, 36.700000, 36.300000, 35.900000, 34.250000, 32.600000,
    30.250000, 27.900000, 26.100000, 24.300000, 22.200000, 20.100000, 18.150000, 16.200000,
    14.700000, 13.200000, 10.900000, 8.600000, 7.350000, 6.100000, 5.150000, 4.200000, 3.050000,
    1.900000, 0.950000, -0.000000, -0.800000, -1.600000, -2.550000, -3.500000, -3.500000,
    -3.500000, -4.650000, -5.800000, -6.500000, -7.200000, -7.900000, -8.600000, -9.050000,
    -9.500000, -10.200000, -10.900000, -10.800000, -10.700000, -11.350000, -12.000000, -13.000000,
    -14.000000, -13.800000, -13.600000, -12.800000, -12.000000, -12.650000, -13.300000, -13.100000,
    -12.900000, -11.750000, -10.600000, -11.100000, -11.600000, -11.900000, -12.200000, -11.200000,
    -10.200000, -9.000000, -7.800000, -9.500000, -11.200000, -10.800000, -10.400000, -10.500000,
    -10.600000, -10.150000, -9.700000, -9.000000, -8.300000, -8.800000, -9.300000, -9.550000,
    -9.800000,
];

pub const CIE_S2: [Float; NUM_CIES_SAMPLES] = [
    0.000000, 1.000000, 2.000000, 3.000000, 4.000000, 6.250000, 8.500000, 8.150000, 7.800000,
    7.250000, 6.700000, 6.000000, 5.300000, 5.700000, 6.100000, 4.550000, 3.000000, 2.100000,
    1.200000, 0.050000, -1.100000, -0.800000, -0.500000, -0.600000, -0.700000, -0.950000,
    -1.200000, -1.900000, -2.600000, -2.750000, -2.900000, -2.850000, -2.800000, -2.700000,
    -2.600000, -2.600000, -2.600000, -2.200000, -1.800000, -1.650000, -1.500000, -1.400000,
    -1.300000, -1.250000, -1.200000, -1.100000, -1.000000, -0.750000, -0.500000, -0.400000,
    -0.300000, -0.150000, 0.000000, 0.100000, 0.200000, 0.350000, 0.500000, 1.300000, 2.100000,
    2.650000, 3.200000, 3.650000, 4.100000, 4.400000, 4.700000, 4.900000, 5.100000, 5.900000,
    6.700000, 7.000000, 7.300000, 7.950000, 8.600000, 9.200000, 9.800000, 10.000000, 10.200000,
    9.250000, 8.300000, 8.950000, 9.600000, 9.050000, 8.500000, 7.750000, 7.000000, 7.300000,
    7.600000, 7.800000, 8.000000, 7.350000, 6.700000, 5.950000, 5.200000, 6.300000, 7.400000,
    7.100000, 6.800000, 6.900000, 7.000000, 6.700000, 6.400000, 5.950000, 5.500000, 5.800000,
    6.100000, 6.300000, 6.500000,
];

const CIE_ILLUM_D5000: [Float; 214] = [
    300.000000, 0.019200, 305.000000, 1.036600, 310.000000, 2.054000, 315.000000, 4.913000,
    320.000000, 7.772000, 325.000000, 11.255700, 330.000000, 14.739500, 335.000000, 16.339001,
    340.000000, 17.938601, 345.000000, 19.466700, 350.000000, 20.994900, 355.000000, 22.459999,
    360.000000, 23.925100, 365.000000, 25.433901, 370.000000, 26.942699, 375.000000, 25.701799,
    380.000000, 24.461000, 385.000000, 27.150700, 390.000000, 29.840401, 395.000000, 39.550301,
    400.000000, 49.664001, 405.000000, 53.155998, 410.000000, 56.647999, 415.000000, 58.445999,
    420.000000, 60.243999, 425.000000, 59.230000, 430.000000, 58.216000, 435.000000, 66.973999,
    440.000000, 75.732002, 445.000000, 81.998001, 450.000000, 88.264000, 455.000000, 89.930000,
    460.000000, 91.596001, 465.000000, 91.940002, 470.000000, 92.283997, 475.000000, 94.155998,
    480.000000, 96.028000, 485.000000, 94.311996, 490.000000, 92.596001, 495.000000, 94.424004,
    500.000000, 96.251999, 505.000000, 96.662003, 510.000000, 97.071999, 515.000000, 97.314003,
    520.000000, 97.556000, 525.000000, 100.005997, 530.000000, 102.456001, 535.000000, 101.694000,
    540.000000, 100.931999, 545.000000, 101.678001, 550.000000, 102.424004, 555.000000, 101.211998,
    560.000000, 100.000000, 565.000000, 98.036697, 570.000000, 96.073402, 575.000000, 95.678398,
    580.000000, 95.283501, 585.000000, 92.577103, 590.000000, 89.870697, 595.000000, 90.772499,
    600.000000, 91.674400, 605.000000, 91.739502, 610.000000, 91.804703, 615.000000, 90.964798,
    620.000000, 90.124901, 625.000000, 87.998299, 630.000000, 85.871696, 635.000000, 86.715302,
    640.000000, 87.558899, 645.000000, 86.069000, 650.000000, 84.579102, 655.000000, 85.167603,
    660.000000, 85.756203, 665.000000, 87.126404, 670.000000, 88.496597, 675.000000, 86.769997,
    680.000000, 85.043404, 685.000000, 79.994698, 690.000000, 74.946098, 695.000000, 76.384598,
    700.000000, 77.823196, 705.000000, 78.671303, 710.000000, 79.519501, 715.000000, 72.694199,
    720.000000, 65.869003, 725.000000, 70.179100, 730.000000, 74.489197, 735.000000, 77.212601,
    740.000000, 79.935997, 745.000000, 73.797401, 750.000000, 67.658897, 755.000000, 58.633598,
    760.000000, 49.608398, 765.000000, 60.462101, 770.000000, 71.315804, 775.000000, 69.405701,
    780.000000, 67.495598, 785.000000, 68.032303, 790.000000, 68.569000, 795.000000, 65.958900,
    800.000000, 63.348801, 805.000000, 59.333599, 810.000000, 55.318501, 815.000000, 58.228600,
    820.000000, 61.138699, 825.000000, 62.712101, 830.000000, 64.285500,
];

const CU_ETA_SAMPLES: [Float; 112] = [
    298.757050, 1.400313, 302.400421, 1.380000, 306.133759, 1.358438, 309.960449, 1.340000,
    313.884003, 1.329063, 317.908142, 1.325000, 322.036835, 1.332500, 326.274139, 1.340000,
    330.624481, 1.334375, 335.092377, 1.325000, 339.682678, 1.317812, 344.400482, 1.310000,
    349.251221, 1.300313, 354.240509, 1.290000, 359.374420, 1.281563, 364.659332, 1.270000,
    370.102020, 1.249062, 375.709625, 1.225000, 381.489777, 1.200000, 387.450562, 1.180000,
    393.600555, 1.174375, 399.948975, 1.175000, 406.505493, 1.177500, 413.280579, 1.180000,
    420.285339, 1.178125, 427.531647, 1.175000, 435.032196, 1.172812, 442.800629, 1.170000,
    450.851562, 1.165312, 459.200653, 1.160000, 467.864838, 1.155312, 476.862213, 1.150000,
    486.212463, 1.142812, 495.936707, 1.135000, 506.057861, 1.131562, 516.600769, 1.120000,
    527.592224, 1.092437, 539.061646, 1.040000, 551.040771, 0.950375, 563.564453, 0.826000,
    576.670593, 0.645875, 590.400818, 0.468000, 604.800842, 0.351250, 619.920898, 0.272000,
    635.816284, 0.230813, 652.548279, 0.214000, 670.184753, 0.209250, 688.800964, 0.213000,
    708.481018, 0.216250, 729.318665, 0.223000, 751.419250, 0.236500, 774.901123, 0.250000,
    799.897949, 0.254188, 826.561157, 0.260000, 855.063293, 0.280000, 885.601257, 0.300000,
];

const CU_K_SAMPLES: [Float; 112] = [
    298.757050, 1.662125, 302.400421, 1.687000, 306.133759, 1.703313, 309.960449, 1.720000,
    313.884003, 1.744563, 317.908142, 1.770000, 322.036835, 1.791625, 326.274139, 1.810000,
    330.624481, 1.822125, 335.092377, 1.834000, 339.682678, 1.851750, 344.400482, 1.872000,
    349.251221, 1.894250, 354.240509, 1.916000, 359.374420, 1.931688, 364.659332, 1.950000,
    370.102020, 1.972438, 375.709625, 2.015000, 381.489777, 2.121562, 387.450562, 2.210000,
    393.600555, 2.177188, 399.948975, 2.130000, 406.505493, 2.160063, 413.280579, 2.210000,
    420.285339, 2.249938, 427.531647, 2.289000, 435.032196, 2.326000, 442.800629, 2.362000,
    450.851562, 2.397625, 459.200653, 2.433000, 467.864838, 2.469187, 476.862213, 2.504000,
    486.212463, 2.535875, 495.936707, 2.564000, 506.057861, 2.589625, 516.600769, 2.605000,
    527.592224, 2.595562, 539.061646, 2.583000, 551.040771, 2.576500, 563.564453, 2.599000,
    576.670593, 2.678062, 590.400818, 2.809000, 604.800842, 3.010750, 619.920898, 3.240000,
    635.816284, 3.458187, 652.548279, 3.670000, 670.184753, 3.863125, 688.800964, 4.050000,
    708.481018, 4.239563, 729.318665, 4.430000, 751.419250, 4.619563, 774.901123, 4.817000,
    799.897949, 5.034125, 826.561157, 5.260000, 855.063293, 5.485625, 885.601257, 5.717000,
];

const AU_ETA_SAMPLES: [Float; 112] = [
    298.757050, 1.795000,   302.400421, 1.812000,   306.133759, 1.822625,   309.960449,
    1.830000,   313.884003, 1.837125,   317.908142, 1.840000,   322.036835, 1.834250,
    326.274139, 1.824000,   330.624481, 1.812000,   335.092377, 1.798000,   339.682678,
    1.782000,   344.400482, 1.766000,   349.251221, 1.752500,   354.240509, 1.740000,
    359.374420, 1.727625,   364.659332, 1.716000,   370.102020, 1.705875,   375.709625,
    1.696000,   381.489777, 1.684750,   387.450562, 1.674000,   393.600555, 1.666000,
    399.948975, 1.658000,   406.505493, 1.647250,   413.280579, 1.636000,   420.285339,
    1.628000,   427.531647, 1.616000,   435.032196, 1.596250,   442.800629, 1.562000,
    450.851562, 1.502125,   459.200653, 1.426000,   467.864838, 1.345875,   476.862213,
    1.242000,   486.212463, 1.086750,   495.936707, 0.916000,   506.057861, 0.754500,
    516.600769, 0.608000,   527.592224, 0.491750,   539.061646, 0.402000,   551.040771,
    0.345500,   563.564453, 0.306000,   576.670593, 0.267625,   590.400818, 0.236000,
    604.800842, 0.212375,   619.920898, 0.194000,   635.816284, 0.177750,   652.548279,
    0.166000,   670.184753, 0.161000,   688.800964, 0.160000,   708.481018, 0.160875,
    729.318665, 0.164000,   751.419250, 0.169500,   774.901123, 0.176000,   799.897949,
    0.181375,   826.561157, 0.188000,   855.063293, 0.198125,   885.601257, 0.210000,
];

const AU_K_SAMPLES: [Float; 112] = [
    298.757050, 1.920375,   302.400421, 1.920000,   306.133759, 1.918875,   309.960449,
    1.916000,   313.884003, 1.911375,   317.908142, 1.904000,   322.036835, 1.891375,
    326.274139, 1.878000,   330.624481, 1.868250,   335.092377, 1.860000,   339.682678,
    1.851750,   344.400482, 1.846000,   349.251221, 1.845250,   354.240509, 1.848000,
    359.374420, 1.852375,   364.659332, 1.862000,   370.102020, 1.883000,   375.709625,
    1.906000,   381.489777, 1.922500,   387.450562, 1.936000,   393.600555, 1.947750,
    399.948975, 1.956000,   406.505493, 1.959375,   413.280579, 1.958000,   420.285339,
    1.951375,   427.531647, 1.940000,   435.032196, 1.924500,   442.800629, 1.904000,
    450.851562, 1.875875,   459.200653, 1.846000,   467.864838, 1.814625,   476.862213,
    1.796000,   486.212463, 1.797375,   495.936707, 1.840000,   506.057861, 1.956500,
    516.600769, 2.120000,   527.592224, 2.326250,   539.061646, 2.540000,   551.040771,
    2.730625,   563.564453, 2.880000,   576.670593, 2.940625,   590.400818, 2.970000,
    604.800842, 3.015000,   619.920898, 3.060000,   635.816284, 3.070000,   652.548279,
    3.150000,   670.184753, 3.445812,   688.800964, 3.800000,   708.481018, 4.087687,
    729.318665, 4.357000,   751.419250, 4.610188,   774.901123, 4.860000,   799.897949,
    5.125813,   826.561157, 5.390000,   855.063293, 5.631250,   885.601257, 5.880000,
];