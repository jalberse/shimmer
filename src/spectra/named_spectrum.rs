/// Provides named spectra.
/// Largely separated from the spectrum module to isolate large amounts of embedded
/// data in its own file; this and spectrum should be re-exported to the top of the
/// module anyways, so this just makes things more readable.
use once_cell::sync::Lazy;

use crate::Float;

use super::{PiecewiseLinear, Spectrum};

// TODO add other named spectra. I'm happy to just have the system in place for
// now though, we can add them as we need them or when I have time to go through
// and copy/generate all the data.
pub enum NamedSpectrum {
    GlassBk7,
    GlassBaf10,
}

// NOTE: These intentionally use static, not const.
// const will compile, but will not work properly. See once_cell documentation.
pub static GLASS_BK7_ETA: Lazy<Spectrum> = Lazy::new(|| {
    Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<58, 24>(
        &GLASS_BK7_ETA_SAMPLES,
        false,
    ))
});

pub static GLASS_BAF10_ETA: Lazy<Spectrum> = Lazy::new(|| {
    Spectrum::PiecewiseLinear(PiecewiseLinear::from_interleaved::<54, 27>(
        &GLASS_BAF10_ETA_SAMPLES,
        false,
    ))
});

const NUM_GLASS_BK7_ETA_SAMPLES: usize = 58;
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
