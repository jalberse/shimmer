/// Provides named spectra.
/// Largely separated from the spectrum module to isolate large amounts of embedded
/// data in its own file; this and spectrum should be re-exported to the top of the
/// module anyways, so this just makes things more readable.
use once_cell::sync::Lazy;

use crate::Float;

use super::{PiecewiseLinear, Spectrum};

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

pub static GLASS_BAF10_ETA: Lazy<Spectrum> =
    Lazy::new(|| Spectrum::PiecewiseLinear(PiecewiseLinear::new(&[2.0, 3.0], &[5.0, 6.0])));

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