/// Provides named spectra.
/// Largely separated from the spectrum module to isolate large amounts of embedded
/// data in its own file; this and spectrum should be re-exported to the top of the
/// module anyways, so this just makes things more readable.
use once_cell::sync::Lazy;

use super::{PiecewiseLinear, Spectrum};

pub enum NamedSpectrum {
    GlassBk7,
    GlassBaf10,
}

// NOTE: These intentionally use static, not const.
// const will compile, but will not work properly. See once_cell documentation.
pub static GLASS_BK7_ETA: Lazy<Spectrum> =
    Lazy::new(|| Spectrum::PiecewiseLinear(PiecewiseLinear::new(&[0.0, 1.0], &[1.0, 2.0])));

pub static GLASS_BAF10_ETA: Lazy<Spectrum> =
    Lazy::new(|| Spectrum::PiecewiseLinear(PiecewiseLinear::new(&[2.0, 3.0], &[5.0, 6.0])));
