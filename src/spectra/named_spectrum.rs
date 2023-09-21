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

// PAPERDOC Embedded spectral data 4.5.3.
// Instead of a map on string keys (which requires evaluating the hash),
// we use an Enum with the names and match on it and return ref to static-lifetimed
// spectra. These are thread-safe read-only single-instance objects.
pub fn get_named_spectrum(spectrum: NamedSpectrum) -> &'static Spectrum {
    match spectrum {
        NamedSpectrum::GLASS_BK7 => Lazy::force(&GLASS_BK7_ETA),
        NamedSpectrum::GLASS_BAF10 => Lazy::force(&GLASS_BAF10_ETA),
    }
}

// TODO and use actual data.
static GLASS_BK7_ETA: Lazy<Spectrum> =
    Lazy::new(|| Spectrum::PiecewiseLinear(PiecewiseLinear::new(&[0.0, 1.0], &[1.0, 2.0])));

static GLASS_BAF10_ETA: Lazy<Spectrum> =
    Lazy::new(|| Spectrum::PiecewiseLinear(PiecewiseLinear::new(&[2.0, 3.0], &[5.0, 6.0])));
