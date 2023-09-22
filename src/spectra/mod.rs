mod cie;
pub mod named_spectrum;
pub mod spectrum;

pub use named_spectrum::NamedSpectrum;
pub use spectrum::Blackbody;
pub use spectrum::Constant;
pub use spectrum::DenselySampled;
pub use spectrum::PiecewiseLinear;
pub use spectrum::Spectrum;
