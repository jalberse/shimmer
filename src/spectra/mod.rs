mod cie;
pub mod named_spectrum;
pub mod sampled_spectrum;
pub mod sampled_wavelengths;
pub mod spectrum;

pub use cie::CIE;
pub use cie::CIE_Y_INTEGRAL;
pub use named_spectrum::NamedSpectrum;
pub use spectrum::inner_product;
pub use spectrum::BlackbodySpectrum;
pub use spectrum::ConstantSpectrum;
pub use spectrum::DenselySampledSpectrum;
pub use spectrum::PiecewiseLinearSpectrum;
pub use spectrum::Spectrum;

pub const NUM_SPECTRUM_SAMPLES: usize = 4;
