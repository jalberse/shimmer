use bitflags::bitflags;

use crate::{
    spectra::sampled_spectrum::SampledSpectrum,
    vecmath::{point::Point2f, Vector3f},
    Float,
};

/// The BxDF Interface.
pub trait BxDFI {
    /// Returns the value of the distribution function for a given pair of directions wi and wo.
    /// Both wi and wo must be provided in terms of the local reflection coordinate system.
    /// The TransportMode dictates whether the outgoing direction wo is toward the camera or
    /// toward the lightsource, which is necessary to handle cases where scattering is non-symmetric.
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum;

    /// Uses importance sampling to draw a direction (wi) from a distribution function that approximates
    /// the scattering function's shape.
    /// u and uc are uniform samples.
    /// uc will generally be used to choose between different kinds of sampling (e.g. reflection or transmission)
    /// and u for the direction.
    fn sample_f(
        wo: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample>;

    /// Returns the value of the PDF for a given pair of directions. This is useful for techniques like
    /// multiple importance sampling that compare probabilities of multiple strategies for obstaining
    /// a given sample.
    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float;

    fn flags(&self) -> BxDFFLags;
}

pub struct BSDFSample {
    /// Value of the BSDF f()
    f: SampledSpectrum,
    /// Sampled direction wi (given wo). BxDF specify wi wrt to the local reflection coordinate system;
    /// BSDF::Sample_f() should transform wi to rendering space before returning, however.
    wi: Vector3f,
    /// PDF of wi with respect to solid angle
    pdf: Float,
    /// Characteristics of the particular sample
    flags: BxDFFLags,
    eta: Float,
    /// Useful for a layered BxDF; in most cases, false.
    pdf_is_proportional: bool,
}

impl BSDFSample {
    pub fn new(f: SampledSpectrum, wi: Vector3f, pdf: Float, flags: BxDFFLags) -> BSDFSample {
        BSDFSample {
            f,
            wi,
            pdf,
            flags,
            eta: 1.0,
            pdf_is_proportional: false,
        }
    }

    pub fn is_reflection(&self) -> bool {
        self.flags.is_reflective()
    }

    pub fn is_transmission(&self) -> bool {
        self.flags.is_transmissive()
    }

    pub fn is_diffuse(&self) -> bool {
        self.flags.is_diffuse()
    }

    pub fn is_glossy(&self) -> bool {
        self.flags.is_glossy()
    }

    pub fn is_specular(&self) -> bool {
        self.flags.is_specular()
    }
}

pub enum TransportMode {
    Radiance,
    Importance,
}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BxDFReflTransFlags: u8
    {
        const UNSET = 0;
        const REFLECTION = 1 << 0;
        const TRANSMISSION = 1 << 1;
        const ALL = Self::REFLECTION.bits() | Self::TRANSMISSION.bits();
    }

}

bitflags! {
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BxDFFLags: u8
    {
        const UNSET = 0;
        const REFLECTION = 1 << 0;
        const TRANSMISSION = 1 << 1;
        const DIFFUSE = 1 << 2;
        const GLOSSY = 1 << 3;
        const SPECULAR = 1 << 4;
        const DIFFUSE_REFLECTION = Self::DIFFUSE.bits() | Self::REFLECTION.bits();
        const DIFFUSE_TRANSMISSION = Self::DIFFUSE.bits() | Self::TRANSMISSION.bits();
        const GLOSSY_REFLECTION = Self::GLOSSY.bits() | Self::REFLECTION.bits();
        const GLOSSY_TRANSMISSION = Self::GLOSSY.bits() | Self::TRANSMISSION.bits();
        const SPECULAR_REFLECTION = Self::SPECULAR.bits() | Self::REFLECTION.bits();
        const SPECULAR_TRANSMISSION = Self::SPECULAR.bits() | Self::TRANSMISSION.bits();
        const ALL = Self::DIFFUSE.bits() | Self::SPECULAR.bits() | Self::REFLECTION.bits() | Self::TRANSMISSION.bits();
    }
}

impl BxDFFLags {
    pub fn is_reflective(&self) -> bool {
        (*self & Self::REFLECTION).bits() != 0
    }

    pub fn is_transmissive(&self) -> bool {
        (*self & Self::TRANSMISSION).bits() != 0
    }

    pub fn is_diffuse(&self) -> bool {
        (*self & Self::DIFFUSE).bits() != 0
    }

    pub fn is_glossy(&self) -> bool {
        (*self & Self::GLOSSY).bits() != 0
    }

    pub fn is_specular(&self) -> bool {
        (*self & Self::SPECULAR).bits() != 0
    }

    pub fn is_non_specular(&self) -> bool {
        (*self & (Self::DIFFUSE | Self::GLOSSY)).bits() != 0
    }
}

#[cfg(test)]
mod tests {
    use super::BxDFFLags;

    #[test]
    fn basic_bxdf_flags() {
        let unset = BxDFFLags::UNSET;
        assert!(!unset.is_diffuse());
        assert!(!unset.is_transmissive());
        assert!(!unset.is_glossy());
        assert!(!unset.is_reflective());

        let gt = BxDFFLags::GLOSSY_TRANSMISSION;
        assert!(gt.is_glossy());
        assert!(gt.is_transmissive());
        assert!(!gt.is_diffuse());
    }
}
