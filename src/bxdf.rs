use bitflags::bitflags;

use crate::{
    float::PI_F,
    sampling::{sample_uniform_hemisphere, sample_uniform_sphere, uniform_hemisphere_pdf},
    spectra::sampled_spectrum::SampledSpectrum,
    vecmath::{point::Point2f, spherical::abs_cos_theta, Vector3f},
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
        &self,
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

    /// Computes the hemispherical-directional reflectance function rho_hd (PBRTv4 4.12)
    fn rho_hd(&self, wo: Vector3f, uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        if wo.z == 0.0 {
            return SampledSpectrum::default();
        }
        let mut r = SampledSpectrum::from_const(0.0);
        debug_assert_eq!(uc.len(), u2.len());
        for i in 0..uc.len() {
            // Compute estimate of rho_hd.
            let bs = self.sample_f(
                wo,
                uc[i],
                u2[i],
                TransportMode::Radiance,
                BxDFReflTransFlags::ALL,
            );
            if let Some(bs) = bs {
                if bs.pdf > 0.0 {
                    r += bs.f * abs_cos_theta(bs.wi) / bs.pdf;
                }
            }
        }
        r / uc.len() as Float
    }

    /// Computes the hemispherical-hemispherical reflectance function rho_hh (PBRTv4 4.13)
    fn rho_hh(&self, u1: &[Point2f], uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        debug_assert_eq!(u1.len(), uc.len());
        debug_assert_eq!(uc.len(), u2.len());
        let mut r = SampledSpectrum::from_const(0.0);
        for i in 0..uc.len() {
            // Compute estimate of rho_hh.
            let wo = sample_uniform_hemisphere(u1[i]);
            if wo.z == 0.0 {
                continue;
            }
            let pdfo = uniform_hemisphere_pdf();
            let bs = self.sample_f(
                wo,
                uc[i],
                u2[i],
                TransportMode::Radiance,
                BxDFReflTransFlags::ALL,
            );
            if let Some(bs) = bs {
                if bs.pdf > 0.0 {
                    r += bs.f * abs_cos_theta(bs.wi) * abs_cos_theta(wo) / (pdfo * bs.pdf);
                }
            }
        }
        r / (PI_F * uc.len() as Float)
    }
}

pub enum BxDF {}

impl BxDFI for BxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        todo!()
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        todo!()
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        todo!()
    }

    fn flags(&self) -> BxDFFLags {
        todo!()
    }
}

pub struct BSDFSample {
    /// Value of the BSDF f()
    pub f: SampledSpectrum,
    /// Sampled direction wi (given wo). BxDF specify wi wrt to the local reflection coordinate system;
    /// BSDF::Sample_f() should transform wi to rendering space before returning, however.
    pub wi: Vector3f,
    /// PDF of wi with respect to solid angle
    pub pdf: Float,
    /// Characteristics of the particular sample
    pub flags: BxDFFLags,
    pub eta: Float,
    /// Useful for a layered BxDF; in most cases, false.
    pub pdf_is_proportional: bool,
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
