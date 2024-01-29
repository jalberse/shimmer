use crate::{
    float::PI_F,
    math::{sqr, INV_PI},
    sampling::{
        cosine_hemisphere_pdf, sample_cosine_hemisphere, sample_uniform_hemisphere,
        uniform_hemisphere_pdf,
    },
    scattering::{
        fresnel_complex_spectral, fresnel_dialectric, reflect, refract, TrowbridgeReitzDistribution,
    },
    spectra::sampled_spectrum::SampledSpectrum,
    vecmath::{
        point::Point2f,
        spherical::{abs_cos_theta, cos_theta, same_hemisphere},
        vector::Vector3,
        Length, Normal3f, Normalize, Tuple3, Vector3f,
    },
    Float,
};
use bitflags::bitflags;

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

pub enum BxDF {
    Diffuse(DiffuseBxDF),
    Conductor(ConductorBxDF),
    Dialectric(DialectricBxDF),
}

impl BxDFI for BxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        match self {
            BxDF::Diffuse(v) => v.f(wo, wi, mode),
            BxDF::Conductor(v) => v.f(wo, wi, mode),
            BxDF::Dialectric(v) => v.f(wo, wi, mode),
        }
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        match self {
            BxDF::Diffuse(v) => v.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Conductor(v) => v.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::Dialectric(v) => v.sample_f(wo, uc, u, mode, sample_flags),
        }
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        match self {
            BxDF::Diffuse(v) => v.pdf(wo, wi, mode, sample_flags),
            BxDF::Conductor(v) => v.pdf(wo, wi, mode, sample_flags),
            BxDF::Dialectric(v) => v.pdf(wo, wi, mode, sample_flags),
        }
    }

    fn flags(&self) -> BxDFFLags {
        match self {
            BxDF::Diffuse(v) => v.flags(),
            BxDF::Conductor(v) => v.flags(),
            BxDF::Dialectric(v) => v.flags(),
        }
    }
}

pub struct DiffuseBxDF {
    /// Values in range [0, 1] that specify the fraction of incident light that is scattered.
    r: SampledSpectrum,
}

impl DiffuseBxDF {
    pub fn new(r: SampledSpectrum) -> DiffuseBxDF {
        DiffuseBxDF { r }
    }
}

impl BxDFI for DiffuseBxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, _mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::from_const(0.0);
        }
        // Normalize so that total integrated reflectance equals R
        self.r * INV_PI
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        _uc: Float,
        u: Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
            return None;
        }
        // Sample cosine-weighted hemisphere to compute wi and pdf
        let mut wi = sample_cosine_hemisphere(u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }
        let pdf = cosine_hemisphere_pdf(abs_cos_theta(wi));

        Some(BSDFSample::new(
            self.r * INV_PI,
            wi,
            pdf,
            BxDFFLags::DIFFUSE_REFLECTION,
        ))
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0
            || !same_hemisphere(wo, wi)
        {
            0.0
        } else {
            cosine_hemisphere_pdf(abs_cos_theta(wi))
        }
    }

    fn flags(&self) -> BxDFFLags {
        if self.r.is_zero() {
            BxDFFLags::UNSET
        } else {
            BxDFFLags::DIFFUSE_REFLECTION
        }
    }
}

pub struct ConductorBxDF {
    mf_distribution: TrowbridgeReitzDistribution,
    eta: SampledSpectrum,
    k: SampledSpectrum,
}

impl ConductorBxDF {
    pub fn new(
        mf_distribution: TrowbridgeReitzDistribution,
        eta: SampledSpectrum,
        k: SampledSpectrum,
    ) -> ConductorBxDF {
        ConductorBxDF {
            mf_distribution,
            eta,
            k,
        }
    }

    pub fn regularize(&mut self) {
        self.mf_distribution.regularize()
    }
}

impl BxDFI for ConductorBxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, _mode: TransportMode) -> SampledSpectrum {
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::from_const(0.0);
        }
        if self.mf_distribution.effectively_smooth() {
            return SampledSpectrum::from_const(0.0);
        }

        // Evaluate rough conductor BRDF
        // Compute cosins and wm for conductor BRDF
        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }
        let wm = wi + wo;
        if wm.length_squared() == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let wm = wm.normalize();

        // Evaluate fresnel factor f for conductor BRDF
        let f = fresnel_complex_spectral(wo.abs_dot(wm), self.eta, self.k);

        self.mf_distribution.d(wm) * f * self.mf_distribution.g(wo, wi)
            / (4.0 * cos_theta_o * cos_theta_i)
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        _uc: Float,
        u: Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if (sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits()) == 0 {
            return None;
        }
        if self.mf_distribution.effectively_smooth() {
            // Sample perfect specular conductor BRDF
            let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
            let f =
                fresnel_complex_spectral(abs_cos_theta(wi), self.eta, self.k) / abs_cos_theta(wi);
            return Some(BSDFSample::new(f, wi, 1.0, BxDFFLags::SPECULAR_REFLECTION));
        }
        // Sample rough conductor BRDF
        // Sample microfacet normal wm and reflected direction wi
        if wo.z == 0.0 {
            return None;
        }
        let wm = self.mf_distribution.sample_wm(wo, u);
        let wi = reflect(wo, wm.into());
        if !same_hemisphere(wo, wi) {
            return None;
        }

        // Compute PDF of wi for microfacet reflection
        let pdf = self.mf_distribution.pdf(wo, wm) / (4.0 * wo.abs_dot(wm));

        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return None;
        }

        // Evaluate fresnel factor f for conductor BRDF
        let f = fresnel_complex_spectral(wo.abs_dot(wm), self.eta, self.k);

        let f = self.mf_distribution.d(wm) * f * self.mf_distribution.g(wo, wi)
            / (4.0 * cos_theta_o * cos_theta_i);
        Some(BSDFSample::new(f, wi, pdf, BxDFFLags::GLOSSY_REFLECTION))
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0
            || !same_hemisphere(wo, wi)
            || self.mf_distribution.effectively_smooth()
        {
            return 0.0;
        }

        // Evaluate sampling PDF of rough conductor BRDF
        let wm = wo + wi;
        if wm.length_squared() == 0.0 {
            return 0.0;
        }
        let wm = wm.normalize().face_forward_n(Normal3f::Z);
        self.mf_distribution.pdf(wo, wm) / (4.0 * wo.abs_dot(wm))
    }

    fn flags(&self) -> BxDFFLags {
        if self.mf_distribution.effectively_smooth() {
            BxDFFLags::SPECULAR_REFLECTION
        } else {
            BxDFFLags::GLOSSY_REFLECTION
        }
    }
}

pub struct DialectricBxDF {
    eta: Float,
    mf_distribution: TrowbridgeReitzDistribution,
}

impl DialectricBxDF {
    pub fn new(eta: Float, mf_distribution: TrowbridgeReitzDistribution) -> DialectricBxDF {
        DialectricBxDF {
            eta,
            mf_distribution,
        }
    }

    pub fn regularize(&mut self) {
        self.mf_distribution.regularize()
    }
}

impl BxDFI for DialectricBxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            return SampledSpectrum::from_const(0.0);
        }

        // Evaluate rough dialectric BSDF
        // Compute generalized half vector _wm_
        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        let reflect = cos_theta_i * cos_theta_o > 0.0;
        let etap = if !reflect {
            if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            }
        } else {
            1.0
        };
        let wm = wi * etap + wo;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || wm.length_squared() == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }

        let f = fresnel_dialectric(wo.dot(wm), self.eta);
        if reflect {
            // Compute reflection at rough dialectric interface
            SampledSpectrum::from_const(
                self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi) * f
                    / Float::abs(4.0 * cos_theta_i * cos_theta_o),
            )
        } else {
            // Compute transmission at rough dialectric interface
            let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap) * cos_theta_i * cos_theta_o;
            let mut ft = self.mf_distribution.d(wm)
                * (1.0 - f)
                * self.mf_distribution.g(wo, wi)
                * Float::abs(wi.dot(wm) * wo.dot(wm) / denom);
            // Account for non-symmetry with transmission to different medium
            if mode == TransportMode::Radiance {
                ft /= sqr(etap)
            }
            SampledSpectrum::from_const(ft)
        }
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            // Sample perfect specular dialectric BSDF
            let r = fresnel_dialectric(cos_theta(wo), self.eta);
            let t = 1.0 - r;
            // Compute probabilities pr and pt for sampling reflection and transmission
            let mut pr = r;
            let mut pt = t;

            if (sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits()) == 0 {
                pr = 0.0;
            }
            if (sample_flags.bits() & BxDFReflTransFlags::TRANSMISSION.bits()) == 0 {
                pt = 0.0;
            }
            if pr == 0.0 && pt == 0.0 {
                return None;
            }

            if uc < pr / (pr + pt) {
                // Sample perfect specular reflection BSDF
                let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
                let fr = SampledSpectrum::from_const(r / abs_cos_theta(wi));
                return Some(BSDFSample::new(
                    fr,
                    wi,
                    pr / (pr + pt),
                    BxDFFLags::SPECULAR_REFLECTION,
                ));
            } else {
                // Sample perfect specular dialectric BTDF
                // Compute ray direction for specular transmission
                if let Some((wi, etap)) = refract(wo, Normal3f::Z, self.eta) {
                    let mut ft = SampledSpectrum::from_const(t / abs_cos_theta(wi));
                    // Account for non-symmetric with transmission to different medium
                    if mode == TransportMode::Radiance {
                        ft /= sqr(etap);
                    }

                    return Some(BSDFSample::new_with_eta(
                        ft,
                        wi,
                        pt / (pr + pt),
                        BxDFFLags::SPECULAR_TRANSMISSION,
                        etap,
                    ));
                } else {
                    return None;
                }
            }
        } else {
            // Sample rough dialectric BSDF
            let wm = self.mf_distribution.sample_wm(wo, u);
            let r = fresnel_dialectric(wo.dot(wm), self.eta);
            let t = 1.0 - r;
            // Compute probabilities pr and pt for sampling reflection and transmission
            let mut pr = r;
            let mut pt = t;
            if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
                pr = 0.0;
            }
            if sample_flags.bits() & BxDFReflTransFlags::TRANSMISSION.bits() == 0 {
                pt = 0.0;
            }
            if pr == 0.0 && pt == 0.0 {
                return None;
            }

            if uc < pr / (pr + pt) {
                // Sample reflection at rough dialectric interface
                let wi = reflect(wo, wm.into());
                if !same_hemisphere(wo, wi) {
                    return None;
                }
                // Compute PDF of rough dialectric reflection
                let pdf =
                    self.mf_distribution.pdf(wo, wm) / (4.0 * wo.abs_dot(wm)) * pr / (pr + pt);

                debug_assert!(!pdf.is_nan());
                let f = SampledSpectrum::from_const(
                    self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi) * r
                        / (4.0 * cos_theta(wi) * cos_theta(wo)),
                );
                Some(BSDFSample::new(f, wi, pdf, BxDFFLags::GLOSSY_REFLECTION))
            } else {
                // Sample transmission at rough dialectric interface
                if let Some((wi, etap)) = refract(wo, wm.into(), self.eta) {
                    if !same_hemisphere(wo, wi) || wi.z == 0.0 {
                        return None;
                    }
                    // Compute PDF of rough dialectric transmission
                    let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);
                    let dwm_dwi = wi.abs_dot(wm) / denom;
                    let pdf = self.mf_distribution.pdf(wo, wm) * dwm_dwi * pt / (pr + pt);
                    debug_assert!(!pdf.is_nan());
                    // Evaluate BRDF and return BDFSample for rough transmission
                    let mut ft = SampledSpectrum::from_const(
                        t * self.mf_distribution.d(wm)
                            * self.mf_distribution.g(wo, wi)
                            * Float::abs(
                                wi.dot(wm) * wo.dot(wm) / (cos_theta(wi) * cos_theta(wo) * denom),
                            ),
                    );

                    // Account for non-symmetry with transmission to different medium
                    if mode == TransportMode::Radiance {
                        ft /= sqr(etap);
                    }
                    Some(BSDFSample::new_with_eta(
                        ft,
                        wi,
                        pdf,
                        BxDFFLags::GLOSSY_TRANSMISSION,
                        etap,
                    ))
                } else {
                    None
                }
            }
        }
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            return 0.0;
        }
        // Evaluate sampling PDF of rough dialectric BSDF
        // Compute generalized half vector wm
        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        let reflect = cos_theta_i * cos_theta_o > 0.0;
        let etap = if !reflect {
            if cos_theta_o > 0.0 {
                self.eta
            } else {
                1.0 / self.eta
            }
        } else {
            1.0
        };
        let wm = wi * etap + wo;
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 || wm.length_squared() == 0.0 {
            return 0.0;
        }
        let wm = wm.normalize().face_forward_n(Normal3f::Z);

        // Discard backfacing microfacets
        if wm.dot(wi) * cos_theta_i < 0.0 || wm.dot(wm) * cos_theta_o < 0.0 {
            return 0.0;
        }

        // Determine Fresnel reflectance of dialectric boundary
        let r = fresnel_dialectric(wo.dot(wm), self.eta);
        let t = 1.0 - r;

        // Compute probabilities pr and pt for sampling reflectance and transmission
        let mut pr = r;
        let mut pt = t;
        if sample_flags.bits() & BxDFReflTransFlags::REFLECTION.bits() == 0 {
            pr = 0.0;
        }
        if sample_flags.bits() & BxDFReflTransFlags::TRANSMISSION.bits() == 0 {
            pt = 0.0;
        }
        if pr == 0.0 && pt == 0.0 {
            return 0.0;
        }

        if reflect {
            // Compute PDF of rough dialectric reflection
            self.mf_distribution.pdf(wo, wm) / (4.0 * wo.abs_dot(wm)) * pr / (pr + pt)
        } else {
            // Compute PDF of rough dialectric transmission
            let denom = sqr(wi.dot(wm) + wo.dot(wm) / etap);
            let dwm_dwi = wi.abs_dot(wm) / denom;
            self.mf_distribution.pdf(wo, wm) * dwm_dwi * pt / (pr + pt)
        }
    }

    fn flags(&self) -> BxDFFLags {
        let flags = if self.eta == 1.0 {
            BxDFFLags::TRANSMISSION
        } else {
            BxDFFLags::REFLECTION | BxDFFLags::TRANSMISSION
        };
        let mf = if self.mf_distribution.effectively_smooth() {
            BxDFFLags::SPECULAR
        } else {
            BxDFFLags::GLOSSY
        };
        flags | mf
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

    pub fn new_with_eta(
        f: SampledSpectrum,
        wi: Vector3f,
        pdf: Float,
        flags: BxDFFLags,
        eta: Float,
    ) -> BSDFSample {
        BSDFSample {
            f,
            wi,
            pdf,
            flags,
            eta,
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

#[derive(Debug, Copy, Clone, PartialEq)]
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
