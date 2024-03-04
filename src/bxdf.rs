use std::{process::exit, rc::Rc, sync::Arc};

use crate::{
    float::{next_float_down, PI_F}, math::{lerp, sqr, INV_PI}, media::HGPhaseFunction, sampling::{
        cosine_hemisphere_pdf, power_heuristic, sample_cosine_hemisphere, sample_exponential, sample_uniform_hemisphere, uniform_hemisphere_pdf
    }, scattering::{
        fresnel_complex_spectral, fresnel_dielectric, reflect, refract, TrowbridgeReitzDistribution,
    }, spectra::sampled_spectrum::SampledSpectrum, vecmath::{
        point::Point2f, spherical::{abs_cos_theta, cos_theta, same_hemisphere}, vector::Vector3, Length, Normal3f, Normalize, Tuple2, Tuple3, Vector3f
    }, Float
};
use bitflags::bitflags;
use itertools::Diff;
use rand::{rngs::SmallRng, Rng, SeedableRng};

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

    fn regularize(&mut self);
}

pub enum BxDF {
    Diffuse(DiffuseBxDF),
    CoatedDiffuse(CoatedDiffuseBxDF),
    Conductor(ConductorBxDF),
    Dielectric(DielectricBxDF),
    ThinDielectric(ThinDielectricBxDF),
}

impl BxDFI for BxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        match self {
            BxDF::Diffuse(v) => v.f(wo, wi, mode),
            BxDF::Conductor(v) => v.f(wo, wi, mode),
            BxDF::Dielectric(v) => v.f(wo, wi, mode),
            BxDF::ThinDielectric(v) => v.f(wo, wi, mode),
            BxDF::CoatedDiffuse(v) => v.f(wo, wi, mode),
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
            BxDF::Dielectric(v) => v.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::ThinDielectric(v) => v.sample_f(wo, uc, u, mode, sample_flags),
            BxDF::CoatedDiffuse(v) => v.sample_f(wo, uc, u, mode, sample_flags),
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
            BxDF::Dielectric(v) => v.pdf(wo, wi, mode, sample_flags),
            BxDF::ThinDielectric(v) => v.pdf(wo, wi, mode, sample_flags),
            BxDF::CoatedDiffuse(v) => v.pdf(wo, wi, mode, sample_flags),
        }
    }

    fn flags(&self) -> BxDFFLags {
        match self {
            BxDF::Diffuse(v) => v.flags(),
            BxDF::Conductor(v) => v.flags(),
            BxDF::Dielectric(v) => v.flags(),
            BxDF::ThinDielectric(v) => v.flags(),
            BxDF::CoatedDiffuse(v) => v.flags(),
        }
    }
    
    fn regularize(&mut self) {
        match self {
            BxDF::Diffuse(f) => f.regularize(),
            BxDF::Conductor(f) => f.regularize(),
            BxDF::Dielectric(f) => f.regularize(),
            BxDF::ThinDielectric(f) => f.regularize(),
            BxDF::CoatedDiffuse(f) => f.regularize(),
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

        // TODO self.r seems to be much too small compared to PBRT?
        //   R is set from the Material in get_bxdf.
        //   It's not the texture evaluator I think,
        //   we both use the universal texture eval.
        // Uhm, I think that the reflectance *should* be the same since we
        //   just grab it as an array from the file?
        // Check that.
        // But then the issue is I guess in evaluating the texture but
        //   that's literally just grabbing from a constant spectrum right now???
        //   Ugh idk I guess go step through it.
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
    
    fn regularize(&mut self) {
        // Do nothing   
    }
}

pub struct CoatedDiffuseBxDF
{
    bxdf: LayeredBxDF<DielectricBxDF, DiffuseBxDF, true>
}

impl CoatedDiffuseBxDF
{
    pub fn new(
        top: DielectricBxDF,
        bottom: DiffuseBxDF,
        thickness: Float,
        albedo: &SampledSpectrum,
        g: Float,
        max_depth: i32,
        n_samples: i32,
    ) -> CoatedDiffuseBxDF
    {
        CoatedDiffuseBxDF {
            bxdf: LayeredBxDF::new(top, bottom, thickness, albedo, g, max_depth, n_samples)
        }
    
    }
}

impl BxDFI for CoatedDiffuseBxDF
{
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        self.bxdf.f(wo, wi, mode)
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        self.bxdf.sample_f(wo, uc, u, mode, sample_flags)
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        self.bxdf.pdf(wo, wi, mode, sample_flags)
    }

    fn flags(&self) -> BxDFFLags {
        self.bxdf.flags()
    }

    fn regularize(&mut self) {
        self.bxdf.regularize()
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

    fn regularize(&mut self) {
        self.mf_distribution.regularize()
    }
}

pub struct DielectricBxDF {
    eta: Float,
    mf_distribution: TrowbridgeReitzDistribution,
}

impl DielectricBxDF {
    pub fn new(eta: Float, mf_distribution: TrowbridgeReitzDistribution) -> DielectricBxDF {
        DielectricBxDF {
            eta,
            mf_distribution,
        }
    }
}

impl BxDFI for DielectricBxDF {
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        if self.eta == 1.0 || self.mf_distribution.effectively_smooth() {
            return SampledSpectrum::from_const(0.0);
        }

        // Evaluate rough dielectric BSDF
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

        let wm = wm.normalize().face_forward_n(Normal3f::Z);
        // Discard backwards facing microfacets
        if wm.dot(wi) * cos_theta_i < 0.0 || wm.dot(wo) * cos_theta_o < 0.0
        {
            return SampledSpectrum::from_const(0.0);
        }

        let f = fresnel_dielectric(wo.dot(wm), self.eta);
        if reflect {
            // Compute reflection at rough dielectric interface
            SampledSpectrum::from_const(
                self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi) * f
                    / Float::abs(4.0 * cos_theta_i * cos_theta_o),
            )
        } else {
            // Compute transmission at rough dielectric interface
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
            // Sample perfect specular dielectric BSDF
            let r = fresnel_dielectric(cos_theta(wo), self.eta);
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
                // Sample perfect specular dielectric BTDF
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
            // Sample rough dielectric BSDF
            let wm = self.mf_distribution.sample_wm(wo, u);
            let r = fresnel_dielectric(wo.dot(wm), self.eta);
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
                // Sample reflection at rough dielectric interface
                let wi = reflect(wo, wm.into());
                if !same_hemisphere(wo, wi) {
                    return None;
                }
                // Compute PDF of rough dielectric reflection
                let pdf =
                    self.mf_distribution.pdf(wo, wm) / (4.0 * wo.abs_dot(wm)) * pr / (pr + pt);

                debug_assert!(!pdf.is_nan());
                let f = SampledSpectrum::from_const(
                    self.mf_distribution.d(wm) * self.mf_distribution.g(wo, wi) * r
                        / (4.0 * cos_theta(wi) * cos_theta(wo)),
                );
                Some(BSDFSample::new(f, wi, pdf, BxDFFLags::GLOSSY_REFLECTION))
            } else {
                // Sample transmission at rough dielectric interface
                if let Some((wi, etap)) = refract(wo, wm.into(), self.eta) {
                    if same_hemisphere(wo, wi) || wi.z == 0.0 {
                        return None;
                    }
                    // Compute PDF of rough dielectric transmission
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
        // Evaluate sampling PDF of rough dielectric BSDF
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

        // Determine Fresnel reflectance of dielectric boundary
        let r = fresnel_dielectric(wo.dot(wm), self.eta);
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
            // Compute PDF of rough dielectric reflection
            self.mf_distribution.pdf(wo, wm) / (4.0 * wo.abs_dot(wm)) * pr / (pr + pt)
        } else {
            // Compute PDF of rough dielectric transmission
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

    fn regularize(&mut self) {
        self.mf_distribution.regularize()
    }
}

pub struct ThinDielectricBxDF {
    eta: Float,
}

impl ThinDielectricBxDF {
    pub fn new(eta: Float) -> ThinDielectricBxDF {
        ThinDielectricBxDF { eta }
    }
}

impl BxDFI for ThinDielectricBxDF {
    fn f(&self, _wo: Vector3f, _wi: Vector3f, _mode: TransportMode) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        uc: Float,
        _u: Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let mut r = fresnel_dielectric(abs_cos_theta(wo), self.eta);
        let mut t = 1.0 - r;
        // Compute r and t accounting for scattering between interfaces
        if r < 1.0 {
            r += sqr(t) * r / (1.0 - sqr(r));
            t = 1.0 - r;
        }

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
            // Sample perfect specular reflection BSDF
            let wi = Vector3f::new(-wo.x, -wo.y, wo.z);
            let fr = SampledSpectrum::from_const(r / abs_cos_theta(wi));
            Some(BSDFSample::new(
                fr,
                wi,
                pr / (pr + pt),
                BxDFFLags::SPECULAR_REFLECTION,
            ))
        } else {
            // Sample perfect specular transmission BSDF
            let wi = -wo;
            let ft = SampledSpectrum::from_const(t / abs_cos_theta(wi));
            Some(BSDFSample::new(
                ft,
                wi,
                pt / (pr + pt),
                BxDFFLags::SPECULAR_TRANSMISSION,
            ))
        }
    }

    fn pdf(
        &self,
        _wo: Vector3f,
        _wi: Vector3f,
        _mode: TransportMode,
        _sample_flags: BxDFReflTransFlags,
    ) -> Float {
        0.0
    }

    fn flags(&self) -> BxDFFLags {
        BxDFFLags::REFLECTION | BxDFFLags::TRANSMISSION | BxDFFLags::SPECULAR
    }

    fn regularize(&mut self) {
        // TODO Should we regularize thin dieletric?
    }
}

struct LayeredBxDF<TopBxDF, BottomBxDF, const TWO_SIDED: bool>
where
    TopBxDF: BxDFI,
    BottomBxDF: BxDFI,
{
    top: TopBxDF,
    bottom: BottomBxDF,
    thickness: Float,
    g: Float,
    albedo: SampledSpectrum,
    max_depth: i32,
    n_samples: i32,
}

impl<TopBxDF, BottomBxDF, const TWO_SIDED: bool> LayeredBxDF<TopBxDF, BottomBxDF, TWO_SIDED>
where
    TopBxDF: BxDFI,
    BottomBxDF: BxDFI,
{
    pub fn new(
        top: TopBxDF,
        bottom: BottomBxDF,
        thickness: Float,
        albedo: &SampledSpectrum,
        g: Float,
        max_depth: i32,
        n_samples: i32,
    ) -> LayeredBxDF<TopBxDF, BottomBxDF, TWO_SIDED>
    {
        LayeredBxDF {
            top,
            bottom,
            thickness,
            g,
            albedo: *albedo,
            max_depth,
            n_samples,
        }
    }

    fn tr(&self, dz: Float, w: Vector3f) -> Float
    {
        if Float::abs(dz) <= Float::MIN{
            1.0
        } else {
            // TODO fast_exp()?
            Float::exp(-Float::abs(dz / w.z))
        }
    }
}

impl<TopBxDF, BottomBxDF, const TWO_SIDED: bool> BxDFI for LayeredBxDF<TopBxDF, BottomBxDF, TWO_SIDED>
where
    TopBxDF: BxDFI,
    BottomBxDF: BxDFI,
{
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        let mut f = SampledSpectrum::from_const(0.0);

        // Estimate layered bxdf value f using random sampling
        // Set wo and wi for layered bsdf evaluation

        let mut wo = wo;
        let mut wi = wi;

        if TWO_SIDED && wo.z < 0.0
        {
            wo = -wo;
            wi = -wi;
        }

        // Determine entrance interface for layered bsdf
        let entered_top = TWO_SIDED || wo.z > 0.0;
        let enter_interface = if entered_top
        {
            TopOrBottomBxDF{
                top: Some(&self.top),
                bottom: None,
            }
        } else {
            TopOrBottomBxDF
            {
                top: None,
                bottom: Some(&self.bottom),
            }
        };

        // Determine exit interface and exit z
        let (exit_interface, non_exit_interface) = if same_hemisphere(wo, wi) ^ entered_top
        {
            let exit_interface = TopOrBottomBxDF
            {
                top: None,
                bottom: Some(&self.bottom),
            };
            let non_exit_interface = TopOrBottomBxDF
            {
                top: Some(&self.top),
                bottom: None,
            };
            (exit_interface, non_exit_interface)
        } else {
            let exit_interface = TopOrBottomBxDF
            {
                top: Some(&self.top),
                bottom: None,
            };
            let non_exit_interface = TopOrBottomBxDF
            {
                top: None,
                bottom: Some(&self.bottom),
            };
            (exit_interface, non_exit_interface)
        };

        let exit_z = if same_hemisphere(wo, wi) ^ entered_top
        {
            0.0
        } else {
            self.thickness
        };

        // Account for reflection at entrance interface
        if same_hemisphere(wo, wi) {
            f = enter_interface.f(wo, wi, mode) * self.n_samples as Float;
        }

        // TODO Use a seed for this
        let rng = &mut SmallRng::from_entropy();
        let mut r = ||
        {
            let v: Float = rng.gen();
            // TODO - Can we have a ONE_MINUS_EPSILON constant instead?
            Float::min(v, next_float_down(1.0))
        };

        for s in 0..self.n_samples
        {
            // Sample random walk through layers to estimate BSDF value
            // Sample transmission direction through entrance interface
            let uc: Float = r();
            let wos = enter_interface.sample_f(
                wo, uc, Point2f::new(r(), r()), mode, BxDFReflTransFlags::TRANSMISSION);
            
            if wos.is_none() {
                continue;
            }
            let wos = wos.unwrap();

            if wos.f.is_zero() || wos.pdf == 0.0 || wos.wi.z == 0.0
            {
                continue;
            }

            // Sample BSDF for virtual light from wi
            let uc = r();
            let wis_mode = match mode
            {
                TransportMode::Radiance => TransportMode::Importance,
                TransportMode::Importance => TransportMode::Radiance,
            };
            let wis = exit_interface.sample_f(wi, uc, Point2f::new(r(), r()), wis_mode, BxDFReflTransFlags::TRANSMISSION);
            if wis.is_none()
            {
                continue;
            }
            let wis = wis.unwrap();
            if wis.f.is_zero() || wis.pdf == 0.0 || wis.wi.z == 0.0
            {
                continue;
            }

            // Declare state for random walk through BSDF layers
            let mut beta = wos.f * abs_cos_theta(wos.wi) / wos.pdf;
            let mut z = if entered_top
            {
                self.thickness
            } else {
                0.0
            };
            let mut w = wos.wi;
            let phase = HGPhaseFunction::new(self.g);

            for depth in 0..self.max_depth
            {
                // Sample next event for layered BSDF random walk
                // Possibly terminate layered BSDF random walk with Russian roulette
                if depth > 3 && beta.max_component_value() < 0.25
                {
                    let q = Float::max(0.0, 1.0 - beta.max_component_value());
                    if r() < q
                    {
                        break;
                    }
                    beta /= 1.0 - q;
                }

                // Account for media between layers and possibly scatter
                if self.albedo.is_zero()
                {
                    // Advance to next layer boundary and update beta for transmittance
                    z = if z == self.thickness
                    {
                        0.0
                    } else {
                        self.thickness
                    };
                    beta = beta * self.tr(self.thickness, w);
                } else {
                    // Sample medium scattering
                    let sigma_t = 1.0;
                    let dz = sample_exponential(r(), sigma_t / Float::abs(w.z));
                    let zp = if w.z > 0.0 
                    {
                        z + dz
                    } else {
                        z - dz
                    };

                    if z == zp
                    {
                        continue;
                    }

                    if 0.0 < zp && zp < self.thickness
                    {
                        // Handle scattering event
                        // Account for scattering through exit_interface suing wis
                        let wt = if !exit_interface.flags().is_specular()
                        {
                            power_heuristic(1, wis.pdf, 1, phase.pdf(-w, -wis.wi))
                        } else {
                            1.0
                        };
                        f += beta * self.albedo * phase.p(-w, -wis.wi) * wt * self.tr(zp - exit_z, wis.wi) * wis.f / wis.pdf;

                        // Sample phase function and update layered path state
                        let u = Point2f::new(r(), r());
                        let ps = phase.sample_p(-w, u);
                        if ps.is_none()
                        {
                            continue;
                        }
                        let ps = ps.unwrap();
                        if ps.pdf == 0.0 || ps.wi.z == 0.0
                        {
                            continue;
                        }
                        beta = beta * (self.albedo * ps.p / ps.pdf);
                        w = ps.wi;
                        z = zp;

                        // Possibly account for scattering through exit_interface
                        if ((z < exit_z && w.z > 0.0) || (z > exit_z && w.z < 0.0)) &&
                        !exit_interface.flags().is_specular()
                        {
                            // Account for scattering through exit_interface 
                            let f_exit = exit_interface.f(-w, wi, mode);
                            if !f_exit.is_zero()
                            {
                                let exit_pdf = exit_interface.pdf(-w, wi, mode, BxDFReflTransFlags::TRANSMISSION);
                                let wt = power_heuristic(1, ps.pdf, 1, exit_pdf);
                                f += beta * self.tr(zp - exit_z, ps.wi) * f_exit * wt;
                            }
                        }

                        continue;
                    }
                    z = Float::clamp(zp, 0.0, self.thickness);
                }

                // Account for scattering at appropriate interface
                if z == exit_z
                {
                    // Account for reflection at exit_interface
                    let uc = r();
                    let bs = exit_interface.sample_f(-w, uc, Point2f::new(r(), r()), mode, BxDFReflTransFlags::REFLECTION);
                    if bs.is_none()
                    {
                        continue;
                    }
                    let bs = bs.unwrap();
                    if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0
                    {
                        continue;
                    }
                    beta = beta * (bs.f * abs_cos_theta(bs.wi) / bs.pdf);
                    w = bs.wi;
                } else {
                    // Account for scattering at non_exit_interface
                    if !non_exit_interface.flags().is_specular()
                    {
                        // Add NEE contribution along presampled wis direction
                        let wt = if !exit_interface.flags().is_specular()
                        {
                            power_heuristic(1, wis.pdf, 1, non_exit_interface.pdf(-w, -wis.wi, mode, BxDFReflTransFlags::ALL))
                        } else {
                            1.0
                        };
                        f += beta * non_exit_interface.f(-w, -wis.wi, mode) * abs_cos_theta(wis.wi) * wt * self.tr(self.thickness, wis.wi) * wis.f / wis.pdf;
                    }
                    // Sample new direction using BSDF at non_exit_interface
                    let uc = r();
                    let u = Point2f::new(r(), r());
                    let bs = non_exit_interface.sample_f(
                        -w, uc, u, mode, BxDFReflTransFlags::REFLECTION);
                    if bs.is_none()
                    {
                        continue;
                    }
                    let bs = bs.unwrap();
                    if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0
                    {
                        continue;
                    }
                    beta = beta * (bs.f * abs_cos_theta(bs.wi) / bs.pdf);
                    w = bs.wi;

                    if !exit_interface.flags().is_specular()
                    {
                        // Add NEE contribution along direction for BSDF sample
                        let f_exit = exit_interface.f(-w, wi, mode);
                        if !f_exit.is_zero()
                        {
                            let wt = if !non_exit_interface.flags().is_specular()
                            {
                                let exit_pdf = exit_interface.pdf(-w, wi, mode, BxDFReflTransFlags::TRANSMISSION);
                                power_heuristic(1, bs.pdf, 1, exit_pdf)
                            } else {
                                1.0
                            };
                            f += beta * self.tr(self.thickness, bs.wi) * f_exit * wt;
                        }
                    }
                }
            }
        }

        f / self.n_samples as Float
    }

    fn sample_f(
        &self,
        wo: Vector3f,
        uc: Float,
        u: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        assert!(sample_flags == BxDFReflTransFlags::ALL);
        let mut wo = wo;
        // Set wo for layered sampling
        let flip_wi = if TWO_SIDED && wo.z < 0.0
        {
            wo = -wo;
            true
        } else {
            false
        };

        // Sample BSDF at entrance interface to get initial direction w
        let entered_top = TWO_SIDED || wo.z > 0.0;
        let mut bs = if entered_top
        {
            self.top.sample_f(wo, uc, u, mode, BxDFReflTransFlags::ALL)?
        } else {
            self.bottom.sample_f(wo, uc, u, mode, BxDFReflTransFlags::ALL)?
        };
        if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0
        {
            return None;
        }

        if bs.is_reflection()
        {
            if flip_wi
            {
                bs.wi = -bs.wi;
            }
            bs.pdf_is_proportional = true;
            return Some(bs);
        }

        let mut w = bs.wi;
        let mut specular_path = bs.is_specular();

        // TODO Use a seed for this
        let rng = &mut SmallRng::from_entropy();
        let mut r = ||
        {
            let v: Float = rng.gen();
            // TODO - Can we have a ONE_MINUS_EPSILON constant instead?
            Float::min(v, next_float_down(1.0))
        };

        let mut f = bs.f * abs_cos_theta(bs.wi);
        let mut pdf =  bs.pdf;
        let mut z = if entered_top { self.thickness } else { 0.0 };
        let phase = HGPhaseFunction::new(self.g);

        for depth in 0..self.max_depth
        {
            // Follow random walk through layers to sample layered BSDF
            // Possibly terminate through russian roulette
            let rr_beta = f.max_component_value() / pdf;
            if depth > 3 && rr_beta < 0.25
            {
                let q = Float::max(0.0, 1.0 - rr_beta);
                if r() < q
                {
                    return None;
                }
                pdf *= 1.0 - q;
            }
            if w.z == 0.0
            {
                return None;

            }

            if !self.albedo.is_zero()
            {
                // Sample potential scattering event in layered medium
                let sigma_t = 1.0;
                let dz = sample_exponential(r(), sigma_t / abs_cos_theta(w));
                let zp = if w.z > 0.0 {
                    z + dz
                } else {
                    z - dz
                };
                if zp == z
                {
                    return None;
                }
                if 0.0 < zp && zp < self.thickness
                {
                    // Update path state for valid scattering event between interfaces
                    let ps = phase.sample_p(-w, Point2f::new(r(), r()))?;
                    if ps.pdf == 0.0 || ps.wi.z == 0.0
                    {
                        return None;
                    }
                    f *= self.albedo * ps.p;
                    pdf *= ps.pdf;
                    specular_path = false;
                    w = ps.wi;
                    z = zp;
                    
                    continue;
                }
                z = Float::clamp(zp, 0.0, self.thickness);
                if z == 0.0
                {
                    debug_assert!(w.z < 0.0);
                } else {
                    debug_assert!(w.z > 0.0);
                }
            } else {
                z = if z == self.thickness { 0.0 } else { self.thickness };
                f = f * self.tr(self.thickness, w);
            }

            // Initialize interface for current interface surface
            let interface = if z == 0.0 
            {
                TopOrBottomBxDF{
                    top: None,
                    bottom: Some(&self.bottom),
                }
            } else {
                TopOrBottomBxDF{
                    top: Some(&self.top),
                    bottom: None,
                }
            };

            // Sample interface BSDF to determine new path direction
            let uc = r();
            let u = Point2f::new(r(), r());
            let bs = interface.sample_f(-w, uc, u, mode, BxDFReflTransFlags::ALL)?;
            if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0
            {
                return None;
            }
            f *= bs.f;
            pdf *= bs.pdf;
            specular_path &= bs.is_specular();
            w = bs.wi;
            
            // Return BSDFSample if the path has left the layers
            if bs.is_transmission()
            {
                let mut flags = if same_hemisphere(wo, w)
                {
                    BxDFFLags::REFLECTION
                } else {
                    BxDFFLags::TRANSMISSION
                };
                flags |= if specular_path { BxDFFLags::SPECULAR } else { BxDFFLags::GLOSSY };
                if flip_wi
                {
                    w = -w;
                }
                return Some(BSDFSample{
                    f,
                    wi: w,
                    pdf,
                    flags,
                    eta: 1.0,
                    pdf_is_proportional: true,
                });
            }

            // Scale f by cosine term after scattering at the interface
            f = f * abs_cos_theta(bs.wi);

        }

        None
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        assert!(sample_flags == BxDFReflTransFlags::ALL);

        let mut wo = wo;
        let mut wi = wi;
        if TWO_SIDED && wo.z < 0.0
        {
            wo = -wo;
            wi = -wi;
        }

        // Declare RNG for layered PDF evaluation
        // TODO Use a seed for this
        let rng = &mut SmallRng::from_entropy();
        let mut r = ||
        {
            let v: Float = rng.gen();
            // TODO - Can we have a ONE_MINUS_EPSILON constant instead?
            Float::min(v, next_float_down(1.0))
        };

        // Update pdf_sum for reflection at the entrance layer
        let entered_top = TWO_SIDED || wo.z > 0.0;
        let mut pdf_sum = 0.0;
        if same_hemisphere(wo, wi)
        {
            let refl_flag = BxDFReflTransFlags::REFLECTION;
            pdf_sum += if entered_top
            {
                self.n_samples as Float * self.top.pdf(wo, wi, mode, refl_flag)
            } else {
                self.n_samples as Float * self.bottom.pdf(wo, wi, mode, refl_flag)
            };
        }

        for s in 0..self.n_samples
        {
            if same_hemisphere(wo, wi)
            {
                let (r_interface, t_interface) = if entered_top
                {
                    let r_interface = TopOrBottomBxDF{
                        top: None,
                        bottom: Some(&self.bottom),
                    };
                    let t_interface = TopOrBottomBxDF{
                        top: Some(&self.top),
                        bottom: None,
                    };
                    (r_interface, t_interface)
                } else {
                    let r_interface = TopOrBottomBxDF{
                        top: Some(&self.top),
                        bottom: None,
                    };
                    let t_interface = TopOrBottomBxDF{
                        top: None,
                        bottom: Some(&self.bottom),
                    };
                    (r_interface, t_interface)
                };

                // Sample t_interface to get direction into the layers
                let trans = BxDFReflTransFlags::TRANSMISSION;
                let wos = t_interface.sample_f(wo, r(), Point2f::new(r(), r()), mode, trans);
                let wis_mode = match mode
                {
                    TransportMode::Radiance => TransportMode::Importance,
                    TransportMode::Importance => TransportMode::Radiance,
                };
                let wis = t_interface.sample_f(wi, r(), Point2f::new(r(), r()), wis_mode, trans);

                // Update pdf_sum accounting for TRT scattering events
                if let (Some(wos), Some(wis)) = (wos, wis)
                {
                    if !wos.f.is_zero() && wos.pdf > 0.0 && !wis.f.is_zero() && wis.pdf > 0.0
                    {
                        if !t_interface.flags().is_non_specular()
                        {
                            pdf_sum += r_interface.pdf(-wos.wi, -wis.wi, mode, BxDFReflTransFlags::ALL);
                        } else {
                            // Use multiple importance sampling to estimate PDF product
                            let rs = r_interface.sample_f(-wos.wi, r(), Point2f::new(r(), r()), mode, BxDFReflTransFlags::ALL);
                            if let Some(rs) = rs
                            {
                                if !r_interface.flags().is_non_specular()
                                {
                                    pdf_sum += t_interface.pdf(-rs.wi, wi, mode, BxDFReflTransFlags::ALL);
                                } else {
                                    let r_pdf = r_interface.pdf(-wos.wi, -wis.wi, mode, BxDFReflTransFlags::ALL);
                                    let wt = power_heuristic(1, wis.pdf, 1, r_pdf);
                                    pdf_sum += wt * r_pdf;

                                    let t_pdf = t_interface.pdf(-rs.wi, wi, mode, BxDFReflTransFlags::ALL);
                                    let wt = power_heuristic(1, rs.pdf, 1, t_pdf);
                                    pdf_sum += wt * t_pdf;
                                }
                            }
                        }
                    }
                }
            } else {
                // Note the same hemisphere
                // Evaluate TT term for PDF estimate

                let (to_interface, ti_interface) = if entered_top
                {
                    let to_interface = TopOrBottomBxDF{
                        top: Some(&self.top),
                        bottom: None,
                    };
                    let ti_interface = TopOrBottomBxDF{
                        top: None,
                        bottom: Some(&self.bottom),
                    };
                    (to_interface, ti_interface)
                } else {
                    let to_interface = TopOrBottomBxDF{
                        top: None,
                        bottom: Some(&self.bottom),
                    };
                    let ti_interface = TopOrBottomBxDF
                    {
                        top: Some(&self.top),
                        bottom: None,
                    };
                    (to_interface, ti_interface)
                };

                let uc = r();
                let u = Point2f::new(r(), r());
                let wos = to_interface.sample_f(wo, uc, u, mode, BxDFReflTransFlags::ALL);
                if wos.is_none()
                {
                    continue;
                }
                let wos = wos.unwrap();
                if wos.f.is_zero() || wos.pdf == 0.0 || wos.wi.z == 0.0 || wos.is_reflection()
                {
                    continue;
                }

                let uc = r();
                let u = Point2f::new(r(), r());
                let wis_mode = match mode
                {
                    TransportMode::Radiance => TransportMode::Importance,
                    TransportMode::Importance => TransportMode::Radiance,
                };
                let wis = ti_interface.sample_f(wi, uc, u, wis_mode, BxDFReflTransFlags::ALL);
                if wis.is_none()
                {
                    continue;
                }
                let wis = wis.unwrap();
                if wis.f.is_zero() || wis.pdf == 0.0 || wis.wi.z == 0.0 || wis.is_reflection()
                {
                    continue;
                }

                if to_interface.flags().is_specular()
                {
                    pdf_sum += ti_interface.pdf(-wos.wi, wi, mode, BxDFReflTransFlags::ALL);
                } else if ti_interface.flags().is_specular()
                {
                    pdf_sum += to_interface.pdf(wo, -wis.wi, mode, BxDFReflTransFlags::ALL);
                } else {
                    pdf_sum += (to_interface.pdf(wo, -wis.wi, mode, BxDFReflTransFlags::ALL)
                        + ti_interface.pdf(-wos.wi, wi, mode, BxDFReflTransFlags::ALL)) / 2.0;
                }
            }
        }

        // Return mixture of PDF estimate and constant PDF
        lerp(0.9, 1.0 / (4.0 * PI_F), pdf_sum / self.n_samples as Float)
    }

    fn flags(&self) -> BxDFFLags {
        let top_flags = self.top.flags();
        let bottom_flags = self.bottom.flags();

        // Otherwise, what are we doing here?
        debug_assert!(top_flags.is_transmissive() || bottom_flags.is_transmissive());

        let mut flags = BxDFFLags::REFLECTION;
        if top_flags.is_specular()
        {
            flags = flags | BxDFFLags::SPECULAR;
        }

        if top_flags.is_diffuse() || bottom_flags.is_diffuse() || !self.albedo.is_zero()
        {
            flags = flags | BxDFFLags::DIFFUSE;
        } else if top_flags.is_glossy() || bottom_flags.is_glossy()
        {
            flags = flags | BxDFFLags::GLOSSY;
        }

        if top_flags.is_transmissive() && bottom_flags.is_transmissive()
        {
            flags = flags | BxDFFLags::TRANSMISSION;
        }

        flags
    }
    
    fn regularize(&mut self) {
        self.top.regularize();
        self.bottom.regularize();
    }
}

struct TopOrBottomBxDF<'a, TopBxDF, BottomBxDF>
where
    TopBxDF: BxDFI,
    BottomBxDF: BxDFI,
{
    top: Option<&'a TopBxDF>,
    bottom: Option<&'a BottomBxDF>,
}

impl<'a, TopBxDF, BottomBxDF> TopOrBottomBxDF<'a, TopBxDF, BottomBxDF>
where 
    TopBxDF: BxDFI,
    BottomBxDF: BxDFI,
{
    fn f(&self, wo: Vector3f, wi: Vector3f, mode: TransportMode) -> SampledSpectrum {
        if let Some(top) = &self.top
        {
            top.f(wo, wi, mode)
        } else if let Some(bottom) = &self.bottom
        {
            bottom.f(wo, wi, mode)
        } else
        {
            panic!("TopOrBottomBxDF: No BxDFs to evaluate");
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
        if let Some(top) = &self.top
        {
            top.sample_f(wo, uc, u, mode, sample_flags)
        } else if let Some(bottom) = &self.bottom
        {
            bottom.sample_f(wo, uc, u, mode, sample_flags)
        } else
        {
            panic!("TopOrBottomBxDF: No BxDFs to sample");
        }
    }

    fn pdf(
        &self,
        wo: Vector3f,
        wi: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if let Some(top) = &self.top
        {
            top.pdf(wo, wi, mode, sample_flags)
        } else if let Some(bottom) = &self.bottom
        {
            bottom.pdf(wo, wi, mode, sample_flags)
        } else
        {
            panic!("TopOrBottomBxDF: No BxDFs to evaluate");
        }
    }

    fn flags(&self) -> BxDFFLags {
        if let Some(top) = &self.top
        {
            top.flags()
        }
        else if let Some(bottom) = &self.bottom
        {
            bottom.flags()
        } else {
            panic!("TopOrBottomBxDF: No BxDFs to evaluate");
        }
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
    use float_cmp::approx_eq;

    use crate::{scattering::TrowbridgeReitzDistribution, vecmath::{Point2f, Tuple2, Tuple3, Vector3f}, Float};

    use super::{BxDFFLags, BxDFI, DielectricBxDF};

    #[test]
    fn mf_distrib()
    {
        let distrib = TrowbridgeReitzDistribution::new(0.0299999993, 0.0299999993);
        let wm = Vector3f::new(
            -0.430063188,
            -0.881908476,
            0.193088099,
        );
        let wi = Vector3f::new(
            0.568110108, 0.816620350, 0.101893365
        );
        let d = distrib.d(wm);
        let g = distrib.g(wm, wi);
        approx_eq!(Float, g, 0.954060972);
        approx_eq!(Float, d, 0.000309075956);
    }

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

    #[test]
    fn dielectric_sample_f()
    {
        let bxdf = DielectricBxDF::new(
            1.5,
            TrowbridgeReitzDistribution::new(0.0, 0.0)
        );

        let sample = bxdf.sample_f(
            Vector3f::new(-0.419299453, -0.656406343, 0.627151370),
            0.237656280,
            Point2f::new(
                0.0488742627,
                0.941848040
            ),
            super::TransportMode::Radiance,
            super::BxDFReflTransFlags::ALL
        );

        assert!(sample.is_some());
        let sample = sample.unwrap();
        assert_eq!(sample.flags, BxDFFLags::SPECULAR_TRANSMISSION);
        approx_eq!(Float, sample.pdf, 0.940032840);
        approx_eq!(Float, sample.eta, 1.5);
        assert!(!sample.pdf_is_proportional);
        approx_eq!(Float, sample.f[0], 0.488867134);
        approx_eq!(Float, sample.f[1], 0.488867134);
        approx_eq!(Float, sample.f[2], 0.488867134);
        approx_eq!(Float, sample.f[3], 0.488867134);
        approx_eq!(Float, sample.wi.x, 0.279532969);
        approx_eq!(Float, sample.wi.y, 0.437604219);
        approx_eq!(Float, sample.wi.z, -0.854613364);
    }
}
