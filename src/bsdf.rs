use crate::{
    bxdf::{BSDFSample, BxDF, BxDFFLags, BxDFI, BxDFReflTransFlags, TransportMode},
    frame::Frame,
    spectra::sampled_spectrum::SampledSpectrum,
    vecmath::{Normal3f, Normalize, Point2f, Vector3f},
    Float,
};

/// BxDF implementations handle computations in a local shading coordinate system;
/// BSDF is a small wrapper around BxDF which handles the conversions between that
/// shading coordinate system and the rendering coordinate system.
pub struct BSDF {
    bxdf: BxDF,
    shading_frame: Frame,
}

impl BSDF {
    // TODO we would like to implement Default() which initializes the bxdf as None,
    // which is useful for transitions between different media that themselves do not scatter light;
    // in which case, a bool() method that checks if bxdf.is_some() would also be useful. pg 544 of PBRTv4

    pub fn new(ns: Normal3f, dpdus: Vector3f, bxdf: BxDF) -> BSDF {
        let shading_frame = Frame::from_xz(dpdus.normalize(), Vector3f::from(ns));
        BSDF {
            bxdf,
            shading_frame,
        }
    }

    /// Provides information about the BSDF's high-level properties.
    pub fn flags(&self) -> BxDFFLags {
        self.bxdf.flags()
    }

    pub fn render_to_local(&self, v: Vector3f) -> Vector3f {
        self.shading_frame.to_local_v(&v)
    }

    pub fn local_to_render(&self, v: Vector3f) -> Vector3f {
        self.shading_frame.from_local_v(&v)
    }

    /// Returns the value of the distribution function for the given pair of directions
    pub fn f(
        &self,
        wo_render: Vector3f,
        wi_render: Vector3f,
        mode: TransportMode,
    ) -> SampledSpectrum {
        let wi = self.render_to_local(wi_render);
        let wo = self.render_to_local(wo_render);
        if wo.z == 0.0 {
            // In the case that wo lies directly on the surface's tangent plane,
            // to avoid NaN propagation, return a zero-valued SampledSpectrum.
            return SampledSpectrum::from_const(0.0);
        }
        self.bxdf.f(wo, wi, mode)
    }

    pub fn sample_f(
        &self,
        wo_render: Vector3f,
        u: Float,
        u2: Point2f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        let wo = self.render_to_local(wo_render);
        if wo.z == 0.0 || !((self.bxdf.flags().bits() & sample_flags.bits()) != 0) {
            // TODO possibly better to return a default BSDFSample instead?
            return None;
        }
        // Sample bxdf and return BSDFSample
        let mut bs = self.bxdf.sample_f(wo, u, u2, mode, sample_flags)?;
        if bs.f.is_zero() || bs.pdf == 0.0 || bs.wi.z == 0.0 {
            return None;
        }
        debug_assert!(bs.pdf >= 0.0);

        bs.wi = self.local_to_render(bs.wi);
        Some(bs)
    }

    pub fn pdf(
        &self,
        wo_render: Vector3f,
        wi_render: Vector3f,
        mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        let wo = self.render_to_local(wo_render);
        let wi = self.render_to_local(wi_render);
        if wo.z == 0.0 {
            return 0.0;
        }
        self.bxdf.pdf(wo, wi, mode, sample_flags)
    }

    pub fn rho_hd(&self, wo_render: Vector3f, uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        let wo = self.render_to_local(wo_render);
        self.bxdf.rho_hd(wo, uc, u2)
    }

    pub fn rho_hh(&self, u1: &[Point2f], uc: &[Float], u2: &[Point2f]) -> SampledSpectrum {
        self.bxdf.rho_hh(u1, uc, u2)
    }

    pub fn regularize(&mut self)
    {
        self.bxdf.regularize();
    }
}
