use crate::{
    bsdf::BSDF,
    bxdf::DiffuseBxDF,
    image::Image,
    interaction::SurfaceInteraction,
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    texture::{FloatTexture, FloatTextureI, SpectrumTexture, SpectrumTextureI, TextureEvalContext},
    vecmath::{Normal3f, Vector3f},
    Float,
};

pub trait MaterialI {
    // TODO issue with this one is that the Enum Material can't choose a constant ConcreteBxDF.
    // Indeed, if enum Material implemented this, then get_bxdf() would return whatever the Material::ConcreteBxDF is
    // rather than whatever the variants' one is.
    // ConcreteBxDF really only makes sense for the concrete variants...
    type ConcreteBxDF;

    // TODO Consider a ScratchBuffer equivalent for these functions; probably necessary for performance.
    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Self::ConcreteBxDF;

    // TODO PBRT gets arcane with GetBSDF. Is there a simple Rust solution?
    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> BSDF;

    // TODO get_bssrdf() for subsurface scattering.

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool;

    // TODO This will use a differerent Image implementation that matches PBRT;
    // we will be returning None for any material we implement right now.
    fn get_normal_map(&self) -> Option<Image>;

    fn get_bump_map(&self) -> Option<FloatTexture>;

    fn has_subsurface_scattering(&self) -> bool;
}

/// Materials evaluate textures to get parameter values that are used to initialize their
/// particular BSFD model.
#[derive(Debug)]
pub enum Material {
    Diffuse(DiffuseMaterial),
}

impl MaterialI for Material {
    type ConcreteBxDF = ();

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        todo!()
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> BSDF {
        match self {
            Material::Diffuse(m) => m.get_bsdf(tex_eval, ctx, lambda),
        }
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        todo!()
    }

    fn get_normal_map(&self) -> Option<Image> {
        todo!()
    }

    fn get_bump_map(&self) -> Option<FloatTexture> {
        todo!()
    }

    fn has_subsurface_scattering(&self) -> bool {
        todo!()
    }
}

#[derive(Debug)]
pub struct DiffuseMaterial {
    reflectance: SpectrumTexture,
    // TODO Add normal map and bump map, and update the getters for those.
}

impl MaterialI for DiffuseMaterial {
    type ConcreteBxDF = DiffuseBxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let r = tex_eval
            .evaluate_spectrum(&self.reflectance, &ctx.tex_ctx, lambda)
            .clamp(0.0, 1.0);
        DiffuseBxDF::new(r)
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, crate::bxdf::BxDF::Diffuse(bxdf))
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        tex_eval.can_evaluate(&[], &[&self.reflectance])
    }

    fn get_normal_map(&self) -> Option<Image> {
        // TODO we should add this later.
        None
    }

    fn get_bump_map(&self) -> Option<FloatTexture> {
        // TODO we should add this later.
        None
    }

    fn has_subsurface_scattering(&self) -> bool {
        false
    }
}

pub struct MaterialEvalContext {
    tex_ctx: TextureEvalContext,
    wo: Vector3f,
    ns: Normal3f,
    dpdus: Vector3f,
}

impl MaterialEvalContext {
    pub fn new(si: &SurfaceInteraction) -> MaterialEvalContext {
        MaterialEvalContext {
            tex_ctx: TextureEvalContext::from(si),
            wo: si.interaction.wo,
            ns: si.shading.n,
            dpdus: si.shading.dpdu,
        }
    }
}

/// A TextureEvaluator is a class that is able to evaluate some or all of the
/// set of Texture types. Its can_evaluate() method reports if it can evaluate all of
/// a set of textures.
/// Why do Material implementations not simply call Texture::evaluate() directly, rather
/// than including this layer of abstraction?
/// Because this can aide performance in a wavefront integrator by separating materials
/// into those that have lightweight textures and those with heavyweight textures to
/// process those separately.
/// For cases where that isn't necessary when using the UniversalTextureEvaluator (which
/// can evaluate all textures and is the default for most scenarios), the compiler can
/// optimize away this abstraction.
pub trait TextureEvaluatorI {
    fn can_evaluate(&self, f_tex: &[&FloatTexture], s_tex: &[&SpectrumTexture]) -> bool;

    fn evaluate_float(&self, tex: &FloatTexture, ctx: &TextureEvalContext) -> Float;

    fn evaluate_spectrum(
        &self,
        tex: &SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;
}

/// A TextureEvaluator which can evaluate all textures; the default in most scenarios.
pub struct UniversalTextureEvaluator {}

impl TextureEvaluatorI for UniversalTextureEvaluator {
    fn can_evaluate(&self, _f_tex: &[&FloatTexture], _s_tex: &[&SpectrumTexture]) -> bool {
        true
    }

    fn evaluate_float(&self, tex: &FloatTexture, ctx: &TextureEvalContext) -> Float {
        tex.evaluate(ctx)
    }

    fn evaluate_spectrum(
        &self,
        tex: &SpectrumTexture,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        tex.evaluate(ctx, lambda)
    }
}
