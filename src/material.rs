use std::{collections::HashMap, sync::Arc};

use crate::{
    bsdf::BSDF,
    bxdf::{BxDF, CoatedDiffuseBxDF, ConductorBxDF, DielectricBxDF, DiffuseBxDF, ThinDielectricBxDF},
    image::Image,
    interaction::SurfaceInteraction,
    loading::{
        paramdict::{NamedTextures, SpectrumType, TextureParameterDictionary},
        parser_target::FileLoc,
    },
    math::Sqrt,
    scattering::TrowbridgeReitzDistribution,
    spectra::{
        sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths,
        spectrum::SpectrumI, ConstantSpectrum, NamedSpectrum, Spectrum,
    },
    texture::{
        FloatTexture, FloatTextureI, SpectrumConstantTexture, SpectrumTexture, SpectrumTextureI,
        TextureEvalContext,
    },
    vecmath::{Normal3f, Vector3f},
    Float,
};

pub trait MaterialI {
    type ConcreteBxDF;

    // TODO Consider a ScratchBuffer equivalent for these functions; probably necessary for performance.
    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF;

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF;

    // TODO get_bssrdf() for subsurface scattering.

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool;

    fn get_normal_map(&self) -> Option<Arc<Image>>;

    fn get_displacement(&self) -> Option<Arc<FloatTexture>>;

    fn has_subsurface_scattering(&self) -> bool;
}

/// Materials evaluate textures to get parameter values that are used to initialize their
/// particular BSFD model.
#[derive(Debug)]
pub enum Material {
    Diffuse(DiffuseMaterial),
    Conductor(ConductorMaterial),
    Dielectric(DielectricMaterial),
    ThinDielectric(ThinDielectricMaterial),
    CoatedDiffuse(CoatedDiffuseMaterial),
}

impl Material {
    pub fn create(
        name: &str,
        parameters: &mut TextureParameterDictionary,
        textures: &NamedTextures,
        normal_map: Option<Arc<Image>>,
        named_materials: &HashMap<String, Arc<Material>>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> Material {
        let material = match name {
            "diffuse" => Material::Diffuse(DiffuseMaterial::create(
                parameters,
                textures,
                normal_map,
                cached_spectra,
                loc,
            )),
            "conductor" => Material::Conductor(ConductorMaterial::create(
                parameters,
                normal_map,
                loc,
                cached_spectra,
                textures,
            )),
            "dielectric" => Material::Dielectric(DielectricMaterial::create(
                parameters,
                normal_map,
                loc,
                cached_spectra,
                textures,
            )),
            "thindielectric" => Material::ThinDielectric(ThinDielectricMaterial::create(
                parameters,
                normal_map,
                loc,
                cached_spectra,
                textures,
            )),
            "coateddiffuse" => Material::CoatedDiffuse(CoatedDiffuseMaterial::create(
                parameters,
                normal_map,
                loc,
                cached_spectra,
                textures,
            )),
            _ => panic!("Material {} unknown.", name),
        };
        material
    }
}

impl MaterialI for Material {
    /// Since this Material encompasses all the Material variants, its concrete BxDF
    /// encompasses all the BxDF variants.
    type ConcreteBxDF = BxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        match self {
            Material::Diffuse(m) => BxDF::Diffuse(m.get_bxdf(tex_eval, ctx, lambda)),
            Material::Conductor(m) => BxDF::Conductor(m.get_bxdf(tex_eval, ctx, lambda)),
            Material::Dielectric(m) => BxDF::Dielectric(m.get_bxdf(tex_eval, ctx, lambda)),
            Material::ThinDielectric(m) => BxDF::ThinDielectric(m.get_bxdf(tex_eval, ctx, lambda)),
            Material::CoatedDiffuse(m) => BxDF::CoatedDiffuse(m.get_bxdf(tex_eval, ctx, lambda)),
        }
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        // PAPERDOC - PBRT's implementation of GetBsdf() involves some semi-arcane C++.
        // We avoid arcane-looking code, but aren't at parity - if we just implement bsdf for each
        // variant, then we violate DRY. Ideally we'd like to move the implementation we have for Diffuse
        // (which should be the same except for the types involved) into the MaterialI trait as a default implementation.
        // But I'm not sure if we can use an associated *variant*, not just an associated type?
        // Look into it.
        // I want to revisit this for using scratch_buffer anyways...
        match self {
            Material::Diffuse(m) => m.get_bsdf(tex_eval, ctx, lambda),
            Material::Conductor(m) => m.get_bsdf(tex_eval, ctx, lambda),
            Material::Dielectric(m) => m.get_bsdf(tex_eval, ctx, lambda),
            Material::ThinDielectric(m) => m.get_bsdf(tex_eval, ctx, lambda),
            Material::CoatedDiffuse(m) => m.get_bsdf(tex_eval, ctx, lambda),
        }
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        match self {
            Material::Diffuse(m) => m.can_evaluate_textures(tex_eval),
            Material::Conductor(m) => m.can_evaluate_textures(tex_eval),
            Material::Dielectric(m) => m.can_evaluate_textures(tex_eval),
            Material::ThinDielectric(m) => m.can_evaluate_textures(tex_eval),
            Material::CoatedDiffuse(m) => m.can_evaluate_textures(tex_eval),
        }
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        match self {
            Material::Diffuse(m) => m.get_normal_map(),
            Material::Conductor(m) => m.get_normal_map(),
            Material::Dielectric(m) => m.get_normal_map(),
            Material::ThinDielectric(m) => m.get_normal_map(),
            Material::CoatedDiffuse(m) => m.get_normal_map(),
        }
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        match self {
            Material::Diffuse(m) => m.get_displacement(),
            Material::Conductor(m) => m.get_displacement(),
            Material::Dielectric(m) => m.get_displacement(),
            Material::ThinDielectric(m) => m.get_displacement(),
            Material::CoatedDiffuse(m) => m.get_displacement(),
        }
    }

    fn has_subsurface_scattering(&self) -> bool {
        match self {
            Material::Diffuse(m) => m.has_subsurface_scattering(),
            Material::Conductor(m) => m.has_subsurface_scattering(),
            Material::Dielectric(m) => m.has_subsurface_scattering(),
            Material::ThinDielectric(m) => m.has_subsurface_scattering(),
            Material::CoatedDiffuse(m) => m.has_subsurface_scattering(),
        }
    }
}

#[derive(Debug)]
pub struct DiffuseMaterial {
    reflectance: Arc<SpectrumTexture>,
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
}

impl DiffuseMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        textures: &NamedTextures,
        normal_map: Option<Arc<Image>>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        _loc: &FileLoc,
    ) -> DiffuseMaterial {
        let reflectance = parameters.get_spectrum_texture(
            "reflectance",
            None,
            SpectrumType::Albedo,
            cached_spectra,
            textures,
        );
        let reflectance = if let Some(reflectance) = reflectance {
            reflectance
        } else {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(
                Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5))),
            )))
        };
        let displacement = Some(parameters.get_float_texture("displacement", 0.0, textures));

        DiffuseMaterial::new(reflectance, displacement, normal_map)
    }

    pub fn new(
        reflectance: Arc<SpectrumTexture>,
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
    ) -> DiffuseMaterial {
        DiffuseMaterial {
            reflectance,
            displacement,
            normal_map,
        }
    }
}

impl MaterialI for DiffuseMaterial {
    type ConcreteBxDF = DiffuseBxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
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
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, crate::bxdf::BxDF::Diffuse(bxdf))
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        tex_eval.can_evaluate(&[], &[&self.reflectance])
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        self.normal_map.clone()
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        self.displacement.clone()
    }

    fn has_subsurface_scattering(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct ConductorMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    eta: Option<Arc<SpectrumTexture>>,
    k: Option<Arc<SpectrumTexture>>,
    reflectance: Option<Arc<SpectrumTexture>>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
}

impl ConductorMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        _loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> ConductorMaterial {
        let mut eta = parameters.get_spectrum_texture_or_none(
            "eta",
            SpectrumType::Unbounded,
            cached_spectra,
            textures,
        );
        let mut k = parameters.get_spectrum_texture(
            "k",
            None,
            SpectrumType::Unbounded,
            cached_spectra,
            textures,
        );
        let reflectance = parameters.get_spectrum_texture(
            "reflectance",
            None,
            SpectrumType::Albedo,
            cached_spectra,
            textures,
        );

        if reflectance.is_some() && (eta.is_some() || k.is_some()) {
            panic!("Cannot specify both reflectance and eta/k for conductor material.");
        }

        if reflectance.is_none() {
            if eta.is_none() {
                eta = Some(Arc::new(SpectrumTexture::Constant(
                    SpectrumConstantTexture::new(Spectrum::get_named_spectrum(
                        NamedSpectrum::CuEta,
                    )),
                )));
            }
            if k.is_none() {
                k = Some(Arc::new(SpectrumTexture::Constant(
                    SpectrumConstantTexture::new(Spectrum::get_named_spectrum(NamedSpectrum::CuK)),
                )));
            }
        }

        let u_roughness =
            if let Some(roughness) = parameters.get_float_texture_or_none("uroughness", textures) {
                roughness
            } else {
                parameters.get_float_texture("roughness", 0.0, textures)
            };
        let v_roughness =
            if let Some(roughness) = parameters.get_float_texture_or_none("vroughness", textures) {
                roughness
            } else {
                parameters.get_float_texture("roughness", 0.0, textures)
            };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        let remap_roughness = parameters.get_one_bool("remaproughness", true);

        ConductorMaterial::new(
            displacement,
            normal_map,
            eta,
            k,
            reflectance,
            u_roughness,
            v_roughness,
            remap_roughness,
        )
    }

    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        // Either eta and k are supplied, or reflectance
        // TODO Enum?
        eta: Option<Arc<SpectrumTexture>>,
        k: Option<Arc<SpectrumTexture>>,
        reflectance: Option<Arc<SpectrumTexture>>,
        u_roughness: Arc<FloatTexture>,
        v_roughness: Arc<FloatTexture>,
        remap_roughness: bool,
    ) -> Self {
        Self {
            displacement,
            normal_map,
            eta,
            k,
            reflectance,
            u_roughness,
            v_roughness,
            remap_roughness,
        }
    }
}

impl MaterialI for ConductorMaterial {
    type ConcreteBxDF = ConductorBxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut u_rough = tex_eval.evaluate_float(&self.u_roughness, &ctx.tex_ctx);
        let mut v_rough = tex_eval.evaluate_float(&self.v_roughness, &ctx.tex_ctx);

        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }

        let (etas, ks) = if let Some(eta) = &self.eta {
            let k = self
                .k
                .as_ref()
                .expect("eta and k should be provided together");
            let etas = tex_eval.evaluate_spectrum(eta, &ctx.tex_ctx, lambda);
            let ks = tex_eval.evaluate_spectrum(&k, &ctx.tex_ctx, lambda);
            (etas, ks)
        } else {
            let r = SampledSpectrum::clamp(
                &tex_eval.evaluate_spectrum(
                    self.reflectance
                        .as_ref()
                        .expect("If eta/k is not present, reflectance should be"),
                    &ctx.tex_ctx,
                    lambda,
                ),
                0.0,
                0.0000,
            );
            let etas = SampledSpectrum::from_const(1.0);
            let ks = 2.0 * SampledSpectrum::sqrt(r)
                / SampledSpectrum::sqrt(SampledSpectrum::clamp_zero(
                    &(SampledSpectrum::from_const(1.0) - r),
                ));
            (etas, ks)
        };
        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough);
        ConductorBxDF::new(distrib, etas, ks)
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, crate::bxdf::BxDF::Conductor(bxdf))
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        let s_tex = if let Some(eta) = &self.eta {
            assert!(self.k.is_some());
            vec![
                self.k
                    .as_ref()
                    .expect("If eta is provided, k should be as well"),
                eta,
            ]
        } else {
            vec![self
                .reflectance
                .as_ref()
                .expect("Expected reflectance without eta/k")]
        };
        tex_eval.can_evaluate(&[&self.u_roughness, &self.v_roughness], &s_tex)
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        self.normal_map.clone()
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        self.displacement.clone()
    }

    fn has_subsurface_scattering(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct DielectricMaterial {
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    remap_roughness: bool,
    eta: Arc<Spectrum>,
}

impl DielectricMaterial {
    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        _loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> DielectricMaterial {
        let eta = if !parameters.get_float_array("eta").is_empty() {
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(
                parameters.get_float_array("eta")[0],
            ))))
        } else {
            parameters.get_one_spectrum("eta", None, SpectrumType::Unbounded, cached_spectra)
        };
        let eta = eta.unwrap_or(Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.5))));

        let u_roughness =
            if let Some(roughness) = parameters.get_float_texture_or_none("uroughness", textures) {
                roughness
            } else {
                parameters.get_float_texture("roughness", 0.0, textures)
            };
        let v_roughness =
            if let Some(roughness) = parameters.get_float_texture_or_none("vroughness", textures) {
                roughness
            } else {
                parameters.get_float_texture("roughness", 0.0, textures)
            };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        let remap_roughness = parameters.get_one_bool("remaproughness", true);

        DielectricMaterial::new(
            displacement,
            normal_map,
            u_roughness,
            v_roughness,
            remap_roughness,
            eta,
        )
    }

    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        u_roughness: Arc<FloatTexture>,
        v_roughness: Arc<FloatTexture>,
        remap_roughness: bool,
        eta: Arc<Spectrum>,
    ) -> Self {
        Self {
            displacement,
            normal_map,
            u_roughness,
            v_roughness,
            remap_roughness,
            eta,
        }
    }
}

impl MaterialI for DielectricMaterial {
    type ConcreteBxDF = DielectricBxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut sampled_eta = self.eta.get(lambda[0]);
        // If the IOR is the same for all wavelengths, then all wavelengths will follow the
        // same path. If the IOR varies, then they will go in different directions - this is dispersion.
        // So, if we do not have a constant IOR, terminate secondary wavelengths in lambda.
        // In the aggregate over many sampled rays, we will be able to see the effects of dispersion.
        // This is a nice feature of spectral rendering that is more difficult to achieve than
        // through RGB rendering. PAPERDOC.
        let is_eta_constant = match self.eta.as_ref() {
            Spectrum::Constant(_) => true,
            _ => false,
        };
        if !is_eta_constant {
            lambda.terminate_secondary();
        }
        // Handle edge case where lambda[0] is beyond the wavelengths stored by the spectrum.
        if sampled_eta == 0.0 {
            sampled_eta = 1.0;
        }

        // Create microfacet distribution for dielectric material
        let mut u_rough = tex_eval.evaluate_float(&self.u_roughness, &ctx.tex_ctx);
        let mut v_rough = tex_eval.evaluate_float(&self.v_roughness, &ctx.tex_ctx);
        if self.remap_roughness {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }
        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough);
        DielectricBxDF::new(sampled_eta, distrib)
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, crate::bxdf::BxDF::Dielectric(bxdf))
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        tex_eval.can_evaluate(&[&self.u_roughness, &self.v_roughness], &[])
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        self.normal_map.clone()
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        self.displacement.clone()
    }

    fn has_subsurface_scattering(&self) -> bool {
        false
    }
}

/// Material for thin dielectric materials such as panes of glass,
/// which requires special handling due to the proximity of the interfaces.
#[derive(Debug)]
pub struct ThinDielectricMaterial
{
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    eta: Arc<Spectrum>,
}

impl ThinDielectricMaterial
{
    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        eta: Arc<Spectrum>,
    ) -> Self {
        Self {
            displacement,
            normal_map,
            eta,
        }
    }

    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        _loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> ThinDielectricMaterial
    {
        let eta = if !parameters.get_float_array("eta").is_empty()
        {
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(parameters.get_float_array("eta")[0]))))
        }
        else
        {
            parameters.get_one_spectrum("eta", None, SpectrumType::Unbounded, cached_spectra)
        };
        let eta = if eta.is_none()
        {
            Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.5)))
        }
        else
        {
            eta.unwrap()
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);

        ThinDielectricMaterial::new(displacement, normal_map, eta)
    }
}

impl MaterialI for ThinDielectricMaterial
{
    type ConcreteBxDF = ThinDielectricBxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        let mut sampled_eta = self.eta.get(lambda[0]);
        match self.eta.as_ref() {
            Spectrum::Constant(_) => {},
            _ => lambda.terminate_secondary(),
        };

        // Handle edge case where lambda[0] is beyond the wavelengths stored by the spectrum.
        if sampled_eta == 0.0
        {
            sampled_eta = 1.0;
        }

        ThinDielectricBxDF::new(sampled_eta)
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, crate::bxdf::BxDF::ThinDielectric(bxdf))
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        true
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        self.normal_map.clone()
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        self.displacement.clone()
    }

    fn has_subsurface_scattering(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct CoatedDiffuseMaterial
{
    displacement: Option<Arc<FloatTexture>>,
    normal_map: Option<Arc<Image>>,
    reflectance: Arc<SpectrumTexture>,
    albedo: Arc<SpectrumTexture>,
    u_roughness: Arc<FloatTexture>,
    v_roughness: Arc<FloatTexture>,
    thickness: Arc<FloatTexture>,
    g: Arc<FloatTexture>,
    eta: Arc<Spectrum>,
    remap_roughness: bool,
    max_depth: i32,
    n_samples: i32,
}

impl CoatedDiffuseMaterial
{
    pub fn new(
        displacement: Option<Arc<FloatTexture>>,
        normal_map: Option<Arc<Image>>,
        reflectance: Arc<SpectrumTexture>,
        albedo: Arc<SpectrumTexture>,
        u_roughness: Arc<FloatTexture>,
        v_roughness: Arc<FloatTexture>,
        thickness: Arc<FloatTexture>,
        g: Arc<FloatTexture>,
        eta: Arc<Spectrum>,
        remap_roughness: bool,
        max_depth: i32,
        n_samples: i32,
    ) -> Self {
        Self {
            displacement,
            normal_map,
            reflectance,
            albedo,
            u_roughness,
            v_roughness,
            thickness,
            g,
            eta,
            remap_roughness,
            max_depth,
            n_samples,
        }
    }

    pub fn create(
        parameters: &mut TextureParameterDictionary,
        normal_map: Option<Arc<Image>>,
        _loc: &FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
    ) -> CoatedDiffuseMaterial
    {
        let reflectance = parameters.get_spectrum_texture("reflectance", None, SpectrumType::Albedo, cached_spectra, textures);
        let reflectance = if reflectance.is_none()
        {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.5))))))
        }
        else
        {
            reflectance.unwrap()
        };

        let u_roughness = parameters.get_float_texture_or_none("uroughness", textures);
        let v_roughness = parameters.get_float_texture_or_none("vroughness", textures);

        let u_roughness = if u_roughness.is_none()
        {
            parameters.get_float_texture("roughness", 0.0, textures)
        }
        else
        {
            u_roughness.unwrap()
        };

        let v_roughness = if v_roughness.is_none()
        {
            parameters.get_float_texture("roughness", 0.0, textures)
        }
        else
        {
            v_roughness.unwrap()
        };

        let thickness = parameters.get_float_texture("thickness", 0.01, textures);

        let eta = if !parameters.get_float_array("eta").is_empty()
        {
            Some(Arc::new(Spectrum::Constant(ConstantSpectrum::new(parameters.get_float_array("eta")[0]))))
        }
        else
        {
            parameters.get_one_spectrum("eta", None, SpectrumType::Unbounded, cached_spectra)
        };
        let eta = if eta.is_none()
        {
            Arc::new(Spectrum::Constant(ConstantSpectrum::new(1.5)))
        }
        else
        {
            eta.unwrap()
        };

        let max_depth = parameters.get_one_int("maxdepth", 10);
        let n_samples = parameters.get_one_int("nsamples", 1);

        let g = parameters.get_float_texture("g", 0.0, textures);
        let albedo = parameters.get_spectrum_texture("albedo", None, SpectrumType::Albedo, cached_spectra, textures);

        let albedo = if albedo.is_none()
        {
            Arc::new(SpectrumTexture::Constant(SpectrumConstantTexture::new(Arc::new(Spectrum::Constant(ConstantSpectrum::new(0.0))))))
        }
        else
        {
            albedo.unwrap()
        };

        let displacement = parameters.get_float_texture_or_none("displacement", textures);
        let remap_roughness = parameters.get_one_bool("remaproughness", true);

        CoatedDiffuseMaterial::new(
            displacement,
            normal_map,
            reflectance,
            albedo,
            u_roughness,
            v_roughness,
            thickness,
            g,
            eta,
            remap_roughness,
            max_depth,
            n_samples
        )
    }
}

impl MaterialI for CoatedDiffuseMaterial
{
    type ConcreteBxDF = CoatedDiffuseBxDF;

    fn get_bxdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> Self::ConcreteBxDF {
        // Initialize diffuse component of plastic material
        let r = SampledSpectrum::clamp(&tex_eval.evaluate_spectrum(&self.reflectance, &ctx.tex_ctx, lambda), 0.0, 1.0);

        // Create microfacet distribution distrib for coated diffuse material
        let mut u_rough = tex_eval.evaluate_float(&self.u_roughness, &ctx.tex_ctx);
        let mut v_rough = tex_eval.evaluate_float(&self.v_roughness, &ctx.tex_ctx);
        if self.remap_roughness
        {
            u_rough = TrowbridgeReitzDistribution::roughness_to_alpha(u_rough);
            v_rough = TrowbridgeReitzDistribution::roughness_to_alpha(v_rough);
        }
        let distrib = TrowbridgeReitzDistribution::new(u_rough, v_rough);

        let thick = tex_eval.evaluate_float(&self.thickness, &ctx.tex_ctx);

        let mut sampled_eta = self.eta.get(lambda[0]);
        match self.eta.as_ref() {
            Spectrum::Constant(_) => {},
            _ => lambda.terminate_secondary()
        }
        if sampled_eta == 0.0
        {
            sampled_eta = 1.0;
        }

        let a = SampledSpectrum::clamp(&tex_eval.evaluate_spectrum(&self.albedo, &ctx.tex_ctx, lambda), 0.0, 1.0);
        let gg = Float::clamp(tex_eval.evaluate_float(&self.g, &ctx.tex_ctx), -1.0, 1.0);

        CoatedDiffuseBxDF::new(
            DielectricBxDF::new(
                sampled_eta,
                distrib,
            ),
            DiffuseBxDF::new(r),
            thick,
            &a,
            gg,
            self.max_depth,
            self.n_samples
        )
    }

    fn get_bsdf<T: TextureEvaluatorI>(
        &self,
        tex_eval: &T,
        ctx: &MaterialEvalContext,
        lambda: &mut SampledWavelengths,
    ) -> BSDF {
        let bxdf = self.get_bxdf(tex_eval, ctx, lambda);
        BSDF::new(ctx.ns, ctx.dpdus, crate::bxdf::BxDF::CoatedDiffuse(bxdf))
    }

    fn can_evaluate_textures<T: TextureEvaluatorI>(&self, tex_eval: &T) -> bool {
        // TODO Here and in other fns, we might make some of these maps optional, in which case, transform.
        // We do something like that for other materials
        tex_eval.can_evaluate(
            &[&self.u_roughness, &self.v_roughness, &self.thickness, &self.g],
            &[&self.reflectance, &self.albedo])
    }

    fn get_normal_map(&self) -> Option<Arc<Image>> {
        self.normal_map.clone()
    }

    fn get_displacement(&self) -> Option<Arc<FloatTexture>> {
        self.displacement.clone()
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

impl From<&SurfaceInteraction> for MaterialEvalContext {
    fn from(si: &SurfaceInteraction) -> Self {
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
    fn can_evaluate(&self, f_tex: &[&Arc<FloatTexture>], s_tex: &[&Arc<SpectrumTexture>]) -> bool;

    fn evaluate_float(&self, tex: &Arc<FloatTexture>, ctx: &TextureEvalContext) -> Float;

    fn evaluate_spectrum(
        &self,
        tex: &Arc<SpectrumTexture>,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;
}

/// A TextureEvaluator which can evaluate all textures; the default in most scenarios.
pub struct UniversalTextureEvaluator {}

impl TextureEvaluatorI for UniversalTextureEvaluator {
    fn can_evaluate(
        &self,
        _f_tex: &[&Arc<FloatTexture>],
        _s_tex: &[&Arc<SpectrumTexture>],
    ) -> bool {
        true
    }

    fn evaluate_float(&self, tex: &Arc<FloatTexture>, ctx: &TextureEvalContext) -> Float {
        tex.evaluate(ctx)
    }

    fn evaluate_spectrum(
        &self,
        tex: &Arc<SpectrumTexture>,
        ctx: &TextureEvalContext,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        tex.evaluate(ctx, lambda)
    }
}
