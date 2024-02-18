use std::{collections::HashMap, sync::Arc};

use spectrum::ConstantSpectrum;

use crate::{
    float::PI_F, interaction::{Interaction, SurfaceInteraction}, loading::{paramdict::{SpectrumType, TextureParameterDictionary}, parser_target::FileLoc}, math::{sqr, INV_2PI, INV_PI}, spectra::{
        sampled_spectrum::SampledSpectrum,
        sampled_wavelengths::SampledWavelengths,
        spectrum::{self, SpectrumI},
        Spectrum,
    }, transform::Transform, vecmath::{spherical::spherical_theta, vector::Vector3, Normal3f, Normalize, Point2f, Point3f, Tuple2, Tuple3, Vector3f}, Float
};

pub trait FloatTextureI {
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float;
}

#[derive(Debug)]
pub enum FloatTexture {
    Constant(FloatConstantTexture),
}

impl FloatTexture {
    pub fn create(
        name: &str,
        render_from_texture: Transform,
        parameters: TextureParameterDictionary,
        loc: &FileLoc,
    ) -> FloatTexture {
        let tex = match name {
            "constant" => {
                let t = FloatConstantTexture::create(render_from_texture, parameters, loc);
                FloatTexture::Constant(t)
            }
            _ => {
                panic!("Texture {} unknown", name);
            }
        };

        // TODO Track number of textures created for stats.
        // TODO Report unused paramters.

        tex
    }
}

impl FloatTextureI for FloatTexture {
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        match self {
            FloatTexture::Constant(t) => t.evaluate(ctx),
        }
    }
}

#[derive(Debug)]
pub struct FloatConstantTexture {
    value: Float,
}

impl FloatConstantTexture {
    pub fn new(value: Float) -> Self {
        Self { value }
    }

    pub fn create(
        _render_from_texture: Transform,
        mut parameters: TextureParameterDictionary,
        _loc: &FileLoc,
    ) -> FloatConstantTexture {
        let v = parameters.get_one_float("value", 1.0);
        FloatConstantTexture::new(v)
    }
}

impl FloatTextureI for FloatConstantTexture {
    fn evaluate(&self, _ctx: &TextureEvalContext) -> Float {
        self.value
    }
}

pub trait SpectrumTextureI {
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum;
}

#[derive(Debug)]
pub enum SpectrumTexture {
    Constant(SpectrumConstantTexture),
}

impl SpectrumTexture {
    pub fn create(
        name: &str,
        render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        loc: &FileLoc,
    ) -> SpectrumTexture {
        let tex = match name {
            "constant" => {
                let t = SpectrumConstantTexture::create(
                    render_from_texture,
                    parameters,
                    spectrum_type,
                    cached_spectra,
                    loc,
                );
                SpectrumTexture::Constant(t)
            }
            _ => {
                panic!("Texture {} unknown", name);
            }
        };

        // TODO Track number of textures created for stats.
        // TODO Report unused paramters.

        tex
    }
}

impl SpectrumTextureI for SpectrumTexture {
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            SpectrumTexture::Constant(t) => t.evaluate(ctx, lambda),
        }
    }
}

#[derive(Debug)]
pub struct SpectrumConstantTexture {
    pub value: Arc<Spectrum>,
}

impl SpectrumTextureI for SpectrumConstantTexture {
    fn evaluate(&self, _ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.value.sample(lambda)
    }
}

impl SpectrumConstantTexture {
    pub fn new(value: Arc<Spectrum>) -> Self {
        Self { value }
    }

    pub fn create(
        render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        _loc: &FileLoc,
    ) -> SpectrumConstantTexture {
        let one = Spectrum::Constant(ConstantSpectrum::new(1.0));
        let c = parameters
            .get_one_spectrum("value", Some(Arc::new(one)), spectrum_type, cached_spectra)
            .unwrap();
        SpectrumConstantTexture::new(c)
    }
}

/// Provides an interface for 2D texture coordinate generation.
pub trait TextureMapping2DI {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D;
}

// TODO let's also provide a 3D mapping equivalent. pg 654.

/// Provides 2D texture coordinate generation.
pub enum TextureMapping2D {
    UV(UVMapping),
    Spherical(SphericalMapping),
    // TODO spherical, cylindrical, and Planar mapping.
}

impl TextureMapping2DI for TextureMapping2D {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        match self {
            TextureMapping2D::UV(m) => m.map(ctx),
            TextureMapping2D::Spherical(m) => m.map(ctx),
            
        }
    }
}

/// Uses the (u, v) coordinates in the TextureEvalContext to compute the texture
/// coordinates, optionally scaling and offsetting their values in each dimension.
pub struct UVMapping {
    /// Scale s from u
    su: Float,
    /// Scale s from v
    sv: Float,
    /// Offset u
    du: Float,
    /// Offset v
    dv: Float,
}

impl Default for UVMapping {
    fn default() -> Self {
        Self {
            su: 1.0,
            sv: 1.0,
            du: 0.0,
            dv: 0.0,
        }
    }
}

impl TextureMapping2DI for UVMapping {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        // Compute texture differentials for 2D (u, v) mapping
        let dsdx = self.su * ctx.dudx;
        let dsdy = self.su * ctx.dudy;
        let dtdx = self.sv * ctx.dvdx;
        let dtdy = self.sv * ctx.dvdy;

        let st = Point2f::new(self.su * ctx.uv[0] + self.du, self.sv * ctx.uv[1] + self.dv);
        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

pub struct SphericalMapping
{
    texture_from_render: Transform,
}

impl TextureMapping2DI for SphericalMapping {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        let pt = self.texture_from_render.apply(&ctx.p);
        let x2y2 = sqr(pt.x) + sqr(pt.y);
        let sqrtx2y2 = x2y2.sqrt();
        let dsdp = Vector3f::new(-pt.y, pt.x, 0.0) / (2.0 * PI_F * x2y2);
        let dtdp = 
            1.0 / (PI_F * (x2y2 + sqr(pt.z))) *
            Vector3f::new(pt.x * pt.z / sqrtx2y2, pt.y * pt.z / sqrtx2y2, -sqrtx2y2);
        
        let dpdx = self.texture_from_render.apply(&ctx.dpdx);
        let dpdy = self.texture_from_render.apply(&ctx.dpdy);

        let dsdx = dsdp.dot(dpdx);
        let dsdy = dsdp.dot(dpdy);
        let dtdx = dtdp.dot(dpdx);
        let dtdy = dtdp.dot(dpdy);

        let vec = (pt - Point3f::ZERO).normalize();
        let st = Point2f::new(
            spherical_theta(vec) * INV_PI,
            spherical_theta(vec) * INV_2PI,
        );
        
        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

/// Stores the (s, t) texture cordinates and estimates for the change in (s, t) w.r.t. pixel
/// x and y coordinates so that textures that using the mapping can determine the (s, t) sampling rate
/// and filter accordingly.
pub struct TexCoord2D {
    st: Point2f,
    dsdx: Float,
    dsdy: Float,
    dtdx: Float,
    dtdy: Float,
}

/// Stores relevant geometric information at the shading point for texture evaluation
pub struct TextureEvalContext {
    p: Point3f,
    dpdx: Vector3f,
    dpdy: Vector3f,
    n: Normal3f,
    uv: Point2f,
    dudx: Float,
    dudy: Float,
    dvdx: Float,
    dvdy: Float,
}

impl TextureEvalContext {
    pub fn new(
        p: Point3f,
        dpdx: Vector3f,
        dpdy: Vector3f,
        n: Normal3f,
        uv: Point2f,
        dudx: Float,
        dudy: Float,
        dvdx: Float,
        dvdy: Float,
    ) -> TextureEvalContext {
        TextureEvalContext {
            p,
            dpdx,
            dpdy,
            n,
            uv,
            dudx,
            dudy,
            dvdx,
            dvdy,
        }
    }
}

impl From<SurfaceInteraction> for TextureEvalContext {
    fn from(value: SurfaceInteraction) -> Self {
        Self {
            p: value.p(),
            dpdx: value.dpdx,
            dpdy: value.dpdy,
            n: value.interaction.n,
            uv: value.interaction.uv,
            dudx: value.dudx,
            dudy: value.dudy,
            dvdx: value.dvdx,
            dvdy: value.dvdy,
        }
    }
}

impl From<&SurfaceInteraction> for TextureEvalContext {
    fn from(value: &SurfaceInteraction) -> Self {
        Self {
            p: value.p(),
            dpdx: value.dpdx,
            dpdy: value.dpdy,
            n: value.interaction.n,
            uv: value.interaction.uv,
            dudx: value.dudx,
            dudy: value.dudy,
            dvdx: value.dvdx,
            dvdy: value.dvdy,
        }
    }
}

impl From<Interaction> for TextureEvalContext {
    fn from(value: Interaction) -> Self {
        Self {
            p: value.p(),
            dpdx: Default::default(),
            dpdy: Default::default(),
            n: Default::default(),
            uv: value.uv,
            dudx: Default::default(),
            dudy: Default::default(),
            dvdx: Default::default(),
            dvdy: Default::default(),
        }
    }
}
