use crate::{
    interaction::{Interaction, SurfaceInteraction},
    spectra::{
        sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths,
        spectrum::SpectrumI, Spectrum,
    },
    vecmath::{Normal3f, Point2f, Point3f, Tuple2, Vector3f},
    Float,
};

pub trait FloatTextureI {
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float;
}

#[derive(Debug)]
pub enum FloatTexture {
    Constant(FloatConstantTexture),
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

impl SpectrumTextureI for SpectrumTexture {
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            SpectrumTexture::Constant(t) => t.evaluate(ctx, lambda),
        }
    }
}

#[derive(Debug)]
pub struct SpectrumConstantTexture {
    value: Spectrum,
}

impl SpectrumTextureI for SpectrumConstantTexture {
    fn evaluate(&self, _ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.value.sample(lambda)
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
    // TODO spherical, cylindrical, and Planar mapping.
}

impl TextureMapping2DI for TextureMapping2D {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        match self {
            TextureMapping2D::UV(m) => m.map(ctx),
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
