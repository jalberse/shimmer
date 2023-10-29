// TODO we'll want to implement the Light interface, but also a LightSampler.
// We can start with just the PointLight and add others later.

use crate::{
    bounding_box::Bounds3f,
    interaction::{Interaction, SurfaceInteraction},
    ray::Ray,
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    vecmath::{point::Point3fi, Normal3f, Point2f, Point3f, Vector3f},
    Float,
};

pub trait LightI {
    /// Returns the total emitted power _phi_
    fn phi(&self, lambda: SampledWavelengths) -> SampledSpectrum;

    /// Soemtimes, knowing the type of light is necessary for efficiency and
    /// correctness. This tells us which type of light this is (NOT which
    /// class implementing LightI this is, but among the LightType categories).
    fn light_type(&self) -> LightType;

    /// Sampling routine that chooses from among only those directions where the
    /// light is potentially visible.
    /// ctx provides information about a reference point in the scene.
    /// u is a uniform sample
    /// lambda are the wavelengths to sample at.
    /// allow_incomplete_pdf allows the sampling routine to skip generating samples
    /// for directions where the light's contribution is small.
    /// Returns a LighLiSample that encapsulates incident radiance, information
    /// about where the light is being emitted from, and the value of the PDF for
    /// the sampled point.
    fn sample_li(
        &self,
        ctx: LightSampleContext,
        u: Point2f,
        lambda: SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample>;

    /// Returns the value of the PDF for sampling given direction wi from the point in ctx.
    /// Assumes that a ray from ctx in the direction wi has already been found to intersect
    /// the light source. PDF is measured w.r.t. solid angle; 0 if the light is described
    /// by a Dirac delta function.
    fn pdf_li(
        &self,
        ctx: LightSampleContext,
        lambda: SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Float;

    /// For area light sources, finds the radiance that is emitted back along the ray.
    /// Takes information about the intersection point and the outgoing direction.
    /// Should only be called for lights with associated geometry.
    /// TODO - Should this be moved to another AreaLightI trait to enforce that it should
    /// not be called for non-area-lights? Same for Le().
    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum;

    /// Only for lights with LightType::Infinite; enables them to contribute radiance to rays
    /// that do not hit geometry in the scene.
    /// TODO can we enforce that it's only for infinite via a different trait?
    fn le(ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum;

    /// Invoked prior to rendering, informs the lights of the scene bounds, which
    /// may not be available whiel constructing the lights.
    fn preprocess(&mut self, scene_bounds: &Bounds3f);

    // TODO bounds() method

    // TODO sample_le() and pdf_le() which are crucial for bidirectional light transport;
    // those are explained in the online version (source code available).
    // We can skip for now, because I don't think we'll be implementing bidirectional soon.
}

/// Provides context for sampling a light via LightI::sample_li().
pub struct LightSampleContext {
    /// A point in the scene
    pub pi: Point3fi,
    /// Surface normal; zero if not on a surface
    pub n: Normal3f,
    /// Shading normal; zero if not on a surface
    pub ns: Normal3f,
}

impl LightSampleContext {
    pub fn new(pi: Point3fi, n: Normal3f, ns: Normal3f) -> LightSampleContext {
        LightSampleContext { pi, n, ns }
    }

    /// Provides the point without intervals.
    pub fn p(&self) -> Point3f {
        Point3f::from(self.pi)
    }
}

impl From<&SurfaceInteraction> for LightSampleContext {
    fn from(value: &SurfaceInteraction) -> Self {
        LightSampleContext {
            pi: value.interaction.pi,
            n: value.interaction.n,
            ns: value.shading.n,
        }
    }
}

impl From<&Interaction> for LightSampleContext {
    fn from(value: &Interaction) -> Self {
        LightSampleContext {
            pi: value.pi,
            n: Normal3f::ZERO,
            ns: Normal3f::ZERO,
        }
    }
}

impl Default for LightSampleContext {
    fn default() -> Self {
        Self {
            pi: Default::default(),
            n: Default::default(),
            ns: Default::default(),
        }
    }
}

pub struct LightLiSample {
    /// The amount of radiance leaving the light toward the receiving point;
    /// it does not include the effect of extinction due to participating media
    /// or occlusion.
    pub l: SampledSpectrum,
    /// The direction along which the light arrives at tthe point specified in the
    /// LightSampleContext.
    pub wi: Vector3f,
    /// The PDF value for the light sample measured w.r.t. solid angle at the receiving point.
    pub pdf: Float,
    /// The point from which light is being emitted
    pub p_light: Interaction,
}

impl LightLiSample {
    pub fn new(
        l: SampledSpectrum,
        wi: Vector3f,
        pdf: Float,
        p_light: Interaction,
    ) -> LightLiSample {
        LightLiSample {
            l,
            wi,
            pdf,
            p_light,
        }
    }
}

impl Default for LightLiSample {
    fn default() -> Self {
        Self {
            l: Default::default(),
            wi: Default::default(),
            pdf: Default::default(),
            p_light: Default::default(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LightType {
    /// Lights that emit solely from a single point in space.
    /// "Delta" refers to the fact that it can be represented by a
    /// Dirac delta distribution
    DeltaPosition,
    /// Lights that emit radiance along a single direction.
    /// "Delta" refers to the fact that it can be represented by a
    /// Dirac delta distribution
    DeltaDirection,
    /// Lights that emit radiance from the surface of a geometric shape.
    Area,
    /// Lights "at infinity" that do not have geometry associated with them but
    /// provide radiance to rays that escape the scene.
    Infinite,
}

impl LightType {
    /// True is the light is defined using a Dirac delta distribution.
    pub fn is_delta(&self) -> bool {
        *self == Self::DeltaDirection || *self == Self::DeltaPosition
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Light {
    // TODO implement. Also, this will likely be an enum.
}
