// TODO Implement LightSampler. Then we can move on from Lights for now,
// and implement them later as needed - but just a PointLight should be sufficient for now
// to render early test scenes.

use std::sync::Arc;

use log::warn;

use crate::{
    bounding_box::Bounds3f,
    float::PI_F,
    interaction::{Interaction, SurfaceInteraction},
    ray::Ray,
    shape::{Shape, ShapeI, ShapeSampleContext},
    spectra::{
        sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths,
        spectrum::SpectrumI, DenselySampledSpectrum, Spectrum,
    },
    transform::Transform,
    vecmath::{
        normal::Normal3,
        point::{Point3, Point3fi},
        Length, Normal3f, Normalize, Point2f, Point3f, Vector3f,
    },
    Float,
};

pub trait LightI {
    /// Returns the total emitted power _phi_
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum;

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
    fn pdf_li(&self, ctx: LightSampleContext, wi: Vector3f, allow_incomplete_pdf: bool) -> Float;

    /// For area light sources, finds the radiance that is emitted back along the ray.
    /// Takes information about the intersection point and the outgoing direction.
    /// Should only be called for lights with associated geometry.
    /// TODO - Should this be moved to another AreaLightI trait to enforce that it should
    /// not be called for non-area-lights? Similar for Le() (but that would be in another trait I think).
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
    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum;

    /// Invoked prior to rendering, informs the lights of the scene bounds, which
    /// may not be available whiel constructing the lights.
    fn preprocess(&mut self, scene_bounds: &Bounds3f);

    // TODO bounds() method

    // TODO sample_le() and pdf_le() which are crucial for bidirectional light transport;
    // those are explained in the online version (source code available).
    // We can skip for now, because I don't think we'll be implementing bidirectional soon.
}

#[derive(Debug, Clone)]
pub enum Light {
    Point(PointLight),
    DiffuseAreaLight(DiffuseAreaLight),
}

impl LightI for Light {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.phi(lambda),
            Light::DiffuseAreaLight(l) => l.phi(lambda),
        }
    }

    fn light_type(&self) -> LightType {
        match self {
            Light::Point(l) => l.light_type(),
            Light::DiffuseAreaLight(l) => l.light_type(),
        }
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        u: Point2f,
        lambda: SampledWavelengths,
        allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        match self {
            Light::Point(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
            Light::DiffuseAreaLight(l) => l.sample_li(ctx, u, lambda, allow_incomplete_pdf),
        }
    }

    fn pdf_li(&self, ctx: LightSampleContext, wi: Vector3f, allow_incomplete_pdf: bool) -> Float {
        match self {
            Light::Point(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
            Light::DiffuseAreaLight(l) => l.pdf_li(ctx, wi, allow_incomplete_pdf),
        }
    }

    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.l(p, n, uv, w, lambda),
            Light::DiffuseAreaLight(l) => l.l(p, n, uv, w, lambda),
        }
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            Light::Point(l) => l.le(ray, lambda),
            Light::DiffuseAreaLight(l) => l.le(ray, lambda),
        }
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        match self {
            Light::Point(l) => l.preprocess(scene_bounds),
            Light::DiffuseAreaLight(l) => l.preprocess(scene_bounds),
        }
    }
}

/// Specific types of lights (e.g. point lights and spotlights) can *have* a LightBase,
/// which provides shared functionality.
#[derive(Debug, Copy, Clone)]
pub struct LightBase {
    light_type: LightType,
    /// Defines the light's coordinate system w.r.t. render space.
    render_from_light: Transform,
    // TODO MediumInterface, when we implement Medium.
}

impl LightBase {
    pub fn light_type(&self) -> LightType {
        self.light_type
    }

    /// Default implementation for LightI::l() so that lights which are not area lights
    /// don't need to implement their own version
    pub fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    /// Defualt implementation for LightI::le() so that lights which are not infinite
    /// don't need to implement their own version.
    pub fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        SampledSpectrum::from_const(0.0)
    }

    // TODO We should implement a caching system for the DenselySampleSpectrum (see pg 745 12.1)
    // When we do, LightBase should also include a LookupSpectrum() function to get a
    // cached DenselySample spectrum.
    // We could also then make Lights impl Copy, and take them out of Rc in e.g. SurfaceInteraction.
}

/// Isotropic point light source that emites the same amount of light in all directions.
#[derive(Debug, Clone)]
pub struct PointLight {
    base: LightBase,
    i: Arc<DenselySampledSpectrum>,
    scale: Float,
}

impl PointLight {
    pub fn new(render_from_light: Transform, i: Spectrum, scale: Float) -> PointLight {
        let base = LightBase {
            light_type: LightType::DeltaPosition,
            render_from_light,
        };
        PointLight {
            base,
            i: Arc::new(DenselySampledSpectrum::new(&i)),
            scale,
        }
    }
}

impl LightI for PointLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        4.0 * PI_F * self.scale * self.i.sample(&lambda)
    }

    fn light_type(&self) -> LightType {
        self.base.light_type()
    }

    // It's technically incorrect to use radiance to describe light arriving at a point
    // from a point light, but the correctness doesn't suffer and this lets us keep one
    // entry point for the light interface.
    fn sample_li(
        &self,
        ctx: LightSampleContext,
        _u: Point2f,
        lambda: SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let p = self.base.render_from_light.apply(&Point3f::ZERO);
        let wi = (p - ctx.p()).normalize();
        let li = self.scale * self.i.sample(&lambda) / p.distance_squared(&ctx.p());
        // TODO I do think this is correct compared to PBRT,
        // but I don't love just leaving many fields default().
        // Can we represent this better? Option?
        let interaction = Interaction {
            pi: p.into(),
            time: Default::default(),
            wo: Default::default(),
            n: Default::default(),
            uv: Default::default(),
        };
        Some(LightLiSample::new(li, wi, 1.0, interaction))
    }

    fn pdf_li(
        &self,
        _ctx: LightSampleContext,
        _wi: Vector3f,
        _allow_incomplete_pdf: bool,
    ) -> Float {
        // Due to delta distribution; won't randonly select an infinitesimal light source.
        0.0
    }

    fn l(
        &self,
        p: Point3f,
        n: Normal3f,
        uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        self.base.l(p, n, uv, w, lambda)
    }

    fn le(&self, ray: &Ray, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.base.le(ray, lambda)
    }

    fn preprocess(&mut self, _scene_bounds: &Bounds3f) {
        // Nothing to do!
    }
}

/// The DiffuseAreaLight descripts an area light where emission at each point on the surface
/// has a uniform directional distribution
#[derive(Debug, Clone)]
pub struct DiffuseAreaLight {
    base: LightBase,
    shape: Shape,
    // TODO alpha: FloatTexture,
    area: Float,
    two_sided: bool,
    l_emit: Arc<DenselySampledSpectrum>,
    scale: Float,
    // TODO image: Image,
    // TODO image color space
}

impl DiffuseAreaLight {
    pub fn new(
        render_from_light: Transform,
        le: Arc<Spectrum>,
        scale: Float,
        shape: Shape,
        two_sided: bool,
    ) -> DiffuseAreaLight {
        let area = shape.area();
        let base = LightBase {
            light_type: LightType::Area,
            render_from_light,
        };
        // TODO Here's an area where we should use lookup_spectrum() instead as we should cache similar
        // densely sampled spectra.
        DiffuseAreaLight {
            base,
            shape,
            area,
            two_sided,
            l_emit: Arc::new(DenselySampledSpectrum::new(le.as_ref())),
            scale,
        }
    }
}

impl LightI for DiffuseAreaLight {
    fn phi(&self, lambda: &SampledWavelengths) -> SampledSpectrum {
        // TODO account for image here
        let l = self.l_emit.sample(lambda) * self.scale;
        let double = if self.two_sided { 2.0 } else { 1.0 };
        PI_F * double * self.area * l
    }

    fn light_type(&self) -> LightType {
        self.base.light_type
    }

    fn sample_li(
        &self,
        ctx: LightSampleContext,
        u: Point2f,
        lambda: SampledWavelengths,
        _allow_incomplete_pdf: bool,
    ) -> Option<LightLiSample> {
        let shape_ctx = ShapeSampleContext::new(ctx.pi, ctx.n, ctx.ns, 0.0);
        let ss = self.shape.sample_with_context(&shape_ctx, u);
        if ss.is_none() {
            return None;
        }
        let ss = ss.unwrap();
        if ss.pdf == 0.0 || (ss.intr.p() - ctx.p()).length_squared() == 0.0 {
            return None;
        }
        debug_assert!(!ss.pdf.is_nan());
        // TODO set the medium interface of ss.

        // TODO check against the alpha texture with alpha_masked().

        // Return LightLiSample for the sampled point on the shape.
        let wi = (ss.intr.p() - ctx.p()).normalize();
        let le = self.l(ss.intr.p(), ss.intr.n, ss.intr.uv, -wi, &lambda);
        if le.is_zero() {
            return None;
        }
        Some(LightLiSample::new(le, wi, ss.pdf, ss.intr))
    }

    fn pdf_li(&self, ctx: LightSampleContext, wi: Vector3f, _allow_incomplete_pdf: bool) -> Float {
        let shape_ctx = ShapeSampleContext::new(ctx.pi, ctx.n, ctx.ns, 0.0);
        self.shape.pdf_with_context(&shape_ctx, wi)
    }

    fn l(
        &self,
        _p: Point3f,
        n: Normal3f,
        _uv: Point2f,
        w: Vector3f,
        lambda: &SampledWavelengths,
    ) -> SampledSpectrum {
        // Check for zero emitted radiance from point on area light
        if !self.two_sided && n.dot_vector(&w) < 0.0 {
            return SampledSpectrum::from_const(0.0);
        }
        // TODO Check alpha mask with alpha texture and UV.

        // TODO If we add image textures, handle here.
        self.scale * self.l_emit.sample(lambda)
    }

    fn le(&self, _ray: &Ray, _lambda: &SampledWavelengths) -> SampledSpectrum {
        warn!("le() should only be called for infinite lights!");
        SampledSpectrum::from_const(0.0)
    }

    fn preprocess(&mut self, scene_bounds: &Bounds3f) {
        // Nothing to do here!
    }
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
