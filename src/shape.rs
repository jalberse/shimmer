// TODO We will need to implement PrimitiveI, but we can implement the Shape interface first.

use crate::{
    bounding_box::Bounds3f,
    direction_cone::DirectionCone,
    interaction::{Interaction, SurfaceInteraction},
    math::radians,
    ray::Ray,
    transform::Transform,
    vecmath::{point::Point3fi, Normal3f, Point2f, Point3f, Tuple3, Vector3f},
    Float,
};

/// The Shape interface provides basic geometric properties of the primitive,
/// such as its surface area and its ray intersection routine. The non-geometric
/// data such as material properties are handled by the PrimitiveI interface.
pub trait ShapeI {
    /// Spatial extent of the shape
    fn bounds(&self) -> Bounds3f;

    /// Bounds the range of surface normals
    fn normal_bounds(&self) -> DirectionCone;

    /// Finds the first ray-shape intersection from (0, t_max), or None if there is no intersection.
    /// The rays are passed in rendering space, so shapes are responsible for transforming them
    /// to object space if needed for intersection tests. the intersection information returned should
    /// be in rendering space.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;

    /// Detects IF an intersection occurs, but does not return information about the intersection.
    /// This saves computation if only a true/false intersection is needed.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool;

    /// Surface area of the shape in rendering space.
    fn area(&self) -> Float;

    /// Samples a point on the surface of the shape.
    /// This is useful for using shapes as emitters.
    fn sample(&self, u: Point2f) -> Option<ShapeSample>;

    /// Probability density for sampling the specified point on the shape
    /// that corresponds to the given interaction.
    /// The interaction should be on the surface of the shape; the caller must ensure this.
    fn pdf(&self, interaction: &Interaction) -> Float;

    /// Like sample(), but takes a reference point from which the shape is being viewed.
    /// This is useful for lighting, since the caller can pass in the point
    /// to be lit and allow shape implementations to ensure they only sample the portion
    /// of the shape that is potentially visible from that point.
    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample>;

    /// Returns the shape's probability of sampling a point on the light such that the incident
    /// direction at the reference point is wi. The density should be with respect to the solid
    /// angle at the reference point. This should only be called for a direction that is known to
    /// intersect with the shape from the reference point.
    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vector3f) -> Float;
}

pub struct ShapeIntersection {
    pub intr: SurfaceInteraction,
    pub t_hit: Float,
}

pub struct QuadricIntersection {
    pub t_hit: Float,
    pub p_obj: Point3f,
    pub phi: Float,
}

pub struct ShapeSample {
    /// Interaction corresponding to the sampled point on the surface
    pub intr: Interaction,
    /// PDF for that sample w.r.t. the surface area of the shape
    pub pdf: Float,
}

pub struct ShapeSampleContext {
    /// Reference point
    pub pi: Point3fi,
    /// Geometric normal. For participating media, zero.
    pub n: Normal3f,
    /// Shading normal. For participating media, zero.
    pub ns: Normal3f,
    pub time: Float,
}

impl ShapeSampleContext {
    pub fn new(pi: Point3fi, n: Normal3f, ns: Normal3f, time: Float) -> ShapeSampleContext {
        ShapeSampleContext { pi, n, ns, time }
    }

    pub fn from_surface_interaction(si: &SurfaceInteraction) -> ShapeSampleContext {
        ShapeSampleContext {
            pi: si.interaction.pi,
            n: si.interaction.n,
            ns: si.shading.n,
            time: si.interaction.time,
        }
    }

    // TODO From medium interaction.

    /// Provides the point for scenarios where numerical error is not important.
    pub fn p(&self) -> Point3f {
        self.pi.into()
    }

    pub fn offset_ray_origin(w: Vector3f) -> Point3f {
        todo!() // TODO these function definitions are in 6.8.6; we can get to them later.
    }

    pub fn offset_ray_origin_pt(pt: Point3f) -> Point3f {
        todo!()
    }

    pub fn spawn_ray(w: Vector3f) -> Ray {
        todo!()
    }
}

pub struct Sphere {
    radius: Float,
    /// Minimum z value; can "cut off" the bottom of the sphere if < radius.
    z_min: Float,
    /// Maximum z value; can "cut off" the top of the sphere if > radius.
    z_max: Float,
    /// The theta corresponding to the minimum z value.
    theta_z_min: Float,
    /// The theta corresponding tot he maximum z value.
    theta_z_max: Float,
    /// phi can be limited to "cut off" to result in a partial sphere covering phi degrees. (0, 360).
    phi_max: Float,
    // TODO we probably want these to be an Rc to a cache of transforms
    render_from_object: Transform,
    object_from_render: Transform,
    /// Reverses the orientation of the surface normals.
    reverse_orientation: bool,
    transform_swaps_handedness: bool,
}

impl Sphere {
    pub fn new(
        render_from_object: Transform,
        object_from_render: Transform,
        reverse_orientation: bool,
        radius: Float,
        z_min: Float,
        z_max: Float,
        phi_max: Float,
    ) -> Sphere {
        Sphere {
            radius,
            z_min: Float::clamp(Float::min(z_min, z_max), -radius, radius),
            z_max: Float::clamp(Float::max(z_min, z_max), -radius, radius),
            theta_z_min: Float::acos(Float::clamp(Float::min(z_min, z_max) / radius, -1.0, 1.0)),
            theta_z_max: Float::acos(Float::clamp(Float::max(z_min, z_max) / radius, -1.0, 1.0)),
            phi_max: radians(Float::clamp(phi_max, 0.0, 360.0)),
            render_from_object,
            object_from_render,
            reverse_orientation,
            transform_swaps_handedness: render_from_object.swaps_handedness(),
        }
    }
}

impl Sphere {
    pub fn basic_intersect(&self, ray: &Ray, t_max: Float) -> Option<QuadricIntersection> {
        let mut phi = 0.0;
        let mut p_hit = 0.0;
        // Transform ray origin and direction to object space
        // TODO we need transformations on Point3fi and Vector3fi I think...
        todo!()
    }

    pub fn interaction_from_intersection(
        &self,
        isect: &QuadricIntersection,
        wo: Vector3f,
        time: Float,
    ) -> SurfaceInteraction {
        todo!()
    }
}

impl ShapeI for Sphere {
    fn bounds(&self) -> Bounds3f {
        // TODO could be made tighter when self.phi_max < 3pi/2.
        self.render_from_object.apply_bb(&Bounds3f::new(
            Point3f::new(-self.radius, -self.radius, self.z_min),
            Point3f::new(self.radius, self.radius, self.z_max),
        ))
    }

    fn normal_bounds(&self) -> DirectionCone {
        DirectionCone::entire_sphere()
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let isect = self.basic_intersect(ray, t_max)?;
        let intr = self.interaction_from_intersection(&isect, -ray.d, ray.time);
        Some(ShapeIntersection {
            intr,
            t_hit: isect.t_hit,
        })
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        todo!()
    }

    fn area(&self) -> Float {
        todo!()
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn pdf(&self, interaction: &Interaction) -> Float {
        todo!()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        todo!()
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vector3f) -> Float {
        todo!()
    }
}
