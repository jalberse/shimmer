// TODO We will need to implement PrimitiveI, but we can implement the Shape interface first.

use crate::{
    bounding_box::Bounds3f,
    direction_cone::DirectionCone,
    float::{gamma, PI_F},
    frame::Frame,
    interaction::{Interaction, SurfaceInteraction},
    interval::Interval,
    math::DifferenceOfProducts,
    math::{radians, safe_acos, safe_sqrt, Sqrt},
    ray::Ray,
    sampling::sample_uniform_sphere,
    transform::Transform,
    vecmath::{
        normal::Normal3,
        point::{Point3, Point3fi},
        vector::{Vector3, Vector3fi},
        Length, Normal3f, Normalize, Point2f, Point3f, Tuple2, Tuple3, Vector3f,
    },
    Float,
};
use log::warn;

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

    pub fn offset_ray_origin(&self, w: Vector3f) -> Point3f {
        todo!() // TODO these function definitions are in 6.8.6; we can get to them later.
    }

    // TODO Could use a trait to "overload" offset_ray_origin() instead.
    pub fn offset_ray_origin_pt(&self, pt: Point3f) -> Point3f {
        todo!()
    }

    pub fn spawn_ray(&self, w: Vector3f) -> Ray {
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
        // Transform ray origin and direction to object space
        let oi: Point3fi = self.object_from_render.apply(&ray.o).into();
        let di: Vector3fi = self.object_from_render.apply(&ray.d).into();
        // Solve quadratic equation to compute sphere t0 and t1

        // Compute sphere quadric coefficients
        let a: Interval = di.x().sqr() + di.y().sqr() + di.z().sqr();
        let b: Interval = 2.0 * (di.x() * oi.x() + di.y() * oi.y() + di.z() * oi.z());
        let c: Interval =
            oi.x().sqr() + oi.y().sqr() + oi.z().sqr() - Interval::from(self.radius).sqr();
        // Compute sphere quadratic discriminant
        let v = Vector3fi::from(oi - b / (2.0 * a) * di);
        let length = v.length();
        if length.is_nan() {
            // This can occur with rays perfectly aimed at the center of the sphere.
            warn!("Sphere intersection failed; intermediate NaN value");
            return None;
        }
        let discrim = 4.0
            * a
            * (Interval::from(self.radius) + length)
            * (Interval::from(self.radius) - length);
        if discrim.lower_bound() < 0.0 {
            return None;
        }
        // Compute quadratic t values
        let root_discrim = discrim.sqrt();
        let q = if Float::from(b) < 0.0 {
            -0.5 * (b - root_discrim)
        } else {
            -0.5 * (b + root_discrim)
        };
        let t0 = q / a;
        let t1 = c / q;
        // Swap quadratic t values so that t0 is the lesser
        let (t0, t1) = if t0.lower_bound() > t1.lower_bound() {
            (t1, t0)
        } else {
            (t0, t1)
        };
        // Check quadric shape for nearest intersection.
        if t0.upper_bound() > t_max || t1.lower_bound() <= 0.0 {
            return None;
        }
        let mut t_shape_hit = t0;
        if t_shape_hit.lower_bound() <= 0.0 {
            t_shape_hit = t1;
            if t_shape_hit.upper_bound() > t_max {
                return None;
            }
        }

        // Compute sphere hit position and phi
        let mut p_hit = Point3f::from(oi) + Float::from(t_shape_hit) * Vector3f::from(di);
        // Refine sphere intersection point
        p_hit *= self.radius / p_hit.distance(&Point3f::ZERO);

        if p_hit.x == 0.0 && p_hit.y == 0.0 {
            p_hit.x = 1e-5 * self.radius;
        }
        let mut phi = Float::atan2(p_hit.y, p_hit.x);
        if phi < 0.0 {
            phi += 2.0 * PI_F;
        }

        // Test sphere intersection against clipping parameters
        if (self.z_min > -self.radius && p_hit.z < self.z_min)
            || (self.z_max < self.radius && p_hit.z > self.z_max)
            || phi > self.phi_max
        {
            // Since t0 isn't valid, try again with t1
            if t_shape_hit == t1 {
                return None;
            }
            if t1.upper_bound() > t_max {
                return None;
            }
            t_shape_hit = t1;
            // Compute sphere hit position and phi
            p_hit = Point3f::from(oi) + Float::from(t_shape_hit) * Vector3f::from(di);
            // Refine sphere intersection point
            p_hit *= self.radius / p_hit.distance(&Point3f::ZERO);

            if p_hit.x == 0.0 && p_hit.y == 0.0 {
                p_hit.x = 1e-5 * self.radius;
            }
            phi = Float::atan2(p_hit.y, p_hit.x);
            if phi < 0.0 {
                phi += 2.0 * PI_F;
            }

            if (self.z_min > -self.radius && p_hit.z < self.z_min)
                || (self.z_max < self.radius && p_hit.z > self.z_max)
                || phi > self.phi_max
            {
                return None;
            }
        }

        Some(QuadricIntersection {
            t_hit: Float::from(t_shape_hit),
            p_obj: p_hit,
            phi,
        })
    }

    pub fn interaction_from_intersection(
        &self,
        isect: &QuadricIntersection,
        wo: Vector3f,
        time: Float,
    ) -> SurfaceInteraction {
        let p_hit = isect.p_obj;
        let phi = isect.phi;
        // Find parametric representation of sphere hit
        let u: Float = phi / self.phi_max;
        let cos_theta = p_hit.z / self.radius;
        let theta = safe_acos(cos_theta);
        let v = (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min);
        // Compute sphere $\dpdu$ and $\dpdv$
        let z_radius = Float::sqrt(p_hit.x() * p_hit.x() + p_hit.y() * p_hit.y());
        let cos_phi = p_hit.x() / z_radius;
        let sin_phi = p_hit.y() / z_radius;
        let dpdu = Vector3f::new(-self.phi_max * p_hit.y(), self.phi_max * p_hit.x(), 0.0);
        let sin_theta = safe_sqrt(1.0 - cos_theta * cos_theta);
        let dpdv = (self.theta_z_max - self.theta_z_min)
            * Vector3f::new(
                p_hit.z() * cos_phi,
                p_hit.z() * sin_phi,
                -self.radius * sin_theta,
            );

        // Compute sphere $\dndu$ and $\dndv$
        let d2pduu = -self.phi_max * self.phi_max * Vector3f::new(p_hit.x(), p_hit.y(), 0.0);
        let d2pduv = (self.theta_z_max - self.theta_z_min)
            * p_hit.z()
            * self.phi_max
            * Vector3f::new(-sin_phi, cos_phi, 0.);
        let d2pdvv = -((self.theta_z_max - self.theta_z_min)
            * (self.theta_z_max - self.theta_z_min))
            * Vector3f::new(p_hit.x(), p_hit.y(), p_hit.z());
        // Compute coefficients for fundamental forms
        let e1 = dpdu.dot(&dpdu);
        let f1 = dpdu.dot(&dpdv);
        let g1 = dpdv.dot(&dpdv);
        let n = dpdu.cross(&dpdv).normalize();
        let e = n.dot(&d2pduu);
        let f = n.dot(&d2pduv);
        let g = n.dot(&d2pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        let egf2 = Float::difference_of_products(e1, g1, f1, f1);
        let env_egf2 = if egf2 == 0.0 { 0.0 } else { 1.0 / egf2 };
        let dndu = Normal3f::from(
            (f * f1 - e * g1) * env_egf2 * dpdu + (e * f1 - f * e1) * env_egf2 * dpdv,
        );
        let dndv = Normal3f::from(
            (g * f1 - f * g1) * env_egf2 * dpdu + (f * f1 - g * e1) * env_egf2 * dpdv,
        );

        // Compute error bounds for sphere intersection
        let p_error = gamma(5) * Vector3f::from(p_hit).abs();

        // Return _SurfaceInteraction_ for quadric intersection
        let flip_normal: bool = self.reverse_orientation ^ self.transform_swaps_handedness;
        let wo_object = self.object_from_render.apply(&wo);

        let si = SurfaceInteraction::new(
            Point3fi::from_value_and_error(p_hit, p_error),
            Point2f::new(u, v),
            wo_object,
            dpdu,
            dpdv,
            dndu,
            dndv,
            time,
            flip_normal,
        );
        self.render_from_object.apply(&si)
    }
}

impl ShapeI for Sphere {
    fn bounds(&self) -> Bounds3f {
        // TODO could be made tighter when self.phi_max < 3pi/2.
        self.render_from_object.apply(&Bounds3f::new(
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
        self.basic_intersect(ray, t_max).is_some()
    }

    fn area(&self) -> Float {
        self.phi_max * self.radius * (self.z_max - self.z_min)
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        let mut p_obj: Point3f = Point3f::ZERO + sample_uniform_sphere(u) * self.radius;
        // Reproject p_obj to sphere surface and compute p_obj_error
        p_obj *= self.radius / p_obj.distance(&Point3f::ZERO);
        let p_obj_error: Vector3f = gamma(5) * Vector3f::from(p_obj).abs();

        // Compute surface normal for sphere sample and return shape sample
        let n_obj = Normal3f::new(p_obj.x, p_obj.y, p_obj.z);
        let normal_sign = if self.reverse_orientation { -1.0 } else { 1.0 };
        let n = normal_sign * self.render_from_object.apply(&n_obj).normalize();

        let theta = safe_acos(p_obj.z / self.radius);
        let mut phi = Float::atan2(p_obj.y, p_obj.x);
        if phi < 0.0 {
            phi += 2.0 * PI_F;
        }
        let uv = Point2f::new(
            phi / self.phi_max,
            (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min),
        );

        let pi = self
            .render_from_object
            .apply(&Point3fi::from_value_and_error(p_obj, p_obj_error));

        Some(ShapeSample {
            intr: Interaction::new(pi, n, uv, Vector3f::default(), 0.0),
            pdf: 1.0 / self.area(),
        })
    }

    fn pdf(&self, _interaction: &Interaction) -> Float {
        1.0 / self.area()
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        let p_center: Point3f = self.render_from_object.apply(&Point3f::ZERO);
        let p_origin: Point3f = ctx.offset_ray_origin_pt(p_center);
        // Sample uniformly on sphere if $\pt{}$ is inside it
        if p_origin.distance_squared(&p_center) <= self.radius * self.radius {
            // Sample shape by area and compute incident direction _wi_
            let mut ss = self.sample(u).expect("Sphere sample() failed!");
            ss.intr.time = ctx.time;
            let wi: Vector3f = ss.intr.p() - ctx.p();
            if wi.length_squared() == 0.0 {
                return None;
            }
            let wi = wi.normalize();

            // Convert area sampling PDF in _ss_ to solid angle measure
            ss.pdf /= ss.intr.n.abs_dot_vector(&-wi) / ctx.p().distance_squared(&ss.intr.p());
            if ss.pdf.is_infinite() {
                return None;
            }
            return Some(ss);
        }

        // Sample sphere uniformly inside subtended cone
        // Compute quantities related to the $\theta_\roman{max}$ for cone
        let sin_theta_max: Float = self.radius / ctx.p().distance(&p_center);
        let sin2_theta_max: Float = sin_theta_max * sin_theta_max;
        let cos_theta_max: Float = safe_sqrt(1.0 - sin2_theta_max);
        let mut one_minus_cos_theta_max: Float = 1.0 - cos_theta_max;

        // Compute $\theta$ and $\phi$ values for sample in cone
        let mut cos_theta: Float = (cos_theta_max - 1.0) * u[0] + 1.0;
        let mut sin2_theta: Float = 1.0 - cos_theta * cos_theta;
        if sin2_theta_max < 0.00068523
        /* sin^2(1.5 deg) */
        {
            // Compute cone sample via Taylor series expansion for small angles
            sin2_theta = sin2_theta_max * u[0];
            cos_theta = Float::sqrt(1.0 - sin2_theta);
            one_minus_cos_theta_max = sin2_theta_max / 2.0;
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        let cos_alpha: Float = sin2_theta / sin_theta_max
            + cos_theta * safe_sqrt(1.0 - sin2_theta / sin_theta_max * sin_theta_max);
        let sin_alpha: Float = safe_sqrt(1.0 - cos_alpha * cos_alpha);

        // Compute surface normal and sampled point on sphere
        let phi = u[1] * 2.0 * PI_F;
        let w = Vector3f::spherical_direction(sin_alpha, cos_alpha, phi);
        let sampling_frame = Frame::from_z((p_center - ctx.p()).normalize());
        let normal_sign = if self.reverse_orientation { -1.0 } else { 1.0 };
        let n = normal_sign * Normal3f::from(sampling_frame.from_local_v(&-w));
        let p = p_center + Vector3f::new(n.x, n.y, n.z) * self.radius;

        // Return _ShapeSample_ for sampled point on sphere
        // Compute _pError_ for sampled point on sphere
        let p_error: Vector3f = gamma(5) * Vector3f::from(p).abs();

        // Compute $(u,v)$ coordinates for sampled point on sphere
        let p_obj = self.object_from_render.apply(&p);
        let theta = safe_acos(p_obj.z / self.radius);
        let mut sphere_phi = Float::atan2(p_obj.y, p_obj.x);
        if sphere_phi < 0.0 {
            sphere_phi += 2.0 * PI_F;
        }
        let uv = Point2f::new(
            sphere_phi / self.phi_max,
            (theta - self.theta_z_min) / (self.theta_z_max - self.theta_z_min),
        );

        debug_assert_ne!(one_minus_cos_theta_max, 0.0); // very small far away sphere
        let pi = Point3fi::from_value_and_error(p, p_error);
        let interaction = Interaction {
            pi,
            time: ctx.time,
            wo: Vector3f::default(),
            n,
            uv,
        };
        Some(ShapeSample {
            intr: interaction,
            pdf: 1.0 / (2.0 * PI_F * one_minus_cos_theta_max),
        })
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vector3f) -> Float {
        let p_center = self.render_from_object.apply(&Point3f::ZERO);
        let p_origin = ctx.offset_ray_origin_pt(p_center);
        if p_origin.distance_squared(&p_center) <= self.radius * self.radius {
            // Return solid angle PDF for point inside sphere
            // Intersect sample ray with shape geometry
            let ray = ctx.spawn_ray(wi);
            let isect = self.intersect(&ray, Float::INFINITY);
            if isect.is_none() {
                return 0.0;
            }
            let isect = isect.unwrap();

            // Compute PDF in solid angle measure from shape intersection point
            let pdf = (1.0 / self.area())
                / isect.intr.interaction.n.abs_dot_vector(&-wi)
                / ctx.p().distance_squared(&isect.intr.p());
            if pdf.is_infinite() {
                return 0.0;
            } else {
                return pdf;
            }
        }
        // Compute general solid angle sphere PDF
        let sin2_theta_max = self.radius * self.radius / ctx.p().distance_squared(&p_center);
        let cos_theta_max = safe_sqrt(1.0 - sin2_theta_max);
        let mut one_minus_cos_theta_max = 1.0 - cos_theta_max;
        // Compute more accurate _oneMinusCosThetaMax_ for small solid angle
        // sin^2(1.5 deg)
        if sin2_theta_max < 0.00068523 {
            one_minus_cos_theta_max = sin2_theta_max / 2.0;
        }
        return 1.0 / (2.90 * PI_F * one_minus_cos_theta_max);
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ray::Ray,
        shape::{ShapeI, Sphere},
        transform::Transform,
        vecmath::{Point3f, Tuple3, Vector3f},
        Float,
    };

    #[test]
    fn sphere_basic() {
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            1.0,
            -1.0,
            1.0,
            360.0,
        );
        let ray = Ray::new(Point3f::new(0.0, 1e-5, -2.0), Vector3f::Z, None);
        assert!(sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(ray.o, -ray.d, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 1.0001, -2.0), Vector3f::Z, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));
    }

    #[test]
    fn sphere_partial_basic() {
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            1.0,
            -0.5,
            0.5,
            360.0,
        );
        let ray = Ray::new(Point3f::new(0.0, -2.0, 1e-5), Vector3f::Y, None);
        assert!(sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(ray.o, -ray.d, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 1e-5, 0.5001), Vector3f::Y, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 1e-5, -0.5001), Vector3f::Y, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));
    }
}
