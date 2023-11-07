use crate::{
    light::Light,
    primitive::{Primitive, PrimitiveI},
    ray::Ray,
    shape::ShapeIntersection,
    vecmath::Vector3f,
    Float,
};

// TODO In places where PBRT uses a ScatchBuffer, I think that an Arena Allocator is a good Rust alternative.

pub trait Integrator {
    fn render(&self);

    /// Traces the given ray into the scene and returns the closest ShapeIntersection if any.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        debug_assert!(ray.d != Vector3f::ZERO);
        if let Some(agg) = self.get_aggregate() {
            agg.intersect(ray, t_max)
        } else {
            None
        }
    }

    /// Like intersect(), but returns only a boolean regarding the existence of an
    /// intersection, rather than information about the intersection. Potentially
    /// more efficient if only the existence of an intersection is needed.
    /// Useful for shadow rays.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        debug_assert!(ray.d != Vector3f::ZERO);
        if let Some(agg) = self.get_aggregate() {
            agg.intersect_predicate(ray, t_max)
        } else {
            false
        }
    }

    fn get_aggregate(&self) -> &Option<Primitive>;

    fn get_lights(&self) -> &Vec<Light>;

    fn get_infinite_lights(&self) -> &Vec<Light>;
}
