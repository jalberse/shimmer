use crate::{bounding_box::Bounds3f, ray::Ray, shape::ShapeIntersection, Float};

pub trait PrimitiveI {
    /// Bounding box of the primitive.
    fn bounds(&self) -> Bounds3f;

    /// Returns information about the ray-primitive intersection if present.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;

    /// Checks *if* a ray-primitive intersection occurs, but does not return
    /// any additional information; potentially cheaper than intersect().
    /// Prefer if no additional information is neeeded about the intersection.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool;
}
