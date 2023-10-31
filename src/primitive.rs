use std::rc::Rc;

use crate::{
    bounding_box::Bounds3f,
    material::Material,
    ray::Ray,
    shape::{Shape, ShapeI, ShapeIntersection},
    Float,
};

pub trait PrimitiveI {
    /// Bounding box of the primitive.
    fn bounds(&self) -> Bounds3f;

    /// Returns information about the ray-primitive intersection if present.
    /// Part of a Primitive's job is to set information in the ShapeIntersection
    /// that can't be known from the Shape::intersect() routine.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;

    /// Checks *if* a ray-primitive intersection occurs, but does not return
    /// any additional information; potentially cheaper than intersect().
    /// Prefer if no additional information is neeeded about the intersection.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool;
}

pub enum Primitive {
    Simple(SimplePrimitive),
}

impl PrimitiveI for Primitive {
    fn bounds(&self) -> Bounds3f {
        match self {
            Primitive::Simple(p) => p.bounds(),
        }
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Primitive::Simple(p) => p.intersect(ray, t_max),
        }
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        match self {
            Primitive::Simple(p) => p.intersect_predicate(ray, t_max),
        }
    }
}

pub struct SimplePrimitive {
    shape: Shape,
    material: Rc<Material>,
}

impl PrimitiveI for SimplePrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut si = self.shape.intersect(ray, t_max)?;
        si.intr.set_intersection_properties(&self.material, &None);
        Some(si)
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        self.shape.intersect_predicate(ray, t_max)
    }
}
