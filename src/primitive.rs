use std::rc::Rc;

use crate::{
    aggregate::BvhAggregate,
    bounding_box::Bounds3f,
    material::Material,
    ray::Ray,
    shape::{Shape, ShapeI, ShapeIntersection},
    transform::Transform,
    vecmath::normal::Normal3,
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
    Transformed(TransformedPrimitive),
    BvhAggregate(BvhAggregate),
}

impl PrimitiveI for Primitive {
    fn bounds(&self) -> Bounds3f {
        match self {
            Primitive::Simple(p) => p.bounds(),
            Primitive::Transformed(p) => p.bounds(),
            Primitive::BvhAggregate(a) => a.bounds(),
        }
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Primitive::Simple(p) => p.intersect(ray, t_max),
            Primitive::Transformed(p) => p.intersect(ray, t_max),
            Primitive::BvhAggregate(a) => a.intersect(ray, t_max),
        }
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        match self {
            Primitive::Simple(p) => p.intersect_predicate(ray, t_max),
            Primitive::Transformed(p) => p.intersect_predicate(ray, t_max),
            Primitive::BvhAggregate(a) => a.intersect_predicate(ray, t_max),
        }
    }
}

/// A Primitive which simply adds material information to the surface interaction of
/// the shape.
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

/// Enables object instancing by wrapping a pointer to a shared primitive with a transform.
pub struct TransformedPrimitive {
    primitive: Rc<Primitive>,
    // TODO We should cache this transform in a pool rather than store a unique one
    render_from_primitive: Transform,
}

impl TransformedPrimitive {
    pub fn new(
        primitive: &Rc<Primitive>,
        render_from_primitive: Transform,
    ) -> TransformedPrimitive {
        TransformedPrimitive {
            primitive: primitive.clone(),
            render_from_primitive,
        }
    }
}

impl PrimitiveI for TransformedPrimitive {
    fn bounds(&self) -> Bounds3f {
        self.render_from_primitive.apply(&self.primitive.bounds())
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let ray = self.render_from_primitive.apply_inv(ray);
        let mut si = self.primitive.intersect(&ray, t_max)?;
        debug_assert!(si.t_hit <= 1.001 * t_max);

        si.intr = self.render_from_primitive.apply(&si.intr);
        debug_assert!(si.intr.interaction.n.dot(&si.intr.shading.n) >= 0.0);

        Some(si)
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        let ray = self.render_from_primitive.apply(ray);
        self.primitive.intersect_predicate(&ray, t_max)
    }
}
