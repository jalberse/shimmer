use std::sync::Arc;

use crate::{
    aggregate::BvhAggregate,
    bounding_box::Bounds3f,
    light::Light,
    material::Material,
    ray::Ray,
    shape::{Shape, ShapeI, ShapeIntersection},
    transform::{InverseTransformRayI, Transform, TransformI, TransformRayI},
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
    Geometric(GeometricPrimitive),
}

impl PrimitiveI for Primitive {
    fn bounds(&self) -> Bounds3f {
        match self {
            Primitive::Simple(p) => p.bounds(),
            Primitive::Transformed(p) => p.bounds(),
            Primitive::BvhAggregate(a) => a.bounds(),
            Primitive::Geometric(p) => p.bounds(),
        }
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Primitive::Simple(p) => p.intersect(ray, t_max),
            Primitive::Transformed(p) => p.intersect(ray, t_max),
            Primitive::BvhAggregate(a) => a.intersect(ray, t_max),
            Primitive::Geometric(p) => p.intersect(ray, t_max),
        }
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        match self {
            Primitive::Simple(p) => p.intersect_predicate(ray, t_max),
            Primitive::Transformed(p) => p.intersect_predicate(ray, t_max),
            Primitive::BvhAggregate(a) => a.intersect_predicate(ray, t_max),
            Primitive::Geometric(p) => p.intersect_predicate(ray, t_max),
        }
    }
}

/// Stores a variety of properties that may be associated with a shape.
pub struct GeometricPrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
    /// Stores the emissive properties if the shape is a light source
    pub area_light: Option<Arc<Light>>,
    // TODO add alpha FloatTexture
    // TODO add medium_interface member
}

impl GeometricPrimitive {
    pub fn new(
        shape: Arc<Shape>,
        material: Arc<Material>,
        area_light: Option<Arc<Light>>,
    ) -> GeometricPrimitive {
        GeometricPrimitive {
            shape,
            material,
            area_light,
        }
    }
}

impl PrimitiveI for GeometricPrimitive {
    fn bounds(&self) -> Bounds3f {
        self.shape.bounds()
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let mut si = self.shape.intersect(ray, t_max)?;
        debug_assert!(si.t_hit < 1.001 * t_max);
        // TODO test intersection against alpha texture if present

        si.intr
            .set_intersection_properties(&self.material, &self.area_light);
        Some(si)
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        // TODO handle alpha. If no alpha, we can actually use shape.intersect_predicate().
        self.intersect(ray, t_max).is_some()
    }
}

/// A Primitive which simply adds material information to the surface interaction of
/// the shape. Used if the extra information in a GeometricPrimitive (e.g. lights)
/// are not needed for this shape.
pub struct SimplePrimitive {
    pub shape: Arc<Shape>,
    pub material: Arc<Material>,
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
    primitive: Arc<Primitive>,
    // TODO We should cache this transform in a pool rather than store a unique one
    render_from_primitive: Transform,
}

impl TransformedPrimitive {
    pub fn new(
        primitive: Arc<Primitive>,
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
        self.render_from_primitive.apply(self.primitive.bounds())
    }

    fn intersect(&self, ray: &Ray, mut t_max: Float) -> Option<ShapeIntersection> {
        let ray = self
            .render_from_primitive
            .apply_ray_inverse(*ray, Some(&mut t_max));
        let mut si = self.primitive.intersect(&ray, t_max)?;
        debug_assert!(si.t_hit <= 1.001 * t_max);

        si.intr = self.render_from_primitive.apply(si.intr);
        debug_assert!(si.intr.interaction.n.dot(si.intr.shading.n) >= 0.0);

        Some(si)
    }

    fn intersect_predicate(&self, ray: &Ray, mut t_max: Float) -> bool {
        let ray = self.render_from_primitive.apply_ray(*ray, Some(&mut t_max));
        self.primitive.intersect_predicate(&ray, t_max)
    }
}
