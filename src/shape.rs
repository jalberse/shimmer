// TODO We will need to implement PrimitiveI, but we can implement the Shape interface first.

use crate::{bounding_box::Bounds3f, direction_cone::DirectionCone};

/// The Shape interface provides basic geometric properties of the primitive,
/// such as its surface area and its ray intersection routine. The non-geometric
/// data such as material properties are handled by the PrimitiveI interface.
pub trait ShapeI {
    /// Spatial extent of the shape
    fn bounds(&self) -> Bounds3f;

    /// Bounds the range of surface normals
    fn normal_bounds(&self) -> DirectionCone;
}
