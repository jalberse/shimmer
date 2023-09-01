//! Why are there distinct types for **Vector** and **Point** and
//! **Normal**? Why not just use a unified vector class? Because they
//! are not the same, so let's capture that with our ~type system~.
//!
//! All **vectors** v in an affine space can be expressed as a linear
//! combination of the basis vectors:
//!
//! v = s_1 * v_1 + ... + s_n * v_n
//!
//! The scalars s_i are the representation of v with respect to the basis.
//!
//! Similarly, for all **points** p and origin p_0, there are unique scalars s_i
//! such that the point can be expressed in terms of the origin p_0 and
//! the basis vectors:
//!
//! p = p_0 + s_1 * v_1 + ... + s_n * v_n
//!
//! So, while vectors and points are both represented by 3 x, y, z values
//! in 3D space, they are distinct mathematical entities and not freely
//! interchangeable. Having separate types can therefore be more easily
//! understood and reasoned with, and allows us to handle them in different
//! manners as necessary (e.g points don't have a length).
//!
//! A **surface normal** (or a **normal**) is a vector perpendicular to a
//! surface at a particular position. Although normals are superficially
//! similar to vectors, it is important to distinguish between them
//! in the type system as well: because normals are defined in terms of
//! their relationship to a particular surface, they behave differently
//! than vectors in some situations, particularly when applying transformations.

pub mod normal;
pub mod point;
mod vec_types;
pub mod vector;

pub use normal::{Normal3f, Normal3i};
pub use point::{Point2f, Point2i, Point3f, Point3i};
pub use vector::{Vector2f, Vector2i, Vector3f, Vector3i};

use crate::{float::Float, math::difference_of_products};

// TODO consider moving away from glam. If nothing else, I don't love not being able to access fields directly
//   as required by the newtype pattern. We could implement optimizations ourselves, and long-term that's likely
//   something we want to do as we e.g. use SIMD to process ray clusters. We likely want more control.
//   In such a case we could get rid of our newtype_macros.rs, since it's only useful specifically for
//   newtype trait implementations (though I'd like to shove that into a repo and keep it, it's situationally useful).
//   We could just use impl_ops instead if we're rolling our own types from scratch instead of using newtypes.

// We use glam as it is a optimized vector math library which includes SIMD optimization.
// We wrap the glam vector classes using the newtype pattern. This accomplishes two things:
//  1. Points, Vectors, and Normals can be defined as distinct types,
//     which is useful as they are distinct mathematical entities and should be treated as such, and
//  2. It allows for us to deviate from glam's implementations of certain operations as needed,
//     and include/exclude operations for each type as appropriate for each type.
//
// A disadvantage of the newtype approach is the associated boilerplate, and some arguably unnecessary abstraction,
// such as being unable to access x, y, z directly (requiring getters or tuple access, i.e. some_vec.0.x).
// But, this trade-off is worth it to be able to leverage the type system for correctness.
// The newtype pattern should have no associated runtime cost here, optimized by the compiler.

// TODO and go add calls to has_nan() in other functions, wrapping in debug_assert().

// TODO our AngleBetween() implementation:
//   We *do* want to roll our own here, as Vec3::angle_between() does not do this accuracy fix.

// TODO These all need accessors, really. Possibly tied to traits?

// TODO note I think that Product trait impl from glam types is the HProd from pbrt, so use that.

/// Define a basic trait that allows us to define shared functions for Vector and Normal types.
/// We won't expose this outside this module, since we'll dictate the permissible behaviors for
/// each type within each type's definition, but this allows shared implementations internally.
/// That's a design decision that could change in the future.
trait Vector3<T> {
    fn new(x: T, y: T, z: T) -> Self;
    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

/// Note that we require 3 generic types because V1 and V2 may be either a Normal or Vector,
/// and the result should always be a Vector. Since the order of the parameters matters,
/// we can't let the output be V1 or V2; so we let the caller explicitly say which should
/// be the output type.
fn cross<V1: Vector3<Float>, V2: Vector3<Float>, V3: Vector3<Float>>(v1: &V1, v2: &V2) -> V3 {
    V3::new(
        difference_of_products(v1.y(), v2.z(), v1.z(), v2.y()),
        difference_of_products(v1.z(), v2.x(), v1.x(), v2.z()),
        difference_of_products(v1.x(), v2.y(), v1.y(), v2.x()),
    )
}
