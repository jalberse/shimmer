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

use std::ops::{Add, Mul, Sub};

pub use normal::{Normal3f, Normal3i};
pub use point::{Point2f, Point2i, Point3f, Point3i};
pub use vector::{Vector2f, Vector2i, Vector3f, Vector3i};

use crate::{
    float::{Float, IsNan, PI_F},
    math::{difference_of_products, safe_asin, Sqrt},
};

// TODO consider moving away from glam. If nothing else, I don't love not being able to access fields directly
//   as required by the newtype pattern. We could implement optimizations ourselves, and long-term that's likely
//   something we want to do as we e.g. use SIMD to process ray clusters. We likely want more control.
//   In such a case we could get rid of our newtype_macros.rs, since it's only useful specifically for
//   newtype trait implementations (though I'd like to shove that into a repo and keep it, it's situationally useful).
//   We could just use impl_ops instead if we're rolling our own types from scratch instead of using newtypes.
//   Also at this point I think that with my necessary traits for Normal/Vector interaction (at least, to follow DRY),
//   and with the optimizations that are not present in glam (because glam is general-purpose), I'm implementing so much
//   myself that I'd rather just roll everything myself.
//
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

// TODO also the list of functions on page 85

/// Define a basic trait that allows us to define shared operations for Vector and Normal types.
trait Vector3<T>
where
    T: Mul<Output = T> + Add<Output = T> + Sqrt + IsNan,
{
    fn new(x: T, y: T, z: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;

    fn has_nan(&self) -> bool {
        self.x().is_nan() || self.y().is_nan() || self.z().is_nan()
    }

    fn length_squared(&self) -> T {
        debug_assert!(!self.has_nan());
        self.x() * self.x() + self.y() * self.y() + self.z() * self.z()
    }

    fn length(&self) -> T {
        // PAPERDOC - PBRTv4 has a discussion on page 88 about an odd usage of std::sqrt().
        // We see here that Rust's trait system obviates the issue.
        self.length_squared().sqrt()
    }

    // TODO bring normalize() into here too I think
}

/// Computes the cross product of two vectors. Generic because we want to be able
/// to use this for Vector and Normal types alike, and combinations of them.
///
/// NOTE: The cross product of two Normals is not valid, so don't do that.
/// That isn't enforced in the type system here, but this is at least private to the module;
/// the public vecmath interface doesn't allow e.g. Normal3f.cross(Normal3f).
///
/// V1: A vector (e.g. Vector3f or a Normal3f).
/// V2: A vector (e.g. Vector3f or a Normal3f).
/// V3: The type of the output cross product.
fn cross<V1, V2, V3>(v1: &V1, v2: &V2) -> V3
where
    V1: Vector3<Float>,
    V2: Vector3<Float>,
    V3: Vector3<Float>,
{
    V3::new(
        difference_of_products(v1.y(), v2.z(), v1.z(), v2.y()),
        difference_of_products(v1.z(), v2.x(), v1.x(), v2.z()),
        difference_of_products(v1.x(), v2.y(), v1.y(), v2.x()),
    )
}

/// Take the dot product of two vectors.
fn dot<V1, V2>(v: V1, w: V2) -> Float
where
    V1: Vector3<Float>,
    V2: Vector3<Float>,
{
    v.x() * w.x() + v.y() * w.y() + v.z() * w.z()
}

/// Computes the angle between two vectors. Generic because we want to be able
/// to use this for Vector and Normal types alike, and combinations of them.
///
/// V1: A vector (e.g. Vector3f or a Normal3f).
/// V2: A vector (e.g. Vector3f or a Normal3f).
/// V3: The resulting type from adding or subtracting V1 and V2.
///   That is typically a Vector3f.
///   We just use the third type to be able to specify that that is the case.
fn angle_between<V1, V2, V3>(v: V1, w: V2) -> Float
where
    V1: Vector3<Float> + Add<V2, Output = V3> + Copy + Clone,
    V2: Vector3<Float> + Add<V1, Output = V1> + Sub<V1, Output = V3> + Copy + Clone,
    V3: Vector3<Float>,
{
    if dot(v, w) < 0.0 {
        PI_F - 2.0 * safe_asin((v + w).length() / 2.0)
    } else {
        2.0 * safe_asin((w - v).length()) / 2.0
    }
}
