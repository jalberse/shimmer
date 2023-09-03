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

// This module contains traits (Tuple, Length, Normalize, etc) which are created
// to allow us to share logic across our various types (Vectors, Normals, Points).
// However, we don't export the traits, and make their functions available within
// the public structs (calling the trait implementation). This way, users don't
// need to import the traits to use the structs.

mod has_nan;
mod length;
pub mod normal;
mod normalize;
pub mod point;
mod tuple;
pub mod vector;

use std::ops::{Add, Mul, Sub};

pub use normal::{Normal3f, Normal3i};
pub use point::{Point2f, Point2i, Point3f, Point3i};
pub use vector::{Vector2f, Vector2i, Vector3f, Vector3i};

use crate::{
    float::{Float, PI_F},
    math::{difference_of_products, safe_asin, Abs},
};

use self::{
    has_nan::HasNan,
    length::Length,
    tuple::{Tuple2, Tuple3},
};

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

// TODO Our impl_op_ex* methods are not all tested; test them.

// TODO Run a test coverage software and ensure full coverage. I know I'm missing a bit. https://lib.rs/crates/cargo-llvm-cov
//   Also hook that up into GitHub actions.

// TODO ensure we debug_assert() with has_nan() where appropriate.

// TODO also implement the list of functions on page 85

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
    V1: Tuple3<Float>,
    V2: Tuple3<Float>,
    V3: Tuple3<Float>,
{
    V3::new(
        difference_of_products(v1.y(), v2.z(), v1.z(), v2.y()),
        difference_of_products(v1.z(), v2.x(), v1.x(), v2.z()),
        difference_of_products(v1.x(), v2.y(), v1.y(), v2.x()),
    )
}

fn cross_i32<V1, V2, V3>(v1: &V1, v2: &V2) -> V3
where
    V1: Tuple3<i32>,
    V2: Tuple3<i32>,
    V3: Tuple3<i32>,
{
    V3::new(
        v1.y() * v2.z() - v1.z() * v2.y(),
        v1.z() * v2.x() - v1.x() * v2.z(),
        v1.x() * v2.y() - v1.y() * v2.x(),
    )
}

/// Take the dot product of two vectors.
fn dot3<V1, V2, T>(v: &V1, w: &V2) -> T
where
    V1: Tuple3<T> + HasNan,
    V2: Tuple3<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T>,
{
    debug_assert!(!v.has_nan());
    debug_assert!(!w.has_nan());
    v.x() * w.x() + v.y() * w.y() + v.z() * w.z()
}

/// Take the dot product of two vectors then take the absolute value.
fn abs_dot3<V1, V2, T>(v: &V1, w: &V2) -> T
where
    V1: Tuple3<T> + HasNan,
    V2: Tuple3<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Abs,
{
    T::abs(dot3(v, w))
}

/// Take the dot product of two vectors.
fn dot2<V1, V2, T>(v: &V1, w: &V2) -> T
where
    V1: Tuple2<T> + HasNan,
    V2: Tuple2<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T>,
{
    debug_assert!(!v.has_nan());
    debug_assert!(!w.has_nan());
    v.x() * w.x() + v.y() * w.y()
}

/// Take the dot product of two vectors then take the absolute value.
fn abs_dot2<V1, V2, T>(v: &V1, w: &V2) -> T
where
    V1: Tuple2<T> + HasNan,
    V2: Tuple2<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Abs,
{
    T::abs(dot2(v, w))
}

// TODO Consider some NormalizedVector, NormalizedNormal type or some other
// mechanism for enforcing that a vector must be normalized for an operation like angle_between.
// It's certainly possible with a type system but that's a lot of code to write.
// You would need the normalize() functions to return the Normalized* type,
// and then define operations on the Normalized* type as appropriate (where scaling
// would for example make it not-normalized).

/// Computes the angle between two vectors in radians. Generic because we want to be able
/// to use this for Vector and Normal types alike, and combinations of them.
/// Uses some numerical methods to be more accurate than a naive method.
///
/// Vectors must be normalized.
///
/// V1: A vector (e.g. Vector3f or a Normal3f).
/// V2: A vector (e.g. Vector3f or a Normal3f).
/// V3: The resulting type from adding or subtracting V1 and V2.
///   That is typically a Vector3f.
///   We just use the third type to be able to specify that that is the case.
fn angle_between<'a, V1, V2, V3>(v1: &'a V1, v2: &'a V2) -> Float
where
    V1: Tuple3<Float> + HasNan,
    V2: Tuple3<Float> + HasNan,
    V3: Tuple3<Float> + Length<Float>,
    &'a V1: Add<&'a V2, Output = V3>,
    &'a V2: Add<&'a V1, Output = V3> + Sub<&'a V1, Output = V3>,
{
    debug_assert!(!v1.has_nan());
    debug_assert!(!v2.has_nan());
    if dot3(v1, v2) < 0.0 {
        PI_F - 2.0 * safe_asin((v1 + v2).length() / 2.0)
    } else {
        2.0 * safe_asin((v2 - v1).length() / 2.0)
    }
}
