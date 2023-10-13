//! A set of functions which help us implement the tuple traits for various types,
//! but that we don't want exposed external to the vecmath module.

use std::{
    ops::{Add, Mul, Neg, Sub},
    process::Output,
};

use crate::{
    float::PI_F,
    math::{safe_asin, DifferenceOfProducts, IsNeg, MulAdd},
    Float,
};

use super::{tuple::TupleElement, Length, Tuple2, Tuple3};

pub fn has_nan3<V, T>(v: &V) -> bool
where
    V: Tuple3<T>,
    T: TupleElement,
{
    v.x().is_nan() || v.y().is_nan() || v.z().is_nan()
}

pub fn has_nan2<V, T>(v: &V) -> bool
where
    V: Tuple2<T>,
    T: TupleElement,
{
    v.x().is_nan() || v.y().is_nan()
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
pub fn cross<V1, V2, V3, E>(v1: &V1, v2: &V2) -> V3
where
    V1: Tuple3<E>,
    V2: Tuple3<E>,
    V3: Tuple3<E>,
    E: TupleElement + DifferenceOfProducts,
{
    V3::new(
        E::difference_of_products(v1.y(), v2.z(), v1.z(), v2.y()),
        E::difference_of_products(v1.z(), v2.x(), v1.x(), v2.z()),
        E::difference_of_products(v1.x(), v2.y(), v1.y(), v2.x()),
    )
}

pub fn cross_i32<V1, V2, V3>(v1: &V1, v2: &V2) -> V3
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
pub fn dot3<V1, V2, E>(v: &V1, w: &V2) -> E
where
    V1: Tuple3<E>,
    V2: Tuple3<E>,
    E: TupleElement + MulAdd + DifferenceOfProducts,
{
    debug_assert!(!v.has_nan());
    debug_assert!(!w.has_nan());

    <E as MulAdd>::mul_add(v.x(), w.x(), E::sum_of_products(v.y(), w.y(), v.z(), w.z()))
}

/// Take the dot product of two vectors.
pub fn dot3i<V1, V2>(v: &V1, w: &V2) -> i32
where
    V1: Tuple3<i32>,
    V2: Tuple3<i32>,
{
    debug_assert!(!v.has_nan());
    debug_assert!(!w.has_nan());

    v.x() * w.x() + v.y() * w.y() + v.z() * w.z()
}

/// Take the dot product of two vectors then take the absolute value.
pub fn abs_dot3<V1, V2, E>(v: &V1, w: &V2) -> E
where
    V1: Tuple3<E>,
    V2: Tuple3<E>,
    E: TupleElement + MulAdd + DifferenceOfProducts,
{
    E::abs(dot3(v, w))
}

/// Take the dot product of two vectors then take the absolute value.
pub fn abs_dot3i<V1, V2>(v: &V1, w: &V2) -> i32
where
    V1: Tuple3<i32>,
    V2: Tuple3<i32>,
{
    i32::abs(dot3i(v, w))
}

/// Take the dot product of two vectors.
pub fn dot2<V1, V2>(v: &V1, w: &V2) -> Float
where
    V1: Tuple2<Float>,
    V2: Tuple2<Float>,
{
    debug_assert!(!v.has_nan());
    debug_assert!(!w.has_nan());
    Float::sum_of_products(v.x(), w.x(), v.y(), w.y())
}

/// Take the dot product of two vectors.
pub fn dot2i<V1, V2>(v: &V1, w: &V2) -> i32
where
    V1: Tuple2<i32>,
    V2: Tuple2<i32>,
{
    debug_assert!(!v.has_nan());
    debug_assert!(!w.has_nan());
    v.x() * w.x() + v.y() * w.y()
}

/// Take the dot product of two vectors then take the absolute value.
pub fn abs_dot2<V1, V2>(v: &V1, w: &V2) -> Float
where
    V1: Tuple2<Float>,
    V2: Tuple2<Float>,
{
    Float::abs(dot2(v, w))
}

/// Take the dot product of two vectors then take the absolute value.
pub fn abs_dot2i<V1, V2>(v: &V1, w: &V2) -> i32
where
    V1: Tuple2<i32>,
    V2: Tuple2<i32>,
{
    i32::abs(dot2i(v, w))
}

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
pub fn angle_between<'a, V1, V2, V3, E>(v1: &'a V1, v2: &'a V2) -> Float
where
    V1: Tuple3<E>,
    V2: Tuple3<E>,
    V3: Tuple3<E> + Length<Float>,
    &'a V1: Add<&'a V2, Output = V3>,
    &'a V2: Add<&'a V1, Output = V3> + Sub<&'a V1, Output = V3>,
    E: TupleElement + MulAdd + DifferenceOfProducts + Into<Float>,
{
    // TODO I think we could simplify the constraints on this function.
    // E might be able to be omitted, and instead specified as a constraint on the ElementType of V1.
    // We could also omit V3 by replacing it with the Output of V1 + V2, I think.
    debug_assert!(!v1.has_nan());
    debug_assert!(!v2.has_nan());
    if dot3(v1, v2).into() < 0.0 {
        PI_F - 2.0 * safe_asin((v1 + v2).length() / 2.0)
    } else {
        2.0 * safe_asin((v2 - v1).length() / 2.0)
    }
}

pub fn angle_between2<'a, V1, V2, V3>(v1: &'a V1, v2: &'a V2) -> Float
where
    V1: Tuple2<Float>,
    V2: Tuple2<Float>,
    V3: Tuple2<Float> + Length<Float>,
    &'a V1: Add<&'a V2, Output = V3>,
    &'a V2: Add<&'a V1, Output = V3> + Sub<&'a V1, Output = V3>,
{
    debug_assert!(!v1.has_nan());
    debug_assert!(!v2.has_nan());
    if dot2(v1, v2) < 0.0 {
        PI_F - 2.0 * safe_asin((v1 + v2).length() / 2.0)
    } else {
        2.0 * safe_asin((v2 - v1).length() / 2.0)
    }
}

pub fn face_forward<T1, T2, E>(a: &T1, b: &T2) -> T1
where
    T1: Tuple3<E> + Neg<Output = T1>,
    T2: Tuple3<E>,
    E: TupleElement + MulAdd + DifferenceOfProducts + IsNeg,
{
    if dot3(a, b).is_neg() {
        -*a
    } else {
        *a
    }
}
