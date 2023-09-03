use std::ops::{Add, Mul, Sub};

use crate::{
    float::{Float, PI_F},
    math::{difference_of_products, safe_asin, Abs},
};

use super::{has_nan::HasNan, length::Length};

/// A tuple with 3 elements.
/// Used for sharing logic across e.g. Vector3f and Normal3f and Point3f.
pub trait Tuple3<T> {
    fn new(x: T, y: T, z: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

/// A tuple with 2 elements.
/// Used for sharing logic across e.g. Vector2f and Normal2f and Point2f.
pub trait Tuple2<T> {
    fn new(x: T, y: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;
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
pub fn cross<V1, V2, V3>(v1: &V1, v2: &V2) -> V3
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
pub fn dot3<V1, V2, T>(v: &V1, w: &V2) -> T
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
pub fn abs_dot3<V1, V2, T>(v: &V1, w: &V2) -> T
where
    V1: Tuple3<T> + HasNan,
    V2: Tuple3<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Abs,
{
    T::abs(dot3(v, w))
}

/// Take the dot product of two vectors.
pub fn dot2<V1, V2, T>(v: &V1, w: &V2) -> T
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
pub fn abs_dot2<V1, V2, T>(v: &V1, w: &V2) -> T
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
pub fn angle_between<'a, V1, V2, V3>(v1: &'a V1, v2: &'a V2) -> Float
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
