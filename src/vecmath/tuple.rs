use std::ops::{Add, Mul, Sub};

use crate::{
    float::{Float, PI_F},
    math::{difference_of_products, safe_asin, Abs, Ceil, Floor},
};

use super::{has_nan::HasNan, length::Length};

/// A tuple with 3 elements.
/// Used for sharing logic across e.g. Vector3f and Normal3f and Point3f.
/// Note that only those functions that are shared across all three types are
/// within this trait; if there's something that only one or two of them have,
/// then that can be represented in a separate trait which they can implement. Composition!
pub trait Tuple3<T>
where
    Self: Sized,
    T: Abs + Ceil + Floor,
{
    fn new(x: T, y: T, z: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;

    fn abs(&self) -> Self {
        Self::new(Abs::abs(self.x()), Abs::abs(self.y()), Abs::abs(self.z()))
    }

    fn ceil(&self) -> Self {
        Self::new(self.x().ceil(), self.y().ceil(), self.z().ceil())
    }

    fn floor(&self) -> Self {
        Self::new(self.x().floor(), self.y().floor(), self.z().floor())
    }

    // Since lerp requires Self: Add<Self>, but we don't want to allow Point + Point
    // and thus can't put that constraint on the trait bounds, we can't have a default
    // implementation here. But we can provide use free common implementation for types
    // which do implement Add<Self>. Though you could make an argument that Points should
    // not be able to be lerp'd if they can't be summed, but it's useful to be able to
    // interpolate points even if we typically can't want to allow summing them.
    fn lerp(t: Float, a: &Self, b: &Self) -> Self;
}

/// A tuple with 2 elements.
/// Used for sharing logic across e.g. Vector2f and Normal2f and Point2f.
pub trait Tuple2<T>
where
    Self: Sized,
    T: Abs + Ceil + Floor,
{
    fn new(x: T, y: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;

    fn abs(&self) -> Self {
        Self::new(Abs::abs(self.x()), Abs::abs(self.y()))
    }

    fn ceil(&self) -> Self {
        Self::new(self.x().ceil(), self.y().ceil())
    }

    fn floor(&self) -> Self {
        Self::new(self.x().floor(), self.y().floor())
    }

    fn lerp(t: Float, a: &Self, b: &Self) -> Self;
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
    T: Mul<Output = T> + Add<Output = T> + Abs + Ceil + Floor,
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
    T: Mul<Output = T> + Add<Output = T> + Abs + Ceil + Floor,
{
    T::abs(dot3(v, w))
}

/// Take the dot product of two vectors.
pub fn dot2<V1, V2, T>(v: &V1, w: &V2) -> T
where
    V1: Tuple2<T> + HasNan,
    V2: Tuple2<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Abs + Ceil + Floor,
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
    T: Mul<Output = T> + Add<Output = T> + Abs + Ceil + Floor,
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
