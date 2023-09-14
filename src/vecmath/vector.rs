use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::has_nan::HasNan;
use super::length::Length;
use super::length_fns::{length2, length3, length_squared2, length_squared3};
use super::normal::Normal3;
use super::normalize::Normalize;
use super::tuple::{Tuple2, Tuple3, TupleElement};
use super::tuple_fns::{
    abs_dot2, abs_dot2i, abs_dot3, abs_dot3i, angle_between, angle_between2, cross, cross_i32,
    dot2, dot2i, dot3, dot3i, has_nan2, has_nan3,
};
use super::{Normal3f, Normal3i, Point2f, Point2i, Point3f, Point3i};
use crate::float::Float;
use crate::math::lerp;
use auto_ops::{impl_op_ex, impl_op_ex_commutative};

pub trait Vector2:
    Tuple2<Self::ElementType>
    + Neg
    + Add<Self, Output = Self>
    + AddAssign<Self::ElementType>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self::ElementType, Output = Self>
    + MulAssign<Self::ElementType>
    + Div<Self::ElementType>
    + DivAssign<Self::ElementType>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
{
    type ElementType: TupleElement;

    /// Compute the dot product.
    fn dot(&self, v: &Self) -> Self::ElementType;

    /// Compute the dot product and take the absolute value.
    fn abs_dot(&self, v: &Self) -> Self::ElementType;

    /// Find the andle between this vector and another vector.
    /// Both vectors must be normalized.
    fn angle_between(&self, v: &Self) -> Float;

    /// Create a new vector orthogonal to w.
    /// w must be normalized.
    /// See PBRTv4 3.2
    fn gram_schmidt(&self, w: &Self) -> Self;
}

pub trait Vector3:
    Tuple3<Self::ElementType>
    + Neg
    + Add<Self, Output = Self>
    + AddAssign<Self::ElementType>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self::ElementType, Output = Self>
    + MulAssign<Self::ElementType>
    + Div<Self::ElementType>
    + DivAssign<Self::ElementType>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
{
    type ElementType: TupleElement;
    type AssociatedNormalType: Normal3;

    /// Compute the dot product.
    fn dot(&self, v: &Self) -> Self::ElementType;

    /// Dot this vector with a normal.
    fn dot_normal(&self, n: &Self::AssociatedNormalType) -> Self::ElementType;

    /// Compute the dot product and take the absolute value.
    fn abs_dot(&self, v: &Self) -> Self::ElementType;

    /// Dot this vector with a normal and take its absolute value.
    fn abs_dot_normal(&self, n: &Self::AssociatedNormalType) -> Self::ElementType;

    /// Take the cross product of this and a vector v.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    fn cross(&self, v: &Self) -> Self;

    /// Take the cross product of this and a normal n.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    fn cross_normal(&self, n: &Self::AssociatedNormalType) -> Self;

    /// Find the andle between this vector and another vector.
    /// Both vectors must be normalized.
    fn angle_between(&self, v: &Self) -> Float;

    /// Find the angle between this vector and a normal
    /// Both vectors must be normalized.
    fn angle_between_normal(&self, n: &Self::AssociatedNormalType) -> Float;

    /// Create a new vector orthogonal to w.
    /// w must be normalized.
    /// See PBRTv4 3.2
    fn gram_schmidt(&self, w: &Self) -> Self;
}

// ---------------------------------------------------------------------------
//        Vector2i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector2i {
    pub x: i32,
    pub y: i32,
}

impl Vector2i {
    /// All zeroes.
    pub const ZERO: Self = Self { x: 0, y: 0 };

    /// All ones.
    pub const ONE: Self = Self { x: 1, y: 1 };

    /// All negative ones.
    pub const NEG_ONE: Self = Self { x: -1, y: -1 };

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self { x: 1, y: 0 };

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self { x: 0, y: 1 };

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self { x: -1, y: 0 };

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self { x: 0, y: -1 };
}

impl Tuple2<i32> for Vector2i {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn x(&self) -> i32 {
        self.x
    }

    fn y(&self) -> i32 {
        self.y
    }

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        lerp(t, a, b)
    }
}

impl Vector2 for Vector2i {
    type ElementType = i32;

    /// Compute the dot product.
    fn dot(&self, v: &Self) -> i32 {
        dot2i(self, v)
    }

    /// Compute the dot product and take the absolute value.
    fn abs_dot(&self, v: &Self) -> i32 {
        abs_dot2i(self, v)
    }

    fn angle_between(&self, v: &Self) -> Float {
        angle_between2(&Vector2f::from(self), &Vector2f::from(v))
    }

    /// Create a new vector orthogonal to w.
    /// w must be normalized.
    /// See PBRTv4 3.2
    fn gram_schmidt(&self, w: &Self) -> Self {
        self - self.dot(w) * w
    }
}

impl HasNan for Vector2i {
    fn has_nan(&self) -> bool {
        false
    }
}

impl Length<i32> for Vector2i {
    fn length_squared(&self) -> i32 {
        length_squared2(self)
    }

    fn length(&self) -> i32 {
        length2(self)
    }
}

impl Default for Vector2i {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|v: &Vector2i| -> Vector2i {
    Vector2i {
        x: v.x.neg(),
        y: v.y.neg(),
    }
});

impl_op_ex!(+|v1: &Vector2i, v2: &Vector2i| -> Vector2i
{
    Vector2i { x: v1.x + v2.x, y: v1.y + v2.y }
});

impl_op_ex!(-|v1: &Vector2i, v2: &Vector2i| -> Vector2i {
    Vector2i {
        x: v1.x - v2.x,
        y: v1.y - v2.y,
    }
});

impl_op_ex_commutative!(*|v: &Vector2i, s: i32| -> Vector2i {
    Vector2i {
        x: v.x * s,
        y: v.y * s,
    }
});

impl_op_ex_commutative!(*|v: &Vector2i, s: Float| -> Vector2i {
    Vector2i {
        x: (v.x as Float * s) as i32,
        y: (v.y as Float * s) as i32,
    }
});

impl_op_ex!(/|v: &Vector2i, s: i32| -> Vector2i
{
    Vector2i { x: v.x / s, y: v.y / s }
});

impl_op_ex!(+=|v1: &mut Vector2i, v: &i32|
{
    v1.x += v;
    v1.y += v;
});

impl_op_ex!(+=|v1: &mut Vector2i, v2: &Vector2i|
{
    v1.x += v2.x;
    v1.y += v2.y;
});

impl_op_ex!(-=|v1: &mut Vector2i, v2: &Vector2i|
{
    v1.x -= v2.x;
    v1.y -= v2.y;
});

impl_op_ex!(*=|v1: &mut Vector2i, v2: &Vector2i|
{
    v1.x *= v2.x;
    v1.y *= v2.y;
});

impl_op_ex!(/|v1: &Vector2i, v2: &Vector2i| -> Vector2i
{
    Vector2i{
        x: v1.x / v2.x,
        y: v1.y / v2.y,
    }
});

impl_op_ex!(/=|v1: &mut Vector2i, v2: &Vector2i|
{
    v1.x /= v2.x;
    v1.y /= v2.y;
});

impl_op_ex!(*|v1: &Vector2i, v2: &Vector2i| -> Vector2i {
    Vector2i {
        x: v1.x * v2.x,
        y: v1.y * v2.y,
    }
});

impl_op_ex!(*=|v1: &mut Vector2i, s: i32|
{
    v1.x *= s;
    v1.y *= s;
});

impl_op_ex!(/=|v1: &mut Vector2i, s: i32|
{
    v1.x /= s;
    v1.y /= s;
});

impl From<Point2i> for Vector2i {
    fn from(value: Point2i) -> Self {
        Vector2i {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<[i32; 2]> for Vector2i {
    fn from(value: [i32; 2]) -> Self {
        Vector2i {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<Vector2i> for [i32; 2] {
    fn from(value: Vector2i) -> Self {
        [value.x, value.y]
    }
}

impl From<(i32, i32)> for Vector2i {
    fn from(value: (i32, i32)) -> Self {
        Vector2i {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<Vector2i> for (i32, i32) {
    fn from(value: Vector2i) -> Self {
        (value.x, value.y)
    }
}

// ---------------------------------------------------------------------------
//        Vector3i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector3i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Vector3i {
    /// All zeroes.
    pub const ZERO: Self = Self { x: 0, y: 0, z: 0 };

    /// All ones.
    pub const ONE: Self = Self { x: 1, y: 1, z: 1 };

    /// All negative ones.
    pub const NEG_ONE: Self = Self {
        x: -1,
        y: -1,
        z: -1,
    };

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self { x: 1, y: 0, z: 0 };

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self { x: 0, y: 1, z: 0 };

    /// A unit-length vector pointing along the positive Z axis.
    pub const Z: Self = Self { x: 0, y: 0, z: 1 };

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self { x: -1, y: 0, z: 0 };

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self { x: 0, y: -1, z: 0 };

    /// A unit-length vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self { x: 0, y: 0, z: -1 };
}

impl Tuple3<i32> for Vector3i {
    fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    fn x(&self) -> i32 {
        self.x
    }

    fn y(&self) -> i32 {
        self.y
    }

    fn z(&self) -> i32 {
        self.z
    }

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        lerp(t, a, b)
    }
}

impl Vector3 for Vector3i {
    type ElementType = i32;
    type AssociatedNormalType = Normal3i;

    /// Compute the dot product
    fn dot(&self, v: &Self) -> i32 {
        dot3i(self, v)
    }

    /// Dot this vector with a normal.
    fn dot_normal(&self, n: &Normal3i) -> i32 {
        dot3i(self, n)
    }

    /// Compute the dot product and take the absolute value.
    fn abs_dot(&self, v: &Self) -> i32 {
        abs_dot3i(self, v)
    }

    /// Dot this vector with a normal and take the absolute value.
    fn abs_dot_normal(&self, n: &Normal3i) -> i32 {
        abs_dot3i(self, n)
    }

    /// Take the cross product of this and a vector v
    fn cross(&self, v: &Self) -> Self {
        // Integer vectors do not need to use EFT methods for accuracy.
        cross_i32(self, v)
    }

    /// Take the cross product of this and a normal n
    fn cross_normal(&self, n: &Normal3i) -> Self {
        cross_i32(self, n)
    }

    fn angle_between(&self, v: &Self) -> Float {
        angle_between(&Vector3f::from(self), &Vector3f::from(v))
    }

    fn angle_between_normal(&self, n: &Self::AssociatedNormalType) -> Float {
        angle_between::<Vector3f, Normal3f, Vector3f>(&Vector3f::from(self), &Normal3f::from(n))
    }

    /// Create a new vector orthogonal to w.
    /// w must be normalized.
    /// See PBRTv4 3.2
    fn gram_schmidt(&self, w: &Self) -> Self {
        self - self.dot(w) * w
    }
}

impl HasNan for Vector3i {
    fn has_nan(&self) -> bool {
        false
    }
}

impl Length<i32> for Vector3i {
    fn length_squared(&self) -> i32 {
        length_squared3(self)
    }

    fn length(&self) -> i32 {
        length3(self)
    }
}

impl Default for Vector3i {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|v: &Vector3i| -> Vector3i {
    Vector3i {
        x: v.x.neg(),
        y: v.y.neg(),
        z: v.z.neg(),
    }
});

impl_op_ex!(+|v1: &Vector3i, v2: &Vector3i| -> Vector3i
{
    Vector3i{
        x: v1.x + v2.x,
        y: v1.y + v2.y,
        z: v1.z + v2.z
    }
});

impl_op_ex!(-|v1: &Vector3i, v2: &Vector3i| -> Vector3i {
    Vector3i {
        x: v1.x - v2.x,
        y: v1.y - v2.y,
        z: v1.z - v2.z,
    }
});

impl_op_ex_commutative!(*|v: &Vector3i, s: i32| -> Vector3i {
    Vector3i {
        x: v.x * s,
        y: v.y * s,
        z: v.z * s,
    }
});

impl_op_ex_commutative!(*|p: &Vector3i, s: Float| -> Vector3i {
    Vector3i {
        x: (p.x as Float * s) as i32,
        y: (p.y as Float * s) as i32,
        z: (p.z as Float * s) as i32,
    }
});

impl_op_ex!(/|v: &Vector3i, s: i32| -> Vector3i
{
    Vector3i { x: v.x / s, y: v.y / s, z: v.z / s }
});

impl_op_ex!(+=|v1: &mut Vector3i, v: &i32|
{
    v1.x += v;
    v1.y += v;
    v1.z += v;
});

impl_op_ex!(+=|v1: &mut Vector3i, v2: &Vector3i|
{
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
});

impl_op_ex!(-=|v1: &mut Vector3i, v2: &Vector3i|
{
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
});

impl_op_ex!(*=|v1: &mut Vector3i, v2: &Vector3i|
{
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
});

impl_op_ex!(/=|v1: &mut Vector3i, v2: &Vector3i|
{
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z;
});

impl_op_ex!(*=|v1: &mut Vector3i, s: i32|
{
    v1.x *= s;
    v1.y *= s;
    v1.z *= s;
});

impl_op_ex!(/=|v1: &mut Vector3i, s: i32|
{
    v1.x /= s;
    v1.y /= s;
    v1.z /= s;
});

impl_op_ex!(/|v1: &Vector3i, v2: &Vector3i| -> Vector3i
{
    Vector3i::new(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z)
});
impl_op_ex!(*|v1: &Vector3i, v2: &Vector3i| -> Vector3i {
    Vector3i::new(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
});

impl From<Point3i> for Vector3i {
    fn from(value: Point3i) -> Self {
        Vector3i {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<Normal3i> for Vector3i {
    fn from(value: Normal3i) -> Self {
        Vector3i {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<[i32; 3]> for Vector3i {
    fn from(value: [i32; 3]) -> Self {
        Vector3i {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Vector3i> for [i32; 3] {
    fn from(value: Vector3i) -> Self {
        [value.x, value.y, value.z]
    }
}

impl From<(i32, i32, i32)> for Vector3i {
    fn from(value: (i32, i32, i32)) -> Self {
        Vector3i {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl From<Vector3i> for (i32, i32, i32) {
    fn from(value: Vector3i) -> Self {
        (value.x, value.y, value.z)
    }
}

// ---------------------------------------------------------------------------
//        Vector2f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2f {
    pub x: Float,
    pub y: Float,
}

impl Vector2f {
    /// All zeroes.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// All ones.
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    /// All negative ones.
    pub const NEG_ONE: Self = Self { x: -1.0, y: -1.0 };

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self { x: 1.0, y: 0.0 };

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self { x: 0.0, y: 1.0 };

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self { x: -1.0, y: 0.0 };

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self { x: 0.0, y: -1.0 };
}

impl Tuple2<Float> for Vector2f {
    fn new(x: Float, y: Float) -> Self {
        Self { x, y }
    }

    fn x(&self) -> Float {
        self.x
    }

    fn y(&self) -> Float {
        self.y
    }

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        lerp(t, a, b)
    }
}

impl Vector2 for Vector2f {
    type ElementType = Float;

    /// Compute the dot product.
    fn dot(&self, v: &Self) -> Float {
        dot2(self, v)
    }

    /// Compute the dot product and take the absolute value.
    fn abs_dot(&self, v: &Self) -> Float {
        abs_dot2(self, v)
    }

    fn angle_between(&self, v: &Self) -> Float {
        angle_between2(self, v)
    }

    /// Create a new vector orthogonal to w.
    /// w must be normalized.
    /// See PBRTv4 3.2
    fn gram_schmidt(&self, w: &Self) -> Self {
        self - self.dot(w) * w
    }
}

impl HasNan for Vector2f {
    fn has_nan(&self) -> bool {
        has_nan2(self)
    }
}

impl Length<Float> for Vector2f {
    fn length_squared(&self) -> Float {
        length_squared2(self)
    }

    fn length(&self) -> Float {
        length2(self)
    }
}

impl Normalize<Float> for Vector2f {}

impl Default for Vector2f {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|v: &Vector2f| -> Vector2f { Vector2f::new(-v.x, -v.y) });
impl_op_ex!(+ |v1: &Vector2f, v2: &Vector2f| -> Vector2f {
    Vector2f::new(v1.x + v2.x, v1.y + v2.y)});
impl_op_ex!(-|v1: &Vector2f, v2: &Vector2f| -> Vector2f {
    Vector2f::new(v1.x - v2.x, v1.y - v2.y)
});
impl_op_ex!(+= |v1: &mut Vector2f, v2: &Vector2f| {
    v1.x += v2.x;
    v1.y += v2.y;
});
impl_op_ex!(-= |n1: &mut Vector2f, n2: &Vector2f| {
    n1.x -= n2.x;
    n1.y -= n2.y;
});
impl_op_ex_commutative!(*|v: &Vector2f, s: Float| -> Vector2f { Vector2f::new(v.x * s, v.y * s) });
impl_op_ex!(/ |v: &Vector2f, s: Float| -> Vector2f {
    Vector2f::new(v.x / s, v.y / s) });
impl_op_ex!(*= |v1: &mut Vector2f, s: Float| {
    v1.x *= s;
    v1.y *= s;
});
impl_op_ex!(/= |v1: &mut Vector2f, s: Float| {
    v1.x /= s;
    v1.y /= s;
});
impl_op_ex!(/= |v1: &mut Vector2f, v2: &Vector2f|
{
    v1.x /= v2.x;
    v1.y /= v2.y;
});
impl_op_ex!(/ |v1: &Vector2f, v2: &Vector2f| -> Vector2f
{
    Vector2f::new(
    v1.x / v2.x,
    v1.y / v2.y)
});
impl_op_ex!(*= |v1: &mut Vector2f, v2: &Vector2f|
{
    v1.x *= v2.x;
    v1.y *= v2.y;
});
impl_op_ex!(*|v1: &Vector2f, v2: &Vector2f| -> Vector2f {
    Vector2f::new(v1.x * v2.x, v1.y * v2.y)
});
impl_op_ex!(+= |v1: &mut Vector2f, v: &Float|
{
    v1.x += v;
    v1.y += v;
});

impl From<Point2f> for Vector2f {
    fn from(value: Point2f) -> Self {
        Self::new(value.x, value.y)
    }
}

impl From<[Float; 2]> for Vector2f {
    fn from(value: [Float; 2]) -> Self {
        Self::new(value[0], value[1])
    }
}

impl From<Vector2f> for [Float; 2] {
    fn from(value: Vector2f) -> Self {
        [value.x, value.y]
    }
}

impl From<(Float, Float)> for Vector2f {
    fn from(value: (Float, Float)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<Vector2f> for (Float, Float) {
    fn from(value: Vector2f) -> Self {
        (value.x, value.y)
    }
}

impl From<&Vector2i> for Vector2f {
    fn from(value: &Vector2i) -> Self {
        Vector2f {
            x: value.x as Float,
            y: value.y as Float,
        }
    }
}

// ---------------------------------------------------------------------------
//        Vector3f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3f {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Vector3f {
    /// All zeroes.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// All ones.
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    /// All negative ones.
    pub const NEG_ONE: Self = Self {
        x: -1.0,
        y: -1.0,
        z: -1.0,
    };

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// A unit-length vector pointing along the positive Z axis.
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };

    /// A unit-length vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };
}

impl Tuple3<Float> for Vector3f {
    fn new(x: Float, y: Float, z: Float) -> Self {
        Self { x, y, z }
    }

    fn x(&self) -> Float {
        self.x
    }

    fn y(&self) -> Float {
        self.y
    }

    fn z(&self) -> Float {
        self.z
    }

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        lerp(t, a, b)
    }
}

impl Vector3 for Vector3f {
    type ElementType = Float;
    type AssociatedNormalType = Normal3f;

    /// Compute the dot product.
    fn dot(&self, v: &Self) -> Float {
        dot3(self, v)
    }

    /// Dot this vector with a normal.
    fn dot_normal(&self, n: &Normal3f) -> Float {
        dot3(self, n)
    }

    /// Compute the dot product and take the absolute value.
    fn abs_dot(&self, v: &Self) -> Float {
        abs_dot3(self, v)
    }

    /// Dot this vector with a normal and take its absolute value.
    fn abs_dot_normal(&self, n: &Normal3f) -> Float {
        abs_dot3(self, n)
    }

    /// Take the cross product of this and a vector v.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    fn cross(&self, v: &Self) -> Self {
        cross::<Vector3f, Vector3f, Vector3f>(self, v)
    }

    /// Take the cross product of this and a normal n.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    fn cross_normal(&self, n: &Normal3f) -> Self {
        cross::<Vector3f, Normal3f, Vector3f>(self, n)
    }

    /// Find the andle between this vector and another vector.
    /// Both vectors must be normalized.
    fn angle_between(&self, v: &Self) -> Float {
        angle_between(self, v)
    }

    /// Find the angle between this vector and a normal
    /// Both vectors must be normalized.
    fn angle_between_normal(&self, n: &Normal3f) -> Float {
        angle_between::<Vector3f, Normal3f, Vector3f>(self, n)
    }

    /// Create a new vector orthogonal to w.
    /// w must be normalized.
    /// See PBRTv4 3.2
    fn gram_schmidt(&self, w: &Self) -> Self {
        self - self.dot(w) * w
    }
}

impl HasNan for Vector3f {
    fn has_nan(&self) -> bool {
        has_nan3(self)
    }
}

impl Length<Float> for Vector3f {
    fn length_squared(&self) -> Float {
        length_squared3(self)
    }

    fn length(&self) -> Float {
        length3(self)
    }
}

impl Normalize<Float> for Vector3f {}

impl Default for Vector3f {
    fn default() -> Self {
        Self::ZERO
    }
}

// Vectors can be negated
impl_op_ex!(-|v: &Vector3f| -> Vector3f { Vector3f::new(-v.x, -v.y, -v.z) });
// Vectors can add and subtract with other vectors
impl_op_ex!(+ |v1: &Vector3f, v2: &Vector3f| -> Vector3f {
    Vector3f::new(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)});
impl_op_ex!(-|v1: &Vector3f, v2: &Vector3f| -> Vector3f {
    Vector3f::new(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)
});
impl_op_ex!(+= |v1: &mut Vector3f, v2: &Vector3f| {
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
});
impl_op_ex!(-= |n1: &mut Vector3f, n2: &Vector3f| {
    n1.x -= n2.x;
    n1.y -= n2.y;
    n1.z -= n2.z;
});

// Vectors can be scaled
impl_op_ex_commutative!(*|v: &Vector3f, s: Float| -> Vector3f {
    Vector3f::new(v.x * s, v.y * s, v.z * s)
});
impl_op_ex!(/ |v: &Vector3f, s: Float| -> Vector3f { Vector3f::new(v.x / s, v.y / s, v.z / s) });
impl_op_ex!(+= |v1: &mut Vector3f, s: Float| {
    v1.x += s;
    v1.y += s;
    v1.z += s;
});
impl_op_ex!(*= |v1: &mut Vector3f, s: Float| {
    v1.x *= s;
    v1.y *= s;
    v1.z *= s;
});
impl_op_ex!(/= |v1: &mut Vector3f, s: Float| {
    v1.x /= s;
    v1.y /= s;
    v1.z /= s;
});
impl_op_ex!(*= |v1: &mut Vector3f, v2: &Vector3f| {
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
});
impl_op_ex!(/= |v1: &mut Vector3f, v2: &Vector3f| {
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z;
});
impl_op_ex!(/ |v1: &Vector3f, v2: &Vector3f| -> Vector3f{
    Vector3f::new(
        v1.x / v2.x,
        v1.y / v2.y,
        v1.z / v2.z)
});
impl_op_ex!(*|v1: &Vector3f, v2: &Vector3f| -> Vector3f {
    Vector3f::new(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
});

impl From<Point3f> for Vector3f {
    fn from(value: Point3f) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

impl From<Normal3f> for Vector3f {
    fn from(value: Normal3f) -> Self {
        Self::new(value.x, value.y, value.z)
    }
}

impl From<[Float; 3]> for Vector3f {
    fn from(value: [Float; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<Vector3f> for [Float; 3] {
    fn from(value: Vector3f) -> Self {
        [value.x, value.y, value.z]
    }
}

impl From<(Float, Float, Float)> for Vector3f {
    fn from(value: (Float, Float, Float)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<Vector3f> for (Float, Float, Float) {
    fn from(value: Vector3f) -> Self {
        (value.x, value.y, value.z)
    }
}

impl From<&Vector3i> for Vector3f {
    fn from(value: &Vector3i) -> Self {
        Self::new(value.x as Float, value.y as Float, value.z as Float)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        vecmath::{
            vector::{Vector2, Vector3},
            HasNan, Length, Normalize, Tuple2, Tuple3,
        },
        Float,
    };

    use super::{
        Normal3f, Normal3i, Point2f, Point2i, Point3f, Point3i, Vector2f, Vector2i, Vector3f,
        Vector3i,
    };

    use float_cmp::approx_eq;

    #[test]
    fn has_nan() {
        let v = Vector2f::new(Float::NAN, 0.0);
        assert!(v.has_nan());

        let v = Vector3f::new(0.0, Float::NAN, 0.0);
        assert!(v.has_nan());
    }

    #[test]
    fn vector_negation() {
        let vec = Vector2f::new(1.0, 2.0);
        assert_eq!(Vector2f::new(-1.0, -2.0), -vec);
        let vec = Vector3f::new(1.0, 2.0, 3.0);
        assert_eq!(Vector3f::new(-1.0, -2.0, -3.0), -vec);
    }

    #[test]
    fn vector_length() {
        let v = Vector2f::new(5.0, 5.0);
        assert_eq!(7.071068, v.length());

        let v = Vector3f::new(5.0, 6.0, 7.0);
        assert_eq!(10.488089, v.length());
    }

    #[test]
    fn vector_length_squared() {
        let v = Vector2f::new(5.0, 5.0);
        assert_eq!(50.0, v.length_squared());

        let v = Vector3f::new(5.0, 6.0, 7.0);
        assert_eq!(110.0, v.length_squared());
    }

    #[test]
    fn vector_normalize() {
        let v = Vector2f::new(10.0, 0.0);
        assert_eq!(Vector2f::new(1.0, 0.0), v.normalize());

        let v = Vector3f::new(0.0, 10.0, 0.0);
        assert_eq!(Vector3f::new(0.0, 1.0, 0.0), v.normalize());
    }

    #[test]
    fn vector_vector_dot() {
        let v1 = Vector3f::new(0.0, 1.0, 2.0);
        let v2 = Vector3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.dot(&v2));

        let v1 = Vector3i::new(0, 1, 2);
        let v2 = Vector3i::new(3, 4, 5);
        assert_eq!(14, v1.dot(&v2));

        let v1 = Vector2f::new(0.0, 1.0);
        let v2 = Vector2f::new(2.0, 3.0);
        assert_eq!(3.0, v1.dot(&v2));

        let v1 = Vector2i::new(0, 1);
        let v2 = Vector2i::new(2, 3);
        assert_eq!(3, v1.dot(&v2));
    }

    #[test]
    fn vector_normal_dot() {
        let v1 = Vector3f::new(0.0, 1.0, 2.0);
        let n = Normal3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.dot_normal(&n));

        let v1 = Vector3i::new(0, 1, 2);
        let n = Normal3i::new(3, 4, 5);
        assert_eq!(14, v1.dot_normal(&n));
    }

    #[test]
    fn vector_vector_abs_dot() {
        let v1 = Vector3f::new(0.0, 1.0, 2.0);
        let v2 = -Vector3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.abs_dot(&v2));

        let v1 = Vector3i::new(0, 1, 2);
        let v2 = -Vector3i::new(3, 4, 5);
        assert_eq!(14, v1.abs_dot(&v2));

        let v1 = Vector2f::new(0.0, 1.0);
        let v2 = -Vector2f::new(2.0, 3.0);
        assert_eq!(3.0, v1.abs_dot(&v2));

        let v1 = Vector2i::new(0, 1);
        let v2 = -Vector2i::new(2, 3);
        assert_eq!(3, v1.abs_dot(&v2));
    }

    #[test]
    fn vector_normal_abs_dot() {
        let v1 = Vector3f::new(0.0, 1.0, 2.0);
        let n = -Normal3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.abs_dot_normal(&n));

        let v1 = Vector3i::new(0, 1, 2);
        let n = -Normal3i::new(3, 4, 5);
        assert_eq!(14, v1.abs_dot_normal(&n));
    }

    #[test]
    fn vector_vector_cross() {
        let v1 = Vector3i::new(3, -3, 1);
        let v2 = Vector3i::new(4, 9, 2);
        assert_eq!(Vector3i::new(-15, -2, 39), v1.cross(&v2));

        let v1 = Vector3f::new(3.0, -3.0, 1.0);
        let v2 = Vector3f::new(4.0, 9.0, 2.0);
        assert_eq!(Vector3f::new(-15.0, -2.0, 39.0), v1.cross(&v2));
    }

    #[test]
    fn vector_normal_cross() {
        let v1 = Vector3i::new(3, -3, 1);
        let n = Normal3i::new(4, 9, 2);
        assert_eq!(Vector3i::new(-15, -2, 39), v1.cross_normal(&n));

        let v1 = Vector3f::new(3.0, -3.0, 1.0);
        let n = Normal3f::new(4.0, 9.0, 2.0);
        assert_eq!(Vector3f::new(-15.0, -2.0, 39.0), v1.cross_normal(&n));
    }

    #[test]
    fn vector_vector_angle_between() {
        let v1 = Vector3f::new(1.0, 2.0, 3.0).normalize();
        let v2 = Vector3f::new(3.0, 4.0, 5.0).normalize();

        assert_eq!(0.18623877, v1.angle_between(&v2));
    }

    #[test]
    fn vector_normal_angle_between() {
        let v = Vector3f::new(1.0, 2.0, 3.0).normalize();
        let n = Normal3f::new(3.0, 4.0, 5.0).normalize();

        assert_eq!(0.18623877, v.angle_between_normal(&n));
    }

    #[test]
    fn vector_coordinate_system() {
        // TODO
    }

    #[test]
    fn vector_normal_face_forward() {
        // TODO
    }

    #[test]
    fn vector_vector_face_forward() {
        // TODO
    }

    #[test]
    fn gram_schmidt() {
        let v1 = Vector3f::new(1.0, -1.0, 1.0);
        let v2 = Vector3f::new(1.0, 0.0, 1.0);

        let v2_schmidted = v2.gram_schmidt(&v1.normalize());

        let expected_x: Float = 1.0 / 3.0;
        let expected_y: Float = 2.0 / 3.0;
        let expected_z: Float = 1.0 / 3.0;

        assert!(approx_eq!(Float, expected_x, v2_schmidted.x));
        assert!(approx_eq!(Float, expected_y, v2_schmidted.y));
        assert!(approx_eq!(Float, expected_z, v2_schmidted.z));
    }

    #[test]
    fn vector_min_component() {
        let v1 = Vector3f::new(10.0, 20.0, 30.0);
        assert_eq!(0, v1.min_component_index());
        assert_eq!(10.0, v1.min_component_value());

        let v1 = Vector3f::new(999.0, 20.0, 30.0);
        assert_eq!(1, v1.min_component_index());
        assert_eq!(20.0, v1.min_component_value());

        let v1 = Vector3f::new(999.0, 999.0, 30.0);
        assert_eq!(2, v1.min_component_index());
        assert_eq!(30.0, v1.min_component_value());
    }

    #[test]
    fn vector_max_component() {
        let v1 = Vector3f::new(10.0, 20.0, 30.0);
        assert_eq!(2, v1.max_component_index());
        assert_eq!(30.0, v1.max_component_value());

        let v1 = Vector3f::new(999.0, 20.0, 30.0);
        assert_eq!(0, v1.max_component_index());
        assert_eq!(999.0, v1.max_component_value());

        let v1 = Vector3f::new(999.0, 1000.0, 30.0);
        assert_eq!(1, v1.max_component_index());
        assert_eq!(1000.0, v1.max_component_value());
    }

    #[test]
    fn vector_permute() {
        let v = Vector3i::new(0, 1, 2);
        assert_eq!(Vector3i::new(2, 1, 0), v.permute((2, 1, 0)));
        assert_eq!(Vector3i::new(2, 0, 1), v.permute((2, 0, 1)));
        assert_eq!(Vector3i::new(1, 2, 0), v.permute((1, 2, 0)));
        assert_eq!(Vector3i::new(1, 0, 2), v.permute((1, 0, 2)));
        assert_eq!(Vector3i::new(0, 1, 2), v.permute((0, 1, 2)));
        assert_eq!(Vector3i::new(0, 2, 1), v.permute((0, 2, 1)));

        let v = Vector2i::new(0, 1);
        assert_eq!(Vector2i::new(0, 1), v.permute((0, 1)));
        assert_eq!(Vector2i::new(1, 0), v.permute((1, 0)));
    }

    #[test]
    fn vector_lerp() {
        let v1 = Vector3f::new(0.0, 0.0, 0.0);
        let v2 = Vector3f::new(1.0, 10.0, 100.0);
        assert_eq!(Vector3f::new(0.5, 5.0, 50.0), Tuple3::lerp(0.5, &v1, &v2))
    }

    #[test]
    fn vector_min() {
        let v1 = Vector3f::new(100.0, 0.0, 0.0);
        let v2 = Vector3f::new(0.0, 10.0, 100.0);
        assert_eq!(Vector3f::new(0.0, 0.0, 0.0), Tuple3::min(&v1, &v2));

        let v1 = Vector3i::new(100, 0, 0);
        let v2 = Vector3i::new(0, 10, 100);
        assert_eq!(Vector3i::new(0, 0, 0), Tuple3::min(&v1, &v2));

        let v1 = Vector2f::new(100.0, 0.0);
        let v2 = Vector2f::new(0.0, 10.0);
        assert_eq!(Vector2f::new(0.0, 0.0), Tuple2::min(&v1, &v2));

        let v1 = Vector2i::new(100, 0);
        let v2 = Vector2i::new(0, 10);
        assert_eq!(Vector2i::new(0, 0), Tuple2::min(&v1, &v2));
    }

    #[test]
    fn vector_max() {
        let v1 = Vector3f::new(100.0, 0.0, 0.0);
        let v2 = Vector3f::new(0.0, 10.0, 100.0);
        assert_eq!(Vector3f::new(100.0, 10.0, 100.0), Tuple3::max(&v1, &v2));

        let v1 = Vector3i::new(100, 0, 0);
        let v2 = Vector3i::new(0, 10, 100);
        assert_eq!(Vector3i::new(100, 10, 100), Tuple3::max(&v1, &v2));

        let v1 = Vector2f::new(100.0, 0.0);
        let v2 = Vector2f::new(0.0, 10.0);
        assert_eq!(Vector2f::new(100.0, 10.0), Tuple2::max(&v1, &v2));

        let v1 = Vector2i::new(100, 0);
        let v2 = Vector2i::new(0, 10);
        assert_eq!(Vector2i::new(100, 10), Tuple2::max(&v1, &v2));
    }

    #[test]
    fn vector_binary_ops() {
        let vec = Vector2i::new(-2, 10);
        // Vector + Vector -> Vector
        assert_eq!(Vector2i::new(-4, 20), vec + vec);
        // Vector - Vecttor -> Vector
        assert_eq!(Vector2i::new(0, 0), vec - vec);
        // Scalar * Vector -> Vector
        assert_eq!(Vector2i::new(-6, 30), vec * 3);
        assert_eq!(Vector2i::new(-6, 30), 3 * vec);
        // Vector / Scalar -> Vector
        assert_eq!(Vector2i::new(-1, 5), vec / 2);

        let vec = Vector3i::new(-2, 10, 20);
        assert_eq!(Vector3i::new(-4, 20, 40), vec + vec);
        assert_eq!(Vector3i::new(0, 0, 0), vec - vec);
        assert_eq!(Vector3i::new(-6, 30, 60), vec * 3);
        assert_eq!(Vector3i::new(-6, 30, 60), 3 * vec);
        assert_eq!(Vector3i::new(-1, 5, 10), vec / 2);

        let vec = Vector2f::new(-1.0, 10.0);
        assert_eq!(Vector2f::new(-2.0, 20.0), vec + vec);
        assert_eq!(Vector2f::new(0.0, 0.0), vec - vec);
        assert_eq!(Vector2f::new(-3.0, 30.0), vec * 3.0);
        assert_eq!(Vector2f::new(-3.0, 30.0), 3.0 * vec);
        assert_eq!(Vector2f::new(-0.5, 5.0), vec / 2.0);

        let vec = Vector3f::new(-1.0, 10.0, 20.0);
        assert_eq!(Vector3f::new(-2.0, 20.0, 40.0), vec + vec);
        assert_eq!(Vector3f::new(0.0, 0.0, 0.0), vec - vec);
        assert_eq!(Vector3f::new(-3.0, 30.0, 60.0), vec * 3.0);
        assert_eq!(Vector3f::new(-3.0, 30.0, 60.0), 3.0 * vec);
        assert_eq!(Vector3f::new(-0.5, 5.0, 10.0), vec / 2.0);
    }

    #[test]
    fn vector_abs() {
        let v = Vector3f::new(-1.0, 2.0, -3.5);
        assert_eq!(Vector3f::new(1.0, 2.0, 3.5), v.abs());

        let v = Vector2f::new(-1.0, 2.0);
        assert_eq!(Vector2f::new(1.0, 2.0), v.abs());
    }

    #[test]
    fn vector_ceil() {
        let v = Vector3f::new(-1.5, 1.5, 1.6);
        assert_eq!(Vector3f::new(-1.0, 2.0, 2.0), v.ceil());

        let v = Vector2f::new(-1.5, 1.5);
        assert_eq!(Vector2f::new(-1.0, 2.0), v.ceil());
    }

    #[test]
    fn vector_floor() {
        let v = Vector3f::new(-1.5, 1.5, 1.6);
        assert_eq!(Vector3f::new(-2.0, 1.0, 1.0), v.floor());

        let v = Vector2f::new(-1.5, 1.5);
        assert_eq!(Vector2f::new(-2.0, 1.0), v.floor());
    }

    #[test]
    fn vector_assignment_ops() {
        // Vector2i
        // +=
        let mut v1 = Vector2i::new(1, 2);
        let v2 = Vector2i::new(3, 4);
        v1 += v2;
        assert_eq!(Vector2i::new(4, 6), v1);

        // -=
        let mut v1 = Vector2i::new(1, 2);
        let v2 = Vector2i::new(3, 4);
        v1 -= v2;
        assert_eq!(Vector2i::new(-2, -2), v1);

        // *=
        let mut v1 = Vector2i::new(1, 2);
        v1 *= 2;
        assert_eq!(Vector2i::new(2, 4), v1);

        // /=
        let mut v1 = Vector2i::new(1, 2);
        v1 /= 2;
        assert_eq!(Vector2i::new(0, 1), v1);

        // Vector3i
        // +=
        let mut v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3i::new(4, 5, 6);
        v1 += v2;
        assert_eq!(Vector3i::new(5, 7, 9), v1);

        // -=
        let mut v1 = Vector3i::new(1, 2, 3);
        let v2 = Vector3i::new(4, 5, 6);
        v1 -= v2;
        assert_eq!(Vector3i::new(-3, -3, -3), v1);

        // *=
        let mut v1 = Vector3i::new(1, 2, 3);
        v1 *= 2;
        assert_eq!(Vector3i::new(2, 4, 6), v1);

        // /=
        let mut v1 = Vector3i::new(1, 2, 3);
        v1 /= 2;
        assert_eq!(Vector3i::new(0, 1, 1), v1);

        // Vector2f
        // +=
        let mut v1 = Vector2f::new(1.0, 2.0);
        let v2 = Vector2f::new(4.0, 5.0);
        v1 += v2;
        assert_eq!(Vector2f::new(5.0, 7.0), v1);

        // -=
        let mut v1 = Vector2f::new(1.0, 2.0);
        let v2 = Vector2f::new(4.0, 5.0);
        v1 -= v2;
        assert_eq!(Vector2f::new(-3.0, -3.0), v1);

        // *=
        let mut v1 = Vector2f::new(1.0, 2.0);
        v1 *= 2.0;
        assert_eq!(Vector2f::new(2.0, 4.0), v1);

        // /=
        let mut v1 = Vector2f::new(1.0, 2.0);
        v1 /= 2.0;
        assert_eq!(Vector2f::new(0.5, 1.0), v1);

        // Vector3f
        // +=
        let mut v1 = Vector3f::new(1.0, 2.0, 3.0);
        let v2 = Vector3f::new(4.0, 5.0, 6.0);
        v1 += v2;
        assert_eq!(Vector3f::new(5.0, 7.0, 9.0), v1);

        // -=
        let mut v1 = Vector3f::new(1.0, 2.0, 3.0);
        let v2 = Vector3f::new(4.0, 5.0, 6.0);
        v1 -= v2;
        assert_eq!(Vector3f::new(-3.0, -3.0, -3.0), v1);

        // *=
        let mut v1 = Vector3f::new(1.0, 2.0, 3.0);
        v1 *= 2.0;
        assert_eq!(Vector3f::new(2.0, 4.0, 6.0), v1);

        // /=
        let mut v1 = Vector3f::new(1.0, 2.0, 3.0);
        v1 /= 2.0;
        assert_eq!(Vector3f::new(0.5, 1.0, 1.5), v1);
    }

    #[test]
    fn vector_from_point() {
        let v1 = Vector2i::new(1, 2);
        let p1 = Point2i::new(1, 2);
        assert_eq!(v1, p1.into());

        let v1 = Vector3i::new(1, 2, 3);
        let p1 = Point3i::new(1, 2, 3);
        assert_eq!(v1, p1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let p1 = Point2f::new(1.0, 2.0);
        assert_eq!(v1, p1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let p1 = Point3f::new(1.0, 2.0, 3.0);
        assert_eq!(v1, p1.into());
    }

    #[test]
    fn vector_from_normal() {
        let v1 = Vector3i::new(1, 2, 3);
        let n1 = Normal3i::new(1, 2, 3);
        assert_eq!(v1, n1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let n1 = Normal3f::new(1.0, 2.0, 3.0);
        assert_eq!(v1, n1.into());
    }

    #[test]
    fn vector_from_into_tuple() {
        let v1 = Vector2i::new(1, 2);
        let t1: (i32, i32) = v1.into();
        assert_eq!((1, 2), t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector3i::new(1, 2, 3);
        let t1: (i32, i32, i32) = v1.into();
        assert_eq!((1, 2, 3), t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let t1: (Float, Float) = v1.into();
        assert_eq!((1.0, 2.0), t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let t1: (Float, Float, Float) = v1.into();
        assert_eq!((1.0, 2.0, 3.0), t1);
        assert_eq!(v1, t1.into());
    }

    #[test]
    fn vector_from_into_array() {
        let v1 = Vector2i::new(1, 2);
        let t1: [i32; 2] = v1.into();
        assert_eq!([1, 2], t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector3i::new(1, 2, 3);
        let t1: [i32; 3] = v1.into();
        assert_eq!([1, 2, 3], t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let t1: [Float; 2] = v1.into();
        assert_eq!([1.0, 2.0], t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let t1: [Float; 3] = v1.into();
        assert_eq!([1.0, 2.0, 3.0], t1);
        assert_eq!(v1, t1.into());
    }
}
