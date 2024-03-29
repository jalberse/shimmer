use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use super::has_nan::HasNan;
use super::length::Length;
use super::length_fns::{length3, length_squared3};
use super::normalize::Normalize;
use super::tuple::{Tuple3, TupleElement};
use super::tuple_fns::{
    abs_dot3, abs_dot3i, angle_between, cross, cross_i32, dot3, dot3i, face_forward, has_nan3,
};
use super::vector::{Vector3, Vector3fi};
use super::{Vector3f, Vector3i};
use crate::float::Float;
use crate::interval::Interval;
use crate::is_nan::IsNan;
use crate::math::lerp;
use auto_ops::*;

pub trait Normal3:
    Tuple3<Self::ElementType>
    + Neg
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self::ElementType, Output = Self>
    + MulAssign<Self::ElementType>
    + Div<Self::ElementType, Output = Self>
    + DivAssign<Self::ElementType>
{
    type ElementType: TupleElement;
    type AssociatedVectorType: Vector3;

    /// Compute the dot product of two normals.
    fn dot(&self, n: Self) -> Self::ElementType;

    /// Compute the dot product with a vector.
    fn dot_vector(&self, v: Self::AssociatedVectorType) -> Self::ElementType;

    /// Compute the dot product of two normals and take the absolute value.
    fn abs_dot(&self, n: Self) -> Self::ElementType;

    /// Compute the dot product with a vector and take the absolute value.
    fn abs_dot_vector(&self, v: Self::AssociatedVectorType) -> Self::ElementType;

    /// Cross this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    fn cross(&self, v: Self::AssociatedVectorType) -> Self::AssociatedVectorType;

    /// Get the angle between this and another normal.
    /// Both must be normalized.
    fn angle_between(&self, n: Self) -> Float;

    /// Get the angle between this normal and a vector.
    fn angle_between_vector(&self, v: Self::AssociatedVectorType) -> Float;

    fn face_forward(&self, n2: Self) -> Self;

    fn face_forward_v(&self, v: Self::AssociatedVectorType) -> Self;
}

// ---------------------------------------------------------------------------
//        Normal3i
// ---------------------------------------------------------------------------

/// Note that Normals are not necessarily normalized.
/// Normals and points cannot be added together, and
/// you cannot take the cross product of two normals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal3i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Normal3i {
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

impl Tuple3<i32> for Normal3i {
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

    fn x_mut(&mut self) -> &mut i32 {
        &mut self.x
    }

    fn y_mut(&mut self) -> &mut i32 {
        &mut self.y
    }

    fn z_mut(&mut self) -> &mut i32 {
        &mut self.z
    }

    fn lerp(t: Float, a: Self, b: Self) -> Self {
        lerp(t, a, b)
    }

    fn x_ref(&self) -> &i32 {
        &self.x
    }

    fn y_ref(&self) -> &i32 {
        &self.y
    }

    fn z_ref(&self) -> &i32 {
        &self.z
    }
}

impl Normal3 for Normal3i {
    type ElementType = i32;
    type AssociatedVectorType = Vector3i;

    /// Compute the dot product of two normals.
    fn dot(&self, n: Self) -> i32 {
        dot3i(*self, n)
    }

    /// Compute the dot product with a vector.
    fn dot_vector(&self, v: Vector3i) -> i32 {
        dot3i(*self, v)
    }

    /// Compute the dot product of two normals and take the absolute value.
    fn abs_dot(&self, n: Self) -> i32 {
        abs_dot3i(*self, n)
    }

    /// Compute the dot product with a vector and take the absolute value.
    fn abs_dot_vector(&self, v: Vector3i) -> i32 {
        abs_dot3i(*self, v)
    }

    /// Cross this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    fn cross(&self, v: Vector3i) -> Vector3i {
        // Note that integer based vectors don't need EFT methods.
        cross_i32(*self, v)
    }

    fn angle_between(&self, n: Normal3i) -> Float {
        angle_between::<Normal3f, Normal3f, Normal3f, Float>(
            Normal3f::from(self).normalize(),
            Normal3f::from(n).normalize(),
        )
    }

    fn angle_between_vector(&self, v: Self::AssociatedVectorType) -> Float {
        angle_between::<Normal3f, Vector3f, Vector3f, Float>(
            Normal3f::from(self).normalize(),
            Vector3f::from(v).normalize(),
        )
    }

    fn face_forward(&self, n2: Self) -> Self {
        face_forward(*self, n2)
    }

    fn face_forward_v(&self, v: Self::AssociatedVectorType) -> Self {
        face_forward(*self, v)
    }
}

impl Index<usize> for Normal3i {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

impl IndexMut<usize> for Normal3i {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

impl HasNan for Normal3i {
    fn has_nan(&self) -> bool {
        false
    }
}

impl Default for Normal3i {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|n: &Normal3i| -> Normal3i {
    Normal3i {
        x: n.x.neg(),
        y: n.y.neg(),
        z: n.z.neg(),
    }
});

// Normals can add and subtract with other normals
impl_op_ex!(+|n1: &Normal3i, n2: &Normal3i| -> Normal3i
{
    Normal3i { x: n1.x + n2.x, y: n1.y + n2.y, z: n1.z + n2.z }
});

impl_op_ex!(+=|n1: &mut Normal3i, n2: &Normal3i|
{
    n1.x += n2.x;
    n1.y += n2.y;
    n1.z += n2.z;
});

impl_op_ex!(-|n1: &Normal3i, n2: &Normal3i| -> Normal3i {
    Normal3i {
        x: n1.x - n2.x,
        y: n1.y - n2.y,
        z: n1.z - n2.z,
    }
});

impl_op_ex!(-=|n1: &mut Normal3i, n2: &Normal3i|
{
    n1.x -= n2.x;
    n1.y -= n2.y;
    n1.z -= n2.z;
});

impl_op_ex_commutative!(*|n: &Normal3i, s: i32| -> Normal3i {
    Normal3i {
        x: n.x * s,
        y: n.y * s,
        z: n.z * s,
    }
});

impl_op_ex_commutative!(*|p: &Normal3i, s: Float| -> Normal3i {
    Normal3i {
        x: (p.x as Float * s) as i32,
        y: (p.y as Float * s) as i32,
        z: (p.z as Float * s) as i32,
    }
});

impl_op_ex!(*=|n1: &mut Normal3i, s: i32|
{
    n1.x *= s;
    n1.y *= s;
    n1.z *= s;
});

impl_op_ex!(/|n: &Normal3i, s: i32| -> Normal3i {
    Normal3i {
        x: n.x / s,
        y: n.y / s,
        z: n.z / s,
    }
});

impl_op_ex!(/=|n1: &mut Normal3i, s: i32|
{
    n1.x /= s;
    n1.y /= s;
    n1.z /= s;
});

impl From<Vector3i> for Normal3i {
    fn from(value: Vector3i) -> Normal3i {
        Normal3i {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<[i32; 3]> for Normal3i {
    fn from(value: [i32; 3]) -> Self {
        Normal3i {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Normal3i> for [i32; 3] {
    fn from(value: Normal3i) -> Self {
        [value.x, value.y, value.z]
    }
}

impl From<(i32, i32, i32)> for Normal3i {
    fn from(value: (i32, i32, i32)) -> Self {
        Normal3i {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl From<Normal3i> for (i32, i32, i32) {
    fn from(value: Normal3i) -> Self {
        (value.x, value.y, value.z)
    }
}

// ---------------------------------------------------------------------------
//        Normal3f
// ---------------------------------------------------------------------------

/// Note that Normals are not necessarily normalized.
/// Normals and points cannot be added together, and
/// you cannot take the cross product of two normals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal3f {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Normal3f {
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

impl Tuple3<Float> for Normal3f {
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

    fn lerp(t: Float, a: Self, b: Self) -> Self {
        lerp(t, a, b)
    }

    fn x_mut(&mut self) -> &mut Float {
        &mut self.x
    }

    fn y_mut(&mut self) -> &mut Float {
        &mut self.y
    }

    fn z_mut(&mut self) -> &mut Float {
        &mut self.z
    }

    fn x_ref(&self) -> &Float {
        &self.x
    }

    fn y_ref(&self) -> &Float {
        &self.y
    }

    fn z_ref(&self) -> &Float {
        &self.z
    }
}

impl Normal3 for Normal3f {
    type ElementType = Float;
    type AssociatedVectorType = Vector3f;

    /// Compute the dot product of two normals.
    fn dot(&self, n: Self) -> Float {
        dot3(*self, n)
    }

    /// Compute the dot with a vector.
    fn dot_vector(&self, v: Vector3f) -> Float {
        dot3(*self, v)
    }

    /// Compute the dot product of two normals and take the absolute value.
    fn abs_dot(&self, n: Self) -> Float {
        abs_dot3(*self, n)
    }

    // TODO Rather than having a *_vector(), we could probably restructure these functions
    // to take a trait and emulate operator overloading in that way. It would simplify the API
    // a bit. This is true for all the types in the vector math module.
    // I was just writing this module after coming back to Rust after quite some time,
    // so I had forgotten some useful patterns like this...

    /// Compute the dot with a vector and take the absolute value.
    fn abs_dot_vector(&self, v: Vector3f) -> Float {
        abs_dot3(*self, v)
    }

    /// Takes the cross of this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    fn cross(&self, v: Vector3f) -> Vector3f {
        cross::<Normal3f, Vector3f, Vector3f, Float>(*self, v)
    }

    /// Get the angle between this and another normal.
    /// Both must be normalized.
    fn angle_between(&self, n: Normal3f) -> Float {
        angle_between::<Normal3f, Normal3f, Normal3f, Float>(*self, n)
    }

    /// Get the angle between this normal and a vector.
    fn angle_between_vector(&self, v: Vector3f) -> Float {
        angle_between::<Normal3f, Vector3f, Vector3f, Float>(*self, v)
    }

    fn face_forward(&self, n2: Self) -> Self {
        face_forward(*self, n2)
    }

    fn face_forward_v(&self, v: Self::AssociatedVectorType) -> Self {
        face_forward(*self, v)
    }
}

impl Index<usize> for Normal3f {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

impl IndexMut<usize> for Normal3f {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

impl HasNan for Normal3f {
    fn has_nan(&self) -> bool {
        has_nan3(*self)
    }
}

impl Length<Float> for Normal3f {
    fn length_squared(&self) -> Float {
        length_squared3(self)
    }

    fn length(&self) -> Float {
        length3(self)
    }
}

impl Normalize<Float> for Normal3f {}

impl Default for Normal3f {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|n: &Normal3f| -> Normal3f { Normal3f::new(-n.x, -n.y, -n.z) });

impl_op_ex!(+ |n1: &Normal3f, n2: &Normal3f| -> Normal3f { Normal3f::new(n1.x + n2.x, n1.y + n2.y, n1.z + n2.z)});

impl_op_ex!(-|n1: &Normal3f, n2: &Normal3f| -> Normal3f {
    Normal3f::new(n1.x - n2.x, n1.y - n2.y, n1.z - n2.z)
});

impl_op_ex!(+= |n1: &mut Normal3f, n2: &Normal3f| {
    n1.x += n2.x;
    n1.y += n2.y;
    n1.z += n2.z;
});

impl_op_ex!(-= |n1: &mut Normal3f, n2: &Normal3f| {
    n1.x -= n2.x;
    n1.y -= n2.y;
    n1.z -= n2.z;
});

impl_op_ex_commutative!(*|n: &Normal3f, s: Float| -> Normal3f {
    Normal3f::new(n.x * s, n.y * s, n.z * s)
});

impl_op_ex!(/ |n: &Normal3f, s: Float| -> Normal3f { Normal3f::new(n.x / s, n.y / s, n.z / s) });

impl_op_ex!(*= |n1: &mut Normal3f, s: Float| {
    n1.x *= s;
    n1.y *= s;
    n1.z *= s;
});

impl_op_ex!(/= |n1: &mut Normal3f, s: Float| {
    n1.x /= s;
    n1.y /= s;
    n1.z /= s;
});

impl_op_ex!(-|n: &Normal3f, v: &Vector3f| -> Vector3f {
    Vector3f::new(n.x - v.x, n.y - v.y, n.z - v.z)
});

impl_op_ex!(-|v: &Vector3f, n: &Normal3f| -> Vector3f {
    Vector3f::new(v.x - n.x, v.y - n.y, v.z - n.z)
});

impl_op_ex_commutative!(+|n: &Normal3f, v: &Vector3f| -> Vector3f {
    Vector3f::new(n.x + v.x, n.y + v.y, n.z + v.z)
});

impl From<Vector3f> for Normal3f {
    fn from(value: Vector3f) -> Normal3f {
        Normal3f::new(value.x, value.y, value.z)
    }
}

impl From<[Float; 3]> for Normal3f {
    fn from(value: [Float; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<Normal3f> for [Float; 3] {
    fn from(value: Normal3f) -> Self {
        [value.x, value.y, value.z]
    }
}

impl From<(Float, Float, Float)> for Normal3f {
    fn from(value: (Float, Float, Float)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<Normal3f> for (Float, Float, Float) {
    fn from(value: Normal3f) -> Self {
        (value.x, value.y, value.z)
    }
}

impl From<Normal3i> for Normal3f {
    fn from(value: Normal3i) -> Self {
        Normal3f {
            x: value.x as Float,
            y: value.y as Float,
            z: value.z as Float,
        }
    }
}

impl From<&Normal3i> for Normal3f {
    fn from(value: &Normal3i) -> Self {
        Normal3f {
            x: value.x as Float,
            y: value.y as Float,
            z: value.z as Float,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Normal3fi {
    x: Interval,
    y: Interval,
    z: Interval,
}

impl Tuple3<Interval> for Normal3fi {
    fn new(x: Interval, y: Interval, z: Interval) -> Self {
        Normal3fi { x, y, z }
    }

    fn x(&self) -> Interval {
        self.x
    }

    fn y(&self) -> Interval {
        self.y
    }

    fn z(&self) -> Interval {
        self.z
    }

    fn x_ref(&self) -> &Interval {
        &self.x
    }

    fn y_ref(&self) -> &Interval {
        &self.y
    }

    fn z_ref(&self) -> &Interval {
        &self.z
    }

    fn x_mut(&mut self) -> &mut Interval {
        &mut self.x
    }

    fn y_mut(&mut self) -> &mut Interval {
        &mut self.y
    }

    fn z_mut(&mut self) -> &mut Interval {
        &mut self.z
    }

    fn lerp(t: Float, a: Self, b: Self) -> Self {
        Normal3fi {
            x: lerp(t, a.x, b.x),
            y: lerp(t, a.y, b.y),
            z: lerp(t, a.z, b.z),
        }
    }
}

impl Normal3 for Normal3fi {
    type ElementType = Interval;

    type AssociatedVectorType = Vector3fi;

    fn dot(&self, n: Self) -> Self::ElementType {
        dot3(*self, n)
    }

    fn dot_vector(&self, v: Self::AssociatedVectorType) -> Self::ElementType {
        dot3(*self, v)
    }

    fn abs_dot(&self, n: Self) -> Self::ElementType {
        abs_dot3(*self, n)
    }

    fn abs_dot_vector(&self, v: Self::AssociatedVectorType) -> Self::ElementType {
        abs_dot3(*self, v)
    }

    fn cross(&self, v: Self::AssociatedVectorType) -> Self::AssociatedVectorType {
        cross::<Normal3fi, Vector3fi, Vector3fi, Interval>(*self, v)
    }

    fn angle_between(&self, n: Self) -> Float {
        angle_between::<Normal3fi, Normal3fi, Normal3fi, Interval>(*self, n)
    }

    fn angle_between_vector(&self, v: Self::AssociatedVectorType) -> Float {
        angle_between::<Normal3fi, Vector3fi, Vector3fi, Interval>(*self, v)
    }

    fn face_forward(&self, n2: Self) -> Self {
        face_forward(*self, n2)
    }

    fn face_forward_v(&self, v: Self::AssociatedVectorType) -> Self {
        face_forward(*self, v)
    }
}

impl Length<Interval> for Normal3fi {
    fn length_squared(&self) -> Interval {
        length_squared3(self).into()
    }

    fn length(&self) -> Interval {
        length3(self).into()
    }
}

impl Index<usize> for Normal3fi {
    type Output = Interval;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index == 0 || index == 1 || index == 2);
        if index == 0 {
            &self.x
        } else if index == 1 {
            &self.y
        } else {
            &self.z
        }
    }
}

impl IndexMut<usize> for Normal3fi {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index == 0 || index == 1 || index == 2);
        if index == 0 {
            &mut self.x
        } else if index == 1 {
            &mut self.y
        } else {
            &mut self.z
        }
    }
}

impl HasNan for Normal3fi {
    fn has_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl_op_ex!(-|n: &Normal3fi| -> Normal3fi { Normal3fi::new(-n.x, -n.y, -n.z) });

impl_op_ex!(+ |n1: &Normal3fi, n2: &Normal3fi| -> Normal3fi { Normal3fi::new(n1.x + n2.x, n1.y + n2.y, n1.z + n2.z)});

impl_op_ex!(-|n1: &Normal3fi, n2: &Normal3fi| -> Normal3fi {
    Normal3fi::new(n1.x - n2.x, n1.y - n2.y, n1.z - n2.z)
});

impl_op_ex!(+= |n1: &mut Normal3fi, n2: &Normal3fi| {
    n1.x += n2.x;
    n1.y += n2.y;
    n1.z += n2.z;
});

impl_op_ex!(-= |n1: &mut Normal3fi, n2: &Normal3fi| {
    n1.x -= n2.x;
    n1.y -= n2.y;
    n1.z -= n2.z;
});

impl_op_ex_commutative!(*|n: &Normal3fi, s: Interval| -> Normal3fi {
    Normal3fi::new(n.x * s, n.y * s, n.z * s)
});

impl_op_ex!(/ |n: &Normal3fi, s: Interval| -> Normal3fi { Normal3fi::new(n.x / s, n.y / s, n.z / s) });

impl_op_ex!(*= |n1: &mut Normal3fi, s: Interval| {
    n1.x *= s;
    n1.y *= s;
    n1.z *= s;
});

impl_op_ex!(/= |n1: &mut Normal3fi, s: Interval| {
    n1.x /= s;
    n1.y /= s;
    n1.z /= s;
});

impl_op_ex!(-|n: &Normal3fi, v: &Vector3fi| -> Vector3fi {
    Vector3fi::new(n.x - v.x(), n.y() - v.y(), n.z - v.z())
});

impl_op_ex!(-|v: &Vector3fi, n: &Normal3fi| -> Vector3fi {
    Vector3fi::new(v.x() - n.x, v.y() - n.y, v.z() - n.z)
});

impl_op_ex_commutative!(+|n: &Normal3fi, v: &Vector3fi| -> Vector3fi {
    Vector3fi::new(n.x + v.x(), n.y + v.y(), n.z + v.z())
});

#[cfg(test)]
mod tests {
    use crate::{
        float::Float,
        vecmath::{normal::Normal3, HasNan, Length, Normalize, Tuple3},
    };

    use super::{Normal3f, Normal3i, Vector3f, Vector3i};

    #[test]
    fn normal_has_nan() {
        let n = Normal3f::new(Float::NAN, 0.0, 0.0);
        assert!(n.has_nan());
    }

    #[test]
    fn normal_negation() {
        let normal = Normal3i::new(1, 2, 3);
        assert_eq!(Normal3i::new(-1, -2, -3), -normal);
        let normal = Normal3f::new(1.0, 2.0, 3.0);
        assert_eq!(Normal3f::new(-1.0, -2.0, -3.0), -normal);
    }

    #[test]
    fn normal_length() {
        let n = Normal3f::new(5.0, 6.0, 7.0);
        assert_eq!(10.488089, n.length());
    }

    #[test]
    fn normal_length_squared() {
        let n = Normal3f::new(5.0, 6.0, 7.0);
        assert_eq!(110.0, n.length_squared());
    }

    #[test]
    fn normal_normalize() {
        let v = Normal3f::new(0.0, 10.0, 0.0);
        assert_eq!(Normal3f::new(0.0, 1.0, 0.0), v.normalize());
    }

    #[test]
    fn normal_normal_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = Normal3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.dot(v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = Normal3i::new(3, 4, 5);
        assert_eq!(14, v1.dot(v2));
    }

    #[test]
    fn normal_vector_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = Vector3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.dot_vector(v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = Vector3i::new(3, 4, 5);
        assert_eq!(14, v1.dot_vector(v2));
    }

    #[test]
    fn normal_normal_abs_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = -Normal3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.abs_dot(v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = -Normal3i::new(3, 4, 5);
        assert_eq!(14, v1.abs_dot(v2));
    }

    #[test]
    fn normal_vector_abs_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = -Vector3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.abs_dot_vector(v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = -Vector3i::new(3, 4, 5);
        assert_eq!(14, v1.abs_dot_vector(v2));
    }

    #[test]
    fn normal_cross_vector() {
        // Note that normals can be crossed with vectors,
        // but you can't cross two normals.

        let n = Normal3i::new(3, -3, 1);
        let v = Vector3i::new(4, 9, 2);
        assert_eq!(Vector3i::new(-15, -2, 39), n.cross(v));

        let n = Normal3f::new(3.0, -3.0, 1.0);
        let v = Vector3f::new(4.0, 9.0, 2.0);
        assert_eq!(Vector3f::new(-15.0, -2.0, 39.0), n.cross(v));
    }

    #[test]
    fn normal_normal_angle_between_i() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2 = Normal3i::new(3, 4, 5);

        assert_eq!(0.18623877, n1.angle_between(n2));
    }

    #[test]
    fn normal_vector_angle_between_i() {
        let n1 = Normal3i::new(1, 2, 3);
        let v2 = Vector3i::new(3, 4, 5);

        assert_eq!(0.18623877, n1.angle_between_vector(v2));
    }

    #[test]
    fn normal_normal_angle_between() {
        let n1 = Normal3f::new(1.0, 2.0, 3.0).normalize();
        let n2 = Normal3f::new(3.0, 4.0, 5.0).normalize();

        assert_eq!(0.18623877, n1.angle_between(n2));
    }

    #[test]
    fn normal_vector_angle_between() {
        let n1 = Normal3f::new(1.0, 2.0, 3.0).normalize();
        let v2 = Vector3f::new(3.0, 4.0, 5.0).normalize();

        assert_eq!(0.18623877, n1.angle_between_vector(v2));
    }

    #[test]
    fn normal_normal_face_forward() {
        // TODO
    }

    #[test]
    fn normal_vector_face_forward() {
        // TODO
    }

    #[test]
    fn normal_assignment_ops() {
        // Normal3i
        // +=
        let mut v1 = Normal3i::new(1, 2, 3);
        let v2 = Normal3i::new(4, 5, 6);
        v1 += v2;
        assert_eq!(Normal3i::new(5, 7, 9), v1);

        // -=
        let mut v1 = Normal3i::new(1, 2, 3);
        let v2 = Normal3i::new(4, 5, 6);
        v1 -= v2;
        assert_eq!(Normal3i::new(-3, -3, -3), v1);

        // *=
        let mut v1 = Normal3i::new(1, 2, 3);
        v1 *= 2;
        assert_eq!(Normal3i::new(2, 4, 6), v1);

        // /=
        let mut v1 = Normal3i::new(1, 2, 3);
        v1 /= 2;
        assert_eq!(Normal3i::new(0, 1, 1), v1);

        // Normal3f
        // +=
        let mut v1 = Normal3f::new(1.0, 2.0, 3.0);
        let v2 = Normal3f::new(4.0, 5.0, 6.0);
        v1 += v2;
        assert_eq!(Normal3f::new(5.0, 7.0, 9.0), v1);

        // -=
        let mut v1 = Normal3f::new(1.0, 2.0, 3.0);
        let v2 = Normal3f::new(4.0, 5.0, 6.0);
        v1 -= v2;
        assert_eq!(Normal3f::new(-3.0, -3.0, -3.0), v1);

        // *=
        let mut v1 = Normal3f::new(1.0, 2.0, 3.0);
        v1 *= 2.0;
        assert_eq!(Normal3f::new(2.0, 4.0, 6.0), v1);

        // /=
        let mut v1 = Normal3f::new(1.0, 2.0, 3.0);
        v1 /= 2.0;
        assert_eq!(Normal3f::new(0.5, 1.0, 1.5), v1);
    }

    #[test]
    fn normal_from_vector() {
        let v1 = Vector3i::new(1, 2, 3);
        let n1 = Normal3i::new(1, 2, 3);
        assert_eq!(n1, v1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let n1 = Normal3f::new(1.0, 2.0, 3.0);
        assert_eq!(n1, v1.into());
    }

    #[test]
    fn normal_from_into_tuples() {
        let v1 = Normal3i::new(1, 2, 3);
        let t1: (i32, i32, i32) = v1.into();
        assert_eq!((1, 2, 3), t1);
        assert_eq!(v1, t1.into());

        let v1 = Normal3f::new(1.0, 2.0, 3.0);
        let t1: (Float, Float, Float) = v1.into();
        assert_eq!((1.0, 2.0, 3.0), t1);
        assert_eq!(v1, t1.into());
    }

    #[test]
    fn normal_from_into_arrays() {
        let v1 = Normal3i::new(1, 2, 3);
        let t1: [i32; 3] = v1.into();
        assert_eq!([1, 2, 3], t1);
        assert_eq!(v1, t1.into());

        let v1 = Normal3f::new(1.0, 2.0, 3.0);
        let t1: [Float; 3] = v1.into();
        assert_eq!([1.0, 2.0, 3.0], t1);
        assert_eq!(v1, t1.into());
    }

    #[test]
    fn normal3f_from_normal3i() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2: Normal3f = n1.into();
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), n2);

        let n3: Normal3f = (&n1).into();
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), n3);
    }

    #[test]
    fn normal3i_div_i32() {
        let n = Normal3i::new(10, 100, 1000);
        assert_eq!(Normal3i::new(5, 50, 500), n / 2);
    }

    #[test]
    fn normal3i_mul_i32() {
        let n = Normal3i::new(10, 100, 1000);
        assert_eq!(Normal3i::new(20, 200, 2000), n * 2);
    }

    #[test]
    fn normal3i_sub_normal3i() {
        let n1 = Normal3i::new(1, 2, 3);
        let n2 = Normal3i::new(11, 12, 13);
        assert_eq!(Normal3i::new(-10, -10, -10), n1 - n2);
    }

    #[test]
    fn normal3i_index() {
        let n = Normal3i::new(1, 2, 3);
        assert_eq!(1, n[0]);
        assert_eq!(2, n[1]);
        assert_eq!(3, n[2]);
    }

    #[test]
    fn normal3i_mut_index() {
        let mut n = Normal3i::new(1, 2, 3);
        n[0] = 11;
        n[1] = 12;
        n[2] = 13;
        assert_eq!(11, n[0]);
        assert_eq!(12, n[1]);
        assert_eq!(13, n[2]);
    }

    #[test]
    fn normal3i_lerp() {
        let n1 = Normal3i::new(0, 100, 1000);
        let n2 = Normal3i::new(10, 200, 2000);
        let n3 = Normal3i::lerp(0.5, n1, n2);
        assert_eq!(5, n3[0]);
        assert_eq!(150, n3[1]);
        assert_eq!(1500, n3[2]);
    }

    #[test]
    fn normal3i_default() {
        let n = Normal3i::default();
        assert_eq!(Normal3i::new(0, 0, 0), n);
    }

    #[test]
    fn normal3f_lerp() {
        let n1 = Normal3f::new(0.0, 100.0, 1000.0);
        let n2 = Normal3f::new(10.0, 200.0, 2000.0);
        let n3 = Normal3f::lerp(0.5, n1, n2);
        assert_eq!(5.0, n3[0]);
        assert_eq!(150.0, n3[1]);
        assert_eq!(1500.0, n3[2]);
    }

    #[test]
    fn normal3f_zero() {
        let n = Normal3f::ZERO;
        assert_eq!(Normal3f::new(0.0, 0.0, 0.0), n);
    }

    #[test]
    fn normal3f_default() {
        let n = Normal3f::default();
        assert_eq!(Normal3f::new(0.0, 0.0, 0.0), n);
    }

    #[test]
    fn normal3f_plus_vector3f() {
        let n = Normal3f::ONE;
        let v = Vector3f::NEG_ONE;
        assert_eq!(Vector3f::ZERO, n + v);
        assert_eq!(Vector3f::ZERO, v + n);
    }

    #[test]
    fn normal3f_index_mut() {
        let mut n = Normal3f::default();
        n[0] = 1.0;
        n[1] = 2.0;
        n[2] = 3.0;
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), n);
    }
}
