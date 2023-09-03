use super::has_nan::{has_nan3, HasNan};
use super::length::{length3, length_squared3, Length};
use super::normalize::Normalize;
use super::{Tuple3, Vector3f, Vector3i};
use crate::float::Float;
use auto_ops::*;

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
    pub const ZERO: Self = Self::splat(0);

    /// All ones.
    pub const ONE: Self = Self::splat(1);

    /// All negative ones.
    pub const NEG_ONE: Self = Self::splat(-1);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self::new(1, 0, 0);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(0, 1, 0);

    /// A unit-length vector pointing along the positive Z axis.
    pub const Z: Self = Self::new(0, 0, 1);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(-1, 0, 0);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(0, -1, 0);

    /// A unit-length vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self::new(0, 0, -1);

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: i32) -> Self {
        Self::new(v, v, v)
    }

    pub fn x(&self) -> i32 {
        Tuple3::x(self)
    }

    pub fn y(&self) -> i32 {
        Tuple3::y(self)
    }

    pub fn z(&self) -> i32 {
        Tuple3::z(self)
    }

    /// Compute the dot product of two normals.
    pub fn dot(&self, n: &Self) -> i32 {
        super::dot3(self, n)
    }

    /// Compute the dot product with a vector.
    pub fn dot_vector(&self, v: &Vector3i) -> i32 {
        super::dot3(self, v)
    }

    /// Compute the dot product of two normals and take the absolute value.
    pub fn abs_dot(&self, n: &Self) -> i32 {
        super::abs_dot3(self, n)
    }

    /// Compute the dot product with a vector and take the absolute value.
    pub fn abs_dot_vector(&self, v: &Vector3i) -> i32 {
        super::abs_dot3(self, v)
    }

    /// Cross this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    pub fn cross(&self, v: &Vector3i) -> Vector3i {
        // Note that integer based vectors don't need EFT methods.
        super::cross_i32(self, v)
    }
}

impl Tuple3<i32> for Normal3i {
    fn new(x: i32, y: i32, z: i32) -> Self {
        Self::new(x, y, z)
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

impl_op_ex!(-|n: Normal3i| -> Normal3i {
    Normal3i {
        x: n.x.neg(),
        y: n.y.neg(),
        z: n.z.neg(),
    }
});

// Normals can add and subtract with other normals
impl_op_ex!(+|n1: Normal3i, n2: Normal3i| -> Normal3i
{
    Normal3i { x: n1.x + n2.y, y: n1.y + n2.y, z: n1.z + n2.z }
});

impl_op_ex!(+=|n1: &mut Normal3i, n2: Normal3i|
{
    n1.x += n2.x;
    n1.y += n2.y;
    n1.z += n2.z;
});

impl_op_ex!(-|n1: Normal3i, n2: Normal3i| -> Normal3i {
    Normal3i {
        x: n1.x - n2.x,
        y: n1.y - n2.y,
        z: n1.z - n2.z,
    }
});

impl_op_ex!(-=|n1: &mut Normal3i, n2: Normal3i|
{
    n1.x -= n2.x;
    n1.y -= n2.y;
    n1.z -= n2.z;
});

impl_op_ex_commutative!(*|n: Normal3i, s: i32| -> Normal3i {
    Normal3i {
        x: n.x * s,
        y: n.y * s,
        z: n.z * s,
    }
});
impl_op_ex!(*=|n1: &mut Normal3i, s: i32|
{
    n1.x *= s;
    n1.y *= s;
    n1.z *= s;
});

impl_op_ex!(/|n: Normal3i, s: i32| -> Normal3i {
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
    pub const ZERO: Self = Self::splat(0.0);

    /// All ones.
    pub const ONE: Self = Self::splat(1.0);

    /// All negative ones.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);

    /// A unit-length vector pointing along the positive Z axis.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0);

    /// A unit-length vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0);

    pub const fn new(x: Float, y: Float, z: Float) -> Self {
        Self { x, y, z }
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: Float) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn x(&self) -> Float {
        Tuple3::x(self)
    }

    pub fn y(&self) -> Float {
        Tuple3::y(self)
    }

    pub fn z(&self) -> Float {
        Tuple3::z(self)
    }

    pub fn has_nan(&self) -> bool {
        HasNan::has_nan(self)
    }

    pub fn length(&self) -> Float {
        Length::length(self)
    }

    pub fn length_squared(&self) -> Float {
        Length::length_squared(self)
    }

    pub fn normalize(self) -> Self {
        Normalize::normalize(self)
    }

    /// Compute the dot product of two normals.
    pub fn dot(&self, n: &Self) -> Float {
        super::dot3(self, n)
    }

    /// Compute the dot with a vector.
    pub fn dot_vector(&self, v: &Vector3f) -> Float {
        super::dot3(self, v)
    }

    /// Compute the dot product of two normals and take the absolute value.
    pub fn abs_dot(&self, n: &Self) -> Float {
        super::abs_dot3(self, n)
    }

    /// Compute the dot with a vector and take the absolute value.
    pub fn abs_dot_vector(&self, v: &Vector3f) -> Float {
        super::abs_dot3(self, v)
    }

    /// Takes the cross of this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    pub fn cross(&self, v: &Vector3f) -> Vector3f {
        super::cross::<Normal3f, Vector3f, Vector3f>(self, v)
    }

    /// Get the angle between this and another normal.
    /// Both must be normalized.
    pub fn angle_between(self, n: Normal3f) -> Float {
        super::angle_between(self, n)
    }

    /// Get the angle between this normal and a vector.
    pub fn angle_between_vector(self, v: Vector3f) -> Float {
        super::angle_between::<Normal3f, Vector3f, Vector3f>(self, v)
    }
}

impl Tuple3<Float> for Normal3f {
    fn new(x: Float, y: Float, z: Float) -> Self {
        Self::new(x, y, z)
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
}

impl HasNan for Normal3f {
    fn has_nan(&self) -> bool {
        has_nan3(self)
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

// Normals can be negated
impl_op_ex!(-|n: Normal3f| -> Normal3f { Normal3f::new(-n.x, -n.y, -n.z) });
// Normals can add and subtract with other normals
impl_op_ex!(+ |n1: Normal3f, n2: Normal3f| -> Normal3f { Normal3f::new(n1.x + n2.x, n1.y + n2.y, n1.z + n2.z)});
impl_op_ex!(-|n1: Normal3f, n2: Normal3f| -> Normal3f {
    Normal3f::new(n1.x - n2.x, n1.y - n2.y, n1.z - n2.z)
});
impl_op_ex!(+= |n1: &mut Normal3f, n2: Normal3f| {
    n1.x += n2.x;
    n1.y += n2.y;
    n1.z += n2.z;
});
impl_op_ex!(-= |n1: &mut Normal3f, n2: Normal3f| {
    n1.x -= n2.x;
    n1.y -= n2.y;
    n1.z -= n2.z;
});

// Normals can be scaled
impl_op_ex_commutative!(*|n: Normal3f, s: Float| -> Normal3f {
    Normal3f::new(n.x * s, n.y * s, n.z * s)
});
impl_op_ex!(/ |n: Normal3f, s: Float| -> Normal3f { Normal3f::new(n.x / s, n.y / s, n.z / s) });
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

// Normals can add and subtract with vectors to create a new vector
impl_op_ex!(-|n: Normal3f, v: Vector3f| -> Vector3f {
    Vector3f::new(n.x - v.x, n.y - v.y, n.z - v.z)
});
impl_op_ex!(-|v: Vector3f, n: Normal3f| -> Vector3f {
    Vector3f::new(v.x - n.x, v.y - n.y, v.z - n.z)
});
impl_op_ex_commutative!(+|n: Normal3f, v: Vector3f| -> Vector3f {
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

#[cfg(test)]
mod tests {
    use crate::float::Float;

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
        assert_eq!(14.0, v1.dot(&v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = Normal3i::new(3, 4, 5);
        assert_eq!(14, v1.dot(&v2));
    }

    #[test]
    fn normal_vector_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = Vector3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.dot_vector(&v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = Vector3i::new(3, 4, 5);
        assert_eq!(14, v1.dot_vector(&v2));
    }

    #[test]
    fn normal_normal_abs_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = -Normal3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.abs_dot(&v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = -Normal3i::new(3, 4, 5);
        assert_eq!(14, v1.abs_dot(&v2));
    }

    #[test]
    fn normal_vector_abs_dot() {
        let v1 = Normal3f::new(0.0, 1.0, 2.0);
        let v2 = -Vector3f::new(3.0, 4.0, 5.0);
        assert_eq!(14.0, v1.abs_dot_vector(&v2));

        let v1 = Normal3i::new(0, 1, 2);
        let v2 = -Vector3i::new(3, 4, 5);
        assert_eq!(14, v1.abs_dot_vector(&v2));
    }

    #[test]
    fn normal_cross_vector() {
        // Note that normals can be crossed with vectors,
        // but you can't cross two normals.

        let n = Normal3i::new(3, -3, 1);
        let v = Vector3i::new(4, 9, 2);
        assert_eq!(Vector3i::new(-15, -2, 39), n.cross(&v));

        let n = Normal3f::new(3.0, -3.0, 1.0);
        let v = Vector3f::new(4.0, 9.0, 2.0);
        assert_eq!(Vector3f::new(-15.0, -2.0, 39.0), n.cross(&v));
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
}
