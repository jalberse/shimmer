use super::vec_types::Vec3f;
use super::{Vector3f, Vector3i};
use crate::float::Float;
use crate::impl_unary_op_for_nt;
use crate::math::difference_of_products;
use crate::newtype_macros::{
    impl_binary_op_assign_for_nt_with_other, impl_binary_op_assign_trait_for_nt,
    impl_binary_op_for_nt_with_other, impl_binary_op_for_other_with_nt,
    impl_binary_op_trait_for_nt,
};
use glam::IVec3;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
//        Normal3i
// ---------------------------------------------------------------------------

/// Note that Normals are not necessarily normalized.
/// Normals and points cannot be added together, and
/// you cannot take the cross product of two normals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal3i(pub IVec3);

impl Normal3i {
    /// All zeroes.
    pub const ZERO: Self = Self(IVec3::ZERO);

    /// All ones.
    pub const ONE: Self = Self(IVec3::ONE);

    /// All negative ones.
    pub const NEG_ONE: Self = Self(IVec3::NEG_ONE);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self(IVec3::X);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self(IVec3::Y);

    /// A unit-length vector pointing along the positive Z axis.
    pub const Z: Self = Self(IVec3::Z);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(IVec3::NEG_X);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(IVec3::NEG_Y);

    /// A unit-length vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self(IVec3::NEG_Z);

    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    #[inline]
    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: i32) -> Self {
        Self(IVec3::splat(v))
    }

    pub fn x(&self) -> i32 {
        self.0.x
    }

    pub fn y(&self) -> i32 {
        self.0.y
    }

    pub fn z(&self) -> i32 {
        self.0.z
    }

    /// Compute the dot product of two normals.
    pub fn dot(self, n: Self) -> i32 {
        self.0.dot(n.0)
    }

    /// Compute the dot product with a vector.
    pub fn dot_vector(self, v: Vector3i) -> i32 {
        self.0.dot(v.0)
    }

    /// Compute the dot product of two normals and take the absolute value.
    pub fn abs_dot(self, n: Self) -> i32 {
        i32::abs(self.dot(n))
    }

    /// Compute the dot product with a vector and take the absolute value.
    pub fn abs_dot_vector(self, v: Vector3i) -> i32 {
        i32::abs(self.dot_vector(v))
    }

    /// Cross this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    pub fn cross(self, v: Vector3i) -> Vector3i {
        // Note that integer based vectors don't need EFT methods.
        Vector3i(self.0.cross(v.0))
    }
}

impl Default for Normal3i {
    fn default() -> Self {
        Self(Default::default())
    }
}

// Normals can add and subtract with other normals
impl_unary_op_for_nt!( impl Neg for Normal3i { fn neg } );
impl_binary_op_trait_for_nt!( impl Add for Normal3i { fn add } );
impl_binary_op_trait_for_nt!( impl Sub for Normal3i { fn sub } );
// Normals can be scaled
impl_binary_op_for_nt_with_other!( impl Mul for Normal3i with i32 { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Normal3i with i32 { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for i32 with Normal3i { fn mul } );
impl_binary_op_assign_trait_for_nt!( impl AddAssign for Normal3i { fn add_assign });
impl_binary_op_assign_trait_for_nt!( impl SubAssign for Normal3i { fn sub_assign });
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Normal3i with i32 { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Normal3i with i32 { fn div_assign });

impl From<Vector3i> for Normal3i {
    fn from(value: Vector3i) -> Normal3i {
        Normal3i(value.0)
    }
}

impl From<[i32; 3]> for Normal3i {
    #[inline]
    fn from(value: [i32; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Normal3i> for [i32; 3] {
    #[inline]
    fn from(value: Normal3i) -> Self {
        value.0.into()
    }
}

impl From<(i32, i32, i32)> for Normal3i {
    #[inline]
    fn from(value: (i32, i32, i32)) -> Self {
        Self(value.into())
    }
}

impl From<Normal3i> for (i32, i32, i32) {
    #[inline]
    fn from(value: Normal3i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Normal3f
// ---------------------------------------------------------------------------

/// Note that Normals are not necessarily normalized.
/// Normals and points cannot be added together, and
/// you cannot take the cross product of two normals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal3f(pub Vec3f);

impl Normal3f {
    /// All zeroes.
    pub const ZERO: Self = Self(Vec3f::ZERO);

    /// All ones.
    pub const ONE: Self = Self(Vec3f::ONE);

    /// All negative ones.
    pub const NEG_ONE: Self = Self(Vec3f::NEG_ONE);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self(Vec3f::X);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self(Vec3f::Y);

    /// A unit-length vector pointing along the positive Z axis.
    pub const Z: Self = Self(Vec3f::Z);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(Vec3f::NEG_X);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(Vec3f::NEG_Y);

    /// A unit-length vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self(Vec3f::NEG_Z);

    #[inline(always)]
    pub const fn new(x: Float, y: Float, z: Float) -> Self {
        Self(Vec3f::new(x, y, z))
    }

    #[inline]
    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: Float) -> Self {
        Self(Vec3f::splat(v))
    }

    pub fn x(&self) -> Float {
        self.0.x
    }

    pub fn y(&self) -> Float {
        self.0.y
    }

    pub fn z(&self) -> Float {
        self.0.z
    }

    pub fn has_nan(self) -> bool {
        self.0.is_nan()
    }

    pub fn length_squared(self) -> Float {
        debug_assert!(!self.has_nan());
        self.0.length_squared()
    }

    pub fn length(self) -> Float {
        debug_assert!(!self.has_nan());
        self.0.length()
    }

    pub fn normalize(self) -> Self {
        debug_assert!(!self.has_nan());
        Self(self.0.normalize())
    }

    /// Compute the dot product of two normals.
    pub fn dot(self, n: Self) -> Float {
        debug_assert!(!self.has_nan());
        debug_assert!(!n.has_nan());
        self.0.dot(n.0)
    }

    /// Compute the dot with a vector.
    pub fn dot_vector(self, v: Vector3f) -> Float {
        debug_assert!(!self.has_nan());
        debug_assert!(!v.has_nan());
        self.0.dot(v.0)
    }

    /// Compute the dot product of two normals and take the absolute value.
    pub fn abs_dot(self, n: Self) -> Float {
        Float::abs(self.dot(n))
    }

    /// Compute the dot with a vector and take the absolute value.
    pub fn abs_dot_vector(self, v: Vector3f) -> Float {
        Float::abs(self.dot_vector(v))
    }

    /// Takes the cross of this normal with a vector.
    /// Note that you cannot take the cross product of two normals.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    pub fn cross(&self, v: &Vector3f) -> Vector3f {
        super::cross::<Normal3f, Vector3f, Vector3f>(self, v)
    }
}

impl super::Vector3<Float> for Normal3f {
    fn new(x: Float, y: Float, z: Float) -> Self {
        Self::new(x, y, z)
    }

    fn x(&self) -> Float {
        self.x()
    }

    fn y(&self) -> Float {
        self.y()
    }

    fn z(&self) -> Float {
        self.z()
    }
}

impl Default for Normal3f {
    fn default() -> Self {
        Self(Default::default())
    }
}

// Normals can add and subtract with other normals
impl_unary_op_for_nt!( impl Neg for Normal3f { fn neg } );
impl_binary_op_trait_for_nt!( impl Add for Normal3f { fn add } );
impl_binary_op_trait_for_nt!( impl Sub for Normal3f { fn sub } );
// Normals can be scaled
impl_binary_op_for_nt_with_other!( impl Mul for Normal3f with Float { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Normal3f with Float { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for Float with Normal3f { fn mul } );
impl_binary_op_assign_trait_for_nt!( impl AddAssign for Normal3f { fn add_assign });
impl_binary_op_assign_trait_for_nt!( impl SubAssign for Normal3f { fn sub_assign });
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Normal3f with Float { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Normal3f with Float { fn div_assign });

impl From<Vector3f> for Normal3f {
    fn from(value: Vector3f) -> Normal3f {
        Normal3f(value.0)
    }
}

impl From<[Float; 3]> for Normal3f {
    #[inline]
    fn from(value: [Float; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Normal3f> for [Float; 3] {
    #[inline]
    fn from(value: Normal3f) -> Self {
        value.0.into()
    }
}

impl From<(Float, Float, Float)> for Normal3f {
    #[inline]
    fn from(value: (Float, Float, Float)) -> Self {
        Self(value.into())
    }
}

impl From<Normal3f> for (Float, Float, Float) {
    #[inline]
    fn from(value: Normal3f) -> Self {
        value.0.into()
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
        assert_eq!(Vector3f::new(-15.0, -2.0, 39.0), n.cross(&v));
    }

    #[test]
    fn normal_normal_angle_between() {
        // TODO this
    }

    #[test]
    fn normal_vector_angle_between() {
        // TODO this
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
