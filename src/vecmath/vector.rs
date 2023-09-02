use super::vec_types::{Vec2f, Vec3f};
use super::{Normal3f, Normal3i, Point2f, Point2i, Point3f, Point3i};
use crate::float::Float;
use crate::impl_unary_op_for_nt;
use crate::newtype_macros::{
    impl_binary_op_assign_for_nt_with_other, impl_binary_op_assign_trait_for_nt,
    impl_binary_op_for_nt_with_other, impl_binary_op_for_other_with_nt,
    impl_binary_op_trait_for_nt,
};
use crate::vecmath::Vector3;
use glam::{IVec2, IVec3};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ---------------------------------------------------------------------------
//        Vector2i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector2i(pub IVec2);

impl Vector2i {
    /// All zeroes.
    pub const ZERO: Self = Self(IVec2::ZERO);

    /// All ones.
    pub const ONE: Self = Self(IVec2::ONE);

    /// All negative ones.
    pub const NEG_ONE: Self = Self(IVec2::NEG_ONE);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self(IVec2::X);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self(IVec2::Y);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(IVec2::NEG_X);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(IVec2::NEG_Y);

    pub const fn new(x: i32, y: i32) -> Self {
        Self(IVec2 { x, y })
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: i32) -> Self {
        Self(IVec2::splat(v))
    }

    pub fn x(&self) -> i32 {
        self.0.x
    }

    pub fn y(&self) -> i32 {
        self.0.y
    }

    /// Compute the dot product.
    pub fn dot(&self, v: &Self) -> i32 {
        self.0.dot(v.0)
    }

    /// Compute the dot product and take the absolute value.
    pub fn abs_dot(&self, v: &Self) -> i32 {
        i32::abs(self.dot(v))
    }
}

impl Default for Vector2i {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Vector2i { fn neg } );
impl_binary_op_trait_for_nt!( impl Add for Vector2i { fn add } );
impl_binary_op_trait_for_nt!( impl Sub for Vector2i { fn sub } );
impl_binary_op_for_nt_with_other!( impl Mul for Vector2i with i32 { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Vector2i with i32 { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for i32 with Vector2i { fn mul } );
impl_binary_op_assign_trait_for_nt!( impl AddAssign for Vector2i { fn add_assign });
impl_binary_op_assign_trait_for_nt!( impl SubAssign for Vector2i { fn sub_assign });
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Vector2i with i32 { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Vector2i with i32 { fn div_assign });

impl From<Point2i> for Vector2i {
    fn from(value: Point2i) -> Self {
        Self(value.0)
    }
}

impl From<[i32; 2]> for Vector2i {
    fn from(value: [i32; 2]) -> Self {
        Self(value.into())
    }
}

impl From<Vector2i> for [i32; 2] {
    fn from(value: Vector2i) -> Self {
        value.0.into()
    }
}

impl From<(i32, i32)> for Vector2i {
    fn from(value: (i32, i32)) -> Self {
        Self(value.into())
    }
}

impl From<Vector2i> for (i32, i32) {
    fn from(value: Vector2i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Vector3i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector3i(pub IVec3);

impl Vector3i {
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

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3 { x, y, z })
    }

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

    /// Compute the dot product
    pub fn dot(&self, v: &Self) -> i32 {
        self.0.dot(v.0)
    }

    /// Dot this vector with a normal.
    pub fn dot_normal(&self, n: &Normal3i) -> i32 {
        self.0.dot(n.0)
    }

    /// Compute the dot product and take the absolute value.
    pub fn abs_dot(&self, v: &Self) -> i32 {
        i32::abs(self.dot(v))
    }

    /// Dot this vector with a normal and take the absolute value.
    pub fn abs_dot_normal(&self, n: &Normal3i) -> i32 {
        i32::abs(self.dot_normal(n))
    }

    /// Take the cross product of this and a vector v
    pub fn cross(&self, v: &Self) -> Self {
        // Integer vectors do not need to use EFT methods for accuracy.
        Self(self.0.cross(v.0))
    }

    /// Take the cross product of this and a normal n
    pub fn cross_normal(&self, n: &Normal3i) -> Self {
        Self(self.0.cross(n.0))
    }
}

impl Default for Vector3i {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Vector3i { fn neg } );
impl_binary_op_trait_for_nt!( impl Add for Vector3i { fn add } );
impl_binary_op_trait_for_nt!( impl Sub for Vector3i { fn sub } );
impl_binary_op_for_nt_with_other!( impl Mul for Vector3i with i32 { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Vector3i with i32 { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for i32 with Vector3i { fn mul } );
impl_binary_op_assign_trait_for_nt!( impl AddAssign for Vector3i { fn add_assign });
impl_binary_op_assign_trait_for_nt!( impl SubAssign for Vector3i { fn sub_assign });
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Vector3i with i32 { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Vector3i with i32 { fn div_assign });

impl From<Point3i> for Vector3i {
    fn from(value: Point3i) -> Self {
        Self(value.0)
    }
}

impl From<Normal3i> for Vector3i {
    fn from(value: Normal3i) -> Self {
        Self(value.0)
    }
}

impl From<[i32; 3]> for Vector3i {
    fn from(value: [i32; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Vector3i> for [i32; 3] {
    fn from(value: Vector3i) -> Self {
        value.0.into()
    }
}

impl From<(i32, i32, i32)> for Vector3i {
    fn from(value: (i32, i32, i32)) -> Self {
        Self(value.into())
    }
}

impl From<Vector3i> for (i32, i32, i32) {
    fn from(value: Vector3i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Vector2f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2f(pub Vec2f);

impl Vector2f {
    /// All zeroes.
    pub const ZERO: Self = Self(Vec2f::ZERO);

    /// All ones.
    pub const ONE: Self = Self(Vec2f::ONE);

    /// All negative ones.
    pub const NEG_ONE: Self = Self(Vec2f::NEG_ONE);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self(Vec2f::X);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self(Vec2f::Y);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(Vec2f::NEG_X);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(Vec2f::NEG_Y);

    pub const fn new(x: Float, y: Float) -> Self {
        Self(Vec2f::new(x, y))
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: Float) -> Self {
        Self(Vec2f::splat(v))
    }

    pub fn x(&self) -> Float {
        self.0.x
    }

    pub fn y(&self) -> Float {
        self.0.y
    }

    pub fn has_nan(&self) -> bool {
        self.0.is_nan()
    }

    pub fn length_squared(&self) -> Float {
        debug_assert!(!self.has_nan());
        self.0.length_squared()
    }

    pub fn length(&self) -> Float {
        debug_assert!(!self.has_nan());
        self.0.length()
    }

    pub fn normalize(&self) -> Self {
        debug_assert!(!self.has_nan());
        Self(self.0.normalize())
    }

    /// Compute the dot product.
    pub fn dot(&self, v: &Self) -> Float {
        debug_assert!(!self.has_nan());
        debug_assert!(!v.has_nan());
        self.0.dot(v.0)
    }

    /// Compute the dot product and take the absolute value.
    pub fn abs_dot(&self, v: &Self) -> Float {
        Float::abs(self.dot(v))
    }
}

impl Default for Vector2f {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Vector2f { fn neg } );
impl_binary_op_trait_for_nt!( impl Add for Vector2f { fn add } );
impl_binary_op_trait_for_nt!( impl Sub for Vector2f { fn sub } );
impl_binary_op_for_nt_with_other!( impl Mul for Vector2f with Float { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Vector2f with Float { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for Float with Vector2f { fn mul } );
impl_binary_op_assign_trait_for_nt!( impl AddAssign for Vector2f { fn add_assign });
impl_binary_op_assign_trait_for_nt!( impl SubAssign for Vector2f { fn sub_assign });
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Vector2f with Float { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Vector2f with Float { fn div_assign });

impl From<Point2f> for Vector2f {
    fn from(value: Point2f) -> Self {
        Self(value.0)
    }
}

impl From<[Float; 2]> for Vector2f {
    fn from(value: [Float; 2]) -> Self {
        Self(value.into())
    }
}

impl From<Vector2f> for [Float; 2] {
    fn from(value: Vector2f) -> Self {
        value.0.into()
    }
}

impl From<(Float, Float)> for Vector2f {
    fn from(value: (Float, Float)) -> Self {
        Self(value.into())
    }
}

impl From<Vector2f> for (Float, Float) {
    fn from(value: Vector2f) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Vector3f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3f(pub Vec3f);

impl Vector3f {
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

    pub const fn new(x: Float, y: Float, z: Float) -> Self {
        Self(Vec3f::new(x, y, z))
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: Float) -> Self {
        Self(Vec3f::splat(v))
    }

    pub fn x(&self) -> Float {
        Vector3::x(self)
    }

    pub fn y(&self) -> Float {
        Vector3::y(self)
    }

    pub fn z(&self) -> Float {
        Vector3::z(self)
    }

    pub fn length(&self) -> Float {
        // To avoid users needing to import the Vector3 trait, which we really only need
        // internal to the vecmath module, we include the length() function here, but
        // just have it call the Trait implementation. Other methods in this and other
        // structs file a similar pattern in this module.
        Vector3::length(self)
    }

    pub fn length_squared(&self) -> Float {
        Vector3::length_squared(self)
    }

    pub fn normalize(&self) -> Self {
        Vector3::normalize(self)
    }

    /// Compute the dot product.
    pub fn dot(&self, v: &Self) -> Float {
        debug_assert!(!self.has_nan());
        debug_assert!(!v.has_nan());
        self.0.dot(v.0)
    }

    /// Dot this vector with a normal.
    pub fn dot_normal(&self, n: &Normal3f) -> Float {
        debug_assert!(!self.has_nan());
        debug_assert!(!n.has_nan());
        self.0.dot(n.0)
    }

    /// Compute the dot product and take the absolute value.
    pub fn abs_dot(&self, v: &Self) -> Float {
        Float::abs(self.dot(v))
    }

    /// Dot this vector with a normal and take its absolute value.
    pub fn abs_dot_normal(&self, n: &Normal3f) -> Float {
        Float::abs(self.dot_normal(n))
    }

    /// Take the cross product of this and a vector v.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    pub fn cross(&self, v: &Self) -> Self {
        super::cross::<Vector3f, Vector3f, Vector3f>(self, v)
    }

    /// Take the cross product of this and a normal n.
    /// Uses an EFT method for calculating the value with minimal error without
    /// casting to f64. See PBRTv4 3.3.2.
    pub fn cross_normal(&self, n: &Normal3f) -> Self {
        super::cross::<Vector3f, Normal3f, Vector3f>(self, n)
    }
}

impl super::Vector3<Float> for Vector3f {
    fn new(x: Float, y: Float, z: Float) -> Self {
        Self::new(x, y, z)
    }

    fn x(&self) -> Float {
        self.0.x
    }

    fn y(&self) -> Float {
        self.0.y
    }

    fn z(&self) -> Float {
        self.0.z
    }
}

impl Default for Vector3f {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Vector3f { fn neg } );
impl_binary_op_trait_for_nt!( impl Add for Vector3f { fn add } );
impl_binary_op_trait_for_nt!( impl Sub for Vector3f { fn sub } );
impl_binary_op_for_nt_with_other!( impl Mul for Vector3f with Float { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Vector3f with Float { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for Float with Vector3f { fn mul } );
impl_binary_op_assign_trait_for_nt!( impl AddAssign for Vector3f { fn add_assign });
impl_binary_op_assign_trait_for_nt!( impl SubAssign for Vector3f { fn sub_assign });
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Vector3f with Float { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Vector3f with Float { fn div_assign });

impl From<Point3f> for Vector3f {
    fn from(value: Point3f) -> Self {
        Self(value.0)
    }
}

impl From<Normal3f> for Vector3f {
    fn from(value: Normal3f) -> Self {
        Self(value.0)
    }
}

impl From<[Float; 3]> for Vector3f {
    fn from(value: [Float; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Vector3f> for [Float; 3] {
    fn from(value: Vector3f) -> Self {
        value.0.into()
    }
}

impl From<(Float, Float, Float)> for Vector3f {
    fn from(value: (Float, Float, Float)) -> Self {
        Self(value.into())
    }
}

impl From<Vector3f> for (Float, Float, Float) {
    fn from(value: Vector3f) -> Self {
        value.0.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::{float::Float, vecmath::Vector3};

    use super::{
        Normal3f, Normal3i, Point2f, Point2i, Point3f, Point3i, Vector2f, Vector2i, Vector3f,
        Vector3i,
    };

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
        // TODO this
    }

    #[test]
    fn vector_normal_angle_between() {
        // TODO this
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
        // TODO
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
