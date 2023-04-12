//! Why are there distinct types for **Vector** and **Point** and
//! **Normal**? Why not just use a unified vector class? Because:
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

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Neg};

use glam::{IVec2, IVec3};

use crate::float::Float;
use crate::impl_unary_op_for_nt;
use crate::newtype_macros::{
    impl_binary_op_assign_for_nt_with_other, impl_binary_op_assign_trait_for_nt,
    impl_binary_op_for_nt_with_other, impl_binary_op_for_other_with_nt,
    impl_binary_op_trait_for_nt,
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

// TODO Get to parity with book - see page 85 for shared fns, and then the other fns mentioned in the book.

// TODO Our cross() implementations:
//   note that f64::mul_add and f32::mul_add are FMA, in std rust!
//   pbrt's FMA actually does use std::fma, which is the C++ equivalent.
//   Their FMA just deals with picking appropriate type.
//   So, our integer structs don't need to bother, use glam's default cross()
//   But for our float structs, create a DifferenceOfProducts() that does that error correction.
//   glam doesn't do this because it's more expensive, and most users of glam probably don't need that precision.

// TODO our AngleBetween() implementation:
//   We *do* want to roll our own here, as Vec3::angle_between() does not do this accuracy fix.

// TODO Ensure things are marked [inline]

// TODO Normal types

// TODO Possibly debug assertions checking for NaN as pbrt does. Obviously requires a nan checking fn

// TODO note I think that Product trait impl from glam types is the HProd from pbrt.

#[cfg(use_f64)]
type Vec2f = glam::DVec2;
#[cfg(not(use_f64))]
type Vec2f = glam::Vec2;
#[cfg(use_f64)]
type Vec3f = glam::DVec3;
#[cfg(not(use_f64))]
type Vec3f = glam::Vec3;

// ---------------------------------------------------------------------------
//        Vector2i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector2i(IVec2);
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

    #[inline(always)]
    pub const fn new(x: i32, y: i32) -> Self {
        Self(IVec2 { x, y })
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    pub const fn splat(v: i32) -> Self {
        Self(IVec2::splat(v))
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
    #[inline]
    fn from(value: Point2i) -> Self {
        Self(value.0)
    }    
}

impl From<[i32; 2]> for Vector2i {
    #[inline]
    fn from(value: [i32; 2]) -> Self {
        Self(value.into())
    }
}

impl From<Vector2i> for [i32; 2] {
    #[inline]
    fn from(value: Vector2i) -> Self {
        value.0.into()
    }
}

impl From<(i32, i32)> for Vector2i {
    #[inline]
    fn from(value: (i32, i32)) -> Self {
        Self(value.into())
    }
}

impl From<Vector2i> for (i32, i32) {
    #[inline]
    fn from(value: Vector2i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Vector3i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector3i(IVec3);

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

    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3 { x, y, z })
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    pub const fn splat(v: i32) -> Self {
        Self(IVec3::splat(v))
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
    #[inline]
    fn from(value: Point3i) -> Self {
        Self(value.0)
    }    
}

impl From<[i32; 3]> for Vector3i {
    #[inline]
    fn from(value: [i32; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Vector3i> for [i32; 3] {
    #[inline]
    fn from(value: Vector3i) -> Self {
        value.0.into()
    }
}


impl From<(i32, i32, i32)> for Vector3i {
    #[inline]
    fn from(value: (i32, i32, i32)) -> Self {
        Self(value.into())
    }
}


impl From<Vector3i> for (i32, i32, i32) {
    #[inline]
    fn from(value: Vector3i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Vector2f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2f(Vec2f);

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

    #[inline(always)]
    pub const fn new(x: Float, y: Float) -> Self {
        Self(Vec2f::new(x, y))
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    pub const fn splat(v: Float) -> Self {
        Self(Vec2f::splat(v))
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
    #[inline]
    fn from(value: Point2f) -> Self {
        Self(value.0)
    }    
}

impl From<[f32; 2]> for Vector2f {
    #[inline]
    fn from(value: [f32; 2]) -> Self {
        Self(value.into())
    }
}

impl From<Vector2f> for [f32; 2] {
    #[inline]
    fn from(value: Vector2f) -> Self {
        value.0.into()
    }
}

impl From<(f32, f32)> for Vector2f {
    #[inline]
    fn from(value: (f32, f32)) -> Self {
        Self(value.into())
    }
}

impl From<Vector2f> for (f32, f32) {
    #[inline]
    fn from(value: Vector2f) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Vector3f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3f(Vec3f);

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

    #[inline(always)]
    pub const fn new(x: Float, y: Float, z: Float) -> Self {
        Self(Vec3f::new(x, y, z))
    }

    /// Creates a vector with all elements set to `v`.
    #[inline(always)]
    pub const fn splat(v: Float) -> Self {
        Self(Vec3f::splat(v))
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
    #[inline]
    fn from(value: Point3f) -> Self {
        Self(value.0)
    }    
}

impl From<[f32; 3]> for Vector3f {
    #[inline]
    fn from(value: [f32; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Vector3f> for [f32; 3] {
    #[inline]
    fn from(value: Vector3f) -> Self {
        value.0.into()
    }
}

impl From<(f32, f32, f32)> for Vector3f {
    #[inline]
    fn from(value: (f32, f32, f32)) -> Self {
        Self(value.into())
    }
}

impl From<Vector3f> for (f32, f32, f32) {
    #[inline]
    fn from(value: Vector3f) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Point2i
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point2i(IVec2);

impl Point2i {
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

    #[inline(always)]
    pub const fn new(x: i32, y: i32) -> Self {
        Self(IVec2::new(x, y))
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    pub const fn splat(v: i32) -> Self {
        Self(IVec2::splat(v))
    }
}

impl Default for Point2i {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Point2i { fn neg } );
impl_binary_op_for_nt_with_other!( impl Mul for Point2i with i32 { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Point2i with i32 { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for i32 with Point2i { fn mul } );
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Point2i with i32 { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Point2i with i32 { fn div_assign });

// Point + Vector -> Point
impl Add<Vector2i> for Point2i {
    type Output = Point2i;
    #[inline]
    fn add(self, rhs: Vector2i) -> Point2i {
        Point2i(self.0 + rhs.0)
    }
}
// Vector + Point -> Point
impl Add<Point2i> for Vector2i {
    type Output = Point2i;
    #[inline]
    fn add(self, rhs: Point2i) -> Point2i {
        Point2i(self.0 + rhs.0)
    }
}

// Point += Vector
impl AddAssign<Vector2i> for Point2i {
    #[inline]
    fn add_assign(&mut self, rhs: Vector2i) {
        self.0 += rhs.0;
    }
}

// Point - Vector -> Point
impl Sub<Vector2i> for Point2i {
    type Output = Point2i;
    #[inline]
    fn sub(self, rhs: Vector2i) -> Point2i {
        Point2i(self.0 - rhs.0)
    }
}

// Point -= Vector
impl SubAssign<Vector2i> for Point2i {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector2i) {
        self.0 -= rhs.0;
    }
}

// Point - Point -> Vector
impl Sub<Point2i> for Point2i {
    type Output = Vector2i;
    #[inline]
    fn sub(self, rhs: Point2i) -> Vector2i {
        Vector2i(self.0 - rhs.0)
    }
}

impl From<Vector2i> for Point2i {
    #[inline]
    fn from(value: Vector2i) -> Self {
        Point2i(value.0)
    }
}

impl From<[i32; 2]> for Point2i {
    #[inline]
    fn from(value: [i32; 2]) -> Self {
        Self(value.into())
    }
}

impl From<Point2i> for [i32; 2] {
    #[inline]
    fn from(value: Point2i) -> Self {
        value.0.into()
    }
}

impl From<(i32, i32)> for Point2i {
    #[inline]
    fn from(value: (i32, i32)) -> Self {
        Self(value.into())
    }
}

impl From<Point2i> for (i32, i32) {
    #[inline]
    fn from(value: Point2i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Point3i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point3i(IVec3);

impl Point3i {
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

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: i32) -> Self {
        Self(IVec3::splat(v))
    }
}

impl Default for Point3i {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Point3i { fn neg } );
impl_binary_op_for_nt_with_other!( impl Mul for Point3i with i32 { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Point3i with i32 { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for i32 with Point3i { fn mul } );
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Point3i with i32 { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Point3i with i32 { fn div_assign });

// Point + Vector -> Point
impl Add<Vector3i> for Point3i {
    type Output = Point3i;
    #[inline]
    fn add(self, rhs: Vector3i) -> Point3i {
        Point3i(self.0 + rhs.0)
    }
}

// Vector + Point -> Point
impl Add<Point3i> for Vector3i {
    type Output = Point3i;
    #[inline]
    fn add(self, rhs: Point3i) -> Point3i {
        Point3i(self.0 + rhs.0)
    }
}

// Point += Vector
impl AddAssign<Vector3i> for Point3i {
    #[inline]
    fn add_assign(&mut self, rhs: Vector3i) {
        self.0 += rhs.0;
    }
}

// Point - Vector -> Point
impl Sub<Vector3i> for Point3i {
    type Output = Point3i;
    #[inline]
    fn sub(self, rhs: Vector3i) -> Point3i {
        Point3i(self.0 - rhs.0)
    }
}

// Point -= Vector
impl SubAssign<Vector3i> for Point3i {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector3i) {
        self.0 -= rhs.0;
    }
}

// Point - Point -> Vector
impl Sub<Point3i> for Point3i {
    type Output = Vector3i;
    #[inline]
    fn sub(self, rhs: Point3i) -> Vector3i {
        Vector3i(self.0 - rhs.0)
    }
}

impl From<Vector3i> for Point3i {
    #[inline]
    fn from(value: Vector3i) -> Self {
        Point3i(value.0)
    }
}

impl From<[i32; 3]> for Point3i {
    #[inline]
    fn from(value: [i32; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Point3i> for [i32; 3] {
    #[inline]
    fn from(value: Point3i) -> Self {
        value.0.into()
    }
}

impl From<(i32, i32, i32)> for Point3i {
    #[inline]
    fn from(value: (i32, i32, i32)) -> Self {
        Self(value.into())
    }
}

impl From<Point3i> for (i32, i32, i32) {
    #[inline]
    fn from(value: Point3i) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Point2f
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2f(Vec2f);

impl Point2f {
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

    #[inline(always)]
    pub const fn new(x: Float, y: Float) -> Self {
        Self(Vec2f::new(x, y))
    }

    /// Creates a vector with all elements set to `v`.
    #[inline]
    pub const fn splat(v: Float) -> Self {
        Self(Vec2f::splat(v))
    }
}

impl Default for Point2f {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Point2f { fn neg } );
impl_binary_op_for_nt_with_other!( impl Mul for Point2f with Float { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Point2f with Float { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for Float with Point2f { fn mul } );
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Point2f with Float { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Point2f with Float { fn div_assign });

// Point + Vector -> Point
impl Add<Vector2f> for Point2f {
    type Output = Point2f;
    #[inline]
    fn add(self, rhs: Vector2f) -> Point2f {
        Point2f(self.0 + rhs.0)
    }
}

// Vector + Point -> Point
impl Add<Point2f> for Vector2f {
    type Output = Point2f;
    #[inline]
    fn add(self, rhs: Point2f) -> Point2f {
        Point2f(self.0 + rhs.0)
    }
}

// Point += Vector
impl AddAssign<Vector2f> for Point2f {
    #[inline]
    fn add_assign(&mut self, rhs: Vector2f) {
        self.0 += rhs.0;
    }
}

// Point - Vector -> Point
impl Sub<Vector2f> for Point2f {
    type Output = Point2f;
    #[inline]
    fn sub(self, rhs: Vector2f) -> Point2f {
        Point2f(self.0 - rhs.0)
    }
}

// Point -= Vector
impl SubAssign<Vector2f> for Point2f {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector2f) {
        self.0 -= rhs.0;
    }
}

// Point - Point -> Vector
impl Sub<Point2f> for Point2f {
    type Output = Vector2f;
    #[inline]
    fn sub(self, rhs: Point2f) -> Vector2f {
        Vector2f(self.0 - rhs.0)
    }
}

impl From<Vector2f> for Point2f {
    fn from(value: Vector2f) -> Self {
        Point2f(value.0)
    }
}

impl From<[f32; 2]> for Point2f {
    #[inline]
    fn from(value: [f32; 2]) -> Self {
        Self(value.into())
    }
}

impl From<Point2f> for [f32; 2] {
    #[inline]
    fn from(value: Point2f) -> Self {
        value.0.into()
    }
}

impl From<(f32, f32)> for Point2f {
    #[inline]
    fn from(value: (f32, f32)) -> Self {
        Self(value.into())
    }
}

impl From<Point2f> for (f32, f32) {
    #[inline]
    fn from(value: Point2f) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
//        Point3f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3f(Vec3f);

impl Point3f {
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

    /// Creates a vector with all elements set to `v`.
    #[inline]
    pub const fn splat(v: Float) -> Self {
        Self(Vec3f::splat(v))
    }
}
impl Default for Point3f {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl_unary_op_for_nt!( impl Neg for Point3f { fn neg } );
impl_binary_op_for_nt_with_other!( impl Mul for Point3f with Float { fn mul } );
impl_binary_op_for_nt_with_other!( impl Div for Point3f with Float { fn div } );
impl_binary_op_for_other_with_nt!( impl Mul for Float with Point3f { fn mul } );
impl_binary_op_assign_for_nt_with_other!( impl MulAssign for Point3f with Float { fn mul_assign });
impl_binary_op_assign_for_nt_with_other!( impl DivAssign for Point3f with Float { fn div_assign });

// Point + Vector -> Point
impl Add<Vector3f> for Point3f {
    type Output = Point3f;
    #[inline]
    fn add(self, rhs: Vector3f) -> Point3f {
        Point3f(self.0 + rhs.0)
    }
}

// Vector + Point -> Point
impl Add<Point3f> for Vector3f {
    type Output = Point3f;
    #[inline]
    fn add(self, rhs: Point3f) -> Point3f {
        Point3f(self.0 + rhs.0)
    }
}

// Point += Vector
impl AddAssign<Vector3f> for Point3f {
    #[inline]
    fn add_assign(&mut self, rhs: Vector3f) {
        self.0 += rhs.0;
    }
}

// Point - Vector -> Point
impl Sub<Vector3f> for Point3f {
    type Output = Point3f;
    #[inline]
    fn sub(self, rhs: Vector3f) -> Point3f {
        Point3f(self.0 - rhs.0)
    }
}

// Point -= Vector
impl SubAssign<Vector3f> for Point3f {
    #[inline]
    fn sub_assign(&mut self, rhs: Vector3f) {
        self.0 -= rhs.0;
    }
}

// Point - Point -> Vector
impl Sub<Point3f> for Point3f {
    type Output = Vector3f;
    #[inline]
    fn sub(self, rhs: Point3f) -> Vector3f {
        Vector3f(self.0 - rhs.0)
    }
}

impl From<Vector3f> for Point3f {
    fn from(value: Vector3f) -> Self {
        Point3f(value.0)
    }
}

impl From<[f32; 3]> for Point3f {
    #[inline]
    fn from(value: [f32; 3]) -> Self {
        Self(value.into())
    }
}

impl From<Point3f> for [f32; 3] {
    #[inline]
    fn from(value: Point3f) -> Self {
        value.0.into()
    }
}

impl From<(f32, f32, f32)> for Point3f {
    #[inline]
    fn from(value: (f32, f32, f32)) -> Self {
        Self(value.into())
    }
}

impl From<Point3f> for (f32, f32, f32) {
    #[inline]
    fn from(value: Point3f) -> Self {
        value.0.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::vecmath::{Vector3i, Point3i, Point2f, Vector3f, Point3f};

    use super::{Vector2f, Vector2i, Point2i};

    #[test]
    fn unary_ops() {
        let vec = Vector2i::new(1, 2);
        assert_eq!(Vector2i::new(-1, -2), -vec);
        let vec = Vector3i::new(1, 2, 3);
        assert_eq!(Vector3i::new(-1, -2, -3), -vec);
        let vec = Point2i::new(1, 2);
        assert_eq!(Point2i::new(-1, -2), -vec);
        let vec = Point3i::new(1, 2, 3);
        assert_eq!(Point3i::new(-1, -2, -3), -vec);

        let vec = Vector2f::new(1.0, 2.0);
        assert_eq!(Vector2f::new(-1.0, -2.0), -vec);
        let vec = Vector3f::new(1.0, 2.0, 3.0);
        assert_eq!(Vector3f::new(-1.0, -2.0, -3.0), -vec);
        let vec = Point2f::new(1.0, 2.0);
        assert_eq!(Point2f::new(-1.0, -2.0), -vec);
        let vec = Point3f::new(1.0, 2.0, 3.0);
        assert_eq!(Point3f::new(-1.0, -2.0, -3.0), -vec);
    }

    #[test]
    fn vector_binary_ops() {
        let vec = Vector2i::new(-2, 10);
        assert_eq!(Vector2i::new(-4, 20), vec + vec);
        assert_eq!(Vector2i::new(0, 0), vec - vec);
        assert_eq!(Vector2i::new(-6, 30), vec * 3);
        assert_eq!(Vector2i::new(-6, 30), 3 * vec);
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
    fn point_binary_ops() {
        let point = Point2i::new(-2, 10);
        assert_eq!(Vector2i::new(0, 0), point - point);
        assert_eq!(Point2i::new(-6, 30), point * 3);
        assert_eq!(Point2i::new(-6, 30), 3 * point);
        assert_eq!(Point2i::new(-1, 5), point / 2);
        let vec = Vector2i::new(1, 0);
        assert_eq!(Point2i::new(-1, 10), point + vec);
        assert_eq!(Point2i::new(-1, 10), vec + point);
        assert_eq!(Point2i::new(-3, 10), point - vec);

        let point = Point3i::new(-2, 10, 20);
        assert_eq!(Vector3i::new(0, 0, 0), point - point);
        assert_eq!(Point3i::new(-6, 30, 60), point * 3);
        assert_eq!(Point3i::new(-6, 30, 60), 3 * point);
        assert_eq!(Point3i::new(-1, 5, 10), point / 2);
        let vec = Vector3i::new(1, 0, 0);
        assert_eq!(Point3i::new(-1, 10, 20), point + vec);
        assert_eq!(Point3i::new(-1, 10, 20), vec + point);
        assert_eq!(Point3i::new(-3, 10, 20), point - vec);

        let point = Point2f::new(-1.0, 10.0);
        assert_eq!(Vector2f::new(0.0, 0.0), point - point);
        assert_eq!(Point2f::new(-3.0, 30.0), point * 3.0);
        assert_eq!(Point2f::new(-3.0, 30.0), 3.0 * point);
        assert_eq!(Point2f::new(-0.5, 5.0), point / 2.0);
        let vec = Vector2f::new(1.0, 0.0);
        assert_eq!(Point2f::new(0.0, 10.0), point + vec);
        assert_eq!(Point2f::new(0.0, 10.0), vec + point);
        assert_eq!(Point2f::new(-2.0, 10.0), point - vec);

        let point = Point3f::new(-1.0, 10.0, 20.0);
        assert_eq!(Vector3f::new(0.0, 0.0, 0.0), point - point);
        assert_eq!(Point3f::new(-3.0, 30.0, 60.0), point * 3.0);
        assert_eq!(Point3f::new(-3.0, 30.0, 60.0), 3.0 * point);
        assert_eq!(Point3f::new(-0.5, 5.0, 10.0), point / 2.0);
        let vec = Vector3f::new(1.0, 0.0, 0.0);
        assert_eq!(Point3f::new(0.0, 10.0, 20.0), point + vec);
        assert_eq!(Point3f::new(0.0, 10.0, 20.0), vec + point);
        assert_eq!(Point3f::new(-2.0, 10.0, 20.0), point - vec);
    }

    #[test]
    fn point_assignment_ops() {

        // *=
        let mut p1 = Point2i::new(1, 2);
        p1 *= 2;
        assert_eq!(Point2i::new(2, 4), p1);
        
        // /=
        let mut p1 = Point2i::new(1, 2);
        p1 /= 2;
        assert_eq!(Point2i::new(0, 1), p1);

        // *=
        let mut p1 = Point3i::new(1, 2, 3);
        p1 *= 2;
        assert_eq!(Point3i::new(2, 4, 6), p1);
        
        // /=
        let mut p1 = Point3i::new(1, 2, 3);
        p1 /= 2;
        assert_eq!(Point3i::new(0, 1, 1), p1);

        // *=
        let mut p1 = Point2f::new(1.0, 2.0);
        p1 *= 2.0;
        assert_eq!(Point2f::new(2.0, 4.0), p1);
        
        // /=
        let mut p1 = Point2f::new(1.0, 2.0);
        p1 /= 2.0;
        assert_eq!(Point2f::new(0.5, 1.0), p1);

        // *=
        let mut p1 = Point3f::new(1.0, 2.0, 3.0);
        p1 *= 2.0;
        assert_eq!(Point3f::new(2.0, 4.0, 6.0), p1);
        
        // /=
        let mut p1 = Point3f::new(1.0, 2.0, 3.0);
        p1 /= 2.0;
        assert_eq!(Point3f::new(0.5, 1.0, 1.5), p1);

        // Point += Vector
        let mut p = Point2i::new(1, 2);
        let v = Vector2i::new(1, 2);
        p += v;
        assert_eq!(Point2i::new(2, 4), p);

        let mut p = Point3i::new(1, 2, 3);
        let v = Vector3i::new(1, 2, 3);
        p += v;
        assert_eq!(Point3i::new(2, 4, 6), p);

        let mut p = Point2f::new(1.0, 2.0);
        let v = Vector2f::new(1.0, 2.0);
        p += v;
        assert_eq!(Point2f::new(2.0, 4.0), p);

        let mut p = Point3f::new(1.0, 2.0, 3.0);
        let v = Vector3f::new(1.0, 2.0, 3.0);
        p += v;
        assert_eq!(Point3f::new(2.0, 4.0, 6.0), p);

        // Point -= Vector
        let mut p = Point2i::new(1, 2);
        let v = Vector2i::new(1, 2);
        p -= v;
        assert_eq!(Point2i::new(0, 0), p);

        let mut p = Point3i::new(1, 2, 3);
        let v = Vector3i::new(1, 2, 3);
        p -= v;
        assert_eq!(Point3i::new(0, 0, 0), p);

        let mut p = Point2f::new(1.0, 2.0);
        let v = Vector2f::new(1.0, 2.0);
        p -= v;
        assert_eq!(Point2f::new(0.0, 0.0), p);

        let mut p = Point3f::new(1.0, 2.0, 3.0);
        let v = Vector3f::new(1.0, 2.0, 3.0);
        p -= v;
        assert_eq!(Point3f::new(0.0, 0.0, 0.0), p);
    }

    #[test]
    fn from_into() {
        // Vec-Point Conversions
        let v1 = Vector2i::new(1, 2);
        let p1: Point2i = v1.into();
        assert_eq!(Point2i::new(1, 2), p1);
        assert_eq!(v1, p1.into());

        let v1 = Vector3i::new(1, 2, 3);
        let p1: Point3i = v1.into();
        assert_eq!(Point3i::new(1, 2, 3), p1);
        assert_eq!(v1, p1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let p1: Point2f = v1.into();
        assert_eq!(Point2f::new(1.0, 2.0), p1);
        assert_eq!(v1, p1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let p1: Point3f = v1.into();
        assert_eq!(Point3f::new(1.0, 2.0, 3.0), p1);
        assert_eq!(v1, p1.into());

        // Tuple-Vec Conversions
        let v1 = Vector2i::new(1, 2);
        let t1: (i32, i32) = v1.into();
        assert_eq!((1, 2), t1);
        assert_eq!(v1, t1.into());
        
        let v1 = Vector3i::new(1, 2, 3);
        let t1: (i32, i32, i32) = v1.into();
        assert_eq!((1, 2, 3), t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let t1: (f32, f32) = v1.into();
        assert_eq!((1.0, 2.0), t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let t1: (f32, f32, f32) = v1.into();
        assert_eq!((1.0, 2.0, 3.0), t1);
        assert_eq!(v1, t1.into());

        // Vec-Array Conversions
        let v1 = Vector2i::new(1, 2);
        let t1: [i32; 2] = v1.into();
        assert_eq!([1, 2], t1);
        assert_eq!(v1, t1.into());
        
        let v1 = Vector3i::new(1, 2, 3);
        let t1: [i32; 3] = v1.into();
        assert_eq!([1, 2, 3], t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let t1: [f32; 2] = v1.into();
        assert_eq!([1.0, 2.0], t1);
        assert_eq!(v1, t1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let t1: [f32; 3] = v1.into();
        assert_eq!([1.0, 2.0, 3.0], t1);
        assert_eq!(v1, t1.into());

        // Point-Tuple Conversions
        let v1 = Point2i::new(1, 2);
        let t1: (i32, i32) = v1.into();
        assert_eq!((1, 2), t1);
        assert_eq!(v1, t1.into());
        
        let v1 = Point3i::new(1, 2, 3);
        let t1: (i32, i32, i32) = v1.into();
        assert_eq!((1, 2, 3), t1);
        assert_eq!(v1, t1.into());

        let v1 = Point2f::new(1.0, 2.0);
        let t1: (f32, f32) = v1.into();
        assert_eq!((1.0, 2.0), t1);
        assert_eq!(v1, t1.into());

        let v1 = Point3f::new(1.0, 2.0, 3.0);
        let t1: (f32, f32, f32) = v1.into();
        assert_eq!((1.0, 2.0, 3.0), t1);
        assert_eq!(v1, t1.into());

        // Point-Array Conversions
        let v1 = Point2i::new(1, 2);
        let t1: [i32; 2] = v1.into();
        assert_eq!([1, 2], t1);
        assert_eq!(v1, t1.into());
        
        let v1 = Point3i::new(1, 2, 3);
        let t1: [i32; 3] = v1.into();
        assert_eq!([1, 2, 3], t1);
        assert_eq!(v1, t1.into());

        let v1 = Point2f::new(1.0, 2.0);
        let t1: [f32; 2] = v1.into();
        assert_eq!([1.0, 2.0], t1);
        assert_eq!(v1, t1.into());

        let v1 = Point3f::new(1.0, 2.0, 3.0);
        let t1: [f32; 3] = v1.into();
        assert_eq!([1.0, 2.0, 3.0], t1);
        assert_eq!(v1, t1.into());

        // TODO when we add Normal types, do those conversion too
    }

    // TODO More and more tests! For each type! For each fn! Yippee!
}
