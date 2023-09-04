use super::has_nan::HasNan;
use super::tuple::{Tuple2, Tuple3};
use super::tuple_fns::{has_nan2, has_nan3};
use super::{Vector2f, Vector2i, Vector3f, Vector3i};
use crate::float::Float;
use crate::math::{self, lerp};
use crate::vecmath::Length;
use auto_ops::{impl_op_ex, impl_op_ex_commutative};

// ---------------------------------------------------------------------------
//        Point2i
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point2i {
    pub x: i32,
    pub y: i32,
}

impl Point2i {
    /// All zeroes.
    pub const ZERO: Self = Self::splat(0);

    /// All ones.
    pub const ONE: Self = Self::splat(1);

    /// All negative ones.
    pub const NEG_ONE: Self = Self::splat(-1);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self::new(1, 0);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(0, 1);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(-1, 0);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(0, -1);

    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: i32) -> Self {
        Self::new(v, v)
    }
}

impl Default for Point2i {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|p: Point2i| -> Point2i {
    Point2i {
        x: p.x.neg(),
        y: p.y.neg(),
    }
});

impl_op_ex_commutative!(*|v: Point2i, s: i32| -> Point2i {
    Point2i {
        x: v.x * s,
        y: v.y * s,
    }
});

impl_op_ex!(/|v: Point2i, s: i32| -> Point2i
{
    Point2i { x: v.x / s, y: v.y / s }
});

impl_op_ex!(*=|p1: &mut Point2i, p2: Point2i|
{
    p1.x *= p2.x;
    p1.y *= p2.y;
});

impl_op_ex!(/=|p1: &mut Point2i, p2: Point2i|
{
    p1.x /= p2.x;
    p1.y /= p2.y;
});

impl_op_ex!(*=|p1: &mut Point2i, s: i32|
{
    p1.x *= s;
    p1.y *= s;
});

impl_op_ex!(/=|p1: &mut Point2i, s: i32|
{
    p1.x /= s;
    p1.y /= s;
});

// Point + Vector -> Point
impl_op_ex_commutative!(+|p: Point2i, v: Vector2i| -> Point2i
{
    Point2i { x: v.x + p.x, y: v.y + p.y }
});

impl_op_ex!(+=|p: &mut Point2i, v: Vector2i|
{
    p.x += v.x;
    p.y += v.y;
});

// Point - Vector -> Point
impl_op_ex_commutative!(-|p: Point2i, v: Vector2i| -> Point2i {
    Point2i {
        x: p.x - v.x,
        y: p.y - v.y,
    }
});

impl_op_ex!(-=|p: &mut Point2i, v: Vector2i|
{
    p.x -= v.x;
    p.y -= v.y;
});

// Point - Point -> Vector
impl_op_ex!(-|p1: Point2i, p2: Point2i| -> Vector2i {
    Vector2i {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
    }
});

impl From<Vector2i> for Point2i {
    fn from(value: Vector2i) -> Self {
        Point2i {
            x: value.x,
            y: value.y,
        }
    }
}

impl From<[i32; 2]> for Point2i {
    fn from(value: [i32; 2]) -> Self {
        Point2i {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<Point2i> for [i32; 2] {
    fn from(value: Point2i) -> Self {
        [value.x, value.y]
    }
}

impl From<(i32, i32)> for Point2i {
    fn from(value: (i32, i32)) -> Self {
        Point2i {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<Point2i> for (i32, i32) {
    fn from(value: Point2i) -> Self {
        (value.x, value.y)
    }
}

// ---------------------------------------------------------------------------
//        Point3i
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point3i {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Point3i {
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
}

impl Tuple3<i32> for Point3i {
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

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        Point3i {
            x: math::lerp(t, &(a.x as Float), &(b.x as Float)) as i32,
            y: math::lerp(t, &(a.y as Float), &(b.y as Float)) as i32,
            z: math::lerp(t, &(a.z as Float), &(b.z as Float)) as i32,
        }
    }
}

impl Default for Point3i {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|p: &Point3i| -> Point3i {
    Point3i {
        x: p.x.neg(),
        y: p.y.neg(),
        z: p.z.neg(),
    }
});

impl_op_ex_commutative!(*|p: &Point3i, s: i32| -> Point3i {
    Point3i {
        x: p.x * s,
        y: p.y * s,
        z: p.z * s,
    }
});

impl_op_ex_commutative!(*|p: &Point3i, s: Float| -> Point3i {
    Point3i {
        x: (p.x as Float * s) as i32,
        y: (p.y as Float * s) as i32,
        z: (p.z as Float * s) as i32,
    }
});

impl_op_ex!(*=|p: &mut Point3i, s: i32|
{
    p.x *= s;
    p.y *= s;
    p.z *= s;
});

impl_op_ex!(/|p: &Point3i, s: i32| -> Point3i
{
    Point3i {
        x: p.x / s,
        y: p.y / s,
        z: p.z / s,
    }
});
impl_op_ex!(/=|p: &mut Point3i, s: i32|
{
    p.x /= s;
    p.y /= s;
    p.z /= s;
});

// Point + Vector -> Point
impl_op_ex_commutative!(+|p: &Point3i, v: &Vector3i| -> Point3i
{
    Point3i { x: p.x + v.x, y: p.y + v.y, z: p.z + v.z }
});

impl_op_ex!(+=|p: &mut Point3i, v: &Vector3i|
{
    p.x += v.x;
    p.y += v.y;
    p.z += v.z;
});

impl_op_ex!(-|p: &Point3i, v: &Vector3i| -> Point3i {
    Point3i {
        x: p.x - v.x,
        y: p.y - v.y,
        z: p.z - v.z,
    }
});

impl_op_ex!(-=|p: &mut Point3i, v: &Vector3i|
{
    p.x -= v.x;
    p.y -= v.y;
    p.z -= v.z;
});

// Point - Point -> Vector
impl_op_ex!(-|p1: &Point3i, p2: &Point3i| -> Vector3i {
    Vector3i {
        x: p1.x - p2.x,
        y: p1.y - p2.y,
        z: p1.z - p2.z,
    }
});

impl From<Vector3i> for Point3i {
    fn from(value: Vector3i) -> Self {
        Point3i {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl From<[i32; 3]> for Point3i {
    fn from(value: [i32; 3]) -> Self {
        Point3i {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }
}

impl From<Point3i> for [i32; 3] {
    fn from(value: Point3i) -> Self {
        [value.x, value.y, value.z]
    }
}

impl From<(i32, i32, i32)> for Point3i {
    fn from(value: (i32, i32, i32)) -> Self {
        Point3i {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl From<Point3i> for (i32, i32, i32) {
    fn from(value: Point3i) -> Self {
        (value.x, value.y, value.z)
    }
}

// ---------------------------------------------------------------------------
//        Point2f
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2f {
    pub x: Float,
    pub y: Float,
}

impl Point2f {
    /// All zeroes.
    pub const ZERO: Self = Self::splat(0.0);

    /// All ones.
    pub const ONE: Self = Self::splat(1.0);

    /// All negative ones.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A unit-length vector pointing along the positive X axis.
    pub const X: Self = Self::new(1.0, 0.0);

    /// A unit-length vector pointing along the positive Y axis.
    pub const Y: Self = Self::new(0.0, 1.0);

    /// A unit-length vector pointing along the negative X axis.
    pub const NEG_X: Self = Self::new(-1.0, 0.0);

    /// A unit-length vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self::new(0.0, -1.0);

    pub const fn new(x: Float, y: Float) -> Self {
        Self { x, y }
    }

    /// Creates a vector with all elements set to `v`.
    pub const fn splat(v: Float) -> Self {
        Self { x: v, y: v }
    }

    // TODO we can make a Distance trait that Points can implement.
    pub fn distance(self, p: Point2f) -> Float {
        debug_assert!(!self.has_nan());
        (self - p).length()
    }

    pub fn distance_squared(self, p: Point2f) -> Float {
        debug_assert!(!self.has_nan());
        (self - p).length_squared()
    }
}

impl Tuple2<Float> for Point2f {
    fn new(x: Float, y: Float) -> Self {
        Self::new(x, y)
    }

    fn x(&self) -> Float {
        self.x
    }

    fn y(&self) -> Float {
        self.y
    }

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        Point2f {
            x: lerp(t, &a.x, &b.x),
            y: lerp(t, &a.y, &b.y),
        }
    }
}

impl HasNan for Point2f {
    fn has_nan(&self) -> bool {
        has_nan2(self)
    }
}

impl Default for Point2f {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|p: &Point2f| -> Point2f { Point2f::new(-p.x, -p.y) });

// Points can be scaled elementwise
impl_op_ex_commutative!(*|p: &Point2f, s: Float| -> Point2f { Point2f::new(p.x * s, p.y * s) });
impl_op_ex!(/ |p: &Point2f, s: Float| -> Point2f {
    Point2f::new(p.x / s, p.y / s) });
impl_op_ex!(*= |p: &mut Point2f, s: Float| {
    p.x *= s;
    p.y *= s;
});
impl_op_ex!(/= |p: &mut Point2f, s: Float| {
    p.x /= s;
    p.y /= s;
});

// Point + Vector -> Point
impl_op_ex_commutative!(+ |p: &Point2f, v: &Vector2f| -> Point2f
{
    Point2f::new(p.x + v.x, p.y + v.y)
});
impl_op_ex!(+=|p: &mut Point2f, v: &Vector2f| {
    p.x += v.x;
    p.y += v.y;
});

// Point - Vector -> Point
impl_op_ex!(-|p: &Point2f, v: &Vector2f| -> Point2f { Point2f::new(p.x - v.x, p.y - v.y) });
impl_op_ex!(-=|p: &mut Point2f, v: &Vector2f| {
    p.x -= v.x;
    p.y -= v.y;
});

// Point - Point -> Vector
impl_op_ex!(-|p1: &Point2f, p2: &Point2f| -> Vector2f { Vector2f::new(p1.x - p2.x, p1.y - p2.y,) });

impl From<Vector2f> for Point2f {
    fn from(value: Vector2f) -> Self {
        Self::new(value.x(), value.y())
    }
}

impl From<[Float; 2]> for Point2f {
    fn from(value: [Float; 2]) -> Self {
        Self::new(value[0], value[1])
    }
}

impl From<Point2f> for [Float; 2] {
    fn from(value: Point2f) -> Self {
        [value.x, value.y]
    }
}

impl From<(Float, Float)> for Point2f {
    fn from(value: (Float, Float)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<Point2f> for (Float, Float) {
    fn from(value: Point2f) -> Self {
        (value.x, value.y)
    }
}

// ---------------------------------------------------------------------------
//        Point3f
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3f {
    pub x: Float,
    pub y: Float,
    pub z: Float,
}

impl Point3f {
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
        Self::new(v, v, v)
    }

    pub fn distance(self, p: Point3f) -> Float {
        debug_assert!(!self.has_nan());
        (self - p).length()
    }

    pub fn distance_squared(self, p: Point3f) -> Float {
        debug_assert!(!self.has_nan());
        (self - p).length_squared()
    }
}

impl Tuple3<Float> for Point3f {
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

    fn lerp(t: Float, a: &Self, b: &Self) -> Self {
        Self {
            x: math::lerp(t, &a.x, &b.x),
            y: math::lerp(t, &a.y, &b.y),
            z: math::lerp(t, &a.z, &b.z),
        }
    }
}

impl HasNan for Point3f {
    fn has_nan(&self) -> bool {
        has_nan3(self)
    }
}

impl Default for Point3f {
    fn default() -> Self {
        Self::ZERO
    }
}

impl_op_ex!(-|v: Point3f| -> Point3f { Point3f::new(-v.x, -v.y, -v.z) });

// Points can be scaled elementwise
impl_op_ex_commutative!(*|p: Point3f, s: Float| -> Point3f {
    Point3f::new(p.x * s, p.y * s, p.z * s)
});
impl_op_ex!(/ |p: Point3f, s: Float| -> Point3f {
    Point3f::new(p.x / s, p.y / s, p.z / s) });
impl_op_ex!(*= |p: &mut Point3f, s: Float| {
    p.x *= s;
    p.y *= s;
    p.z *= s;
});
impl_op_ex!(/= |p: &mut Point3f, s: Float| {
    p.x /= s;
    p.y /= s;
    p.z /= s;
});

// Point + Vector -> Point
impl_op_ex_commutative!(+ |p: Point3f, v: Vector3f| -> Point3f
{
    Point3f::new(p.x + v.x, p.y + v.y, p.z + v.z)
});

impl_op_ex!(+=|p: &mut Point3f, v: Vector3f| {
    p.x += v.x;
    p.y += v.y;
    p.z += v.z;
});

// Point - Vector -> Point
impl_op_ex!(-|p: Point3f, v: Vector3f| -> Point3f {
    Point3f::new(p.x - v.x, p.y - v.y, p.z - v.z)
});
impl_op_ex!(-=|p: &mut Point3f, v: Vector3f| {
    p.x -= v.x;
    p.y -= v.y;
    p.z -= v.z;
});

// Point - Point -> Vector
impl_op_ex!(-|p1: Point3f, p2: Point3f| -> Vector3f {
    Vector3f::new(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z)
});

impl From<Vector3f> for Point3f {
    fn from(value: Vector3f) -> Self {
        Point3f::new(value.x, value.y, value.z)
    }
}

impl From<[Float; 3]> for Point3f {
    fn from(value: [Float; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<Point3f> for [Float; 3] {
    fn from(value: Point3f) -> Self {
        [value.x, value.y, value.z]
    }
}

impl From<(Float, Float, Float)> for Point3f {
    fn from(value: (Float, Float, Float)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl From<Point3f> for (Float, Float, Float) {
    fn from(value: Point3f) -> Self {
        (value.x, value.y, value.z)
    }
}

#[cfg(test)]
mod tests {
    use crate::{vecmath::HasNan, Float};

    use super::{Point2f, Point2i, Point3f, Point3i, Vector2f, Vector2i, Vector3f, Vector3i};

    #[test]
    fn point_has_nan() {
        let p = Point2f::new(Float::NAN, 0.0);
        assert!(p.has_nan());

        let p = Point3f::new(Float::NAN, 0.0, 10.0);
        assert!(p.has_nan());
    }

    #[test]
    fn point_negation() {
        let point = Point2i::new(1, 2);
        assert_eq!(Point2i::new(-1, -2), -point);
        let point = Point3i::new(1, 2, 3);
        assert_eq!(Point3i::new(-1, -2, -3), -point);
    }

    #[test]
    fn point_point_distance() {
        let p1 = Point2f::new(0.0, 0.0);
        let p2 = Point2f::new(3.0, 4.0);
        assert_eq!(5.0, p1.distance(p2));
    }

    #[test]
    fn point_point_distance_squared() {
        let p1 = Point2f::new(0.0, 0.0);
        let p2 = Point2f::new(3.0, 4.0);
        assert_eq!(25.0, p1.distance_squared(p2));
    }

    #[test]
    fn point_binary_ops() {
        let point = Point2i::new(-2, 10);
        // Point - Point -> Vector
        assert_eq!(Vector2i::new(0, 0), point - point);
        // Point * Scalar -> Point
        assert_eq!(Point2i::new(-6, 30), point * 3);
        assert_eq!(Point2i::new(-6, 30), 3 * point);
        // Point / Scalar -> Point
        assert_eq!(Point2i::new(-1, 5), point / 2);
        let vec = Vector2i::new(1, 0);
        // Point + Vector -> Point
        assert_eq!(Point2i::new(-1, 10), point + vec);
        assert_eq!(Point2i::new(-1, 10), vec + point);
        // Point - Vector -> Point
        assert_eq!(Point2i::new(-3, 10), point - vec);

        // Similarly for other types.
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

        // Note that points and normals cannot be summed.
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
    fn point_from_vector() {
        let v1 = Vector2i::new(1, 2);
        let p1 = Point2i::new(1, 2);
        assert_eq!(p1, v1.into());

        let v1 = Vector3i::new(1, 2, 3);
        let p1 = Point3i::new(1, 2, 3);
        assert_eq!(p1, v1.into());

        let v1 = Vector2f::new(1.0, 2.0);
        let p1 = Point2f::new(1.0, 2.0);
        assert_eq!(p1, v1.into());

        let v1 = Vector3f::new(1.0, 2.0, 3.0);
        let p1 = Point3f::new(1.0, 2.0, 3.0);
        assert_eq!(p1, v1.into());
    }

    #[test]
    fn point_from_into_tuple() {
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
        let t1: (Float, Float) = v1.into();
        assert_eq!((1.0, 2.0), t1);
        assert_eq!(v1, t1.into());

        let v1 = Point3f::new(1.0, 2.0, 3.0);
        let t1: (Float, Float, Float) = v1.into();
        assert_eq!((1.0, 2.0, 3.0), t1);
        assert_eq!(v1, t1.into());
    }

    #[test]
    fn point_from_into_array() {
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
        let t1: [Float; 2] = v1.into();
        assert_eq!([1.0, 2.0], t1);
        assert_eq!(v1, t1.into());

        let v1 = Point3f::new(1.0, 2.0, 3.0);
        let t1: [Float; 3] = v1.into();
        assert_eq!([1.0, 2.0, 3.0], t1);
        assert_eq!(v1, t1.into());
    }
}
