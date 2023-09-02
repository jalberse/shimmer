/// A mathematical vector of length 3 (a tuple with 3 elements).
/// Used for sharing logic across e.g. Vector3f and Normal3f and Point3f.
pub trait Tuple3<T> {
    fn new(x: T, y: T, z: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;
}

/// A mathematical vector of length 2 (a tuple with 2 elements).
/// Used for sharing logic across e.g. Vector2f and Normal2f and Point2f.
pub trait Tuple2<T> {
    fn new(x: T, y: T) -> Self;

    fn x(&self) -> T;
    fn y(&self) -> T;
}

// TODO the Point classes should also use these traits to share stuff
