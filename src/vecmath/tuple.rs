use crate::{
    float::Float,
    math::{Abs, Ceil, Floor},
};

/// A tuple with 3 elements.
/// Used for sharing logic across e.g. Vector3f and Normal3f and Point3f.
/// Note that only those functions that are shared across all three types are
/// within this trait; if there's something that only one or two of them have,
/// then that can be represented in a separate trait which they can implement. Composition!
pub trait Tuple3<T>: Sized
where
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
pub trait Tuple2<T>: Sized
where
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
