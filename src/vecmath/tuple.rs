use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use crate::{
    float::Float,
    is_nan::IsNan,
    math::{Abs, Ceil, Floor, Max, Min, NumericLimit, Sqrt},
};

use super::HasNan;

/// A TupleElement satisfies all the necessary traits to be an element of a Tuple.
pub trait TupleElement:
    Abs
    + Ceil
    + Floor
    + Min
    + Max
    + PartialOrd
    + Copy
    + Clone
    + NumericLimit
    + Sqrt
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Add<Self, Output = Self>
    + IsNan
{
    fn from_i32(val: i32) -> Self;
    fn into_float(self) -> Float;
    fn from_float(v: Float) -> Self;
    fn zero() -> Self;
}

impl TupleElement for Float {
    fn from_i32(val: i32) -> Self {
        val as Self
    }

    fn zero() -> Self {
        0.0
    }

    fn into_float(self) -> Float {
        self
    }

    fn from_float(v: Float) -> Self {
        v
    }
}

impl TupleElement for i32 {
    fn from_i32(val: i32) -> Self {
        val as Self
    }

    fn zero() -> Self {
        0
    }

    fn into_float(self) -> Float {
        self as Float
    }

    fn from_float(v: Float) -> Self {
        v as i32
    }
}

/// A tuple with 3 elements.
/// Used for sharing logic across e.g. Vector3f and Normal3f and Point3f.
/// Note that only those functions that are shared across all three types are
/// within this trait; if there's something that only one or two of them have,
/// then that can be represented in a separate trait which they can implement. Composition!
pub trait Tuple3<T>: Sized + Copy + Clone + HasNan + Index<usize> + IndexMut<usize>
where
    T: TupleElement,
{
    fn new(x: T, y: T, z: T) -> Self;

    fn splat(v: T) -> Self {
        Self::new(v, v, v)
    }

    fn x(&self) -> T;
    fn y(&self) -> T;
    fn z(&self) -> T;

    fn x_ref(&self) -> &T;
    fn y_ref(&self) -> &T;
    fn z_ref(&self) -> &T;

    fn x_mut(&mut self) -> &mut T;
    fn y_mut(&mut self) -> &mut T;
    fn z_mut(&mut self) -> &mut T;

    fn get(&self, index: usize) -> &T {
        debug_assert!(index < 3);
        if index == 0 {
            self.x_ref()
        } else if index == 1 {
            self.y_ref()
        } else {
            self.z_ref()
        }
    }

    fn get_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < 3);
        if index == 0 {
            self.x_mut()
        } else if index == 1 {
            self.y_mut()
        } else {
            self.z_mut()
        }
    }

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

    // Take the componentwise minimum of the two tuples.
    fn min(a: &Self, b: &Self) -> Self {
        Self::new(
            T::min(a.x(), b.x()),
            T::min(a.y(), b.y()),
            T::min(a.z(), b.z()),
        )
    }

    // Take the componentwise minimum of the two tuples.
    fn max(a: &Self, b: &Self) -> Self {
        Self::new(
            T::max(a.x(), b.x()),
            T::max(a.y(), b.y()),
            T::max(a.z(), b.z()),
        )
    }

    fn min_component_value(&self) -> T {
        T::min(self.x(), T::min(self.y(), self.z()))
    }

    fn min_component_index(&self) -> usize {
        if self.x() < self.y() {
            if self.x() < self.z() {
                return 0;
            } else {
                return 2;
            }
        } else {
            if self.y() < self.z() {
                return 1;
            } else {
                return 2;
            }
        }
    }

    fn max_component_value(&self) -> T {
        T::max(self.x(), T::max(self.y(), self.z()))
    }

    fn max_component_index(&self) -> usize {
        if self.x() > self.y() {
            if self.x() > self.z() {
                return 0;
            } else {
                return 2;
            }
        } else {
            if self.y() > self.z() {
                return 1;
            } else {
                return 2;
            }
        }
    }

    fn permute(self, permutation: (usize, usize, usize)) -> Self {
        // TODO We could likely implement this more efficiently if we used some accessor/Indexing
        // rather than branching. But it's not simle to impl Index for Tuple due to Tuple
        // requiring Sized. So without evidence this really matters, this is fine for now.
        let x = if permutation.0 == 0 {
            self.x()
        } else {
            if permutation.0 == 1 {
                self.y()
            } else {
                self.z()
            }
        };

        let y = if permutation.1 == 0 {
            self.x()
        } else {
            if permutation.1 == 1 {
                self.y()
            } else {
                self.z()
            }
        };

        let z = if permutation.2 == 0 {
            self.x()
        } else {
            if permutation.2 == 1 {
                self.y()
            } else {
                self.z()
            }
        };

        Self::new(x, y, z)
    }
}

/// A tuple with 2 elements.
/// Used for sharing logic across e.g. Vector2f and Normal2f and Point2f.
pub trait Tuple2<T>: Sized + Copy + Clone + HasNan + Index<usize> + IndexMut<usize>
where
    T: TupleElement,
{
    fn new(x: T, y: T) -> Self;

    fn splat(v: T) -> Self {
        Self::new(v, v)
    }

    fn x(&self) -> T;
    fn y(&self) -> T;

    fn x_ref(&self) -> &T;
    fn y_ref(&self) -> &T;

    fn x_mut(&mut self) -> &mut T;
    fn y_mut(&mut self) -> &mut T;

    fn get(&self, index: usize) -> &T {
        debug_assert!(index < 2);
        if index == 0 {
            self.x_ref()
        } else {
            self.y_ref()
        }
    }

    fn get_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < 2);
        if index == 0 {
            self.x_mut()
        } else {
            self.y_mut()
        }
    }

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

    // Take the componentwise minimum of the two tuples.
    fn min(a: &Self, b: &Self) -> Self {
        Self::new(T::min(a.x(), b.x()), T::min(a.y(), b.y()))
    }

    // Take the componentwise maximum of the two tuples.
    fn max(a: &Self, b: &Self) -> Self {
        Self::new(T::max(a.x(), b.x()), T::max(a.y(), b.y()))
    }

    fn min_component_value(&self) -> T {
        T::min(self.x(), self.y())
    }

    fn min_component_index(&self) -> usize {
        if self.x() < self.y() {
            0
        } else {
            1
        }
    }

    fn max_component_value(&self) -> T {
        T::max(self.x(), self.y())
    }

    fn max_component_index(&self) -> usize {
        if self.x() > self.y() {
            0
        } else {
            1
        }
    }

    fn permute(self, permutation: (usize, usize)) -> Self {
        let x = if permutation.0 == 0 {
            self.x()
        } else {
            self.y()
        };

        let y = if permutation.1 == 0 {
            self.x()
        } else {
            self.y()
        };

        Self::new(x, y)
    }
}