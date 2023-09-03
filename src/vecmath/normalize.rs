use std::ops::{Add, Div, Mul};

use crate::math::Sqrt;

use super::{has_nan::HasNan, length::Length};

pub trait Normalize<T>: HasNan + Length<T>
where
    Self: Sized + Div<T, Output = Self> + Copy + Clone,
    T: Mul<T, Output = T> + Add<T, Output = T> + Sqrt,
{
    // TODO I think we can take a reference here instead.
    fn normalize(self) -> Self {
        debug_assert!(!self.has_nan());
        self / self.length()
    }
}
