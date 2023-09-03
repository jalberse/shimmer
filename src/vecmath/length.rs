use std::ops::{Add, Mul};

use crate::math::Sqrt;

use super::has_nan::HasNan;

pub trait Length<T>: HasNan
where
    T: Mul<Output = T> + Add<Output = T> + Sqrt,
{
    fn length_squared(&self) -> T;
    fn length(&self) -> T;
}
