use std::ops::{Add, Mul};

use crate::{float::Float, math::Sqrt};

use super::{has_nan::HasNan, tuple::Tuple3};

pub trait Length<T>: HasNan
where
    T: Mul<Output = T> + Add<Output = T> + Sqrt,
{
    fn length_squared(&self) -> T;
    fn length(&self) -> T;
}

pub fn length_squared3<V, T>(v: &V) -> T
where
    V: Tuple3<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T>,
{
    debug_assert!(!v.has_nan());
    v.x() * v.x() + v.y() * v.y() + v.z() * v.z()
}

pub fn length3<V, T>(v: &V) -> T
where
    V: Tuple3<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Sqrt,
{
    // PAPERDOC - PBRTv4 has a discussion on page 88 about an odd usage of std::sqrt().
    // We see here that Rust's trait system obviates the issue.
    length_squared3::<V, T>(v).sqrt()
}

// TODO equivalent length functions for vectors of length 2
