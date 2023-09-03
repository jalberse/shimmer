use std::ops::{Add, Mul};

use crate::math::{Abs, Sqrt};

use super::{
    has_nan::HasNan,
    tuple::{Tuple2, Tuple3},
};

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
    T: Mul<Output = T> + Add<Output = T> + Abs,
{
    debug_assert!(!v.has_nan());
    v.x() * v.x() + v.y() * v.y() + v.z() * v.z()
}

pub fn length3<V, T>(v: &V) -> T
where
    V: Tuple3<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Sqrt + Abs,
{
    // PAPERDOC - PBRTv4 has a discussion on page 88 about an odd usage of std::sqrt().
    // We see here that Rust's trait system obviates the issue.
    length_squared3(v).sqrt()
}

pub fn length_squared2<V, T>(v: &V) -> T
where
    V: Tuple2<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Abs,
{
    debug_assert!(!v.has_nan());
    v.x() * v.x() + v.y() * v.y()
}

pub fn length2<V, T>(v: &V) -> T
where
    V: Tuple2<T> + HasNan,
    T: Mul<Output = T> + Add<Output = T> + Sqrt + Abs,
{
    length_squared2(v).sqrt()
}
