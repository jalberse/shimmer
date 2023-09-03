use crate::{is_nan::IsNan, math::Abs};

use super::tuple::{Tuple2, Tuple3};

pub trait HasNan {
    fn has_nan(&self) -> bool;
}

pub fn has_nan3<V, T>(v: &V) -> bool
where
    V: Tuple3<T>,
    T: IsNan + Abs,
{
    v.x().is_nan() || v.y().is_nan() || v.z().is_nan()
}

pub fn has_nan2<V, T>(v: &V) -> bool
where
    V: Tuple2<T>,
    T: IsNan + Abs,
{
    v.x().is_nan() || v.y().is_nan()
}
