//! A set of functions which help us implement the Length trait for various types,
//! but that we don't want exposed external to the vecmath module.

use super::{tuple::TupleElement, HasNan, Tuple2, Tuple3};

pub fn length_squared3<V, T>(v: &V) -> T
where
    V: Tuple3<T>,
    T: TupleElement,
{
    debug_assert!(!v.has_nan());
    v.x() * v.x() + v.y() * v.y() + v.z() * v.z()
}

pub fn length3<V, T>(v: &V) -> T
where
    V: Tuple3<T>,
    T: TupleElement,
{
    // PAPERDOC - PBRTv4 has a discussion on page 88 about an odd usage of std::sqrt().
    // We see here that Rust's trait system obviates the issue.
    length_squared3(v).sqrt()
}

pub fn length_squared2<V, T>(v: &V) -> T
where
    V: Tuple2<T> + HasNan,
    T: TupleElement,
{
    debug_assert!(!v.has_nan());
    v.x() * v.x() + v.y() * v.y()
}

pub fn length2<V, T>(v: &V) -> T
where
    V: Tuple2<T> + HasNan,
    T: TupleElement,
{
    length_squared2(v).sqrt()
}