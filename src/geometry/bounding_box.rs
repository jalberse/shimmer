use std::marker::PhantomData;

use crate::Float;

use super::vecmath::{
    point::{Point2, Point3},
    tuple::TupleElement,
    Point2f, Point2i, Point3f, Point3i,
};

pub type Bounds2i = Bounds2<Point2i, i32>;
pub type Bounds2f = Bounds2<Point2f, Float>;
pub type Bounds3i = Bounds2<Point3i, i32>;
pub type Bounds3f = Bounds2<Point3f, Float>;

struct Bounds2<P, T>
where
    P: Point2<T>,
    T: TupleElement,
{
    pub min: P,
    pub max: P,
    // https://doc.rust-lang.org/std/marker/struct.PhantomData.html#unused-type-parameters
    point_element_type: PhantomData<T>,
}

struct Bounds3<P, T>
where
    P: Point3<T>,
    T: TupleElement,
{
    pub min: P,
    pub max: P,
    // https://doc.rust-lang.org/std/marker/struct.PhantomData.html#unused-type-parameters
    point_element_type: PhantomData<T>,
}
