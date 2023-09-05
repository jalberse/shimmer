use std::marker::PhantomData;

use super::vecmath::{
    point::{Point2, Point3},
    tuple::TupleElement,
};

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
