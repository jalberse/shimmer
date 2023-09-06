use std::marker::PhantomData;

use crate::{math::NumericLimit, Float};

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
    point_element_type: PhantomData<T>,
}

impl<P, T> Default for Bounds2<P, T>
where
    P: Point2<T>,
    T: TupleElement,
{
    fn default() -> Self {
        let min_num = NumericLimit::MIN;
        let max_num = NumericLimit::MAX;
        Self {
            min: P::new(min_num, min_num),
            max: P::new(max_num, max_num),
            point_element_type: Default::default(),
        }
    }
}

struct Bounds3<P, T>
where
    P: Point3<T>,
    T: TupleElement,
{
    pub min: P,
    pub max: P,
    point_element_type: PhantomData<T>,
}

impl<P, T> Default for Bounds3<P, T>
where
    P: Point3<T>,
    T: TupleElement,
{
    fn default() -> Self {
        let min_num = NumericLimit::MIN;
        let max_num = NumericLimit::MAX;
        Self {
            min: P::new(min_num, min_num, min_num),
            max: P::new(max_num, max_num, max_num),
            point_element_type: Default::default(),
        }
    }
}
