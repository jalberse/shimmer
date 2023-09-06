use std::{marker::PhantomData, ops::Index};

use crate::{math::NumericLimit, Float};

use super::vecmath::{
    point::{Point2, Point3},
    tuple::TupleElement,
    Point2f, Point2i, Point3f, Point3i, Tuple2, Tuple3,
};

pub type Bounds2i = Bounds2<Point2i, i32>;
pub type Bounds2f = Bounds2<Point2f, Float>;
pub type Bounds3i = Bounds3<Point3i, i32>;
pub type Bounds3f = Bounds3<Point3f, Float>;

struct Bounds2<P, T>
where
    P: Point2<T>,
    T: TupleElement,
{
    pub min: P,
    pub max: P,
    point_element_type: PhantomData<T>,
}

impl<P, T> Bounds2<P, T>
where
    P: Point2<T>,
    T: TupleElement,
{
    fn from_point(point: P) -> Self {
        Self {
            min: point,
            max: point,
            point_element_type: Default::default(),
        }
    }

    fn from_points(p1: P, p2: P) -> Self {
        Self {
            min: Tuple2::min(&p1, &p2),
            max: Tuple2::max(&p1, &p2),
            point_element_type: Default::default(),
        }
    }
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

impl<P: Point2<T>, T: TupleElement> Index<usize> for Bounds2<P, T> {
    type Output = P;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 2);
        if index == 0 {
            &self.min
        } else {
            &self.max
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

impl<P, T> Bounds3<P, T>
where
    P: Point3<T>,
    T: TupleElement,
{
    fn from_point(point: P) -> Self {
        Self {
            min: point,
            max: point,
            point_element_type: Default::default(),
        }
    }

    fn from_points(p1: P, p2: P) -> Self {
        Self {
            min: Tuple3::min(&p1, &p2),
            max: Tuple3::max(&p1, &p2),
            point_element_type: Default::default(),
        }
    }
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

impl<P: Point3<T>, T: TupleElement> Index<usize> for Bounds3<P, T> {
    type Output = P;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 2);
        if index == 0 {
            &self.min
        } else {
            &self.max
        }
    }
}

mod tests {
    use crate::{
        geometry::{
            bounding_box::{Bounds2f, Bounds3f},
            vecmath::{Point2f, Point3f},
        },
        Float,
    };

    #[test]
    fn bounds2_default() {
        let bounds = Bounds2f::default();
        assert_eq!(Float::MIN, bounds.min.x);
        assert_eq!(Float::MIN, bounds.min.y);
        assert_eq!(Float::MAX, bounds.max.x);
        assert_eq!(Float::MAX, bounds.max.y);
    }

    #[test]
    fn bounds3_default() {
        let bounds = Bounds3f::default();
        assert_eq!(Float::MIN, bounds.min.x);
        assert_eq!(Float::MIN, bounds.min.y);
        assert_eq!(Float::MIN, bounds.min.z);
        assert_eq!(Float::MAX, bounds.max.x);
        assert_eq!(Float::MAX, bounds.max.y);
        assert_eq!(Float::MAX, bounds.max.z);
    }

    #[test]
    fn bounds2_from_point() {
        let p = Point2f::new(0.0, 10.0);
        let bounds = Bounds2f::from_point(p);
        assert_eq!(p, bounds.min);
        assert_eq!(p, bounds.max);
    }

    #[test]
    fn bounds3_from_point() {
        let p = Point3f::new(0.0, 10.0, 100.0);
        let bounds = Bounds3f::from_point(p);
        assert_eq!(p, bounds.min);
        assert_eq!(p, bounds.max);
    }

    #[test]
    fn bounds2_from_points() {
        let p1 = Point2f::new(0.0, 10.0);
        let p2 = Point2f::new(10.0, 0.0);
        let bounds = Bounds2f::from_points(p1, p2);
        assert_eq!(Point2f::new(0.0, 00.0), bounds.min);
        assert_eq!(Point2f::new(10.0, 10.0), bounds.max);
    }

    #[test]
    fn bounds3_from_points() {
        let p1 = Point3f::new(0.0, 10.0, 100.0);
        let p2 = Point3f::new(10.0, 0.0, 5.0);
        let bounds = Bounds3f::from_points(p1, p2);
        assert_eq!(Point3f::new(0.0, 00.0, 5.0), bounds.min);
        assert_eq!(Point3f::new(10.0, 10.0, 100.0), bounds.max);
    }

    #[test]
    fn bounds2_index() {
        let p1 = Point2f::new(0.0, 10.0);
        let p2 = Point2f::new(10.0, 0.0);
        let bounds = Bounds2f::from_points(p1, p2);
        assert_eq!(Point2f::new(0.0, 00.0), bounds[0]);
        assert_eq!(Point2f::new(10.0, 10.0), bounds[1]);
    }

    #[test]
    fn bounds3_index() {
        let p1 = Point3f::new(0.0, 10.0, 100.0);
        let p2 = Point3f::new(10.0, 0.0, 5.0);
        let bounds = Bounds3f::from_points(p1, p2);
        assert_eq!(Point3f::new(0.0, 00.0, 5.0), bounds[0]);
        assert_eq!(Point3f::new(10.0, 10.0, 100.0), bounds[1]);
    }
}
