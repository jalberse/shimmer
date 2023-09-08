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

#[derive(Debug, PartialEq)]
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
    fn new(p1: P, p2: P) -> Self {
        Self {
            min: Tuple2::min(&p1, &p2),
            max: Tuple2::max(&p1, &p2),
            point_element_type: Default::default(),
        }
    }

    fn from_point(point: P) -> Self {
        Self {
            min: point,
            max: point,
            point_element_type: Default::default(),
        }
    }

    /// Returns a corner of the bounding box specified by `corner`.
    fn corner(&self, corner: usize) -> P {
        debug_assert!(corner < 4);
        P::new(self[corner & 1].x(), self[(corner & 2 != 0) as usize].y())
    }

    fn union_point(&self, p: &P) -> Self {
        let min = Tuple2::min(&self.min, &p);
        let max = Tuple2::max(&self.max, &p);
        // Set values directly to maintain degeneracy and avoid infinite extents.
        // See PBRTv4 pg 99.
        Self {
            min,
            max,
            point_element_type: Default::default(),
        }
    }

    fn union(&self, other: &Self) -> Self {
        let min = Tuple2::min(&self.min, &other.min);
        let max = Tuple2::max(&self.max, &other.max);
        // Set values directly to maintain degeneracy and avoid infinite extents.
        // See PBRTv4 pg 99.
        Self {
            min,
            max,
            point_element_type: Default::default(),
        }
    }

    /// None if the bounds do not intersect.
    /// Else, the intersection of the two bounds.
    fn intersect(&self, other: &Self) -> Option<Self> {
        let min = Tuple2::max(&self.min, &other.min);
        let max = Tuple2::min(&self.max, &other.max);

        if min.x() >= max.x() || min.y() >= max.y() {
            return None;
        }

        Some(Self {
            min,
            max,
            point_element_type: Default::default(),
        })
    }

    fn overlaps(&self, other: &Self) -> bool {
        let x_overlap = self.max.x() >= other.min.x() && self.min.x() <= other.max.x();
        let y_overlap = self.max.y() >= other.min.y() && self.min.y() <= other.max.y();
        x_overlap && y_overlap
    }

    /// Checks if the point is inside the bounds (inclusive)
    fn inside(&self, p: &P) -> bool {
        p.x() >= self.min.x()
            && p.x() <= self.max.x()
            && p.y() >= self.min.y()
            && p.y() <= self.max.y()
    }

    /// Checks if the point is inside the bounds, with exclusive upper bounds.
    fn inside_exclusive(&self, p: &P) -> bool {
        p.x() >= self.min.x()
            && p.x() < self.max.x()
            && p.y() >= self.min.y()
            && p.y() < self.max.y()
    }

    /// Zero if the point is inside the bounds, else the squared distance from
    /// the point to the bounding box.
    fn distance_squared(&self, p: &P) -> T {
        let dx = T::max(
            T::max(T::zero(), self.min.x() - p.x()),
            p.x() - self.max.x(),
        );
        let dy = T::max(
            T::max(T::zero(), self.min.y() - p.y()),
            p.y() - self.max.y(),
        );

        dx * dx + dy * dy
    }

    fn distance(&self, p: &P) -> T {
        // PAPERDOC - We don't require an intermediate type here as PBRTv4 does.
        T::sqrt(self.distance_squared(p))
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

impl<P, T> Index<usize> for Bounds2<P, T>
where
    P: Point2<T>,
    T: TupleElement,
{
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

#[derive(Debug, PartialEq)]
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
    fn new(p1: P, p2: P) -> Self {
        Self {
            min: Tuple3::min(&p1, &p2),
            max: Tuple3::max(&p1, &p2),
            point_element_type: Default::default(),
        }
    }

    fn from_point(point: P) -> Self {
        Self {
            min: point,
            max: point,
            point_element_type: Default::default(),
        }
    }

    /// Returns a corner of the bounding box specified by `corner`.
    fn corner(&self, corner: usize) -> P {
        debug_assert!(corner < 8);
        P::new(
            self[corner & 1].x(),
            self[(corner & 2 != 0) as usize].y(),
            self[(corner & 4 != 0) as usize].z(),
        )
    }

    fn union_point(&self, p: &P) -> Self {
        let min = Tuple3::min(&self.min, &p);
        let max = Tuple3::max(&self.max, &p);
        // Set values directly to maintain degeneracy and avoid infinite extents.
        // See PBRTv4 pg 99.
        Self {
            min,
            max,
            point_element_type: Default::default(),
        }
    }

    fn union(&self, other: &Self) -> Self {
        let min = Tuple3::min(&self.min, &other.min);
        let max = Tuple3::max(&self.max, &other.max);
        // Set values directly to maintain degeneracy and avoid infinite extents.
        // See PBRTv4 pg 99.
        Self {
            min,
            max,
            point_element_type: Default::default(),
        }
    }

    /// None if the bounds do not intersect.
    /// Else, the intersection of the two bounds.
    fn intersect(&self, other: &Self) -> Option<Self> {
        let min = Tuple3::max(&self.min, &other.min);
        let max = Tuple3::min(&self.max, &other.max);

        // PAPERDOC - PBRTv4 has an IsEmpty() function that must be called after this
        // function in case the bounds don't intersect; but that intent is not clear
        // from the function signature. An Option result type is more clear.
        if min.x() >= max.x() || min.y() >= max.y() || min.z() >= max.z() {
            return None;
        }

        Some(Self {
            min,
            max,
            point_element_type: Default::default(),
        })
    }

    fn overlaps(&self, other: &Self) -> bool {
        let x_overlap = self.max.x() >= other.min.x() && self.min.x() <= other.max.x();
        let y_overlap = self.max.y() >= other.min.y() && self.min.y() <= other.max.y();
        let z_overlap = self.max.z() >= other.min.z() && self.min.z() <= other.max.z();
        x_overlap && y_overlap && z_overlap
    }

    /// Checks if the point is inside the bounds (inclusive)
    fn inside(&self, p: &P) -> bool {
        p.x() >= self.min.x()
            && p.x() <= self.max.x()
            && p.y() >= self.min.y()
            && p.y() <= self.max.y()
            && p.z() >= self.min.z()
            && p.z() <= self.max.z()
    }

    /// Checks if the point is inside the bounds, with exclusive upper bounds.
    fn inside_exclusive(&self, p: &P) -> bool {
        p.x() >= self.min.x()
            && p.x() < self.max.x()
            && p.y() >= self.min.y()
            && p.y() < self.max.y()
            && p.z() >= self.min.z()
            && p.z() < self.max.z()
    }

    /// Zero if the point is inside the bounds, else the squared distance from
    /// the point to the bounding box.
    fn distance_squared(&self, p: &P) -> T {
        let dx = T::max(
            T::max(T::zero(), self.min.x() - p.x()),
            p.x() - self.max.x(),
        );
        let dy = T::max(
            T::max(T::zero(), self.min.y() - p.y()),
            p.y() - self.max.y(),
        );
        let dz = T::max(
            T::max(T::zero(), self.min.z() - p.z()),
            p.z() - self.max.z(),
        );

        dx * dx + dy * dy + dz * dz
    }

    fn distance(&self, p: &P) -> T {
        // PAPERDOC - We don't require an intermediate type here as PBRTv4 does.
        T::sqrt(self.distance_squared(p))
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
            bounding_box::{Bounds2f, Bounds3f, Bounds3i},
            vecmath::{Point2f, Point2i, Point3f, Point3i},
        },
        Float,
    };

    use super::Bounds2i;

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
        let bounds = Bounds2f::new(p1, p2);
        assert_eq!(Point2f::new(0.0, 00.0), bounds.min);
        assert_eq!(Point2f::new(10.0, 10.0), bounds.max);
    }

    #[test]
    fn bounds3_from_points() {
        let p1 = Point3f::new(0.0, 10.0, 100.0);
        let p2 = Point3f::new(10.0, 0.0, 5.0);
        let bounds = Bounds3f::new(p1, p2);
        assert_eq!(Point3f::new(0.0, 00.0, 5.0), bounds.min);
        assert_eq!(Point3f::new(10.0, 10.0, 100.0), bounds.max);
    }

    #[test]
    fn bounds2_index() {
        let p1 = Point2f::new(0.0, 10.0);
        let p2 = Point2f::new(10.0, 0.0);
        let bounds = Bounds2f::new(p1, p2);
        assert_eq!(Point2f::new(0.0, 00.0), bounds[0]);
        assert_eq!(Point2f::new(10.0, 10.0), bounds[1]);
    }

    #[test]
    fn bounds3_index() {
        let p1 = Point3f::new(0.0, 10.0, 100.0);
        let p2 = Point3f::new(10.0, 0.0, 5.0);
        let bounds = Bounds3f::new(p1, p2);
        assert_eq!(Point3f::new(0.0, 00.0, 5.0), bounds[0]);
        assert_eq!(Point3f::new(10.0, 10.0, 100.0), bounds[1]);
    }

    #[test]
    fn bounds2_corner() {
        let p1 = Point2i::new(0, 0);
        let p2 = Point2i::new(1, 1);
        let bounds = Bounds2i::new(p1, p2);
        assert_eq!(Point2i::new(0, 0), bounds.corner(0));
        assert_eq!(Point2i::new(1, 0), bounds.corner(1));
        assert_eq!(Point2i::new(0, 1), bounds.corner(2));
        assert_eq!(Point2i::new(1, 1), bounds.corner(3));
    }

    #[test]
    fn bounds3_corner() {
        let p1 = Point3i::new(0, 0, 0);
        let p2 = Point3i::new(1, 1, 1);
        let bounds = Bounds3i::new(p1, p2);
        assert_eq!(Point3i::new(0, 0, 0), bounds.corner(0));
        assert_eq!(Point3i::new(1, 0, 0), bounds.corner(1));
        assert_eq!(Point3i::new(0, 1, 0), bounds.corner(2));
        assert_eq!(Point3i::new(1, 1, 0), bounds.corner(3));
        assert_eq!(Point3i::new(0, 0, 1), bounds.corner(4));
        assert_eq!(Point3i::new(1, 0, 1), bounds.corner(5));
        assert_eq!(Point3i::new(0, 1, 1), bounds.corner(6));
        assert_eq!(Point3i::new(1, 1, 1), bounds.corner(7));
    }

    #[test]
    fn bounds2_union_point() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(1, 1);
        let bounds = Bounds2i::new(min, max);
        let new_point = Point2i::new(-1, -1);
        let union = bounds.union_point(&new_point);
        assert_eq!(Point2i::new(-1, -1), union.min);
        assert_eq!(Point2i::new(1, 1), union.max);
    }

    #[test]
    fn bounds3_union_point() {
        let min = Point3i::new(0, 0, 1);
        let max = Point3i::new(1, 1, 1);
        let bounds = Bounds3i::new(min, max);
        let new_point = Point3i::new(-1, -1, -1);
        let union = bounds.union_point(&new_point);
        assert_eq!(Point3i::new(-1, -1, -1), union.min);
        assert_eq!(Point3i::new(1, 1, 1), union.max);
    }

    #[test]
    fn bounds2_union() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(1, 1);
        let bounds = Bounds2i::new(min, max);
        let min2 = Point2i::new(10, 10);
        let max2 = Point2i::new(11, 11);
        let bounds2 = Bounds2i::new(min2, max2);
        let union = bounds.union(&bounds2);
        assert_eq!(Point2i::new(0, 0), union.min);
        assert_eq!(Point2i::new(11, 11), union.max);
    }

    #[test]
    fn bounds3_union() {
        let min = Point3i::new(0, 0, 0);
        let max = Point3i::new(1, 1, 0);
        let bounds = Bounds3i::new(min, max);
        let min2 = Point3i::new(10, 10, 10);
        let max2 = Point3i::new(11, 11, 11);
        let bounds2 = Bounds3i::new(min2, max2);
        let union = bounds.union(&bounds2);
        assert_eq!(Point3i::new(0, 0, 0), union.min);
        assert_eq!(Point3i::new(11, 11, 11), union.max);
    }

    #[test]
    fn bounds2_intersect_none() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(1, 1);
        let bounds = Bounds2i::new(min, max);
        let min2 = Point2i::new(10, 10);
        let max2 = Point2i::new(11, 11);
        let bounds2 = Bounds2i::new(min2, max2);

        assert!(bounds.intersect(&bounds2).is_none());
    }

    #[test]
    fn bounds2_intersect_some() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(7, 7);
        let bounds = Bounds2i::new(min, max);
        let min2 = Point2i::new(3, 3);
        let max2 = Point2i::new(11, 11);
        let bounds2 = Bounds2i::new(min2, max2);

        let intersection = bounds.intersect(&bounds2);
        assert!(intersection.is_some());
        assert_eq!(
            Bounds2i::new(Point2i::new(3, 3), Point2i::new(7, 7)),
            intersection.unwrap()
        )
    }

    #[test]
    fn bounds3_intersect_none() {
        let min = Point3i::new(0, 0, 0);
        let max = Point3i::new(1, 1, 1);
        let bounds = Bounds3i::new(min, max);
        let min2 = Point3i::new(10, 10, 10);
        let max2 = Point3i::new(11, 11, 11);
        let bounds2 = Bounds3i::new(min2, max2);

        assert!(bounds.intersect(&bounds2).is_none());
    }

    #[test]
    fn bounds3_intersect_some() {
        let min = Point3i::new(0, 0, 0);
        let max = Point3i::new(7, 7, 7);
        let bounds = Bounds3i::new(min, max);
        let min2 = Point3i::new(3, 3, 3);
        let max2 = Point3i::new(11, 11, 11);
        let bounds2 = Bounds3i::new(min2, max2);

        let intersection = bounds.intersect(&bounds2);
        assert!(intersection.is_some());
        assert_eq!(
            Bounds3i::new(Point3i::new(3, 3, 3), Point3i::new(7, 7, 7)),
            intersection.unwrap()
        )
    }

    #[test]
    fn bounds2_overlap() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(4, 4);
        let bounds = Bounds2i::new(min, max);
        let min2 = Point2i::new(5, 5);
        let max2 = Point2i::new(11, 11);
        let bounds2 = Bounds2i::new(min2, max2);

        assert!(!bounds.overlaps(&bounds2));

        let min = Point2i::new(0, 0);
        let max = Point2i::new(7, 7);
        let bounds = Bounds2i::new(min, max);
        let min2 = Point2i::new(3, 3);
        let max2 = Point2i::new(11, 11);
        let bounds2 = Bounds2i::new(min2, max2);

        assert!(bounds.overlaps(&bounds2));
    }

    #[test]
    fn bounds3_overlap() {
        let min = Point3i::new(0, 0, 0);
        let max = Point3i::new(7, 7, 7);
        let bounds = Bounds3i::new(min, max);
        let min2 = Point3i::new(10, 10, 10);
        let max2 = Point3i::new(11, 11, 11);
        let bounds2 = Bounds3i::new(min2, max2);

        assert!(!bounds.overlaps(&bounds2));

        let min = Point3i::new(4, 4, 4);
        let max = Point3i::new(7, 7, 7);
        let bounds = Bounds3i::new(min, max);
        let min2 = Point3i::new(3, 3, 3);
        let max2 = Point3i::new(11, 11, 11);
        let bounds2 = Bounds3i::new(min2, max2);

        assert!(bounds.overlaps(&bounds2));
    }

    #[test]
    fn bounds2_inside() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(4, 4);
        let bounds = Bounds2i::new(min, max);
        let inside_point = Point2i::new(1, 1);
        let outside_point = Point2i::new(1, 5);
        assert!(bounds.inside(&inside_point));
        assert!(!bounds.inside(&outside_point))
    }

    #[test]
    fn bounds3_inside() {
        let min = Point3i::new(0, 0, 0);
        let max = Point3i::new(4, 4, 4);
        let bounds = Bounds3i::new(min, max);
        let inside_point = Point3i::new(1, 1, 1);
        let outside_point = Point3i::new(1, 5, 5);
        assert!(bounds.inside(&inside_point));
        assert!(!bounds.inside(&outside_point))
    }

    #[test]
    fn bounds2_inside_exclusive() {
        let min = Point2i::new(0, 0);
        let max = Point2i::new(4, 4);
        let bounds = Bounds2i::new(min, max);
        let inside_point = Point2i::new(0, 0);
        let outside_point = Point2i::new(4, 4);
        assert!(bounds.inside_exclusive(&inside_point));
        assert!(!bounds.inside_exclusive(&outside_point))
    }

    #[test]
    fn bounds3_inside_exclusive() {
        let min = Point3i::new(0, 0, 0);
        let max = Point3i::new(4, 4, 4);
        let bounds = Bounds3i::new(min, max);
        let inside_point = Point3i::new(0, 0, 0);
        let outside_point = Point3i::new(4, 4, 4);
        assert!(bounds.inside_exclusive(&inside_point));
        assert!(!bounds.inside_exclusive(&outside_point))
    }

    #[test]
    fn bounds2_distance() {
        let min = Point2f::new(0.0, 0.0);
        let max = Point2f::new(4.0, 4.0);
        let bounds = Bounds2f::new(min, max);
        let inside_point = Point2f::new(1.0, 1.0);
        let outside_point = Point2f::new(6.0, 4.0);
        assert_eq!(0.0, bounds.distance_squared(&inside_point));
        assert_eq!(4.0, bounds.distance_squared(&outside_point));
        assert_eq!(2.0, bounds.distance(&outside_point));
    }

    #[test]
    fn bounds3_distance() {
        let min = Point3f::new(0.0, 0.0, 0.0);
        let max = Point3f::new(4.0, 4.0, 4.0);
        let bounds = Bounds3f::new(min, max);
        let inside_point = Point3f::new(1.0, 1.0, 1.0);
        let outside_point = Point3f::new(6.0, 4.0, 4.0);
        assert_eq!(0.0, bounds.distance_squared(&inside_point));
        assert_eq!(4.0, bounds.distance_squared(&outside_point));
        assert_eq!(2.0, bounds.distance(&outside_point));
    }
}
