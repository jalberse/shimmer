use super::vecmath::point::Point3;

/// This is a simple Sphere which just encapsulates the center and the radius;
/// this is not to be confused with shape::Sphere which encapsulates additional
/// functionality like ray intersection. This class is mostly useful for e.g.
/// bounding spheres that don't need that extra information or routines.
pub struct Sphere<C, T> {
    pub center: C,
    pub radius: T,
}

impl<C, T> Sphere<C, T>
where
    C: Point3<ElementType = T>,
{
    pub fn new(center: C, radius: T) -> Self {
        Self { center, radius }
    }
}
