use super::vecmath::point::Point3;

// TODO I used this for finding the bounding sphere of a Bounding Box, but I think that
// we should replace that use case with shape::Sphere instead.
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
