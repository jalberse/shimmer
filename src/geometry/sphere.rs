use super::vecmath::point::Point3;

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
