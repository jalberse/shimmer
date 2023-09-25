use crate::{
    medium::Medium,
    vecmath::{Point3f, Vector3f},
    Float,
};

pub struct Ray {
    /// Origin of the ray
    pub o: Point3f,
    /// Direction of the ray
    pub d: Vector3f,
    /// The time for the ray; used for animated scenes
    pub time: Float,
    // TODO we're making this Optional for now, but I think when Medium becomes a proper enum
    // (it's a TaggedPointer in PBRT), one of the variants of THAT enum should be None.
    /// The medium at the ray's origin.
    pub medium: Option<Medium>,
}

impl Ray {
    pub fn new(origin: Point3f, direction: Vector3f, medium: Option<Medium>) -> Ray {
        Ray {
            o: origin,
            d: direction,
            time: 0.0,
            medium: medium,
        }
    }

    pub fn new_with_time(
        origin: Point3f,
        direction: Vector3f,
        time: Float,
        medium: Option<Medium>,
    ) -> Ray {
        Ray {
            o: origin,
            d: direction,
            time,
            medium: medium,
        }
    }

    pub fn get(&self, t: Float) -> Point3f {
        self.o + self.d * t
    }
}

impl Default for Ray {
    fn default() -> Self {
        Self {
            o: Point3f::ZERO,
            d: Vector3f::ZERO,
            time: 0.0,
            medium: None,
        }
    }
}
