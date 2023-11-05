use crate::{
    medium::Medium,
    vecmath::{HasNan, Point3f, Vector3f},
    Float,
};

pub trait RayI {
    fn get(&self, t: Float) -> Point3f;
}

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
}

impl RayI for Ray {
    fn get(&self, t: Float) -> Point3f {
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

impl HasNan for Ray {
    fn has_nan(&self) -> bool {
        self.o.has_nan() || self.d.has_nan() || self.time.is_nan()
    }
}

pub struct RayDifferential {
    pub ray: Ray,
    pub auxiliary: Option<AuxiliaryRays>,
}

impl RayDifferential {
    pub fn new(ray: Ray, auxiliary: Option<AuxiliaryRays>) -> RayDifferential {
        RayDifferential { ray, auxiliary }
    }

    pub fn scale_differentials(&mut self, s: Float) {
        if let Some(aux) = &mut self.auxiliary {
            aux.rx_origin = self.ray.o + (aux.rx_origin - self.ray.o) * s;
            aux.ry_origin = self.ray.o + (aux.ry_origin - self.ray.o) * s;

            aux.rx_direction = self.ray.d + (aux.rx_direction - self.ray.d) * s;
            aux.ry_direction = self.ray.d + (aux.ry_direction - self.ray.d) * s;
        }
    }
}

impl RayI for RayDifferential {
    fn get(&self, t: Float) -> Point3f {
        self.ray.get(t)
    }
}

impl Default for RayDifferential {
    fn default() -> Self {
        Self {
            ray: Default::default(),
            auxiliary: Default::default(),
        }
    }
}

impl HasNan for RayDifferential {
    fn has_nan(&self) -> bool {
        self.ray.has_nan()
            || (if let Some(aux) = &self.auxiliary {
                aux.has_nan()
            } else {
                false
            })
    }
}

// PAPERDOC PBRTv4 uses a flag which we must remember to check (hasDifferentials).
// The inclusion of the Option type is just as efficient (we store a flag anyways)
// but is easier for the programmer.

/// We wrap the differential information in a single struct so that
/// we can guarantee all or none are present in RayDifferential via a single Option member.
pub struct AuxiliaryRays {
    pub rx_origin: Point3f,
    pub rx_direction: Vector3f,
    pub ry_origin: Point3f,
    pub ry_direction: Vector3f,
}

impl AuxiliaryRays {
    pub fn new(
        rx_origin: Point3f,
        rx_direction: Vector3f,
        ry_origin: Point3f,
        ry_direction: Vector3f,
    ) -> AuxiliaryRays {
        AuxiliaryRays {
            rx_origin,
            rx_direction,
            ry_origin,
            ry_direction,
        }
    }
}

impl Default for AuxiliaryRays {
    fn default() -> Self {
        Self {
            rx_origin: Default::default(),
            rx_direction: Default::default(),
            ry_origin: Default::default(),
            ry_direction: Default::default(),
        }
    }
}

impl HasNan for AuxiliaryRays {
    fn has_nan(&self) -> bool {
        self.rx_origin.has_nan()
            || self.rx_direction.has_nan()
            || self.ry_origin.has_nan()
            || self.ry_direction.has_nan()
    }
}

#[cfg(test)]
mod tests {
    use crate::vecmath::{Point3f, Tuple3, Vector3f};

    use super::{Ray, RayI};

    #[test]
    fn get_pos_from_ray() {
        let ray = Ray::new(Point3f::ZERO, Vector3f::ONE, None);
        assert_eq!(Point3f::new(3.0, 3.0, 3.0), ray.get(3.0));
    }
}
