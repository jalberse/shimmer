use crate::{
    float::{next_float_down, next_float_up},
    media::Medium,
    vecmath::{
        normal::Normal3, point::Point3fi, vector::Vector3, HasNan, Normal3f, Point3f, Tuple3,
        Vector3f,
    },
    Float,
};

pub trait RayI {
    fn get(&self, t: Float) -> Point3f;
}

#[derive(Debug, Copy, Clone)]
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

    pub fn offset_ray_origin(pi: Point3fi, n: Normal3f, w: Vector3f) -> Point3f {
        // Find vector offset to corner of error bounds and compute initial p0
        let d = n.abs().dot_vector(pi.error());
        let mut offset = d * Vector3f::from(n);
        if w.dot_normal(n) < 0.0 {
            offset = -offset;
        }
        let mut po = Point3f::from(pi) + offset;

        // Round offset point po away from p
        for i in 0..3 {
            if offset[i] > 0.0 {
                po[i] = next_float_up(po[i]);
            } else if offset[i] < 0.0 {
                po[i] = next_float_down(po[i])
            }
        }

        po
    }

    pub fn spawn_ray(pi: Point3fi, n: Normal3f, time: Float, d: Vector3f) -> Ray {
        Ray::new_with_time(Ray::offset_ray_origin(pi, n, d), d, time, None)
    }

    pub fn spawn_ray_to(p_from: Point3fi, n: Normal3f, time: Float, p_to: Point3f) -> Ray {
        let d = p_to - Point3f::from(p_from);
        Self::spawn_ray(p_from, n, time, d)
    }

    pub fn spawn_ray_to_both_offset(
        p_from: Point3fi,
        n_from: Normal3f,
        time: Float,
        p_to: Point3fi,
        n_to: Normal3f,
    ) -> Ray {
        let pf =
            Self::offset_ray_origin(p_from, n_from, Point3f::from(p_to) - Point3f::from(p_from));
        let pt = Self::offset_ray_origin(p_to, n_to, pf - Point3f::from(p_to));
        Ray {
            o: pf,
            d: pt - pf,
            time,
            medium: None,
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


#[derive(Debug, Copy, Clone)]
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

/// We wrap the differential information in a single struct so that
/// we can guarantee all or none are present in RayDifferential via a single Option member.
#[derive(Debug, Copy, Clone)]
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
