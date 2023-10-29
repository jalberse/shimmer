use crate::{
    bounding_box::Bounds3f,
    float::PI_F,
    math::{degrees, safe_acos, safe_sqrt},
    transform::Transform,
    vecmath::{point::Point3, vector::Vector3, Length, Normalize, Point3f, Vector3f},
    Float,
};

/// Similar to a bounding region of space, it can be useful to bound
/// a set of directions. For example, if a light source only emits illumination
/// in some directions but not others, that information can be useful to cull
/// the light source from being included in lighting calculations at points
/// in space that it does not illuminate.
/// This struct provides information to allow this.
#[derive(Debug, Copy, Clone)]
pub struct DirectionCone {
    /// Central direction of cone
    pub w: Vector3f,
    /// The cosine of the spread angle, from w to the extent of the cone.
    /// Storing as the cosine rather than as the raw angle makes certain
    /// calculations more efficient.
    pub cos_theta: Float,
}

impl DirectionCone {
    pub fn new(w: Vector3f, cos_theta: Float) -> DirectionCone {
        DirectionCone {
            w: w.normalize(),
            cos_theta,
        }
    }

    pub fn from_angle(w: Vector3f) -> DirectionCone {
        DirectionCone {
            w: w.normalize(),
            cos_theta: 1.0,
        }
    }

    pub fn entire_sphere() -> DirectionCone {
        DirectionCone {
            w: Vector3f::Z,
            cos_theta: -1.0,
        }
    }

    /// Returns a DirectionCone that bounds the directions subtended by a given bounding
    /// box with respect to a point p
    pub fn bound_subtended_directions(b: &Bounds3f, p: Point3f) -> DirectionCone {
        let bounding_sphere = b.bounding_sphere();
        // Check if the point is inside the bounding sphere
        if p.distance_squared(&bounding_sphere.center)
            < bounding_sphere.radius * bounding_sphere.radius
        {
            return DirectionCone::entire_sphere();
        }
        // Compute and return the DirectionCone for the bounding sphere
        let w = (bounding_sphere.center - p).normalize();
        let sin2_theta_max = bounding_sphere.radius * bounding_sphere.radius
            / bounding_sphere.center.distance_squared(&p);

        let cos_theta_max = safe_sqrt(1.0 - sin2_theta_max);
        DirectionCone::new(w, cos_theta_max)
    }

    pub fn is_empty(&self) -> bool {
        self.cos_theta == Float::INFINITY
    }

    /// Returns true if the vector w is within the bounding direction
    pub fn inside(&self, w: Vector3f) -> bool {
        !self.is_empty() && self.w.dot(&w.normalize()) >= self.cos_theta
    }

    pub fn union(&self, other: &DirectionCone) -> DirectionCone {
        // Handle cases where one or both cones are empty
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        // Handle the case where one cone is inside the other
        let theta_a = safe_acos(self.cos_theta);
        let theta_b = safe_acos(other.cos_theta);
        let theta_d = self.w.angle_between(&other.w);
        if Float::min(theta_d + theta_b, PI_F) <= theta_a {
            return self.clone();
        }
        if Float::min(theta_d + theta_a, PI_F) <= theta_b {
            return other.clone();
        }
        // Compute the spread angle of the merged cone
        let theta_o = (theta_a + theta_b + theta_d) / 2.0;
        if theta_o >= PI_F {
            return DirectionCone::entire_sphere();
        }
        // Find the merged cones' axis and return the cone union
        let theta_r = theta_o - theta_a;
        let wr = self.w.cross(&other.w);
        if wr.length_squared() == 0.0 {
            return DirectionCone::entire_sphere();
        }
        let w = Transform::rotate(degrees(theta_r), &wr).apply(&self.w);
        DirectionCone::new(w, Float::cos(theta_o))
    }
}

impl Default for DirectionCone {
    fn default() -> Self {
        Self {
            w: Default::default(),
            cos_theta: Float::INFINITY, // i.e. zero degrees
        }
    }
}
