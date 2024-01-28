use crate::{
    math::safe_sqrt,
    vecmath::{normal::Normal3, vector::Vector3, Normal3f, Vector3f},
    Float,
};

#[inline]
pub fn reflect(wo: Vector3f, n: Normal3f) -> Vector3f {
    -wo + 2.0 * wo.dot_normal(n) * Vector3f::from(n)
}

// PAPERDOC - Easy example of where Option is useful, compared to returning a boolean.
// Option ensures both cases are handled, where callers might ignore the boolean.
/// Returns (wt, etap) where wt is the refracted direction and etap is the adjusted relative IOR.
/// wi: Incident direction
/// n: Surface normal in same hemisphere as wi
/// eta: Relative index of refraction
#[inline]
pub fn refract(wi: Vector3f, mut n: Normal3f, mut eta: Float) -> Option<(Vector3f, Float)> {
    let mut cos_theta_i = n.dot_vector(wi);
    // Potentially flip interface orientation for Snell's law
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        n = -n;
    }

    let sin2_theta_i = Float::max(0.0, 1.0 - cos_theta_i * cos_theta_i);
    let sin2_theta_t = eta * eta / sin2_theta_i;

    // Handle total internal reflection case
    if sin2_theta_t >= 1.0 {
        return None;
    }

    let cos_theta_t = Float::sqrt(1.0 - sin2_theta_t);

    let wt = -wi / eta + (cos_theta_i / eta - cos_theta_t) * Vector3f::from(n);
    let etap = eta;
    Some((wt, etap))
}

/// cos_theta_i: Cosine of the angle between the incident direction and the normal.
/// eta: Relative index of refraction.
/// output: Unpolorized fresnel reflection of a dialectric interface
#[inline]
pub fn fresnel_dialectric(cos_theta_i: Float, mut eta: Float) -> Float {
    let mut cos_theta_i = Float::clamp(cos_theta_i, -1.0, 1.0);

    // Potentially flip interface orientation
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin2_theta_t = sin2_theta_i / (eta * eta);
    if sin2_theta_t >= 1.0 {
        return 1.0;
    }

    let cos_theta_t = safe_sqrt(1.0 - sin2_theta_t);

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    0.5 * (r_parl * r_parl + r_perp * r_perp)
}
