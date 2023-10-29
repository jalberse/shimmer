use crate::{float::PI_F, math::safe_acos, Float};

use super::{vector::Vector3, Tuple3, Vector3f};

pub fn spherical_triangle_area(a: Vector3f, b: Vector3f, c: Vector3f) -> Float {
    Float::abs(2.0 * Float::atan2(a.dot(&b.cross(&c)), 1.0 + a.dot(&b) + a.dot(&c) + b.dot(&c)))
}
pub fn spherical_direction(sin_theta: Float, cos_theta: Float, phi: Float) -> Vector3f {
    Vector3f::new(
        sin_theta.clamp(-1.0, 1.0) * Float::cos(phi),
        sin_theta.clamp(-1.0, 1.0) * Float::sin(phi),
        cos_theta.clamp(-1.0, 1.0),
    )
}

pub fn spherical_theta(v: Vector3f) -> Float {
    safe_acos(v.z)
}

pub fn spherical_phi(v: Vector3f) -> Float {
    let p = Float::atan2(v.y, v.x);
    if p < 0.0 {
        p + 2.0 * PI_F
    } else {
        p
    }
}

pub fn cos_theta(w: Vector3f) -> Float {
    w.z
}

pub fn cos2_theta(w: Vector3f) -> Float {
    w.z * w.z
}

pub fn abs_cos_theta(w: Vector3f) -> Float {
    Float::abs(w.z)
}

pub fn sin2_theta(w: Vector3f) -> Float {
    Float::max(0.0, 1.0 - cos2_theta(w))
}

pub fn sin_theta(w: Vector3f) -> Float {
    Float::sqrt(sin2_theta(w))
}

pub fn tan_theta(w: Vector3f) -> Float {
    sin_theta(w) / cos_theta(w)
}

pub fn tan2_theta(w: Vector3f) -> Float {
    sin2_theta(w) / cos2_theta(w)
}

pub fn cos_phi(w: Vector3f) -> Float {
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        1.0
    } else {
        Float::clamp(w.x / sin_theta, -1.0, 1.0)
    }
}

pub fn sin_phi(w: Vector3f) -> Float {
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        1.0
    } else {
        Float::clamp(w.y / sin_theta, -1.0, 1.0)
    }
}

pub fn cos_d_phi(wa: Vector3f, wb: Vector3f) -> Float {
    let waxy = wa.x * wa.x + wa.y * wa.y;
    let wbxy = wb.x * wb.x + wb.y * wb.y;
    if waxy == 0.0 || wbxy == 0.0 {
        return 1.0;
    }
    Float::clamp(
        (wa.x * wb.x + wa.y * wb.y) / Float::sqrt(waxy * wbxy),
        -1.0,
        1.0,
    )
}
