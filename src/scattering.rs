use crate::vecmath::{vector::Vector3, Normal3f, Vector3f};

#[inline]
pub fn reflect(wo: Vector3f, n: Normal3f) -> Vector3f {
    -wo + 2.0 * wo.dot_normal(n) * Vector3f::from(n)
}
