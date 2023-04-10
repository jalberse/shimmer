use crate::float::Float;

#[inline]
pub fn lerp(x: Float, a: Float, b: Float) -> Float {
    (1.0 - x) * a + x * b
}
