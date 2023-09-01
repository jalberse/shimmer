use crate::float::Float;

pub fn lerp(x: Float, a: Float, b: Float) -> Float {
    (1.0 - x) * a + x * b
}

/// Computes a * b - c * d using an error-free transformation (EFT) method.
/// See PBRT B.2.9.
pub fn difference_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
    let cd = c * d;
    let difference = Float::mul_add(a, b, -cd);
    let error = Float::mul_add(-c, d, cd);
    difference + error
}

mod tests {
    #[test]
    fn lerp() {
        let a = 0.0;
        let b = 10.0;
        let x = 0.45;
        assert_eq!(4.5, super::lerp(x, a, b));
    }

    #[test]
    fn test_difference_of_products() {
        // This won't test that this is more accurate than a regular difference necessarily,
        // we're just showing general correctness. We'll trust the literature on it being more
        // accurate for now.
        let a = 10.0;
        let b = 10.0;
        let c = 5.0;
        let d = 5.0;
        assert_eq!(75.0, super::difference_of_products(a, b, c, d));
    }
}
