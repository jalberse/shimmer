use std::ops::{Add, Mul};

use crate::float::Float;

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

impl Sqrt for Float {
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }
}

pub trait Abs {
    fn abs(self) -> Self;
}

impl Abs for Float {
    fn abs(self) -> Self {
        Float::abs(self)
    }
}

impl Abs for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }
}

pub trait Ceil {
    fn ceil(self) -> Self;
}

impl Ceil for Float {
    fn ceil(self) -> Self {
        Float::ceil(self)
    }
}

impl Ceil for i32 {
    fn ceil(self) -> Self {
        self
    }
}

pub trait Floor {
    fn floor(self) -> Self;
}

impl Floor for Float {
    fn floor(self) -> Self {
        Float::floor(self)
    }
}

impl Floor for i32 {
    fn floor(self) -> Self {
        self
    }
}

pub fn lerp<'a, T>(t: Float, a: &'a T, b: &'a T) -> T
where
    T: Add<T, Output = T>,
    &'a T: Mul<Float, Output = T>,
{
    a * (1.0 - t) + b * t
}

/// Computes a * b - c * d using an error-free transformation (EFT) method.
/// See PBRT B.2.9.
pub fn difference_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
    let cd = c * d;
    let difference = Float::mul_add(a, b, -cd);
    let error = Float::mul_add(-c, d, cd);
    difference + error
}

/// asin, with a check to ensure output is not slightly outside the legal range [-1, 1]
/// See PBRTv4 8.2.3
pub fn safe_asin(x: Float) -> Float {
    Float::asin(Float::clamp(x, -1.0, 1.0))
}

/// acos, with a check to ensure output is not slightly outside the legal range [-1, 1]
/// See PBRTv4 8.2.3
pub fn safe_acos(x: Float) -> Float {
    Float::asin(Float::clamp(x, -1.0, 1.0))
}

mod tests {
    #[test]
    fn lerp() {
        let a = 0.0;
        let b = 10.0;
        let x = 0.45;
        assert_eq!(4.5, super::lerp(x, &a, &b));
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
