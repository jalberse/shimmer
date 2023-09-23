use std::{
    ops::{Add, Mul},
    path::Component,
};

use crate::{compensated_float::CompensatedFloat, float::Float};

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

impl Sqrt for Float {
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }
}

impl Sqrt for i32 {
    fn sqrt(self) -> Self {
        (self as f32).sqrt() as i32
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

pub trait Min {
    // Take the minimum of self and a
    fn min(self, a: Self) -> Self;
}

impl Min for Float {
    fn min(self, a: Self) -> Self {
        Float::min(self, a)
    }
}

impl Min for i32 {
    fn min(self, a: Self) -> Self {
        <i32 as Ord>::min(self, a)
    }
}
pub trait Max {
    // Take the maximum of self and a
    fn max(self, a: Self) -> Self;
}

impl Max for Float {
    fn max(self, a: Self) -> Self {
        Float::max(self, a)
    }
}

impl Max for i32 {
    fn max(self, a: Self) -> Self {
        <i32 as Ord>::max(self, a)
    }
}

/// Provides the maximum and minimum possible representable number
pub trait NumericLimit {
    const MIN: Self;
    const MAX: Self;
}

impl NumericLimit for i32 {
    const MIN: i32 = i32::MIN;
    const MAX: i32 = i32::MAX;
}

impl NumericLimit for Float {
    const MIN: Float = Float::MIN;
    const MAX: Float = Float::MAX;
}

/// Provides the equivalent of f32::mul_add for the specified type.
/// Really just provided so that we can use a common interface for Float
/// and integer types, even if the integer types won't benefit from this form.
pub trait MulAdd {
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl MulAdd for Float {
    fn mul_add(self, a: Self, b: Self) -> Self {
        Float::mul_add(self, a, b)
    }
}

impl MulAdd for i32 {
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
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

/// Computes a * b + c * d using an error-free transformation (EFT) method.
/// See PBRT B.2.9.
pub fn sum_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
    difference_of_products(a, b, -c, d)
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

/// Inner Products with error free transformations.
/// Computes the inner product using f32 precision, with error
/// equivalent to f64 precision.
///
///  The xs and ys slices provide the following values, in pairs:
/// x0 * y0 + x1 * y1 + x2 * y2 + ...
/// xs and ys must be of the same length.
///
/// Accurate dot products with FMA: Graillat et al.,
/// https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf
///
/// Accurate summation, dot product and polynomial evaluation in complex
/// floating point arithmetic, Graillat and Menissier-Morain.
pub fn inner_product(xs: &[Float], ys: &[Float]) -> CompensatedFloat {
    // PAPERDOC this is arguable a more elegant solution with slices than variadic parameters.
    debug_assert!(xs.len() == ys.len());
    if xs.len() == 1 {
        // Base case
        return two_prod(xs[0], ys[0]);
    } else {
        let ab = two_prod(xs[0], ys[0]);
        let tp = inner_product(&xs[1..xs.len()], &ys[1..ys.len()]);
        let sum = two_sum(ab.v, tp.v);
        return CompensatedFloat::new(sum.v, ab.err + tp.err + sum.err);
    }
}

fn two_prod(a: Float, b: Float) -> CompensatedFloat {
    let ab = a * b;
    let err = Float::mul_add(a, b, -ab);
    CompensatedFloat::new(ab, err)
}

fn two_sum(a: Float, b: Float) -> CompensatedFloat {
    let s = a + b;
    let delta = s - a;
    let err = (a - (s - delta)) + (b - delta);
    CompensatedFloat::new(s, err)
}

pub fn find_interval<T>(size: usize, pred: T)
where
    T: Fn(&[Float], usize) -> usize,
{
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
