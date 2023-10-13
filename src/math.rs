use std::ops::{Add, Mul};

use crate::{
    compensated_float::CompensatedFloat,
    float::Float,
    interval::Interval,
    square_matrix::{Invertible, SquareMatrix},
};

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

impl NumericLimit for Interval {
    const MIN: Interval = Interval::from_val(Float::NEG_INFINITY);
    const MAX: Interval = Interval::from_val(Float::INFINITY);
}

/// Provides the equivalent of f32::mul_add for the specified type.
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

pub trait DifferenceOfProducts {
    fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self;

    fn sum_of_products(a: Self, b: Self, c: Self, d: Self) -> Self;
}

impl DifferenceOfProducts for Float {
    /// Computes a * b - c * d using an error-free transformation (EFT) method.
    /// See PBRT B.2.9.
    fn difference_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
        let cd = c * d;
        let difference = Float::mul_add(a, b, -cd);
        let error = Float::mul_add(-c, d, cd);
        difference + error
    }

    /// Computes a * b + c * d using an error-free transformation (EFT) method.
    /// See PBRT B.2.9.
    fn sum_of_products(a: Float, b: Float, c: Float, d: Float) -> Float {
        Self::difference_of_products(a, b, -c, d)
    }
}

impl DifferenceOfProducts for i32 {
    fn difference_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        a * b - c * d
    }

    fn sum_of_products(a: Self, b: Self, c: Self, d: Self) -> Self {
        a * b + c * d
    }
}

pub trait IsNeg {
    fn is_neg(&self) -> bool;
}

impl IsNeg for i32 {
    fn is_neg(&self) -> bool {
        self < &0
    }
}

impl IsNeg for Float {
    fn is_neg(&self) -> bool {
        self < &0.0
    }
}

impl IsNeg for Interval {
    /// Returns true if the midpoint is negative, rather than if the entire interval is
    /// negative. This is useful for e.g. checking if the dot of two Vector3<Interval> classes
    /// is negative.
    fn is_neg(&self) -> bool {
        Into::<Float>::into(*self) < 0.0
    }
}

pub fn lerp<'a, T>(t: Float, a: &'a T, b: &'a T) -> T
where
    T: Add<T, Output = T>,
    &'a T: Mul<Float, Output = T>,
{
    a * (1.0 - t) + b * t
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

pub fn evaluate_polynomial(t: Float, c: &[Float]) -> Float {
    // TODO Consider using FMA as PBRT does. I previous did that but was getting the wrong answer; we'll go with KISS for now.
    t * t * c[0] + t * c[1] + c[2]
}

pub fn find_interval(size: usize, pred: impl Fn(usize) -> bool) -> usize {
    let mut first = 1;
    let mut last = size as i32 - 2;
    while last > 0 {
        let half = last >> 1;
        let middle = first + half;
        let pred_result = pred(middle.try_into().unwrap());
        first = if pred_result { middle + 1 } else { first };
        last = if pred_result { last - (half + 1) } else { half };
    }
    i32::clamp(first - 1, 0, size as i32 - 2) as usize
}

pub fn linear_least_squares_3<const ROWS: usize>(
    a: &[[Float; 3]; ROWS],
    b: &[[Float; 3]; ROWS],
) -> Option<SquareMatrix<3>> {
    let (at_a, at_b) = linear_least_squares_helper::<3, ROWS>(a, b);
    let at_ai = at_a.inverse()?;
    Some((at_ai * at_b).transpose())
}

// TODO test
pub fn linear_least_squares_4<const ROWS: usize>(
    a: &[[Float; 4]; ROWS],
    b: &[[Float; 4]; ROWS],
) -> Option<SquareMatrix<4>> {
    let (at_a, at_b) = linear_least_squares_helper::<4, ROWS>(a, b);

    // We don't implement Invertible for all N, so we don't implement Invertible for a generic N,
    // only for specific values e.g. SquareMatrix<3> and SquareMatrix<4>.
    // This is due to Rust not supporting specialization for const generics.
    // Here's someone running into the exact same issue, actually: https://stackoverflow.com/questions/74761968/rust-matrix-type-with-specialized-and-generic-functions
    let at_ai = at_a.inverse()?;
    Some((at_ai * at_b).transpose())
}

// This section of linear least squares can be generic over N, so let's do that.
pub fn linear_least_squares_helper<const N: usize, const ROWS: usize>(
    a: &[[Float; N]; ROWS],
    b: &[[Float; N]; ROWS],
) -> (SquareMatrix<N>, SquareMatrix<N>) {
    let mut AtA = SquareMatrix::<N>::default();
    let mut AtB = SquareMatrix::<N>::default();
    for i in 0..N {
        for j in 0..N {
            for r in 0..ROWS {
                AtA[i][j] += a[r][i] * a[r][j];
                AtB[i][j] += a[r][i] * b[r][j];
            }
        }
    }
    (AtA, AtB)
}

#[cfg(test)]
mod tests {
    use super::DifferenceOfProducts;
    use crate::Float;

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
        assert_eq!(75.0, Float::difference_of_products(a, b, c, d));
    }
}
