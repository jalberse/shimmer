use std::{mem, ops::{Add, Div, Mul, Sub}};

use fast_polynomial::{poly_array, polynomials};
use num::Complex;

use crate::{
    compensated_float::CompensatedFloat,
    float::{Float, PI_F},
    interval::Interval,
    square_matrix::{Invertible, SquareMatrix},
    vecmath::{Length, Point2f, Tuple2, Tuple3, Vector3f},
};

pub const INV_PI: Float = 0.31830988618379067154;
pub const INV_2PI: Float = 0.15915494309189533577;
pub const INV_4PI: Float = 0.07957747154594766788;
pub const PI_OVER_4: Float = 0.78539816339744830961;
pub const PI_OVER_2: Float = 1.57079632679489661923;

#[inline]
pub fn sqr<T>(x: T) -> T
where
    T: Mul<T, Output = T> + Copy,
{
    x * x
}

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

impl DifferenceOfProducts for f32 {
    /// Computes a * b - c * d using an error-free transformation (EFT) method.
    /// See PBRT B.2.9.
    fn difference_of_products(a: f32, b: f32, c: f32, d: f32) -> f32 {
        let cd = c * d;
        let difference = f32::mul_add(a, b, -cd);
        let error = f32::mul_add(-c, d, cd);
        difference + error
    }

    /// Computes a * b + c * d using an error-free transformation (EFT) method.
    /// See PBRT B.2.9.
    fn sum_of_products(a: f32, b: f32, c: f32, d: f32) -> f32 {
        Self::difference_of_products(a, b, -c, d)
    }
}

impl DifferenceOfProducts for f64 {
    /// Computes a * b - c * d using an error-free transformation (EFT) method.
    /// See PBRT B.2.9.
    fn difference_of_products(a: f64, b: f64, c: f64, d: f64) -> f64 {
        let cd = c * d;
        let difference = f64::mul_add(a, b, -cd);
        let error = f64::mul_add(-c, d, cd);
        difference + error
    }

    /// Computes a * b + c * d using an error-free transformation (EFT) method.
    /// See PBRT B.2.9.
    fn sum_of_products(a: f64, b: f64, c: f64, d: f64) -> f64 {
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

pub fn difference_of_products_float_vec(a: Float, b: Vector3f, c: Float, d: Vector3f) -> Vector3f {
    let cd = c * d;
    let difference = a * b - cd;
    let error = -c * d + cd;
    difference + error
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

pub fn lerp<T>(t: Float, a: T, b: T) -> T
where
    T: Add<T, Output = T>,
    T: Mul<Float, Output = T>,
{
    a * (1.0 - t) + b * t
}

/// Radians from degrees
pub fn radians(deg: Float) -> Float {
    (PI_F / 180.0) * deg
}

/// Degrees from radians
pub fn degrees(rad: Float) -> Float {
    (180.0 / PI_F) * rad
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

/// Returns the square root of the value, clamping it to zero in case of rounding errors
/// causing the value to be slightly negative.
pub fn safe_sqrt(x: Float) -> Float {
    debug_assert!(x >= -1e-3); // not too negative
    Float::sqrt(Float::max(0.0, x))
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
    let mut a_t_a = SquareMatrix::<N>::default();
    let mut a_t_b = SquareMatrix::<N>::default();
    for i in 0..N {
        for j in 0..N {
            for r in 0..ROWS {
                a_t_a[i][j] += a[r][i] * a[r][j];
                a_t_b[i][j] += a[r][i] * b[r][j];
            }
        }
    }
    (a_t_a, a_t_b)
}

pub fn quadratic(a: Float, b: Float, c: Float) -> Option<(Float, Float)>
{
    // Handle the case of a == 0 for quadratic solution
    if a == 0.0
    {
        if b == 0.0
        {
            return None;
        }
        return Some((-c / b, -c / b));
    }

    // Find quadratic discriminant
    let discrim = Float::difference_of_products(b, b, 4.0 * a, c);
    if discrim < 0.0
    {
        return None;
    }

    let root_discrim = Float::sqrt(discrim);

    // Compute quadratic _t_ values
    let q = -0.5 * (b + Float::copysign(root_discrim, b));
    let t0 = q / a;
    let t1 = c / q;
    if t0 > t1
    {
        Some((t1, t0))
    }
    else
    {
        Some((t0, t1))
    }
}

// http://www.plunk.org/~hatch/rightway.html
pub fn sin_over_x(x: Float) -> Float
{
    if 1.0 - x * x == 1.0
    {
        return 1.0;
    }
    x.sin() / x
}

pub fn sinc(x: Float) -> Float 
{
    sin_over_x(PI_F * x)
}

pub fn windowed_sinc(x: Float, radius: Float, tau: Float) -> Float 
{
    if x.abs() > radius
    {
        return 0.0;
    }
    sinc(x) * sinc(x / tau)
}

/// Provides modulo operator, making the result positive.
/// This is useful for e.g. repeating textures, where Rust's default modulo operator
/// would keep the sign the same as the divident, but where we want a positive value.
pub fn modulo<T>(a: T, b: T) -> T
where
    T: Div<T, Output = T> + Mul<T, Output = T> + Add<T, Output = T> + Sub<T, Output = T> + Copy + std::cmp::PartialOrd<i32>,
{
    let result = a - (a / b) * b;
    if result < 0
    {
        result + b
    }
    else
    {
        result
    }
}

/// Square-sphere mapping function definition
/// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
pub fn equal_area_square_to_sphere(p: Point2f) -> Vector3f
{
    assert!(p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0);

    // Transform p to [-1, 1]^2 amd compute abs
    let u = 2.0 * p.x - 1.0;
    let v = 2.0 * p.y - 1.0;
    let up = u.abs();
    let vp = v.abs();

    // Compute radius r as signed distance from diagonal
    let signed_distance = 1.0 - (up + vp);
    let d = signed_distance.abs();
    let r = 1.0 - d;

    // Compute angle phi for square to sphere mapping
    let phi = if r == 0.0 { 1.0 } else { vp - up / r + 1.0} * PI_F / 4.0;

    // Find z coordinate for spherical direction
    let z = Float::copysign(1.0 - sqr(r), signed_distance);

    // Compute cos(phi) and sin(phi) for original quadrant and return the vector
    let cos_phi = Float::copysign(Float::cos(phi), u);
    let sin_phi = Float::copysign(Float::sin(phi), v);
    Vector3f::new(
        cos_phi * r * safe_sqrt(2.0 - sqr(r)),
        sin_phi * r * safe_sqrt(2.0 - sqr(r)),
        z
    )
}

// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
pub fn equal_area_sphere_to_square(d: Vector3f) -> Point2f
{
    debug_assert!(d.length_squared() > 0.999 && d.length_squared() < 1.001);

    let x = d.x.abs();
    let y = d.y.abs();
    let z = d.z.abs();

    let r = safe_sqrt(1.0 - z);

    let a = Float::max(x, y);
    let b = Float::min(x, y);
    let b = if a == 0.0 { 0.0 } else { b / a };

    // // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    let t1 = 0.406758566246788489601959989e-5;
    let t2 = 0.636226545274016134946890922156;
    let t3 = 0.61572017898280213493197203466e-2;
    let t4 = -0.247333733281268944196501420480;
    let t5 = 0.881770664775316294736387951347e-1;
    let t6 = 0.419038818029165735901852432784e-1;
    let t7 = -0.251390972343483509333252996350e-1;

    // TODO Is the order of this okay?
    let mut phi = poly_array(b, &[t1, t2, t3, t4, t5, t6, t7]);

    if x < y{
        phi = 1.0 - phi;
    }

    let mut v = phi * r;
    let mut u = r - v;

    if d.z < 0.0
    {
        // Southern hemisphere, mirror uv
        mem::swap(&mut u, &mut v);
        u = 1.0 - u;
        v = 1.0 - v;
    }

    // Move uv to the correct quadrant based on the signs
    u = Float::copysign(u, d.x);
    v = Float::copysign(v, d.y);

    // Transform from [-1, 1] to [0, 1]
    Point2f::new(
        0.5 * (u + 1.0),
        0.5 * (v + 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::DifferenceOfProducts;
    use crate::Float;
    use fast_polynomial::poly;

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
        assert_eq!(75.0, Float::difference_of_products(a, b, c, d));
    }

    #[test]
    fn test_evaluate_polynomial() {
        // Note that poly takes the coefficients in an order you might not expect.
        // 1 + 2 * x + 3 * x
        let c = [1.0, 2.0, 3.0];
        // 1 * 2^0 + 2 * 2^1 + 3 * 2^2
        assert_eq!(17.0, poly(2.0, &c));
    }
}
