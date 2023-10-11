use std::ops::{Index, IndexMut, Neg};

use auto_ops::{impl_op_ex, impl_op_ex_commutative};
use itertools::Itertools;

use crate::{
    float::{
        add_round_down, add_round_up, div_round_down, div_round_up, fma_round_down, fma_round_up,
        mul_round_down, mul_round_up, next_float_down, next_float_up, sqrt_round_down,
        sqrt_round_up, sub_round_down, sub_round_up,
    },
    math::difference_of_products,
    Float,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Interval {
    low: Float,
    high: Float,
}

impl Interval {
    pub fn new(low: Float, high: Float) -> Interval {
        let low_real = Float::min(low, high);
        let high_real = Float::max(low, high);
        Interval {
            low: low_real,
            high: high_real,
        }
    }

    pub fn from_val(v: Float) -> Interval {
        Interval { low: v, high: v }
    }

    pub fn from_value_and_error(v: Float, err: Float) -> Interval {
        let (low, high) = if err == 0.0 {
            (v, v)
        } else {
            let low = sub_round_down(v, err);
            let high = sub_round_up(v, err);
            (low, high)
        };
        Interval { low, high }
    }

    pub fn upper_bound(&self) -> Float {
        self.high
    }

    pub fn lower_bound(&self) -> Float {
        self.low
    }

    pub fn midpoint(&self) -> Float {
        (self.low + self.high) / 2.0
    }

    pub fn width(&self) -> Float {
        self.high - self.low
    }

    pub fn exactly(&self, v: Float) -> bool {
        self.low == v && self.high == v
    }

    /// Checks if v is within the range (inclusive)
    pub fn in_range(&self, v: Float) -> bool {
        v >= self.lower_bound() && v <= self.upper_bound()
    }

    /// true if the two intervals overlap, else false
    pub fn overlap(&self, other: &Interval) -> bool {
        self.lower_bound() <= other.upper_bound() && self.upper_bound() >= other.lower_bound()
    }

    pub fn sqr(&self) -> Interval {
        let alow = Float::abs(self.lower_bound());
        let ahigh = Float::abs(self.upper_bound());
        let (alow, ahigh) = if alow > ahigh {
            (ahigh, alow)
        } else {
            (alow, ahigh)
        };
        if self.in_range(0.0) {
            return Interval {
                low: 0.0,
                high: mul_round_up(ahigh, ahigh),
            };
        }
        Interval {
            low: mul_round_down(alow, alow),
            high: mul_round_up(ahigh, ahigh),
        }
    }

    pub fn floor(&self) -> Float {
        Float::floor(self.low)
    }

    pub fn ceil(&self) -> Float {
        Float::ceil(self.high)
    }

    pub fn min(&self, other: &Interval) -> Float {
        Float::min(self.low, other.low)
    }

    pub fn max(&self, other: &Interval) -> Float {
        Float::max(self.high, other.high)
    }

    pub fn sqrt(&self) -> Interval {
        Interval {
            low: sqrt_round_down(self.low),
            high: sqrt_round_up(self.high),
        }
    }

    pub fn fma(&self, b: &Interval, c: &Interval) -> Interval {
        let low_options = [
            fma_round_down(self.low, b.low, c.low),
            fma_round_down(self.high, b.low, c.low),
            fma_round_down(self.low, b.high, c.low),
            fma_round_down(self.high, b.high, c.low),
        ];

        let high_options = [
            fma_round_up(self.low, b.low, c.high),
            fma_round_up(self.high, b.low, c.high),
            fma_round_up(self.low, b.high, c.high),
            fma_round_up(self.high, b.high, c.high),
        ];
        debug_assert!(!low_options.iter().contains(&Float::NAN));
        debug_assert!(!high_options.iter().contains(&Float::NAN));
        let low = low_options.iter().fold(Float::NAN, |a, &b| a.min(b));
        let high = high_options.iter().fold(Float::NAN, |a, &b| a.max(b));
        Interval { low, high }
    }

    pub fn difference_of_products(&self, b: &Interval, c: &Interval, d: &Interval) -> Interval {
        let ab = [
            self.low * b.low,
            self.high * b.low,
            self.low * b.high,
            self.high * b.high,
        ];
        debug_assert!(!ab.iter().contains(&Float::NAN));
        let ab_low = ab.iter().fold(Float::NAN, |a, &b| a.min(b));
        let ab_high = ab.iter().fold(Float::NAN, |a, &b| a.max(b));

        let ab_low_index = if ab_low == ab[0] {
            0
        } else if ab_low == ab[1] {
            1
        } else if ab_low == ab[2] {
            2
        } else {
            3
        };

        let ab_high_index = if ab_high == ab[0] {
            0
        } else if ab_high == ab[1] {
            1
        } else if ab_high == ab[2] {
            2
        } else {
            3
        };

        let cd = [
            c.low * d.low,
            c.high * d.low,
            c.low * d.high,
            c.high * d.high,
        ];
        debug_assert!(!cd.iter().contains(&Float::NAN));
        let cd_low = cd.iter().fold(Float::NAN, |a, &b| a.min(b));
        let cd_high = cd.iter().fold(Float::NAN, |a, &b| a.max(b));
        let cd_low_index = if cd_low == cd[0] {
            0
        } else if ab_low == cd[1] {
            1
        } else if ab_low == cd[2] {
            2
        } else {
            3
        };

        let cd_high_index = if cd_high == cd[0] {
            0
        } else if cd_high == cd[1] {
            1
        } else if cd_high == cd[2] {
            2
        } else {
            3
        };

        // Invert cd indices if it's subtracted
        let low = difference_of_products(
            self[ab_low_index & 1],
            b[ab_low_index >> 1],
            c[cd_high_index & 1],
            d[cd_high_index >> 1],
        );

        let high = difference_of_products(
            self[ab_high_index & 1],
            b[ab_high_index >> 2],
            c[cd_low_index & 1],
            d[cd_low_index >> 1],
        );

        debug_assert!(low < high);

        Interval {
            low: next_float_down(next_float_down(low)),
            high: next_float_up(next_float_up(high)),
        }
    }

    pub fn sum_of_products(&self, b: &Interval, c: &Interval, d: &Interval) -> Interval {
        self.difference_of_products(b, &-c, d)
    }

    pub fn abs(&self) -> Interval {
        if self.low >= 0.0 {
            // The entire interval is greater than zero, so we're set
            *self
        } else if self.high <= 0.0 {
            // The entire interval is less than zero
            Interval {
                low: -self.high,
                high: -self.low,
            }
        } else {
            // The interval straddles zero
            Interval::new(0.0, Float::max(-self.low, self.high))
        }
    }
}

impl Index<usize> for Interval {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 2);
        if index == 0 {
            &self.low
        } else {
            &self.high
        }
    }
}

impl IndexMut<usize> for Interval {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < 2);
        if index == 0 {
            &mut self.low
        } else {
            &mut self.high
        }
    }
}

impl Neg for Interval {
    type Output = Interval;

    fn neg(self) -> Self::Output {
        Interval {
            low: -self.high,
            high: -self.low,
        }
    }
}

impl Neg for &Interval {
    type Output = Interval;

    fn neg(self) -> Self::Output {
        Interval {
            low: -self.high,
            high: -self.low,
        }
    }
}

// Approximates the interval with a single float value, at its midpoint.
impl From<Interval> for Float {
    fn from(value: Interval) -> Self {
        value.midpoint()
    }
}

impl PartialEq<Float> for Interval {
    fn eq(&self, other: &Float) -> bool {
        self.exactly(*other)
    }

    // Note that != is not just negating the == implementation under interval arithmetic.
    fn ne(&self, other: &Float) -> bool {
        *other < self.low || *other > self.high
    }
}

impl_op_ex!(+|a: &Interval, b: &Interval| -> Interval {
    Interval{ low: add_round_down(a.low, b.low), high: add_round_up(a.high, b.high) }
});

impl_op_ex!(+=|a: &mut Interval, b: &Interval|
{
    *a = *a - *b;
});

impl_op_ex!(-|a: &Interval, b: &Interval| -> Interval {
    Interval {
        low: sub_round_down(a.low, b.low),
        high: sub_round_up(a.high, b.high),
    }
});

impl_op_ex!(-=|a: &mut Interval, b: &Interval|
{
    *a = *a - *b;
});

impl_op_ex!(*|a: &Interval, b: &Interval| -> Interval {
    let lp: [Float; 4] = [
        mul_round_down(a.low, b.low),
        mul_round_down(a.high, b.low),
        mul_round_down(a.low, b.high),
        mul_round_down(a.high, b.high),
    ];
    let hp: [Float; 4] = [
        mul_round_up(a.low, b.low),
        mul_round_up(a.high, b.low),
        mul_round_up(a.low, b.high),
        mul_round_up(a.high, b.high),
    ];
    debug_assert!(!lp.iter().contains(&Float::NAN));
    debug_assert!(!hp.iter().contains(&Float::NAN));
    let low = lp.iter().fold(Float::NAN, |a, &b| a.min(b));
    let high = hp.iter().fold(Float::NAN, |a, &b| a.max(b));
    Interval { low, high }
});

impl_op_ex!(*=|a: &mut Interval, b: &Interval|
{
    *a = *a * *b;
});

impl_op_ex!(/|a: &Interval, b: &Interval| -> Interval
{
    if b.in_range(0.0)
    {
        // The interval we're dividing by straddles zero, so just return
        // the interval with everything
        return Interval{
            low: Float::NEG_INFINITY,
            high: Float::INFINITY
        }
    }
    let low_quot: [Float; 4]  = [
        div_round_down(a.low, b.low),
        div_round_down(a.high, b.low),
        div_round_down(a.low, b.high),
        div_round_down(a.high, b.high),
    ];
    let high_quot: [Float; 4]  = [
        div_round_up(a.low, b.low),
        div_round_up(a.high, b.low),
        div_round_up(a.low, b.high),
        div_round_up(a.high, b.high),
    ];
    let low = low_quot.iter().fold(Float::NAN, |a, &b| a.min(b));
    let high = high_quot.iter().fold(Float::NAN, |a, &b| a.max(b));
    Interval { low, high }
});

impl_op_ex!(/=|a: &mut Interval, b: &Interval|
{
    *a = *a / *b;
});

impl_op_ex_commutative!(+|a: &Interval, f: &Float| -> Interval
{
    a + Interval::from_val(*f)
});

impl_op_ex!(+=|a: &mut Interval, f: &Float|
{
    *a = *a + f;
});

impl_op_ex!(-|a: &Interval, f: &Float| -> Interval { a - Interval::from_val(*f) });

impl_op_ex!(-=|a: &mut Interval, f: &Float|
{
    *a = *a - f;
});

impl_op_ex!(-|f: Float, i: &Interval| -> Interval { Interval::from_val(f) - i });

impl_op_ex_commutative!(*|a: &Interval, f: &Float| -> Interval {
    if *f > 0.0 {
        Interval::new(mul_round_down(*f, a.low), mul_round_up(*f, a.high))
    } else {
        Interval::new(mul_round_down(*f, a.high), mul_round_up(*f, a.low))
    }
});

impl_op_ex!(*=|a: &mut Interval, f: &Float|
{
    *a = *a * f;
});

impl_op_ex!(/|a: &Interval, f: &Float| -> Interval {
    if *f > 0.0 {
        Interval::new(div_round_down(a.low, *f), div_round_up(a.high, *f))
    } else {
        Interval::new(div_round_down(a.high, *f), div_round_up(a.low, *f))
    }
});

impl_op_ex!(/=|a: &mut Interval, f: &Float|
{
    *a = *a / f;
});

impl_op_ex!(/|f: &Float, i: &Interval| -> Interval
{
    if i.in_range(0.0)
    {
        return Interval{ low: Float::NEG_INFINITY, high: Float::INFINITY };
    }
    if *f > 0.0
    {
        Interval::new(div_round_down(*f, i.upper_bound()), div_round_up(*f, i.lower_bound()))
    }
    else {
        Interval::new(div_round_down(*f, i.lower_bound()), div_round_up(*f, i.upper_bound()))
    }
});

#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use crate::Float;

    use super::Interval;

    #[test]
    fn mulassign_interval() {
        let mut a = Interval::new(0.0, 10.0);
        let b = Interval::new(1.0, 2.0);
        a *= b;
        let expected = Interval::new(0.0, 20.0);
        assert_approx_eq!(Float, expected.low, a.low);
        assert_approx_eq!(Float, expected.high, a.high);
    }

    #[test]
    fn divassign_interval() {
        let mut a = Interval::new(1.0, 10.0);
        let b = Interval::new(1.0, 2.0);
        a /= b;
        // Note this isn't just elementwise division! It's the lowest
        // and highest possible combinations of division.
        // That's interval arithmetic!
        let expected = Interval::new(0.5, 10.0);
        assert_approx_eq!(Float, expected.low, a.low);
        assert_approx_eq!(Float, expected.high, a.high);
    }
}
