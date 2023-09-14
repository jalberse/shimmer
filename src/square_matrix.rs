use std::ops::{Add, Div, DivAssign, Index, Mul, MulAssign};

use auto_ops::impl_op_ex;

use crate::{float::Float, math::MulAdd};

// PAPERDOC - PBRTv4 must implement ==, <, !=.
//   In Rust, you can often derive a trait like so instead. Easier.
#[derive(Debug, PartialEq, PartialOrd)]
pub struct SquareMatrix<const N: usize> {
    pub m: [[Float; N]; N],
}

impl<const N: usize> SquareMatrix<N> {
    pub fn new(m: [[Float; N]; N]) -> Self {
        Self { m }
    }

    pub fn zero() -> Self {
        let mut m: [[Float; N]; N] = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                m[i][j] = 0.0;
            }
        }
        Self::new(m)
    }

    /// Sets the diagonal of the matrix to the provided values with other values 0.
    pub fn diag(vals: [Float; N]) -> Self {
        // PAPERDOC - Rust const generics allow this nice method of passing values in,
        // where PBRTv4 uses variable length parameter lists.
        let mut m: [[Float; N]; N] = [[0.0; N]; N];
        for i in 0..N {
            m[i][i] = vals[i];
        }
        Self::new(m)
    }

    /// Gets N for an N x N matrix.
    pub fn dim(&self) -> usize {
        N
    }

    pub fn is_identity(&self) -> bool {
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    if self.m[i][j] != 1.0 {
                        return false;
                    }
                } else if self.m[i][j] != 0.0 {
                    return false;
                }
            }
        }
        true
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Self::zero();
        for i in 0..N {
            for j in 0..N {
                transposed.m[i][j] = self[j][i];
            }
        }
        transposed
    }

    fn add_mat(&self, other: &SquareMatrix<N>) -> SquareMatrix<N> {
        let mut m = SquareMatrix::zero();
        for i in 0..N {
            for j in 0..N {
                m.m[i][j] = self[i][j] + other[i][j];
            }
        }
        m
    }

    fn mul_float(&self, v: Float) -> SquareMatrix<N> {
        let mut m = SquareMatrix::zero();
        for i in 0..N {
            for j in 0..N {
                m.m[i][j] = self[i][j] * v;
            }
        }
        m
    }

    fn mul_assign_float(&mut self, v: Float) {
        for i in 0..N {
            for j in 0..N {
                self.m[i][j] *= v;
            }
        }
    }

    fn div_float(&self, v: Float) -> SquareMatrix<N> {
        let mut m = SquareMatrix::zero();
        for i in 0..N {
            for j in 0..N {
                m.m[i][j] = self[i][j] / v;
            }
        }
        m
    }

    fn div_assign_float(&mut self, v: Float) {
        for i in 0..N {
            for j in 0..N {
                self.m[i][j] /= v;
            }
        }
    }
}

impl<const N: usize> Default for SquareMatrix<N> {
    fn default() -> Self {
        let mut m: [[Float; N]; N] = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                m[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }
        Self::new(m)
    }
}

impl<const N: usize> Index<usize> for SquareMatrix<N> {
    type Output = [Float; N];

    fn index(&self, index: usize) -> &Self::Output {
        &self.m[index]
    }
}

impl<const N: usize> Add for &SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_mat(rhs)
    }
}

impl<const N: usize> Add for SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_mat(&rhs)
    }
}

impl<const N: usize> Mul<Float> for &SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn mul(self, rhs: Float) -> Self::Output {
        self.mul_float(rhs)
    }
}

impl<const N: usize> Mul<Float> for SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn mul(self, rhs: Float) -> Self::Output {
        self.mul_float(rhs)
    }
}

impl<const N: usize> Mul<SquareMatrix<N>> for Float {
    type Output = SquareMatrix<N>;

    fn mul(self, rhs: SquareMatrix<N>) -> Self::Output {
        rhs.mul_float(self)
    }
}

impl<const N: usize> Div<Float> for &SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn div(self, rhs: Float) -> Self::Output {
        self.div_float(rhs)
    }
}

impl<const N: usize> Div<Float> for SquareMatrix<N> {
    type Output = SquareMatrix<N>;

    fn div(self, rhs: Float) -> Self::Output {
        self.div_float(rhs)
    }
}

impl<const N: usize> MulAssign<Float> for SquareMatrix<N> {
    fn mul_assign(&mut self, rhs: Float) {
        self.mul_assign_float(rhs);
    }
}

impl<const N: usize> DivAssign<Float> for SquareMatrix<N> {
    fn div_assign(&mut self, rhs: Float) {
        self.div_assign_float(rhs);
    }
}

// TODO multiply two matrices - I think can't be generic over N?

// TODO invert matrix (we don't need invertOrFail - caller can unwrap and panic if they want)
//  I think can't be generic over N.

// TODO Detemrinant of matrix. While I think this *can* be generic over N,
//      we don't need N > 4, and specific implementations for N < 4 can be more efficient,
//      so let's implement it for each type N = 1, 2, 3, 4.

// TODO multiplication with a vector

mod tests {
    use crate::Float;

    use super::SquareMatrix;

    #[test]
    fn zero() {
        let m = SquareMatrix::<4>::zero();
        for i in 0..m.dim() {
            for j in 0..m.dim() {
                assert_eq!(0.0, m[i][j]);
            }
        }
    }

    #[test]
    fn diag() {
        let m = SquareMatrix::<4>::diag([0.0, 1.0, 2.0, 3.0]);
        for i in 0..m.dim() {
            for j in 0..m.dim() {
                if i == j {
                    assert_eq!(i as Float, m[i][j]);
                } else {
                    assert_eq!(0.0, m[i][j])
                }
            }
        }
    }

    #[test]
    fn index() {
        let m = SquareMatrix::<4>::diag([0.0, 1.0, 2.0, 3.0]);
        let row1 = m[1];
        assert_eq!([0.0, 1.0, 0.0, 0.0], row1);
        let bottom_corner = m[3][3];
        assert_eq!(3.0, bottom_corner);
    }

    #[test]
    fn is_identity() {
        let m = SquareMatrix::<4>::default();
        assert!(m.is_identity());

        let m = SquareMatrix::<4>::diag([0.0, 1.0, 2.0, 3.0]);
        assert!(!m.is_identity());

        let m = SquareMatrix::<4>::zero();
        assert!(!m.is_identity());
    }

    #[test]
    fn add_matrices() {
        let m1 = SquareMatrix::<4>::diag([1.0, 2.0, 3.0, 4.0]);
        let m2 = SquareMatrix::<4>::diag([11.0, 12.0, 13.0, 14.0]);
        let sum = m1 + m2;
        assert_eq!(SquareMatrix::<4>::diag([12.0, 14.0, 16.0, 18.0]), sum);
    }

    #[test]
    fn mat_mul_float() {
        let m = SquareMatrix::<4>::diag([1.0, 2.0, 3.0, 4.0]);
        let scaled = m * 2.0;
        assert_eq!(SquareMatrix::<4>::diag([2.0, 4.0, 6.0, 8.0]), scaled);
        let m = SquareMatrix::<4>::diag([1.0, 2.0, 3.0, 4.0]);
        let scaled = 2.0 * m;
        assert_eq!(SquareMatrix::<4>::diag([2.0, 4.0, 6.0, 8.0]), scaled);
    }

    #[test]
    fn mat_div_float() {
        let m = SquareMatrix::<4>::diag([2.0, 4.0, 6.0, 8.0]);
        let scaled = m / 2.0;
        assert_eq!(SquareMatrix::<4>::diag([1.0, 2.0, 3.0, 4.0]), scaled);
    }

    #[test]
    fn mat_mul_assign_float() {
        let mut m = SquareMatrix::<4>::diag([1.0, 2.0, 3.0, 4.0]);
        m *= 2.0;
        assert_eq!(SquareMatrix::<4>::diag([2.0, 4.0, 6.0, 8.0]), m);
    }

    #[test]
    fn mat_div_assign_float() {
        let mut m = SquareMatrix::<4>::diag([4.0, 8.0, 12.0, 16.0]);
        m /= 2.0;
        assert_eq!(SquareMatrix::<4>::diag([2.0, 4.0, 6.0, 8.0]), m);
    }

    #[test]
    fn mat_transpose() {
        let m = SquareMatrix::<4>::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let transposed = m.transpose();
        assert_eq!(
            SquareMatrix::<4>::new([
                [1.0, 5.0, 9.0, 13.0],
                [2.0, 6.0, 10.0, 14.0],
                [3.0, 7.0, 11.0, 15.0],
                [4.0, 8.0, 12.0, 16.0],
            ]),
            transposed
        );
    }
}
