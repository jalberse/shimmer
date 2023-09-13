use std::ops::{Add, Index, Mul};

use auto_ops::impl_op_ex;

use crate::float::Float;

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

// TODO divide matrix by scalar (for convenience)

// TODO transpose

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
}
