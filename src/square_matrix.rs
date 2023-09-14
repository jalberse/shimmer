use std::ops::{Add, Div, DivAssign, Index, Mul, MulAssign};

use crate::{float::Float, math::difference_of_products};

pub trait Invertible: Sized {
    fn inverse(&self) -> Option<Self>;
}

pub trait Determinant {
    fn determinant(&self) -> Float;
}

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

// PAPERDOC - In PBRTv4/C++, this would typically be done via function
// overloading for the different values of N. That doesn't allow us to e.g.
// constrain a generic on Determinant.
impl Determinant for SquareMatrix<1> {
    fn determinant(&self) -> Float {
        self[0][0]
    }
}

impl Determinant for SquareMatrix<2> {
    fn determinant(&self) -> Float {
        difference_of_products(self[0][0], self[1][1], self[0][1], self[1][0])
    }
}

impl Determinant for SquareMatrix<3> {
    fn determinant(&self) -> Float {
        let minor12 = difference_of_products(self[1][1], self[2][2], self[1][2], self[2][1]);
        let minor02 = difference_of_products(self[1][0], self[2][2], self[1][2], self[2][0]);
        let minor01 = difference_of_products(self[1][0], self[2][1], self[1][1], self[2][0]);
        Float::mul_add(
            self[0][2],
            minor01,
            difference_of_products(self[0][0], minor12, self[0][1], minor02),
        )
    }
}

impl Invertible for SquareMatrix<3> {
    fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }

        let inv_det = 1.0 / det;

        let mut out = SquareMatrix::<3>::zero();

        out.m[0][0] =
            inv_det * difference_of_products(self[1][1], self[2][2], self[1][2], self[2][1]);
        out.m[1][0] =
            inv_det * difference_of_products(self[1][2], self[2][0], self[1][0], self[2][2]);
        out.m[2][0] =
            inv_det * difference_of_products(self[1][0], self[2][1], self[1][1], self[2][0]);
        out.m[0][1] =
            inv_det * difference_of_products(self[0][2], self[2][1], self[0][1], self[2][2]);
        out.m[1][1] =
            inv_det * difference_of_products(self[0][0], self[2][2], self[0][2], self[2][0]);
        out.m[2][1] =
            inv_det * difference_of_products(self[0][1], self[2][0], self[0][0], self[2][1]);
        out.m[0][2] =
            inv_det * difference_of_products(self[0][1], self[1][2], self[0][2], self[1][1]);
        out.m[1][2] =
            inv_det * difference_of_products(self[0][2], self[1][0], self[0][0], self[1][2]);
        out.m[2][2] =
            inv_det * difference_of_products(self[0][0], self[1][1], self[0][1], self[1][0]);

        Some(out)
    }
}

impl Invertible for SquareMatrix<4> {
    fn inverse(&self) -> Option<Self> {
        // Via: https://github.com/google/ion/blob/master/ion/math/matrixutils.cc,
        // (c) Google, Apache license.

        // For 4x4 do not compute the adjugate as the transpose of the cofactor
        // matrix, because this results in extra work. Several calculations can be
        // shared across the sub-determinants.
        //
        // This approach is explained in David Eberly's Geometric Tools book,
        // excerpted here:
        //   http://www.geometrictools.com/Documentation/LaplaceExpansionTheorem.pdf

        let s0 = difference_of_products(self[0][0], self[1][1], self[1][0], self[0][1]);
        let s1 = difference_of_products(self[0][0], self[1][2], self[1][0], self[0][2]);
        let s2 = difference_of_products(self[0][0], self[1][3], self[1][0], self[0][3]);
        let s3 = difference_of_products(self[0][1], self[1][2], self[1][1], self[0][2]);
        let s4 = difference_of_products(self[0][1], self[1][3], self[1][1], self[0][3]);
        let s5 = difference_of_products(self[0][2], self[1][3], self[1][2], self[0][3]);
        let c0 = difference_of_products(self[2][0], self[3][1], self[3][0], self[2][1]);
        let c1 = difference_of_products(self[2][0], self[3][2], self[3][0], self[2][2]);
        let c2 = difference_of_products(self[2][0], self[3][3], self[3][0], self[2][3]);
        let c3 = difference_of_products(self[2][1], self[3][2], self[3][1], self[2][2]);
        let c4 = difference_of_products(self[2][1], self[3][3], self[3][1], self[2][3]);
        let c5 = difference_of_products(self[2][2], self[3][3], self[3][2], self[2][3]);

        // TODO We need InnerProduct. Have made a CompensatedFloat; need to use it for implementing InnerProduct.

        todo!()
    }
}

impl Determinant for SquareMatrix<4> {
    fn determinant(&self) -> Float {
        let s0 = difference_of_products(self[0][0], self[1][1], self[1][0], self[0][1]);
        let s1 = difference_of_products(self[0][0], self[1][2], self[1][0], self[0][2]);
        let s2 = difference_of_products(self[0][0], self[1][3], self[1][0], self[0][3]);

        let s3 = difference_of_products(self[0][1], self[1][2], self[1][1], self[0][2]);
        let s4 = difference_of_products(self[0][1], self[1][3], self[1][1], self[0][3]);
        let s5 = difference_of_products(self[0][2], self[1][3], self[1][2], self[0][3]);

        let c0 = difference_of_products(self[2][0], self[3][1], self[3][0], self[2][1]);
        let c1 = difference_of_products(self[2][0], self[3][2], self[3][0], self[2][2]);
        let c2 = difference_of_products(self[2][0], self[3][3], self[3][0], self[2][3]);

        let c3 = difference_of_products(self[2][1], self[3][2], self[3][1], self[2][2]);
        let c4 = difference_of_products(self[2][1], self[3][3], self[3][1], self[2][3]);
        let c5 = difference_of_products(self[2][2], self[3][3], self[3][2], self[2][3]);

        difference_of_products(s0, c5, s1, c4)
            + difference_of_products(s2, c3, -s3, c2)
            + difference_of_products(s5, c0, s4, c1)
    }
}

// TODO multiplication with a vector

mod tests {
    use crate::{square_matrix::Determinant, Float};

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

    #[test]
    fn determinants() {
        let m1 = SquareMatrix::<1>::new([[1.0]]);
        assert_eq!(1.0, m1.determinant());
        let m2 = SquareMatrix::<2>::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(-2.0, m2.determinant());
        let m3 = SquareMatrix::<3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(0.0, m3.determinant());
        let m4 = SquareMatrix::<4>::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        assert_eq!(0.0, m4.determinant());

        // Test less-trivial examples.
        let m5 = SquareMatrix::<4>::new([
            [4.0, 3.0, 2.0, 2.0],
            [0.0, 1.0, -3.0, 3.0],
            [0.0, -1.0, 3.0, 3.0],
            [0.0, 3.0, 1.0, 1.0],
        ]);
        assert_eq!(-240.0, m5.determinant());

        let m6 = SquareMatrix::<3>::new([[2.0, -3.0, 1.0], [2.0, 0.0, -1.0], [1.0, 4.0, 5.0]]);
        assert_eq!(49.0, m6.determinant());
    }
}
