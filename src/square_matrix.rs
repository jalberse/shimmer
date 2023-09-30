use std::ops::{Add, Div, DivAssign, Index, IndexMut, Mul, MulAssign};

use auto_ops::impl_op_ex;

use crate::{
    color::XYZ,
    float::Float,
    math::{difference_of_products, inner_product},
};

pub trait Invertible: Sized {
    fn inverse(&self) -> Option<Self>;
}

pub trait Determinant {
    fn determinant(&self) -> Float;
}

// PAPERDOC - PBRTv4 must implement ==, <, !=.
//   In Rust, you can often derive a trait like so instead. Easier.
#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
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
        // I think this is also possible in C++, though.
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

impl_op_ex!(
    *|m1: &SquareMatrix<3>, m2: &SquareMatrix<3>| -> SquareMatrix<3> {
        let mut out = SquareMatrix::<3>::zero();
        for i in 0..3 {
            for j in 0..3 {
                out.m[i][j] = Float::from(inner_product(
                    &[m1[i][0], m1[i][1], m1[i][2]],
                    &[m2[0][j], m2[1][j], m2[2][j]],
                ));
            }
        }
        out
    }
);

impl_op_ex!(
    *|m1: &SquareMatrix<4>, m2: &SquareMatrix<4>| -> SquareMatrix<4> {
        let mut out = SquareMatrix::<4>::zero();
        for i in 0..4 {
            for j in 0..4 {
                out.m[i][j] = Float::from(inner_product(
                    &[m1[i][0], m1[i][1], m1[i][2], m1[i][3]],
                    &[m2[0][j], m2[1][j], m2[2][j], m2[3][j]],
                ));
            }
        }
        out
    }
);

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

        let det: Float =
            inner_product(&[s0, -s1, s2, s3, s5, -s4], &[c5, c4, c3, c2, c0, c1]).into();
        if det == 0.0 {
            return None;
        }

        let s = 1.0 / det;

        let inv: [[Float; 4]; 4] = [
            [
                s * Float::from(inner_product(
                    &[self[1][1], self[1][3], -self[1][2]],
                    &[c5, c3, c4],
                )),
                s * Float::from(inner_product(
                    &[-self[0][1], self[0][2], -self[0][3]],
                    &[c5, c4, c3],
                )),
                s * Float::from(inner_product(
                    &[self[3][1], self[3][3], -self[3][2]],
                    &[s5, s3, s4],
                )),
                s * Float::from(inner_product(
                    &[-self[2][1], self[2][2], -self[2][3]],
                    &[s5, s4, s3],
                )),
            ],
            [
                s * Float::from(inner_product(
                    &[-self[1][0], self[1][2], -self[1][3]],
                    &[c5, c2, c1],
                )),
                s * Float::from(inner_product(
                    &[self[0][0], self[0][3], -self[0][2]],
                    &[c5, c1, c2],
                )),
                s * Float::from(inner_product(
                    &[-self[3][0], self[3][2], -self[3][3]],
                    &[s5, s2, s1],
                )),
                s * Float::from(inner_product(
                    &[self[2][0], self[2][3], -self[2][2]],
                    &[s5, s1, s2],
                )),
            ],
            [
                s * Float::from(inner_product(
                    &[self[1][0], self[1][3], -self[1][1]],
                    &[c4, c0, c2],
                )),
                s * Float::from(inner_product(
                    &[-self[0][0], self[0][1], -self[0][3]],
                    &[c4, c2, c0],
                )),
                s * Float::from(inner_product(
                    &[self[3][0], self[3][3], -self[3][1]],
                    &[s4, s0, s2],
                )),
                s * Float::from(inner_product(
                    &[-self[2][0], self[2][1], -self[2][3]],
                    &[s4, s2, s0],
                )),
            ],
            [
                s * Float::from(inner_product(
                    &[-self[1][0], self[1][1], -self[1][2]],
                    &[c3, c1, c0],
                )),
                s * Float::from(inner_product(
                    &[self[0][0], self[0][2], -self[0][1]],
                    &[c3, c0, c1],
                )),
                s * Float::from(inner_product(
                    &[-self[3][0], self[3][1], -self[3][2]],
                    &[s3, s1, s0],
                )),
                s * Float::from(inner_product(
                    &[self[2][0], self[2][2], -self[2][1]],
                    &[s3, s0, s1],
                )),
            ],
        ];

        Some(SquareMatrix::<4>::new(inv))
    }
}

/// Multiply a matrix with a vector and output another type.
/// The output type U is distinct from V because this can be used to e.g.
/// convert an RGB color representation into an XYZ color representation.
/// Note that it might be best to impl Mul for various types (using
/// this function as a helper function) rather than calling this directly,
/// just for the sake of more readable code.
pub fn mul_mat_vec<const N: usize, V, VResult>(m: &SquareMatrix<N>, v: &V) -> VResult
where
    V: Index<usize, Output = Float>,
    VResult: IndexMut<usize, Output = Float> + Default,
{
    // TODO consider requiring V to have some const generic N as well,
    // so that our type system can enforce N == N.

    // PAPREDOC - trait bounds make this generic code much more clear (if more verbose) than PBRTv4 page 1050.
    // It makes clear what the generic types must satisfy.
    let mut out: VResult = Default::default();
    for i in 0..N {
        for j in 0..N {
            out[i] += m[i][j] * v[j];
        }
    }
    out
}

impl Mul<XYZ> for SquareMatrix<3> {
    type Output = XYZ;

    fn mul(self, rhs: XYZ) -> Self::Output {
        mul_mat_vec::<3, XYZ, XYZ>(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::vecmath::tuple::Tuple3;
    use crate::{square_matrix::Determinant, vecmath::Vector3f, Float};

    use super::{mul_mat_vec, Invertible, SquareMatrix};

    use float_cmp::approx_eq;

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

    #[test]
    fn inverse_3x3() {
        let m = SquareMatrix::<3>::new([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 1.0, 3.0]]);
        let inverse = m.inverse().unwrap();
        let expected = SquareMatrix::<3>::new([
            [-5.0 / 12.0, 1.0 / 4.0, 1.0 / 3.0],
            [7.0 / 12.0, 1.0 / 4.0, -2.0 / 3.0],
            [1.0 / 12.0, -1.0 / 4.0, 1.0 / 3.0],
        ]);
        assert!(approx_eq!(Float, expected[0][0], inverse[0][0]));
        assert!(approx_eq!(Float, expected[0][1], inverse[0][1]));
        assert!(approx_eq!(Float, expected[0][2], inverse[0][2]));

        assert!(approx_eq!(Float, expected[1][0], inverse[1][0]));
        assert!(approx_eq!(Float, expected[1][1], inverse[1][1]));
        assert!(approx_eq!(Float, expected[1][2], inverse[1][2]));

        assert!(approx_eq!(Float, expected[2][0], inverse[2][0]));
        assert!(approx_eq!(Float, expected[2][1], inverse[2][1]));
        assert!(approx_eq!(Float, expected[2][2], inverse[2][2]));

        // Now, a non-invertible (singular) matrix...
        let m = SquareMatrix::<3>::new([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0], [3.0, 2.0, -1.0]]);
        let inverse = m.inverse();
        assert!(inverse.is_none());
    }

    #[test]
    fn inverse_4x4() {
        let m = SquareMatrix::<4>::new([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]);
        let inverse = m.inverse().unwrap();
        let expected = SquareMatrix::<4>::new([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0, -2.0],
            [1.0, -1.0, -2.0, 2.0],
        ]);

        assert!(approx_eq!(Float, expected[0][0], inverse[0][0]));
        assert!(approx_eq!(Float, expected[0][1], inverse[0][1]));
        assert!(approx_eq!(Float, expected[0][2], inverse[0][2]));
        assert!(approx_eq!(Float, expected[0][3], inverse[0][3]));

        assert!(approx_eq!(Float, expected[1][0], inverse[1][0]));
        assert!(approx_eq!(Float, expected[1][1], inverse[1][1]));
        assert!(approx_eq!(Float, expected[1][2], inverse[1][2]));
        assert!(approx_eq!(Float, expected[1][3], inverse[1][3]));

        assert!(approx_eq!(Float, expected[2][0], inverse[2][0]));
        assert!(approx_eq!(Float, expected[2][1], inverse[2][1]));
        assert!(approx_eq!(Float, expected[2][2], inverse[2][2]));
        assert!(approx_eq!(Float, expected[2][3], inverse[2][3]));

        assert!(approx_eq!(Float, expected[3][0], inverse[3][0]));
        assert!(approx_eq!(Float, expected[3][1], inverse[3][1]));
        assert!(approx_eq!(Float, expected[3][2], inverse[3][2]));
        assert!(approx_eq!(Float, expected[3][3], inverse[3][3]));

        // Now, a singular matrix...
        let m = SquareMatrix::<4>::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [0.0, 0.0, 0.0, 0.0], // All zeroes, det == 0, thus singular
            [1.0, 2.0, 3.0, 4.0],
        ]);
        let inverse = m.inverse();
        assert!(inverse.is_none());
    }

    #[test]
    fn mat_vec_mul() {
        let m = SquareMatrix::<3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let v = Vector3f::new(2.0, 1.0, 3.0);
        let result = mul_mat_vec::<3, Vector3f, Vector3f>(&m, &v);
        assert_eq!(Vector3f::new(13.0, 31.0, 49.0), result);
    }

    #[test]
    fn mat_mul() {
        let m1 = SquareMatrix::<3>::new([[1.0, 2.0, -1.0], [3.0, 2.0, 0.0], [-4.0, 0.0, 2.0]]);
        let m2 = SquareMatrix::<3>::new([[3.0, 4.0, 2.0], [0.0, 1.0, 0.0], [-2.0, 0.0, 1.0]]);
        let result = m1 * m2;
        let expected =
            SquareMatrix::<3>::new([[5.0, 6.0, 1.0], [9.0, 14.0, 6.0], [-16.0, -16.0, -6.0]]);
        assert_eq!(expected, result);

        let m1 = SquareMatrix::<4>::new([
            [5.0, 7.0, 9.0, 10.0],
            [2.0, 3.0, 3.0, 8.0],
            [8.0, 10.0, 2.0, 3.0],
            [3.0, 3.0, 4.0, 8.0],
        ]);
        let m2 = SquareMatrix::<4>::new([
            [3.0, 10.0, 12.0, 18.0],
            [12.0, 1.0, 4.0, 9.0],
            [9.0, 10.0, 12.0, 2.0],
            [3.0, 12.0, 4.0, 10.0],
        ]);
        let result = m1 * m2;
        let expected = SquareMatrix::<4>::new([
            [210.0, 267.0, 236.0, 271.0],
            [93.0, 149.0, 104.0, 149.0],
            [171.0, 146.0, 172.0, 268.0],
            [105.0, 169.0, 128.0, 169.0],
        ]);
        assert_eq!(expected, result);
    }
}
