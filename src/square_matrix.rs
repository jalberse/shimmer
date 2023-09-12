use crate::float::Float;

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

mod tests {
    use crate::Float;

    use super::SquareMatrix;

    #[test]
    fn zero() {
        let m = SquareMatrix::<4>::zero();
        for i in 0..m.dim() {
            for j in 0..m.dim() {
                assert_eq!(0.0, m.m[i][j]);
            }
        }
    }

    #[test]
    fn diag() {
        let m = SquareMatrix::<4>::diag([0.0, 1.0, 2.0, 3.0]);
        for i in 0..m.dim() {
            for j in 0..m.dim() {
                if i == j {
                    assert_eq!(i as Float, m.m[i][j]);
                } else {
                    assert_eq!(0.0, m.m[i][j])
                }
            }
        }
    }
}
