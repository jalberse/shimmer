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

    pub fn dim(&self) -> usize {
        N
    }
}

impl<const N: usize> Default for SquareMatrix<N> {
    fn default() -> Self {
        // TODO check that this is correct for identity - does identity include homogeneous coordinate?
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
    use super::SquareMatrix;

    #[test]
    fn zero() {
        let m = SquareMatrix::<4>::zero();
        // TODO use m.dim()
    }
}
