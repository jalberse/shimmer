use crate::{
    square_matrix::{Invertible, SquareMatrix},
    Float,
};

// TODO Note that since transforms are relatively memory hungry, it can be good to
// de-duplicate them in use. C.2.3 InternCache. We likely want something like that,
// but for now we will abstain.

#[derive(PartialEq, PartialOrd, Copy, Clone)]
pub struct Transform {
    m: SquareMatrix<4>,
    // Inverse of m
    m_inv: SquareMatrix<4>,
}

impl Transform {
    /// The caller must ensure that m_inv is the correct inverse of m.
    /// This is the most common ctor, since if we can provide a simple
    /// inverse, we can save calculating the general form.
    pub fn new(m: SquareMatrix<4>, m_inv: SquareMatrix<4>) -> Transform {
        Self { m, m_inv }
    }

    /// Creates a transform from m, computing the inverse using the general form.
    /// new() should be preferred to avoid expensive inverse calculation when possible.
    pub fn new_calc_inverse(m: SquareMatrix<4>) -> Transform {
        let m_inv = match m.inverse() {
            Some(inverse) => inverse,
            None => {
                let mut m = SquareMatrix::<4>::zero();
                for i in 0..4 {
                    for j in 0..4 {
                        m.m[i][j] = Float::NAN;
                    }
                }
                m
            }
        };

        // PAPERDOC - It might be idiomatic in Rust to store the inverse in Option here,
        // rather than populate with NaN as PBRTv4 does. This would ensure the programmer
        // must check for NaN, rather than let it poison anything.
        // However, then each transform would take an additional byte of memory to store
        // the discriminator. Maybe this is a case of premature optimization, but I'll elect
        // to also store NaN in the inverse if the matrix was non-invertible instead.
        // This should hopefully be quite self-contained and any NaN poisoning would be
        // extremely obvious, so we'll accept it.
        Transform { m, m_inv }
    }

    pub fn from_2d(m: [[Float; 4]; 4]) -> Transform {
        Self::new_calc_inverse(SquareMatrix::new(m))
    }

    pub fn get_matrix(&self) -> &SquareMatrix<4> {
        &self.m
    }

    pub fn get_inverse_matrix(&self) -> &SquareMatrix<4> {
        &self.m_inv
    }

    pub fn inverse(&self) -> Transform {
        Transform {
            m: self.m,
            m_inv: self.m_inv,
        }
    }

    pub fn transpose(&self) -> Transform {
        Transform {
            m: self.m.transpose(),
            m_inv: self.m_inv.transpose(),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.m.is_identity()
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            m: Default::default(),
            m_inv: Default::default(),
        }
    }
}
