use crate::{
    square_matrix::{Invertible, SquareMatrix},
    vecmath::{normal::Normal3, point::Point3, vector::Vector3, Length, Tuple3, Vector3f},
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

    pub fn translate(delta: Vector3f) -> Transform {
        let m = SquareMatrix::<4>::new([
            [1.0, 0.0, 0.0, delta.x],
            [0.0, 1.0, 0.0, delta.y],
            [0.0, 0.0, 1.0, delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = SquareMatrix::<4>::new([
            [1.0, 0.0, 0.0, -delta.x],
            [0.0, 1.0, 0.0, -delta.y],
            [0.0, 0.0, 1.0, -delta.z],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        Self::new(m, m_inv)
    }

    pub fn scale(x: Float, y: Float, z: Float) -> Transform {
        let m = SquareMatrix::<4>::new([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let m_inv = SquareMatrix::<4>::new([
            [1.0 / x, 0.0, 0.0, 0.0],
            [0.0, 1.0 / y, 0.0, 0.0],
            [0.0, 0.0, 1.0 / z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        Self::new(m, m_inv)
    }

    pub fn has_scale(&self) -> bool {
        self.has_scale_tolerance(1e-3)
    }

    pub fn has_scale_tolerance(&self, tolerance: Float) -> bool {
        let la2 = self.apply_v(&Vector3f::new(1.0, 0.0, 0.0)).length_squared();
        let lb2 = self.apply_v(&Vector3f::new(0.0, 1.0, 0.0)).length_squared();
        let lc2 = self.apply_v(&Vector3f::new(0.0, 0.0, 1.0)).length_squared();

        la2.abs() - 1.0 > tolerance || lb2.abs() > tolerance || lc2.abs() > tolerance
    }

    /// theta: radians
    pub fn rotate_x(theta: Float) -> Transform {
        let sin_theta = Float::sin(theta);
        let cos_theta = Float::cos(theta);

        let m = SquareMatrix::<4>::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Transform {
            m,
            m_inv: m.transpose(),
        }
    }

    /// theta: radians
    pub fn rotate_y(theta: Float) -> Transform {
        let sin_theta = Float::sin(theta);
        let cos_theta = Float::cos(theta);

        let m = SquareMatrix::<4>::new([
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Transform {
            m,
            m_inv: m.transpose(),
        }
    }

    /// theta: radians
    pub fn rotate_z(theta: Float) -> Transform {
        let sin_theta = Float::sin(theta);
        let cos_theta = Float::cos(theta);

        let m = SquareMatrix::<4>::new([
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);

        Transform {
            m,
            m_inv: m.transpose(),
        }
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

    pub fn apply_p<P: Point3>(&self, p: &P) -> P {
        todo!()
    }

    pub fn apply_v<V: Vector3>(&self, v: &V) -> V {
        todo!()
    }

    pub fn apply_n<N: Normal3>(&self, n: &N) -> N {
        todo!()
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
