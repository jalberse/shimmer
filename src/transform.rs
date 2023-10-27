use auto_ops::impl_op_ex;

use crate::{
    bounding_box::Bounds3f,
    float::gamma,
    frame::Frame,
    ray::{AuxiliaryRays, Ray, RayDifferential},
    square_matrix::{Determinant, Invertible, SquareMatrix},
    vecmath::{
        point::Point3fi,
        vector::{Vector3, Vector3fi},
        Length, Normal3f, Normalize, Point3f, Tuple3, Vector3f,
    },
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
        // must check for non-invertible, rather than let it poison anything.
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

    pub fn from_frame(frame: &Frame) -> Transform {
        let m = SquareMatrix::<4>::new([
            [frame.x.x, frame.x.y, frame.x.z, 0.0],
            [frame.y.x, frame.y.y, frame.y.z, 0.0],
            [frame.z.x, frame.z.y, frame.z.z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        Transform::new_calc_inverse(m)
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

    pub fn orthographic(z_near: Float, z_far: Float) -> Transform {
        Transform::scale(1.0, 1.0, 1.0 / (z_far - z_near))
            * Transform::translate(Vector3f::new(0.0, 0.0, -z_near))
    }

    pub fn has_scale(&self) -> bool {
        self.has_scale_tolerance(1e-3)
    }

    pub fn has_scale_tolerance(&self, tolerance: Float) -> bool {
        let la2 = self.apply(&Vector3f::new(1.0, 0.0, 0.0)).length_squared();
        let lb2 = self.apply(&Vector3f::new(0.0, 1.0, 0.0)).length_squared();
        let lc2 = self.apply(&Vector3f::new(0.0, 0.0, 1.0)).length_squared();

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

    fn rotate_helper(sin_theta: Float, cos_theta: Float, axis: &Vector3f) -> Transform {
        let a = axis.normalize();
        let mut m = SquareMatrix::<4>::zero();
        // Coompute the rotation of each basis vector in turn.
        m.m[0][0] = a.x * a.x + (1.0 - a.x * a.x) + cos_theta;
        m.m[0][1] = a.x * a.y * (1.0 * cos_theta) - a.z * sin_theta;
        m.m[0][2] = a.x * a.z * (1.0 - cos_theta) + a.y * sin_theta;
        m.m[0][3] = 0.0;

        m.m[1][0] = a.x * a.y * (1.0 - cos_theta) + a.z * sin_theta;
        m.m[1][1] = a.y * a.y + (1.0 - a.y * a.y) * cos_theta;
        m.m[1][2] = a.y * a.z * (1.0 - cos_theta) - a.x * sin_theta;
        m.m[1][3] = 0.0;

        m.m[2][0] = a.x * a.z * (1.0 - cos_theta) - a.y * sin_theta;
        m.m[2][1] = a.y * a.z * (1.0 - cos_theta) + a.x * sin_theta;
        m.m[2][2] = a.z * a.z + (1.0 - a.z * a.z) * cos_theta;
        m.m[2][3] = 0.0;

        Transform {
            m,
            m_inv: m.transpose(),
        }
    }

    pub fn rotate(theta: Float, axis: &Vector3f) -> Transform {
        Transform::rotate_helper(Float::sin(theta), Float::cos(theta), axis)
    }

    /// Gets the rotation matrix that would transform from to to.
    /// both from and to should be normalized.
    pub fn rotate_from_to(from: &Vector3f, to: &Vector3f) -> Transform {
        // Compute intermediate vector for vector reflection
        // PAPERDOC - Example of where Rust being expression-based
        // allows for easy const correctness where PBRTv4 does not allow const.
        // In C++, a common pattern to maintain const correctness here is to define and call a lambda inline,
        // which is overcomplicated syntax.
        let ref1: Vector3f = if from.x.abs() < 0.72 && to.x < 0.72 {
            Vector3f::X
        } else if from.y < 0.72 && to.y < 0.72 {
            Vector3f::Y
        } else {
            Vector3f::Z
        };

        let u = ref1 - from;
        let v = ref1 - to;

        let mut r = SquareMatrix::<4>::zero();
        for i in 0..3 {
            for j in 0..3 {
                let kronecker = if i == j { 1.0 } else { 0.0 };
                r.m[i][j] =
                    kronecker - 2.0 / u.dot(&u) * u[i] * u[j] - 2.0 / v.dot(&v) * v[i] * v[j]
                        + 4.0 * u.dot(&v) / (u.dot(&u) * v.dot(&v)) * v[i] * u[j];
            }
        }

        Transform {
            m: r,
            m_inv: r.transpose(),
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

    pub fn look_at(pos: &Point3f, look_at: &Point3f, up: &Vector3f) -> Transform {
        let mut world_from_camera = SquareMatrix::<4>::zero();

        world_from_camera.m[0][3] = pos.x;
        world_from_camera.m[1][3] = pos.y;
        world_from_camera.m[2][3] = pos.z;
        world_from_camera.m[3][3] = 1.0;

        let dir = (look_at - pos).normalize();
        let right = up.normalize().cross(&dir).normalize();
        let new_up = dir.cross(&right);

        world_from_camera.m[0][0] = right.x;
        world_from_camera.m[1][0] = right.y;
        world_from_camera.m[2][0] = right.z;
        world_from_camera.m[3][0] = 0.0;

        world_from_camera.m[0][1] = new_up.x;
        world_from_camera.m[1][1] = new_up.y;
        world_from_camera.m[2][1] = new_up.z;
        world_from_camera.m[3][1] = 0.0;

        world_from_camera.m[0][2] = dir.x;
        world_from_camera.m[1][2] = dir.y;
        world_from_camera.m[2][2] = dir.z;
        world_from_camera.m[3][2] = 0.0;

        let camera_from_world = world_from_camera.inverse().expect("Uninvertible look_at!");

        Transform {
            m: camera_from_world,
            m_inv: world_from_camera,
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

    pub fn swaps_handedness(&self) -> bool {
        // Create a 3x3 for cheaper determinant calculation
        let s = SquareMatrix::<3>::new([
            [self.m[0][0], self.m[0][1], self.m[0][2]],
            [self.m[1][0], self.m[1][1], self.m[1][2]],
            [self.m[2][0], self.m[2][1], self.m[2][2]],
        ]);
        s.determinant() < 0.0
    }

    pub fn apply<T: Transformable>(&self, val: &T) -> T {
        val.apply(&self)
    }

    pub fn apply_inv<T: InverseTransformable>(&self, val: &T) -> T {
        val.apply_inverse(&self)
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

// Allow composition of transformations!
impl_op_ex!(*|t1: &Transform, t2: &Transform| -> Transform {
    Transform {
        m: t1.m * t2.m,
        m_inv: t2.m_inv * t1.m_inv,
    }
});

pub trait Transformable {
    fn apply(&self, transform: &Transform) -> Self;
}

impl Transformable for Point3f {
    fn apply(&self, transform: &Transform) -> Self {
        apply_point_helper(&transform.m, self)
    }
}

impl Transformable for Vector3f {
    fn apply(&self, transform: &Transform) -> Self {
        apply_vector_helper(&transform.m, self)
    }
}

impl Transformable for Normal3f {
    fn apply(&self, transform: &Transform) -> Self {
        // Note that we pass m_inv. This is intentional; normals are
        // transformed by the inverse.
        apply_normal_helper(&transform.m_inv, self)
    }
}

impl Transformable for Point3fi {
    fn apply(&self, transform: &Transform) -> Self {
        let x: Float = self.x().into();
        let y: Float = self.y().into();
        let z: Float = self.z().into();
        // Compute transformed coordinates
        let xp: Float = (transform.m[0][0] * x + transform.m[0][1] * y)
            + (transform.m[0][2] * z + transform.m[0][3]);
        let yp: Float = (transform.m[1][0] * x + transform.m[1][1] * y)
            + (transform.m[1][2] * z + transform.m[1][3]);
        let zp: Float = (transform.m[2][0] * x + transform.m[2][1] * y)
            + (transform.m[2][2] * z + transform.m[2][3]);
        let wp: Float = (transform.m[3][0] * x + transform.m[3][1] * y)
            + (transform.m[3][2] * z + transform.m[3][3]);

        // Compute absolute error for transformed point
        let p_error: Vector3f = if self.is_exact() {
            // Compute error for transformed exact _p_
            let err_x = Float::abs(transform.m[0][0] * x)
                + Float::abs(transform.m[0][1] * y)
                + Float::abs(transform.m[0][2] * z)
                + Float::abs(transform.m[0][3]);
            let err_y = Float::abs(transform.m[1][0] * x)
                + Float::abs(transform.m[1][1] * y)
                + Float::abs(transform.m[1][2] * z)
                + Float::abs(transform.m[1][3]);
            let err_z = Float::abs(transform.m[2][0] * x)
                + Float::abs(transform.m[2][1] * y)
                + Float::abs(transform.m[2][2] * z)
                + Float::abs(transform.m[2][3]);
            Vector3f::new(err_x, err_y, err_z)
        } else {
            // Compute error for transformed approximate _p_
            let p_in_error = self.error();
            let err_x = (gamma(3) + 1.0)
                * (Float::abs(transform.m[0][0]) * p_in_error.x
                    + Float::abs(transform.m[0][1]) * p_in_error.y
                    + Float::abs(transform.m[0][2]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(transform.m[0][0] * x)
                        + Float::abs(transform.m[0][1] * y)
                        + Float::abs(transform.m[0][2] * z)
                        + Float::abs(transform.m[0][3]));
            let err_y = (gamma(3) + 1.0)
                * (Float::abs(transform.m[1][0]) * p_in_error.x
                    + Float::abs(transform.m[1][1]) * p_in_error.y
                    + Float::abs(transform.m[1][2]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(transform.m[1][0] * x)
                        + Float::abs(transform.m[1][1] * y)
                        + Float::abs(transform.m[1][2] * z)
                        + Float::abs(transform.m[1][3]));
            let err_z = (gamma(3) + 1.0)
                * (Float::abs(transform.m[2][0]) * p_in_error.x
                    + Float::abs(transform.m[2][1]) * p_in_error.y
                    + Float::abs(transform.m[2][2]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(transform.m[2][0] * x)
                        + Float::abs(transform.m[2][1] * y)
                        + Float::abs(transform.m[2][2] * z)
                        + Float::abs(transform.m[2][3]));
            Vector3f::new(err_x, err_y, err_z)
        };
        if wp == 1.0 {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_error)
        } else {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_error) / wp.into()
        }
    }
}

impl Transformable for Vector3fi {
    fn apply(&self, transform: &Transform) -> Self {
        let x: Float = self.x().into();
        let y: Float = self.y().into();
        let z: Float = self.z().into();
        let v_out_err = if self.is_exact() {
            let x_err = gamma(3)
                * (Float::abs(transform.m[0][0] * x)
                    + Float::abs(transform.m[0][1] * y)
                    + Float::abs(transform.m[0][2] * z));
            let y_err = gamma(3)
                * (Float::abs(transform.m[1][0] * x)
                    + Float::abs(transform.m[1][1] * y)
                    + Float::abs(transform.m[1][2] * z));
            let z_err = gamma(3)
                * (Float::abs(transform.m[2][0] * x)
                    + Float::abs(transform.m[2][1] * y)
                    + Float::abs(transform.m[2][2] * z));
            Vector3f::new(x_err, y_err, z_err)
        } else {
            let v_in_error = self.error();
            let x_err = (gamma(3) + 1.0)
                * (Float::abs(transform.m[0][0]) * v_in_error.x
                    + Float::abs(transform.m[0][1]) * v_in_error.y
                    + Float::abs(transform.m[0][2]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(transform.m[0][0] * x)
                        + Float::abs(transform.m[0][1] * y)
                        + Float::abs(transform.m[0][2] * z));
            let y_err = (gamma(3) + 1.0)
                * (Float::abs(transform.m[1][0]) * v_in_error.x
                    + Float::abs(transform.m[1][1]) * v_in_error.y
                    + Float::abs(transform.m[1][2]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(transform.m[1][0] * x)
                        + Float::abs(transform.m[1][1] * y)
                        + Float::abs(transform.m[1][2] * z));
            let z_err = (gamma(3) + 1.0)
                * (Float::abs(transform.m[2][0]) * v_in_error.x
                    + Float::abs(transform.m[2][1]) * v_in_error.y
                    + Float::abs(transform.m[2][2]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(transform.m[2][0] * x)
                        + Float::abs(transform.m[2][1] * y)
                        + Float::abs(transform.m[2][2] * z));
            Vector3f::new(x_err, y_err, z_err)
        };

        let xp: Float = transform.m[0][0] * x + transform.m[0][1] * y + transform.m[0][2] * z;
        let yp: Float = transform.m[1][0] * x + transform.m[1][1] * y + transform.m[1][2] * z;
        let zp: Float = transform.m[2][0] * x + transform.m[2][1] * y + transform.m[2][2] * z;

        Vector3fi::from_value_and_error(Vector3f::new(xp, yp, zp), v_out_err)
    }
}

impl Transformable for Ray {
    fn apply(&self, transform: &Transform) -> Self {
        let o: Point3fi = transform.apply(&self.o).into();
        let d: Vector3fi = transform.apply(&self.d).into();
        // Offset ray origin to edge of error bounds and compute t_max
        let length_squared = d.length_squared();
        let o: Point3fi = if length_squared > 0.0 {
            let dt = d.abs().dot(&o.error().into()) / length_squared;
            o + d * dt
        } else {
            o
        };
        Ray::new_with_time(o.into(), d.into(), self.time, self.medium)
    }
}

impl Transformable for RayDifferential {
    // TODO note we may also wanta versiont hat calculates the new t_max, or add that to this one
    fn apply(&self, transform: &Transform) -> Self {
        // Get the transformed base ray
        let tr: Ray = transform.apply(&self.ray);
        // Get the transformed aux rays, if any
        let auxiliary: Option<AuxiliaryRays> = if let Some(aux) = &self.auxiliary {
            let rx_origin = transform.apply(&aux.rx_origin);
            let rx_direction = transform.apply(&aux.rx_direction);
            let ry_origin = transform.apply(&aux.ry_origin);
            let ry_direction = transform.apply(&aux.ry_direction);
            Some(AuxiliaryRays::new(
                rx_origin,
                rx_direction,
                ry_origin,
                ry_direction,
            ))
        } else {
            None
        };
        RayDifferential { ray: tr, auxiliary }
    }
}

impl Transformable for Bounds3f {
    fn apply(&self, transform: &Transform) -> Self {
        // TODO this could be made more efficient.
        let mut out = Bounds3f::new(
            transform.apply(&self.corner(0)),
            transform.apply(&self.corner(1)),
        );

        for i in 2..8 {
            out = out.union_point(&transform.apply(&self.corner(i)));
        }

        out
    }
}

pub trait InverseTransformable {
    fn apply_inverse(&self, transform: &Transform) -> Self;
}

impl InverseTransformable for Point3f {
    fn apply_inverse(&self, transform: &Transform) -> Self {
        apply_point_helper(&transform.m_inv, self)
    }
}

impl InverseTransformable for Vector3f {
    fn apply_inverse(&self, transform: &Transform) -> Self {
        apply_vector_helper(&transform.m_inv, self)
    }
}

impl InverseTransformable for Normal3f {
    fn apply_inverse(&self, transform: &Transform) -> Self {
        // See PBRTv4 page 131 - we haven't passed the wrong matrix!
        // Normals must be transformed by the inverse transform of the transformation matrix.
        apply_normal_helper(&transform.m, self)
    }
}

impl InverseTransformable for Point3fi {
    fn apply_inverse(&self, transform: &Transform) -> Self {
        let x: Float = self.x().into();
        let y: Float = self.y().into();
        let z: Float = self.z().into();
        // Compute transformed coordinates from point _pt_
        let xp: Float = (transform.m_inv[0][0] * x + transform.m_inv[0][1] * y)
            + (transform.m_inv[0][2] * z + transform.m_inv[0][3]);
        let yp: Float = (transform.m_inv[1][0] * x + transform.m_inv[1][1] * y)
            + (transform.m_inv[1][2] * z + transform.m_inv[1][3]);
        let zp: Float = (transform.m_inv[2][0] * x + transform.m_inv[2][1] * y)
            + (transform.m_inv[2][2] * z + transform.m_inv[2][3]);
        let wp: Float = (transform.m_inv[3][0] * x + transform.m_inv[3][1] * y)
            + (transform.m_inv[3][2] * z + transform.m_inv[3][3]);

        // Compute absolute error for transformed point
        let p_out_error = if self.is_exact() {
            let x_err = gamma(3)
                * (Float::abs(transform.m_inv[0][0] * x)
                    + Float::abs(transform.m_inv[0][1] * y)
                    + Float::abs(transform.m_inv[0][2] * z));
            let y_err = gamma(3)
                * (Float::abs(transform.m_inv[1][0] * x)
                    + Float::abs(transform.m_inv[1][1] * y)
                    + Float::abs(transform.m_inv[1][2] * z));
            let z_err = gamma(3)
                * (Float::abs(transform.m_inv[2][0] * x)
                    + Float::abs(transform.m_inv[2][1] * y)
                    + Float::abs(transform.m_inv[2][2] * z));
            Vector3f::new(x_err, y_err, z_err)
        } else {
            let p_in_err = self.error();
            let x_err = (gamma(3) + 1.0)
                * (Float::abs(transform.m_inv[0][0]) * p_in_err.x
                    + Float::abs(transform.m_inv[0][1]) * p_in_err.y
                    + Float::abs(transform.m_inv[0][2]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(transform.m_inv[0][0] * x)
                        + Float::abs(transform.m_inv[0][1] * y)
                        + Float::abs(transform.m_inv[0][2] * z)
                        + Float::abs(transform.m_inv[0][3]));
            let y_err = (gamma(3) + 1.0)
                * (Float::abs(transform.m_inv[1][0]) * p_in_err.x
                    + Float::abs(transform.m_inv[1][1]) * p_in_err.y
                    + Float::abs(transform.m_inv[1][2]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(transform.m_inv[1][0] * x)
                        + Float::abs(transform.m_inv[1][1] * y)
                        + Float::abs(transform.m_inv[1][2] * z)
                        + Float::abs(transform.m_inv[1][3]));
            let z_err = (gamma(3) + 1.0)
                * (Float::abs(transform.m_inv[2][0]) * p_in_err.x
                    + Float::abs(transform.m_inv[2][1]) * p_in_err.y
                    + Float::abs(transform.m_inv[2][2]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(transform.m_inv[2][0] * x)
                        + Float::abs(transform.m_inv[2][1] * y)
                        + Float::abs(transform.m_inv[2][2] * z)
                        + Float::abs(transform.m_inv[2][3]));
            Vector3f::new(x_err, y_err, z_err)
        };

        if wp == 1.0 {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_out_error)
        } else {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_out_error) / wp.into()
        }
    }
}

impl InverseTransformable for Ray {
    // TODO we likely want to include a t_max computation in here.
    fn apply_inverse(&self, transform: &Transform) -> Self {
        let o: Point3fi = Point3fi::from(self.o).apply_inverse(transform);
        let d: Vector3f = self.d.apply_inverse(transform);
        // Offset ray origin to edge of error bounds
        // TODO And compute t_max
        let length_squared = d.length_squared();
        let o = if length_squared > 0.0 {
            let o_error = Vector3f::new(
                o.x().width() / 2.0,
                o.y().width() / 2.0,
                o.z().width() / 2.0,
            );
            let dt = d.abs().dot(&o_error) / length_squared;
            o + (d * dt).into()
        } else {
            o
        };
        Ray::new_with_time(Point3f::from(o), d, self.time, self.medium)
    }
}

impl InverseTransformable for RayDifferential {
    fn apply_inverse(&self, transform: &Transform) -> Self {
        // Get the transformed base ray
        let tr: Ray = transform.apply_inv(&self.ray);
        // Get the transformed aux rays, if any
        let auxiliary: Option<AuxiliaryRays> = if let Some(aux) = &self.auxiliary {
            let rx_origin = transform.apply_inv(&aux.rx_origin);
            let rx_direction = transform.apply_inv(&aux.rx_direction);
            let ry_origin = transform.apply_inv(&aux.ry_origin);
            let ry_direction = transform.apply_inv(&aux.ry_direction);
            Some(AuxiliaryRays::new(
                rx_origin,
                rx_direction,
                ry_origin,
                ry_direction,
            ))
        } else {
            None
        };
        RayDifferential { ray: tr, auxiliary }
    }
}

/// Helper function to share transform and inverse transform implementation.
fn apply_point_helper(m: &SquareMatrix<4>, p: &Point3f) -> Point3f {
    let xp = m[0][0] * p.x() + m[0][1] * p.y() + m[0][2] * p.z() + m[0][3];
    let yp = m[1][0] * p.x() + m[1][1] * p.y() + m[1][2] * p.z() + m[1][3];
    let zp = m[2][0] * p.x() + m[2][1] * p.y() + m[2][2] * p.z() + m[2][3];
    let wp = m[3][0] * p.x() + m[3][1] * p.y() + m[3][2] * p.z() + m[3][3];
    if wp == 1.0 {
        Point3f::new(xp, yp, zp)
    } else {
        debug_assert!(wp != 0.0);
        Point3f::new(xp, yp, zp) / wp
    }
}

/// Helper function to share transform and inverse transform implementation.
fn apply_vector_helper(m: &SquareMatrix<4>, v: &Vector3f) -> Vector3f {
    Vector3f::new(
        m[0][0] * v.x() + m[0][1] * v.y() + m[0][2] * v.z(),
        m[1][0] * v.x() + m[1][1] * v.y() + m[1][2] * v.z(),
        m[2][0] * v.x() + m[2][1] * v.y() + m[2][2] * v.z(),
    )
}

/// Helper function to share transform and inverse transform implementation.
fn apply_normal_helper(m: &SquareMatrix<4>, n: &Normal3f) -> Normal3f {
    // Notice indices are different to get transpose (compare to Vector transform)
    Normal3f::new(
        m[0][0] * n.x() + m[1][0] * n.y() + m[2][0] * n.z(),
        m[0][1] * n.x() + m[1][1] * n.y() + m[2][1] * n.z(),
        m[0][2] * n.x() + m[1][2] * n.y() + m[2][2] * n.z(),
    )
}

#[cfg(test)]
mod tests {
    use crate::{
        bounding_box::Bounds3f,
        vecmath::{Normal3f, Point3f, Tuple3, Vector3f},
    };

    use super::Transform;

    #[test]
    fn translate_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply(&p);
        assert_eq!(Point3f::new(11.0, 22.0, 43.0), new);
    }

    #[test]
    fn translate_inverse_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_inv(&p);
        assert_eq!(Point3f::new(-9.0, -18.0, -37.0), new);
    }

    #[test]
    fn translate_vector() {
        let v = Vector3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Vector3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_vector_inv() {
        let v = Vector3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_inv(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Vector3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_normal() {
        let v = Normal3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        // Note this is applying the inverse transpose still! But...
        let new = translate.apply(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_normal_inv() {
        let v = Normal3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_inv(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn scale_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply(&p);
        assert_eq!(Point3f::new(2.0, 6.0, 12.0), scaled);
        let back_again = scale.apply_inv(&scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn scale_vector() {
        let p = Vector3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply(&p);
        assert_eq!(Vector3f::new(2.0, 6.0, 12.0), scaled);
        let back_again = scale.apply_inv(&scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn scale_normal() {
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply(&p);
        // Note how this differs from vectors - we must transform by the inverse transpose!
        assert_eq!(Normal3f::new(0.5, 0.6666667, 0.75), scaled);
        let back_again = scale.apply_inv(&scaled);
        assert_eq!(p, back_again);

        // Again, a bit more simply
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 2.0, 2.0);
        let scaled = scale.apply(&p);
        // Note how this differs from vectors - we must transform by the inverse transpose!
        assert_eq!(Normal3f::new(0.5, 1.0, 1.5), scaled);
        let back_again = scale.apply_inv(&scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn apply_bb_transform() {
        let bounds = Bounds3f::new(Point3f::ZERO, Point3f::ONE);
        let translate = Transform::translate(Vector3f::ONE);
        let translated = translate.apply(&bounds);
        assert_eq!(Bounds3f::new(Point3f::ONE, Point3f::ONE * 2.0), translated);
    }

    #[test]
    fn transform_composition() {
        let t1 = Transform::translate(Vector3f::ONE);
        let t2 = Transform::scale(1.0, 2.0, 3.0);
        // This will scale then translate
        let composed = t1 * t2;
        let p = Point3f::ONE;
        let new = composed.apply(&p);
        assert_eq!(Point3f::new(2.0, 3.0, 4.0), new);
        let reverted = composed.apply_inv(&new);
        assert_eq!(p, reverted);
    }

    #[test]
    fn rotate_from_to() {
        let from = Vector3f::Z;
        let to = Vector3f::Z;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(&from);
        assert_eq!(to, to_new);

        let from = Vector3f::Z;
        let to = Vector3f::X;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(&from);
        assert_eq!(to, to_new);

        let from = Vector3f::Z;
        let to = Vector3f::Y;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(&from);
        assert_eq!(to, to_new);
    }

    // TODO test rotations inverse
}
