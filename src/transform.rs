use auto_ops::impl_op_ex;

use crate::{
    bounding_box::Bounds3f,
    float::gamma,
    frame::Frame,
    interaction::{Interaction, SurfaceInteraction, SurfaceInteractionShading},
    math::radians,
    ray::{AuxiliaryRays, Ray, RayDifferential},
    square_matrix::{Determinant, Invertible, SquareMatrix},
    vecmath::{
        normal::Normal3,
        point::Point3fi,
        vector::{Vector3, Vector3fi},
        Length, Normal3f, Normalize, Point3f, Tuple3, Vector3f,
    },
    Float,
};

// TODO Actually, this trait can be implemented just for Transform, if we use a generic type T for the trait.
// THen Transform can implement it for various <T>.
// That's probably cleaner...
pub trait TransformI<T> {
    fn apply(&self, val: T) -> T;
}

pub trait TransformRayI<T> {
    /// Applies the transformation, returning the transformed ray
    /// and the new t_max, if provided. The t_max is adjusted to account for
    /// error correction necessary for ray transformations.
    fn apply_ray(&self, val: T, t_max: Option<&mut Float>) -> T;
}

pub trait InverseTransformI<T> {
    fn apply_inverse(&self, val: T) -> T;
}

pub trait InverseTransformRayI<T> {
    /// Applies the inverse transformation, returning the transformed ray
    /// and the new t_max, if provided. The t_max is adjusted to account for
    /// error correction necessary for ray transformations.
    fn apply_ray_inverse(&self, val: T, t_max: Option<&mut Float>) -> T;
}

// TODO Note that since transforms are relatively memory hungry, it can be good to
// de-duplicate them in use. C.2.3 InternCache. We likely want something like that,
// but for now we will abstain.

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
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

        Transform { m, m_inv }
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
        let la2 = self.apply(Vector3f::new(1.0, 0.0, 0.0)).length_squared();
        let lb2 = self.apply(Vector3f::new(0.0, 1.0, 0.0)).length_squared();
        let lc2 = self.apply(Vector3f::new(0.0, 0.0, 1.0)).length_squared();

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
        let mut m = SquareMatrix::<4>::default();
        // Compute the rotation of each basis vector in turn.
        m.m[0][0] = a.x * a.x + (1.0 - a.x * a.x) * cos_theta;
        m.m[0][1] = a.x * a.y * (1.0 - cos_theta) - a.z * sin_theta;
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
        Transform::rotate_helper(Float::sin(radians(theta)), Float::cos(radians(theta)), axis)
    }

    /// Gets the rotation matrix that would transform from to to.
    /// both from and to should be normalized.
    pub fn rotate_from_to(from: &Vector3f, to: &Vector3f) -> Transform {
        // Compute intermediate vector for vector reflection
        let ref1: Vector3f = if from.x.abs() < 0.72 && to.x.abs() < 0.72 {
            Vector3f::X
        } else if from.y.abs() < 0.72 && to.y.abs() < 0.72 {
            Vector3f::Y
        } else {
            Vector3f::Z
        };

        let u = ref1 - from;
        let v = ref1 - to;

        let mut r = SquareMatrix::<4>::identity();
        for i in 0..3 {
            for j in 0..3 {
                let kronecker = if i == j { 1.0 } else { 0.0 };
                r.m[i][j] = kronecker - 2.0 / u.dot(u) * u[i] * u[j] - 2.0 / v.dot(v) * v[i] * v[j]
                    + 4.0 * u.dot(v) / (u.dot(u) * v.dot(v)) * v[i] * u[j];
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
            m: self.m_inv,
            m_inv: self.m,
        }
    }

    pub fn look_at(pos: &Point3f, look_at: &Point3f, up: &Vector3f) -> Transform {
        let mut world_from_camera = SquareMatrix::<4>::default();

        world_from_camera.m[0][3] = pos.x;
        world_from_camera.m[1][3] = pos.y;
        world_from_camera.m[2][3] = pos.z;
        world_from_camera.m[3][3] = 1.0;

        let dir = (look_at - pos).normalize();
        let right = up.normalize().cross(dir).normalize();
        let new_up = dir.cross(right);

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

    pub fn perspective(fov: Float, n: Float, f: Float) -> Transform {
        let persp: SquareMatrix<4> = SquareMatrix {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, f / (f - n), -f * n / (f - n)],
                [0.0, 0.0, 1.0, 0.0],
            ],
        };
        let inv_tan_ang = 1.0 / Float::tan(radians(fov) / 2.0);
        Transform::scale(inv_tan_ang, inv_tan_ang, 1.0) * Transform::new_calc_inverse(persp)
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
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            m: Default::default(),
            m_inv: Default::default(),
        }
    }
}

impl From<[[Float; 4]; 4]> for Transform {
    fn from(value: [[Float; 4]; 4]) -> Self {
        Self::new_calc_inverse(SquareMatrix::new(value))
    }
}

// Allow composition of transformations!
impl_op_ex!(*|t1: &Transform, t2: &Transform| -> Transform {
    Transform {
        m: t1.m * t2.m,
        m_inv: t2.m_inv * t1.m_inv,
    }
});

impl TransformI<Point3f> for Transform
{
    fn apply(&self, val: Point3f) -> Point3f {
        apply_point_helper(&self.m, &val)
    }
}

impl TransformI<Vector3f> for Transform
{
    fn apply(&self, val: Vector3f) -> Vector3f {
        apply_vector_helper(&self.m, &val)
    }
}

impl TransformI<Normal3f> for Transform {
    fn apply(&self, val: Normal3f) -> Normal3f {
        // Note that we pass m_inv. This is intentional; normals are
        // transformed by the inverse.
        apply_normal_helper(&self.m_inv, &val)
    }
}

impl TransformI<Point3fi> for Transform {
    fn apply(&self, val: Point3fi) -> Point3fi {
        let x: Float = val.x().into();
        let y: Float = val.y().into();
        let z: Float = val.z().into();
        // Compute transformed coordinates
        let xp: Float = (self.m[0][0] * x + self.m[0][1] * y)
            + (self.m[0][2] * z + self.m[0][3]);
        let yp: Float = (self.m[1][0] * x + self.m[1][1] * y)
            + (self.m[1][2] * z + self.m[1][3]);
        let zp: Float = (self.m[2][0] * x + self.m[2][1] * y)
            + (self.m[2][2] * z + self.m[2][3]);
        let wp: Float = (self.m[3][0] * x + self.m[3][1] * y)
            + (self.m[3][2] * z + self.m[3][3]);

        // Compute absolute error for transformed point
        let p_error: Vector3f = if val.is_exact() {
            // Compute error for transformed exact _p_
            let err_x = gamma(3)
                * (Float::abs(self.m[0][0] * x)
                    + Float::abs(self.m[0][1] * y)
                    + Float::abs(self.m[0][2] * z)
                    + Float::abs(self.m[0][3]));
            let err_y = gamma(3)
                * (Float::abs(self.m[1][0] * x)
                    + Float::abs(self.m[1][1] * y)
                    + Float::abs(self.m[1][2] * z)
                    + Float::abs(self.m[1][3]));
            let err_z = gamma(3)
                * (Float::abs(self.m[2][0] * x)
                    + Float::abs(self.m[2][1] * y)
                    + Float::abs(self.m[2][2] * z)
                    + Float::abs(self.m[2][3]));
            Vector3f::new(err_x, err_y, err_z)
        } else {
            // Compute error for transformed approximate _p_
            let p_in_error = val.error();
            let err_x = (gamma(3) + 1.0)
                * (Float::abs(self.m[0][0]) * p_in_error.x
                    + Float::abs(self.m[0][1]) * p_in_error.y
                    + Float::abs(self.m[0][2]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[0][0] * x)
                        + Float::abs(self.m[0][1] * y)
                        + Float::abs(self.m[0][2] * z)
                        + Float::abs(self.m[0][3]));
            let err_y = (gamma(3) + 1.0)
                * (Float::abs(self.m[1][0]) * p_in_error.x
                    + Float::abs(self.m[1][1]) * p_in_error.y
                    + Float::abs(self.m[1][2]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[1][0] * x)
                        + Float::abs(self.m[1][1] * y)
                        + Float::abs(self.m[1][2] * z)
                        + Float::abs(self.m[1][3]));
            let err_z = (gamma(3) + 1.0)
                * (Float::abs(self.m[2][0]) * p_in_error.x
                    + Float::abs(self.m[2][1]) * p_in_error.y
                    + Float::abs(self.m[2][2]) * p_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[2][0] * x)
                        + Float::abs(self.m[2][1] * y)
                        + Float::abs(self.m[2][2] * z)
                        + Float::abs(self.m[2][3]));
            Vector3f::new(err_x, err_y, err_z)
        };
        if wp == 1.0 {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_error)
        } else {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_error) / wp.into()
        }
    }
}

impl TransformI<Vector3fi> for Transform {
    fn apply(&self, val: Vector3fi) -> Vector3fi {
        let x: Float = val.x().into();
        let y: Float = val.y().into();
        let z: Float = val.z().into();
        let v_out_err = if val.is_exact() {
            let x_err = gamma(3)
                * (Float::abs(self.m[0][0] * x)
                    + Float::abs(self.m[0][1] * y)
                    + Float::abs(self.m[0][2] * z));
            let y_err = gamma(3)
                * (Float::abs(self.m[1][0] * x)
                    + Float::abs(self.m[1][1] * y)
                    + Float::abs(self.m[1][2] * z));
            let z_err = gamma(3)
                * (Float::abs(self.m[2][0] * x)
                    + Float::abs(self.m[2][1] * y)
                    + Float::abs(self.m[2][2] * z));
            Vector3f::new(x_err, y_err, z_err)
        } else {
            let v_in_error = val.error();
            let x_err = (gamma(3) + 1.0)
                * (Float::abs(self.m[0][0]) * v_in_error.x
                    + Float::abs(self.m[0][1]) * v_in_error.y
                    + Float::abs(self.m[0][2]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[0][0] * x)
                        + Float::abs(self.m[0][1] * y)
                        + Float::abs(self.m[0][2] * z));
            let y_err = (gamma(3) + 1.0)
                * (Float::abs(self.m[1][0]) * v_in_error.x
                    + Float::abs(self.m[1][1]) * v_in_error.y
                    + Float::abs(self.m[1][2]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[1][0] * x)
                        + Float::abs(self.m[1][1] * y)
                        + Float::abs(self.m[1][2] * z));
            let z_err = (gamma(3) + 1.0)
                * (Float::abs(self.m[2][0]) * v_in_error.x
                    + Float::abs(self.m[2][1]) * v_in_error.y
                    + Float::abs(self.m[2][2]) * v_in_error.z)
                + gamma(3)
                    * (Float::abs(self.m[2][0] * x)
                        + Float::abs(self.m[2][1] * y)
                        + Float::abs(self.m[2][2] * z));
            Vector3f::new(x_err, y_err, z_err)
        };

        let xp: Float = self.m[0][0] * x + self.m[0][1] * y + self.m[0][2] * z;
        let yp: Float = self.m[1][0] * x + self.m[1][1] * y + self.m[1][2] * z;
        let zp: Float = self.m[2][0] * x + self.m[2][1] * y + self.m[2][2] * z;

        Vector3fi::from_value_and_error(Vector3f::new(xp, yp, zp), v_out_err)
    }
}

impl TransformRayI<Ray> for Transform {
    fn apply_ray(&self, val: Ray, t_max: Option<&mut Float>) -> Ray {
        let o: Point3fi = self.apply(val.o).into();
        let d: Vector3f = self.apply(val.d);
        // Offset ray origin to edge of error bounds and compute t_max
        let length_squared = d.length_squared();
        let o: Point3fi = if length_squared > 0.0 {
            let dt = d.abs().dot(o.error().into()) / length_squared;
            if let Some(t_max) = t_max {
                *t_max = *t_max - dt;
            }
            o + (d * dt).into()
        } else {
            o
        };
        Ray::new_with_time(o.into(), d.into(), val.time, val.medium)
    }
}

impl TransformRayI<RayDifferential> for Transform {
    fn apply_ray(&self, val: RayDifferential, t_max: Option<&mut Float>) -> RayDifferential {
        // Get the transformed base ray
        let tr = self.apply_ray(val.ray, t_max);
        // Get the transformed aux rays, if any
        let auxiliary: Option<AuxiliaryRays> = if let Some(aux) = &val.auxiliary {
            let rx_origin = self.apply(aux.rx_origin);
            let rx_direction = self.apply(aux.rx_direction);
            let ry_origin = self.apply(aux.ry_origin);
            let ry_direction = self.apply(aux.ry_direction);
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

impl TransformI<Bounds3f> for Transform {
    fn apply(&self, val: Bounds3f) -> Bounds3f {
        // TODO this could be made more efficient.
        let mut out = Bounds3f::new(
            self.apply(val.corner(0)),
            self.apply(val.corner(1)),
        );

        for i in 2..8 {
            out = out.union_point(self.apply(val.corner(i)));
        }

        out
    }
}

impl TransformI<SurfaceInteraction> for Transform {
    fn apply(&self, val: SurfaceInteraction) -> SurfaceInteraction {
        let t = self.inverse();

        let n = t.apply(val.interaction.n).normalize();

        SurfaceInteraction {
            interaction: Interaction {
                pi: self.apply(val.interaction.pi),
                time: val.interaction.time,
                wo: t.apply(val.interaction.wo).normalize(),
                n,
                uv: val.interaction.uv,
            },
            dpdu: t.apply(val.dpdu),
            dpdv: t.apply(val.dpdv),
            dndu: t.apply(val.dndu),
            dndv: t.apply(val.dndv),
            shading: SurfaceInteractionShading {
                n: t.apply(val.shading.n).normalize().face_forward(n),
                dpdu: t.apply(val.shading.dpdu),
                dpdv: t.apply(val.shading.dpdv),
                dndu: t.apply(val.shading.dndu),
                dndv: t.apply(val.shading.dndv),
            },
            face_index: val.face_index,
            material: val.material.clone(),
            area_light: val.area_light.clone(),
            dpdx: t.apply(val.dpdx),
            dpdy: t.apply(val.dpdy),
            dudx: val.dudx,
            dvdx: val.dvdx,
            dudy: val.dudy,
            dvdy: val.dvdy,
        }
    }
}

impl InverseTransformI<Point3f> for Transform {
    fn apply_inverse(&self, val: Point3f) -> Point3f {
        apply_point_helper(&self.m_inv, &val)
    }
}

impl InverseTransformI<Vector3f> for Transform {
    fn apply_inverse(&self, val: Vector3f) -> Vector3f {
        apply_vector_helper(&self.m_inv, &val)
    }
}

impl InverseTransformI<Normal3f> for Transform {
    fn apply_inverse(&self, val: Normal3f) -> Normal3f {
        // See PBRTv4 page 131 - we haven't passed the wrong matrix!
        // Normals must be transformed by the inverse transform of the transformation matrix.
        apply_normal_helper(&self.m, &val)
    }
}

impl InverseTransformI<Point3fi> for Transform {
    fn apply_inverse(&self, val: Point3fi) -> Point3fi {
        let x: Float = val.x().into();
        let y: Float = val.y().into();
        let z: Float = val.z().into();
        // Compute transformed coordinates from point _pt_
        let xp: Float = (self.m_inv[0][0] * x + self.m_inv[0][1] * y)
            + (self.m_inv[0][2] * z + self.m_inv[0][3]);
        let yp: Float = (self.m_inv[1][0] * x + self.m_inv[1][1] * y)
            + (self.m_inv[1][2] * z + self.m_inv[1][3]);
        let zp: Float = (self.m_inv[2][0] * x + self.m_inv[2][1] * y)
            + (self.m_inv[2][2] * z + self.m_inv[2][3]);
        let wp: Float = (self.m_inv[3][0] * x + self.m_inv[3][1] * y)
            + (self.m_inv[3][2] * z + self.m_inv[3][3]);

        // Compute absolute error for transformed point
        let p_out_error = if val.is_exact() {
            let x_err = gamma(3)
                * (Float::abs(self.m_inv[0][0] * x)
                    + Float::abs(self.m_inv[0][1] * y)
                    + Float::abs(self.m_inv[0][2] * z));
            let y_err = gamma(3)
                * (Float::abs(self.m_inv[1][0] * x)
                    + Float::abs(self.m_inv[1][1] * y)
                    + Float::abs(self.m_inv[1][2] * z));
            let z_err = gamma(3)
                * (Float::abs(self.m_inv[2][0] * x)
                    + Float::abs(self.m_inv[2][1] * y)
                    + Float::abs(self.m_inv[2][2] * z));
            Vector3f::new(x_err, y_err, z_err)
        } else {
            let p_in_err = val.error();
            let x_err = (gamma(3) + 1.0)
                * (Float::abs(self.m_inv[0][0]) * p_in_err.x
                    + Float::abs(self.m_inv[0][1]) * p_in_err.y
                    + Float::abs(self.m_inv[0][2]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(self.m_inv[0][0] * x)
                        + Float::abs(self.m_inv[0][1] * y)
                        + Float::abs(self.m_inv[0][2] * z)
                        + Float::abs(self.m_inv[0][3]));
            let y_err = (gamma(3) + 1.0)
                * (Float::abs(self.m_inv[1][0]) * p_in_err.x
                    + Float::abs(self.m_inv[1][1]) * p_in_err.y
                    + Float::abs(self.m_inv[1][2]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(self.m_inv[1][0] * x)
                        + Float::abs(self.m_inv[1][1] * y)
                        + Float::abs(self.m_inv[1][2] * z)
                        + Float::abs(self.m_inv[1][3]));
            let z_err = (gamma(3) + 1.0)
                * (Float::abs(self.m_inv[2][0]) * p_in_err.x
                    + Float::abs(self.m_inv[2][1]) * p_in_err.y
                    + Float::abs(self.m_inv[2][2]) * p_in_err.z)
                + gamma(3)
                    * (Float::abs(self.m_inv[2][0] * x)
                        + Float::abs(self.m_inv[2][1] * y)
                        + Float::abs(self.m_inv[2][2] * z)
                        + Float::abs(self.m_inv[2][3]));
            Vector3f::new(x_err, y_err, z_err)
        };

        if wp == 1.0 {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_out_error)
        } else {
            Point3fi::from_value_and_error(Point3f::new(xp, yp, zp), p_out_error) / wp.into()
        }
    }
}

impl InverseTransformRayI<Ray> for Transform {
    fn apply_ray_inverse(&self, val: Ray, t_max: Option<&mut Float>) -> Ray {
        let o: Point3fi = self.apply_inverse(Point3fi::from(val.o));
        let d: Vector3f = self.apply_inverse(val.d);
        // Offset ray origin to edge of error bounds
        let length_squared = d.length_squared();
        let o = if length_squared > 0.0 {
            let o_error = Vector3f::new(
                o.x().width() / 2.0,
                o.y().width() / 2.0,
                o.z().width() / 2.0,
            );
            let dt = d.abs().dot(o_error) / length_squared;
            if let Some(t_max) = t_max {
                *t_max = *t_max - dt;
            }
            o + (d * dt).into()
        } else {
            o
        };
        Ray::new_with_time(Point3f::from(o), d, val.time, val.medium)
    }
}

impl InverseTransformRayI<RayDifferential> for Transform {
    fn apply_ray_inverse(
        &self,
        val: RayDifferential,
        t_max: Option<&mut Float>,
    ) -> RayDifferential {
        // Get the transformed base ray
        let tr = self.apply_ray_inverse(val.ray, t_max);
        // Get the transformed aux rays, if any
        let auxiliary: Option<AuxiliaryRays> = if let Some(aux) = &val.auxiliary {
            let rx_origin = self.apply_inverse(aux.rx_origin);
            let rx_direction = self.apply_inverse(aux.rx_direction);
            let ry_origin = self.apply_inverse(aux.ry_origin);
            let ry_direction = self.apply_inverse(aux.ry_direction);
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
    // Implicitly converts to the homogeneous column vector [x y z 1]^T.
    // Transforms by multiplying that vector with the transformation matrix.
    let xp = m[0][0] * p.x() + m[0][1] * p.y() + m[0][2] * p.z() + m[0][3];
    let yp = m[1][0] * p.x() + m[1][1] * p.y() + m[1][2] * p.z() + m[1][3];
    let zp = m[2][0] * p.x() + m[2][1] * p.y() + m[2][2] * p.z() + m[2][3];
    let wp = m[3][0] * p.x() + m[3][1] * p.y() + m[3][2] * p.z() + m[3][3];
    // ... and then converts back to the nonhomogeneous point representation by dividing by wp.
    if wp == 1.0 {
        // For efficiency, skips division if the weight is 1.
        Point3f::new(xp, yp, zp)
    } else {
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
    use float_cmp::assert_approx_eq;

    use crate::{
        bounding_box::Bounds3f,
        vecmath::{Normal3f, Normalize, Point3f, Tuple3, Vector3f},
        Float,
    };

    use super::Transform;
    use super::{TransformI, InverseTransformI, TransformRayI, InverseTransformRayI};

    #[test]
    fn translate_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply(p);
        assert_eq!(Point3f::new(11.0, 22.0, 43.0), new);
    }

    #[test]
    fn translate_inverse_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_inverse(p);
        assert_eq!(Point3f::new(-9.0, -18.0, -37.0), new);
    }

    #[test]
    fn translate_vector() {
        let v = Vector3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply(v);
        // Translation does not effect vectors or normals!
        assert_eq!(Vector3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_vector_inv() {
        let v = Vector3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_inverse(v);
        // Translation does not effect vectors or normals!
        assert_eq!(Vector3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_normal() {
        let v = Normal3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        // Note this is applying the inverse transpose still! But...
        let new = translate.apply(v);
        // Translation does not effect vectors or normals!
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_normal_inv() {
        let v = Normal3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_inverse(v);
        // Translation does not effect vectors or normals!
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn scale_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply(p);
        assert_eq!(Point3f::new(2.0, 6.0, 12.0), scaled);
        let back_again = scale.apply_inverse(scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn scale_vector() {
        let p = Vector3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply(p);
        assert_eq!(Vector3f::new(2.0, 6.0, 12.0), scaled);
        let back_again = scale.apply_inverse(scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn scale_normal() {
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply(p);
        // Note how this differs from vectors - we must transform by the inverse transpose!
        assert_eq!(Normal3f::new(0.5, 0.6666667, 0.75), scaled);
        let back_again = scale.apply_inverse(scaled);
        assert_eq!(p, back_again);

        // Again, a bit more simply
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 2.0, 2.0);
        let scaled = scale.apply(p);
        // Note how this differs from vectors - we must transform by the inverse transpose!
        assert_eq!(Normal3f::new(0.5, 1.0, 1.5), scaled);
        let back_again = scale.apply_inverse(scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn apply_bb_transform() {
        let bounds = Bounds3f::new(Point3f::ZERO, Point3f::ONE);
        let translate = Transform::translate(Vector3f::ONE);
        let translated = translate.apply(bounds);
        assert_eq!(Bounds3f::new(Point3f::ONE, Point3f::ONE * 2.0), translated);
    }

    #[test]
    fn transform_composition() {
        let t1 = Transform::translate(Vector3f::ONE);
        let t2 = Transform::scale(1.0, 2.0, 3.0);
        // This will scale then translate
        let composed = t1 * t2;
        let p = Point3f::ONE;
        let new = composed.apply(p);
        assert_eq!(Point3f::new(2.0, 3.0, 4.0), new);
        let reverted = composed.apply_inverse(new);
        assert_eq!(p, reverted);
    }

    #[test]
    fn rotate_from_to() {
        let from = Vector3f::Z;
        let to = Vector3f::Z;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(from);
        assert_eq!(to, to_new);

        let from = Vector3f::Z;
        let to = Vector3f::X;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(from);
        assert_eq!(to, to_new);

        let from = Vector3f::Z;
        let to = Vector3f::Y;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(from);
        assert_eq!(to, to_new);

        // Note that rotate_from_to() expects normalized vectors.
        let from = Vector3f::new(0.1, 0.2, 0.3).normalize();
        let to = Vector3f::new(0.4, 0.5, 0.6).normalize();
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply(from);
        assert_approx_eq!(Float, to.x, to_new.x);
        assert_approx_eq!(Float, to.y, to_new.y);
        assert_approx_eq!(Float, to.z, to_new.z);
    }
}
