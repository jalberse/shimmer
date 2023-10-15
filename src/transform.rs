use auto_ops::impl_op_ex;

use crate::{
    bounding_box::Bounds3f,
    frame::Frame,
    ray::{Ray, RayDifferential},
    square_matrix::{Determinant, Invertible, SquareMatrix},
    vecmath::{vector::Vector3, Length, Normal3f, Normalize, Point3f, Tuple3, Vector3f},
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

    pub fn rotation(theta: Float, axis: &Vector3f) -> Transform {
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

    fn apply_p_helper(m: &SquareMatrix<4>, p: &Point3f) -> Point3f {
        // TODO We may need this to be generic on P: Point3. Until then, let's
        // stick with Point3f as concrete.
        //  Else we need TupleElement: Mul<Float>.
        // In this case, we might switch out i32 in e.g. Point3i for a NewType so that we can impl
        //  Mul<Float> on it legally. ITupleElem could be the struct name or something.
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

    pub fn apply_p(&self, p: &Point3f) -> Point3f {
        Self::apply_p_helper(&self.m, p)
    }

    pub fn apply_p_inv(&self, p: &Point3f) -> Point3f {
        Self::apply_p_helper(&self.m_inv, p)
    }

    fn apply_v_helper(m: &SquareMatrix<4>, v: &Vector3f) -> Vector3f {
        Vector3f::new(
            m[0][0] * v.x() + m[0][1] * v.y() + m[0][2] * v.z(),
            m[1][0] * v.x() + m[1][1] * v.y() + m[1][2] * v.z(),
            m[2][0] * v.x() + m[2][1] * v.y() + m[2][2] * v.z(),
        )
    }

    pub fn apply_v(&self, v: &Vector3f) -> Vector3f {
        Self::apply_v_helper(&self.m, v)
    }

    pub fn apply_v_inv(&self, v: &Vector3f) -> Vector3f {
        Self::apply_v_helper(&self.m_inv, v)
    }

    fn apply_n_helper(m: &SquareMatrix<4>, n: &Normal3f) -> Normal3f {
        // Notice indices are different to get transpose (compare to Vector transform)
        Normal3f::new(
            m[0][0] * n.x() + m[1][0] * n.y() + m[2][0] * n.z(),
            m[0][1] * n.x() + m[1][1] * n.y() + m[2][1] * n.z(),
            m[0][2] * n.x() + m[1][2] * n.y() + m[2][2] * n.z(),
        )
    }

    pub fn apply_n(&self, n: &Normal3f) -> Normal3f {
        // See PBRTv4 page 131 - we haven't passed the wrong matrix!
        // Normals must be transformed by the inverse transform of the transformation matrix.
        Self::apply_n_helper(&self.m_inv, n)
    }

    pub fn apply_n_inv(&self, n: &Normal3f) -> Normal3f {
        // See PBRTv4 page 131 - we haven't passed the wrong matrix!
        // Normals must be transformed by the inverse transform of the transformation matrix.
        Self::apply_n_helper(&self.m, n)
    }

    // TODO ray transforms. Requires Interval, and Point<Interval>
    pub fn apply_r(&self, r: &Ray) -> Ray {
        todo!()
    }

    pub fn apply_r_inv(&self, r: &Ray) -> Ray {
        todo!()
    }

    pub fn apply_rd(&self, r: &RayDifferential) -> RayDifferential {
        todo!()
    }

    pub fn apply_rd_inv(&self, r: &RayDifferential) -> RayDifferential {
        todo!()
    }

    pub fn apply_bb(&self, bb: &Bounds3f) -> Bounds3f {
        // TODO this could be made more efficient.
        let mut out = Bounds3f::new(self.apply_p(&bb.corner(0)), self.apply_p(&bb.corner(1)));

        for i in 2..8 {
            out = out.union_point(&self.apply_p(&bb.corner(i)));
        }

        out
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
        let new = translate.apply_p(&p);
        assert_eq!(Point3f::new(11.0, 22.0, 43.0), new);
    }

    #[test]
    fn translate_inverse_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_p_inv(&p);
        assert_eq!(Point3f::new(-9.0, -18.0, -37.0), new);
    }

    #[test]
    fn translate_vector() {
        let v = Vector3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_v(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Vector3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_vector_inv() {
        let v = Vector3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_v_inv(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Vector3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_normal() {
        let v = Normal3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        // Note this is applying the inverse transpose still! But...
        let new = translate.apply_n(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn translate_normal_inv() {
        let v = Normal3f::new(1.0, 2.0, 3.0);
        let translate = Transform::translate(Vector3f::new(10.0, 20.0, 40.0));
        let new = translate.apply_n_inv(&v);
        // Translation does not effect vectors or normals!
        assert_eq!(Normal3f::new(1.0, 2.0, 3.0), new);
    }

    #[test]
    fn scale_point() {
        let p = Point3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply_p(&p);
        assert_eq!(Point3f::new(2.0, 6.0, 12.0), scaled);
        let back_again = scale.apply_p_inv(&scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn scale_vector() {
        let p = Vector3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply_v(&p);
        assert_eq!(Vector3f::new(2.0, 6.0, 12.0), scaled);
        let back_again = scale.apply_v_inv(&scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn scale_normal() {
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 3.0, 4.0);
        let scaled = scale.apply_n(&p);
        // Note how this differs from vectors - we must transform by the inverse transpose!
        assert_eq!(Normal3f::new(0.5, 0.6666667, 0.75), scaled);
        let back_again = scale.apply_n_inv(&scaled);
        assert_eq!(p, back_again);

        // Again, a bit more simply
        let p = Normal3f::new(1.0, 2.0, 3.0);
        let scale = Transform::scale(2.0, 2.0, 2.0);
        let scaled = scale.apply_n(&p);
        // Note how this differs from vectors - we must transform by the inverse transpose!
        assert_eq!(Normal3f::new(0.5, 1.0, 1.5), scaled);
        let back_again = scale.apply_n_inv(&scaled);
        assert_eq!(p, back_again);
    }

    #[test]
    fn apply_bb_transform() {
        let bounds = Bounds3f::new(Point3f::ZERO, Point3f::ONE);
        let translate = Transform::translate(Vector3f::ONE);
        let translated = translate.apply_bb(&bounds);
        assert_eq!(Bounds3f::new(Point3f::ONE, Point3f::ONE * 2.0), translated);
    }

    #[test]
    fn transform_composition() {
        let t1 = Transform::translate(Vector3f::ONE);
        let t2 = Transform::scale(1.0, 2.0, 3.0);
        // This will scale then translate
        let composed = t1 * t2;
        let p = Point3f::ONE;
        let new = composed.apply_p(&p);
        assert_eq!(Point3f::new(2.0, 3.0, 4.0), new);
        let reverted = composed.apply_p_inv(&new);
        assert_eq!(p, reverted);
    }

    #[test]
    fn rotate_from_to() {
        let from = Vector3f::Z;
        let to = Vector3f::Z;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply_v(&from);
        assert_eq!(to, to_new);

        let from = Vector3f::Z;
        let to = Vector3f::X;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply_v(&from);
        assert_eq!(to, to_new);

        let from = Vector3f::Z;
        let to = Vector3f::Y;
        let r = Transform::rotate_from_to(&from, &to);
        let to_new = r.apply_v(&from);
        assert_eq!(to, to_new);
    }

    // TODO test rotations inverse
}
