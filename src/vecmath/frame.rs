use crate::{
    vecmath::{vector::Vector3, Length},
    Float,
};

use super::{normal::Normal3, Normal3f, Vector3f};

struct Frame {
    x: Vector3f,
    y: Vector3f,
    z: Vector3f,
}

impl Frame {
    pub fn new(x: Vector3f, y: Vector3f, z: Vector3f) -> Frame {
        debug_assert!(Float::abs(x.length_squared() - 1.0) < 1e-4);
        debug_assert!(Float::abs(y.length_squared() - 1.0) < 1e-4);
        debug_assert!(Float::abs(z.length_squared() - 1.0) < 1e-4);
        debug_assert!(x.abs_dot(y) < 1e-4);
        debug_assert!(y.abs_dot(z) < 1e-4);
        debug_assert!(z.abs_dot(x) < 1e-4);
        Frame { x, y, z }
    }

    pub fn from_xz(x: Vector3f, z: Vector3f) -> Frame {
        Frame {
            x,
            y: z.cross(x),
            z,
        }
    }

    pub fn from_xy(x: Vector3f, y: Vector3f) -> Frame {
        Frame {
            x,
            y,
            z: x.cross(y),
        }
    }

    pub fn from_z<T: Into<Vector3f> + Copy>(z: T) -> Frame {
        let (x, y) = Vector3f::coordinate_system(&z.into());
        Frame { x, y, z: z.into() }
    }

    pub fn from_x<T: Into<Vector3f> + Copy>(x: T) -> Frame {
        let (y, z) = Vector3f::coordinate_system(&x.into());
        Frame { x: x.into(), y, z }
    }

    pub fn from_y<T: Into<Vector3f> + Copy>(y: T) -> Frame {
        let (x, z) = Vector3f::coordinate_system(&y.into());
        Frame { x, y: y.into(), z }
    }

    pub fn to_local(&self, v: Vector3f) -> Vector3f {
        Vector3f {
            x: v.dot(self.x),
            y: v.dot(self.y),
            z: v.dot(self.z),
        }
    }

    // TODO API here can improve by making generic T and combining with vector impl - but
    // requires improving dot() in a similar manner. Traits as a better alternative to overloading.
    pub fn to_local_n(&self, n: Normal3f) -> Normal3f {
        Normal3f {
            x: n.dot_vector(self.x),
            y: n.dot_vector(self.y),
            z: n.dot_vector(self.z),
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self {
            x: Vector3f::X,
            y: Vector3f::Y,
            z: Vector3f::Z,
        }
    }
}
