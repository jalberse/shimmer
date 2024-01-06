use crate::vecmath::{normal::Normal3, vector::Vector3, Normal3f, Tuple3, Vector3f};

pub struct Frame {
    pub x: Vector3f,
    pub y: Vector3f,
    pub z: Vector3f,
}

impl Frame {
    pub fn new(x: Vector3f, y: Vector3f, z: Vector3f) -> Frame {
        Frame { x, y, z }
    }

    pub fn from_xz(x: Vector3f, z: Vector3f) -> Frame {
        let y = z.cross(&x);
        Frame::new(x, y, z)
    }

    pub fn from_xy(x: Vector3f, y: Vector3f) -> Frame {
        let z = x.cross(&y);
        Frame::new(x, y, z)
    }

    pub fn from_z(z: Vector3f) -> Frame {
        let (x, y) = z.coordinate_system();
        Frame { x, y, z }
    }

    pub fn from_y(y: Vector3f) -> Frame {
        let (x, z) = y.coordinate_system();
        Frame { x, y, z }
    }

    pub fn from_x(x: Vector3f) -> Frame {
        let (y, z) = x.coordinate_system();
        Frame { x, y, z }
    }

    pub fn to_local_v(&self, v: &Vector3f) -> Vector3f {
        Vector3f::new(v.dot(&self.x), v.dot(&self.y), v.dot(&self.z))
    }

    pub fn to_local_n(&self, n: &Normal3f) -> Normal3f {
        Normal3f::new(
            n.dot_vector(&self.x),
            n.dot_vector(&self.y),
            n.dot_vector(&self.z),
        )
    }

    pub fn from_local_v(&self, v: &Vector3f) -> Vector3f {
        v.x * self.x + v.y * self.y + v.z * self.z
    }

    pub fn from_local_n(&self, n: &Normal3f) -> Normal3f {
        n.x * Normal3f::from(self.x) + n.y * Normal3f::from(self.y) + Normal3f::from(n.z * self.z)
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
