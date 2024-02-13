pub mod mesh;
pub mod shape;
pub mod sphere;
pub mod triangle;
pub mod bilinear_patch;

pub use mesh::TriangleMesh;
pub use shape::{Shape, ShapeI, ShapeIntersection, ShapeSample, ShapeSampleContext};
pub use triangle::Triangle;
pub use bilinear_patch::BilinearPatch;