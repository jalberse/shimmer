pub mod mesh;
pub mod shape;
pub mod sphere;
pub mod triangle;

pub use mesh::TriangleMesh;
pub use shape::{Shape, ShapeI, ShapeIntersection, ShapeSample, ShapeSampleContext};
pub use triangle::Triangle;
