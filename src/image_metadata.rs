use crate::{square_matrix::SquareMatrix, transform::Transform};

pub struct ImageMetadata {
    pub camera_from_world: Option<SquareMatrix<4>>,
    pub ndc_from_world: Option<SquareMatrix<4>>,
}
