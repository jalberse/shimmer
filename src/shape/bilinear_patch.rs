use std::sync::Arc;

use crate::Float;

use super::mesh::BilinearPatchMesh;


pub struct BilinearPatch{
    mesh: Arc<BilinearPatchMesh>,
    blp_index: usize,
    area: Float,
}

impl BilinearPatch{
    pub fn new(mesh: Arc<BilinearPatchMesh>, blp_index: usize) -> BilinearPatch{
        let v = mesh.vertex_indices[blp_index * 4];
        let p00 = mesh.p[v];
        let v = mesh.vertex_indices[blp_index * 4 + 1];
        let p10 = mesh.p[v];
        let v = mesh.vertex_indices[blp_index * 4 + 2];
        let p01 = mesh.p[v];
        let v = mesh.vertex_indices[blp_index * 4 + 3];
        let p11 = mesh.p[v];

        // TODO calculate area; would also use IsRectangle.

        BilinearPatch{
            mesh,
            blp_index,
        }
    }
}