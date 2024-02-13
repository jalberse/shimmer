use clap::parser::Indices;
use itertools::Itertools;

use crate::{
    transform::Transform,
    vecmath::{Normal3f, Point2f, Point3f, Vector3f},
};

#[derive(Debug, Clone)]
pub struct TriangleMesh {
    pub n_triangles: usize,
    pub n_vertices: usize,
    pub vertex_indices: Vec<usize>,
    pub p: Vec<Point3f>,
    pub n: Vec<Normal3f>,
    pub s: Vec<Vector3f>,
    pub uv: Vec<Point2f>,
    pub face_indices: Vec<usize>,
    pub reverse_orientation: bool,
    pub transform_swaps_handedness: bool,
}

impl TriangleMesh {
    pub fn new(
        render_from_object: &Transform,
        reverse_orientation: bool,
        indices: Vec<usize>,
        p: Vec<Point3f>,
        s: Vec<Vector3f>,
        n: Vec<Normal3f>,
        uv: Vec<Point2f>,
        face_indices: Vec<usize>,
    ) -> TriangleMesh {
        let n_triangles = indices.len() / 3;
        let n_vertices = p.len();
        debug_assert!(indices.len() % 3 == 0);

        let vertex_indices = indices;

        // Transform mesh vertices to rendering space an initialize self.p.
        // We transform to render space to avoid many transforms on rays during rendering;
        // other Shape implementations like Spheres will stay in local space.
        // Which is better depends on the specific intersection routine.
        let p = p
            .iter()
            .map(|pt| render_from_object.apply(pt))
            .collect_vec();

        if !uv.is_empty() {
            debug_assert_eq!(n_vertices, uv.len());
        }

        let n = if !n.is_empty() {
            debug_assert_eq!(n_vertices, n.len());
            n.iter()
                .map(|nn| {
                    let nn = render_from_object.apply(nn);
                    if reverse_orientation {
                        -nn
                    } else {
                        nn
                    }
                })
                .collect_vec()
        } else {
            n
        };

        let s = if !s.is_empty() {
            debug_assert_eq!(n_vertices, s.len());
            s.iter()
                .map(|ss| render_from_object.apply(ss))
                .collect_vec()
        } else {
            s
        };

        if !face_indices.is_empty() {
            debug_assert_eq!(n_triangles, face_indices.len());
        }

        TriangleMesh {
            n_triangles,
            n_vertices,
            vertex_indices,
            p,
            n,
            s,
            uv,
            face_indices,
            reverse_orientation,
            transform_swaps_handedness: render_from_object.swaps_handedness(),
        }
    }
}


pub struct BilinearPatchMesh
{
    pub reverse_orientation: bool,
    pub transform_swaps_handedness: bool,
    pub n_patches: usize,
    pub n_vertices: usize,
    pub vertex_indices: Vec<usize>,
    pub p: Vec<Point3f>,
    pub n: Vec<Normal3f>,
    pub uv: Vec<Point2f>,
}

impl BilinearPatchMesh{
    pub fn new(
        render_from_object: &Transform,
        reverse_orientation: bool,
        vertex_indices: Vec<usize>,
        p: Vec<Point3f>,
        mut n: Vec<Normal3f>,
        uv: Vec<Point2f>,
        face_indices: Vec<usize>,
        // TODO imageDist
    ) -> BilinearPatchMesh
    {
        assert_eq!(vertex_indices.len() % 4, 0);

        let n_vertices = p.len();
        let n_patches = vertex_indices.len() / 4;
        let transform_swaps_handedness = render_from_object.swaps_handedness();
        
        // TODO We'd like to use a buffercache for the vertex indices to avoid repeats.

        // Transform mesh vertices to rendering space
        let p = p
            .iter()
            .map(|pt| render_from_object.apply(pt))
            .collect_vec();
        
        // Copy UV and n vertex data, if present
        if !uv.is_empty()
        {
            assert_eq!(n_vertices, uv.len());
            // TODO Use cache
        }

        if !n.is_empty()
        {
            assert_eq!(n_vertices, n.len());
            for nn in n.iter_mut()
            {
                *nn = render_from_object.apply(nn);
                if reverse_orientation
                {
                    *nn = -(*nn);
                }
            }
            // TODO Use cache
        }

        if !face_indices.is_empty()
        {
            assert_eq!(n_patches, face_indices.len());
            // TODO Use cache
        }

        BilinearPatchMesh
        {
            reverse_orientation,
            transform_swaps_handedness,
            n_patches,
            n_vertices,
            vertex_indices,
            p,
            n,
            uv,
        }
    }
}