use itertools::Itertools;
use ply_rs::{parser::Parser, ply};

use crate::{
    file, transform::Transform, vecmath::{Normal3f, Point2f, Point3f, Tuple2, Tuple3, Vector3f}
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


#[derive(Debug)]
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
    pub face_indices: Vec<usize>,
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
            face_indices
        }
    }
}

/// TriQuadMesh is not used for rendering, but as an intermediate mesh format
/// when reading e.g. PLY files. It is converted to a TriangleMesh and BilinearPatchMesh
/// for rendering purposes.
pub struct TriQuadMesh
{
    pub p: Vec<Point3f>,
    pub n: Vec<Normal3f>,
    pub uv: Vec<Point2f>,
    pub face_indices: Vec<i32>,
    pub tri_indices: Vec<i32>,
    pub quad_indices: Vec<i32>,
}

impl TriQuadMesh
{
    pub fn read_ply(filename: &str) -> TriQuadMesh
    {
        // TODO Handle if this is a .ply.gz file?
        // https://stackoverflow.com/questions/47048037/how-to-iterate-stream-a-gzip-file-containing-a-single-csv
        // Would just need to make a gzdecoder and pass it to BufReader instead of f, if we end in .gz.

        let f = std::fs::File::open(filename).expect("Unable to read PLY file");
        let mut f = std::io::BufReader::new(f);

        let vertex_parser = Parser::<PlyVertex>::new();
        let face_parser = Parser::<PlyFace>::new();

        // First, consume the header;
        // we could also use 'face_parser', the configuration is the parser's only state.
        // The reading position only depends on f.
        let header = vertex_parser.read_header(&mut f).unwrap();

        let mut vertex_list = Vec::new();
        let mut face_list = Vec::new();
        for (_, element) in &header.elements
        {
            match element.name.as_ref()
            {
                "vertex" => vertex_list = vertex_parser.read_payload_for_element(&mut f, element, &header).unwrap(),
                "face" => face_list = face_parser.read_payload_for_element(&mut f, element, &header).unwrap(),
                _ => panic!("Unexpected element: {}", element.name),
            }
        }

        // Transform from the PLY representation to the representation our mesh uses.
        let (p, n, uv): (Vec<Point3f>, Vec<Normal3f>, Vec<Point2f>) = vertex_list.into_iter().map(|v| {
            let p = Point3f::new(v.x, v.y, v.z);
            let n = Normal3f::new(v.nx, v.ny, v.nz);
            let uv = Point2f::new(v.u, v.v);
            (p, n, uv)
        }).multiunzip();

        // TODO could reserve these with the proper length.
        let mut tri_indices = Vec::new();
        let mut quad_indices = Vec::new();
        let mut face_indices = Vec::new();
        for f in &face_list {
            // We expect either vertex_index or face_index to be non-empty.
            assert!(f.face_indices.len() > 0 && f.vertex_indices.len() == 0
                || f.face_indices.len() == 0 && f.vertex_indices.len() > 0);

            if f.vertex_indices.len() == 3
            {
                debug_assert!(f.face_indices.len() == 0);
                tri_indices.push(f.vertex_indices[0]);
                tri_indices.push(f.vertex_indices[1]);
                tri_indices.push(f.vertex_indices[2]);
            } else if f.vertex_indices.len() == 4
            {
                debug_assert!(f.face_indices.len() == 0);
                // Note that order is modified as we represent as a BilienarPatch.
                quad_indices.push(f.vertex_indices[0]);
                quad_indices.push(f.vertex_indices[1]);
                quad_indices.push(f.vertex_indices[3]);
                quad_indices.push(f.vertex_indices[2]);
            } else if f.face_indices.len() == 0
            {
                panic!("Only tris and quads are supported");
            }

            if f.face_indices.len() > 0
            {
                debug_assert!(f.vertex_indices.len() == 0);
                face_indices.push(f.face_indices[0]);
            }
        };

        for idx in &tri_indices
        {
            assert!(*idx >= 0);
            assert!(idx < &(p.len() as i32));
        }
        for idx in &quad_indices
        {
            assert!(*idx >= 0);
            assert!(idx < &(p.len() as i32));
        }

        TriQuadMesh
        {
            p,
            n,
            uv,
            face_indices,
            tri_indices,
            quad_indices,
        }
    }
}

#[derive(Debug)]
struct PlyVertex
{
    x: f32,
    y: f32,
    z: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    u: f32,
    v: f32,
}

impl ply::PropertyAccess for PlyVertex {
    fn new() -> Self {
        PlyVertex {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            u: 0.0,
            v: 0.0,
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.x = v,
            ("y", ply::Property::Float(v)) => self.y = v,
            ("z", ply::Property::Float(v)) => self.z = v,
            ("nx", ply::Property::Float(v)) => self.nx = v,
            ("ny", ply::Property::Float(v)) => self.ny = v,
            ("nz", ply::Property::Float(v)) => self.nz = v,
            ("u", ply::Property::Float(v)) => self.u = v,
            ("v", ply::Property::Float(v)) => self.v = v,
            ("s", ply::Property::Float(v)) => self.u = v,
            ("t", ply::Property::Float(v)) => self.v = v,
            ("texture_u", ply::Property::Float(v)) => self.u = v,
            ("texture_v", ply::Property::Float(v)) => self.v = v,
            ("texture_s", ply::Property::Float(v)) => self.u = v,
            ("texture_t", ply::Property::Float(v)) => self.v = v,
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }
}

struct PlyFace
{
    vertex_indices: Vec<i32>,
    face_indices: Vec<i32>,
}

impl ply::PropertyAccess for PlyFace {
    fn new() -> Self {
        PlyFace {
            vertex_indices: Vec::new(),
            face_indices: Vec::new(),
        }
    }
    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("vertex_indices", ply::Property::ListInt(vec)) => self.vertex_indices = vec,
            ("vertex_index", ply::Property::ListInt(vec)) => self.vertex_indices = vec,
            ("face_indices", ply::Property::ListInt(vec)) => self.face_indices = vec,
            (k, _) => panic!("Face: Unexpected key/value combination: key: {}", k),
        }
    }
}

#[cfg(test)]
mod tests
{
    use std::path::PathBuf;

    #[test]
    fn basic_ply_read()
    {
        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push("test_files/cube.ply");
        let filename = d.to_str().unwrap();
        let mesh = super::TriQuadMesh::read_ply(filename);

        assert!(mesh.p.len() == 8);
        // This mesh is 6 quads, so our tri indices are empty
        assert!(mesh.tri_indices.is_empty());
        assert!(mesh.quad_indices.len() == 6 * 4);
    }
}