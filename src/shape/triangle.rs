use std::sync::Arc;

use crate::{
    bounding_box::Bounds3f,
    direction_cone::DirectionCone,
    vecmath::{
        spherical::spherical_triangle_area, vector::Vector3, Length, Normal3f, Normalize, Point3f,
    },
    Float,
};

use super::{ShapeI, TriangleMesh};

///
pub struct Triangle {
    // TODO Consider following PBRT and only storing an offset into a vector of meshes;
    // that could save on space. But maintaining that "global" list of meshes
    // is a bit of a design problem, so we'll go with a simple approach here instead.
    mesh: Arc<TriangleMesh>,
    tri_index: i32,
}

impl Triangle {
    const MIN_SPHERICAL_SAMPLE_AREA: Float = 3e-4;
    const MAX_SPHERICAL_SAMPLE_AREA: Float = 6.22;

    pub fn new(mesh: Arc<TriangleMesh>, tri_index: i32) -> Triangle {
        Triangle { mesh, tri_index }
    }

    pub fn get_mesh(&self) -> &Arc<TriangleMesh> {
        &self.mesh
    }

    fn get_points(&self) -> (Point3f, Point3f, Point3f) {
        let v = self.mesh.vertex_indices[3 * self.tri_index as usize];
        let p0 = self.mesh.p[v];
        let p1 = self.mesh.p[v + 1];
        let p2 = self.mesh.p[v + 2];
        (p0, p1, p2)
    }

    /// Calculates the solid angle that the triangle subtends from p.
    pub fn solid_angle(&self, p: Point3f) -> Float {
        let (p0, p1, p2) = self.get_points();
        spherical_triangle_area(
            (p0 - p).normalize(),
            (p1 - p).normalize(),
            (p2 - p).normalize(),
        )
    }
}

impl ShapeI for Triangle {
    fn bounds(&self) -> Bounds3f {
        let (p0, p1, p2) = self.get_points();
        Bounds3f::new(p0, p1).union_point(&p2)
    }

    fn normal_bounds(&self) -> crate::direction_cone::DirectionCone {
        let v = self.mesh.vertex_indices[3 * self.tri_index as usize];
        let (p0, p1, p2) = self.get_points();
        let n = (p1 - p0).cross(&(p2 - p0)).normalize();
        // Ensure correct orientation of geometric normal for normal bounds
        let n = if !self.mesh.n.is_empty() {
            let ns = self.mesh.n[v] + self.mesh.n[v + 1] + self.mesh.n[v + 2];
            n.face_forward_n(&ns)
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            -n
        } else {
            n
        };
        DirectionCone::from_angle(n.into())
    }

    fn intersect(&self, ray: &crate::ray::Ray, t_max: Float) -> Option<super::ShapeIntersection> {
        todo!()
    }

    fn intersect_predicate(&self, ray: &crate::ray::Ray, t_max: Float) -> bool {
        todo!()
    }

    fn area(&self) -> Float {
        let (p0, p1, p2) = self.get_points();
        0.5 * (p1 - p0).cross(&(p2 - p0)).length()
    }

    fn sample(&self, u: crate::vecmath::Point2f) -> Option<super::ShapeSample> {
        todo!()
    }

    fn pdf(&self, interaction: &crate::interaction::Interaction) -> Float {
        todo!()
    }

    fn sample_with_context(
        &self,
        ctx: &super::ShapeSampleContext,
        u: crate::vecmath::Point2f,
    ) -> Option<super::ShapeSample> {
        todo!()
    }

    fn pdf_with_context(
        &self,
        ctx: &super::ShapeSampleContext,
        wi: crate::vecmath::Vector3f,
    ) -> Float {
        todo!()
    }
}
