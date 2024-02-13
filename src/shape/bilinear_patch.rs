use std::sync::Arc;

use crate::bounding_box::Bounds3f;
use crate::direction_cone::DirectionCone;
use crate::math::lerp;
use crate::vecmath::point::Point3;
use crate::vecmath::spherical::cos_theta;
use crate::vecmath::{Length, Point3f, Vector3f};
use crate::Float;
use crate::vecmath::vector::Vector3;
use crate::vecmath::normalize::Normalize;

use super::mesh::BilinearPatchMesh;
use super::ShapeI;


pub struct BilinearPatch{
    mesh: Arc<BilinearPatchMesh>,
    blp_index: usize,
    area: Float,
}

impl BilinearPatch{
    pub fn new(mesh: Arc<BilinearPatchMesh>, blp_index: usize) -> BilinearPatch{
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&mesh, blp_index);

        let area = if BilinearPatch::is_rectangle(&mesh, blp_index)
        {
            p00.distance(p01) * p00.distance(p10)
        } else {
            // Compute approximate area of bilinear patch
            const NA: usize = 3;
            let mut p = [[Point3f::ZERO; NA + 1]; NA + 1];
            for i in 0..NA + 1
            {
                let u = i as Float / NA as Float;
                for j in 0..NA + 1
                {
                    let v = j as Float / NA as Float;
                    p[i][j] = lerp(u, lerp(v, Vector3f::from(p00), Vector3f::from(p01)), lerp(v, Vector3f::from(p10), Vector3f::from(p11))).into();
                }
            }
            let mut area = 0.0;
            for i in 0..NA
            {
                for j in 0..NA 
                {
                    area += 0.5 * (p[i + 1][j + 1] - p[i][j]).cross(p[i + 1][j] - p[i][j + 1]).length();
                }
            }
            area
        };

        BilinearPatch{
            mesh,
            blp_index,
            area,
        }
    }

    fn get_points(mesh: &Arc<BilinearPatchMesh>, blp_index: usize) -> (Point3f, Point3f, Point3f, Point3f)
    {
        let v = mesh.vertex_indices[blp_index * 4];
        let p00 = mesh.p[v];
        let v = mesh.vertex_indices[blp_index * 4 + 1];
        let p10 = mesh.p[v];
        let v = mesh.vertex_indices[blp_index * 4 + 2];
        let p01 = mesh.p[v];
        let v = mesh.vertex_indices[blp_index * 4 + 3];
        let p11 = mesh.p[v];
        (p00, p10, p01, p11)
    }
    fn get_vertex_indices(mesh: &Arc<BilinearPatchMesh>, blp_index: usize) -> (usize, usize, usize, usize)
    {
        (mesh.vertex_indices[blp_index * 4],
        mesh.vertex_indices[blp_index * 4 + 1],
        mesh.vertex_indices[blp_index * 4 + 2],
        mesh.vertex_indices[blp_index * 4 + 3])
    }

    fn is_rectangle(mesh: &Arc<BilinearPatchMesh>, blp_index: usize) -> bool 
    {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(mesh, blp_index);

        if p00 == p01 || p01 == p11 || p11 == p10 || p10 == p00
        {
            return false;
        }


        // Check if bilinear patch vertices are coplanar
        let n = (p10 - p00).cross(p01 - p00).normalize();

        if (p11 - p00).normalize().abs_dot(n) > 1e-5
        {
            return false;
        }

        // CHeck if planar vertices form a rectangle
        let p_center = (p00 + p01.into() + p10.into() + p11.into()) * 0.25;
        let d2: [Float; 4] = [
            (p00 - p_center).length_squared(),
            (p01 - p_center).length_squared(),
            (p10 - p_center).length_squared(),
            (p11 - p_center).length_squared(),
        ];
        for i in 1..4
        {
            if Float::abs(d2[i] - d2[0]) / d2[0] > 1e-4
            {
                return false;
            }
        }
        true
    }
}

impl ShapeI for BilinearPatch
{
    fn bounds(&self) -> crate::bounding_box::Bounds3f {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        Bounds3f::new(p00, p01).union(&Bounds3f::new(p10, p11))
    }

    fn normal_bounds(&self) -> crate::direction_cone::DirectionCone {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let(v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(&self.mesh, self.blp_index);

        // If the patch is a triangle, return bounds for single surface normal.
        if p00 == p10 || p10 == p11 || p11 == p01 || p01 == p00
        {
            let dpdu = lerp::<Vector3f>(0.5, p10.into(), p11.into()) - lerp::<Vector3f>(0.5, p00.into(), p01.into());
            let dpdv = lerp::<Vector3f>(0.5, p01.into(), p11.into()) - lerp::<Vector3f>(0.5, p00.into(), p10.into());
            let n = dpdu.cross(dpdv).normalize();
            let n = if !self.mesh.n.is_empty()
            {
                let n00 = self.mesh.n[v0];
                let n10 = self.mesh.n[v1];
                let n01 = self.mesh.n[v2];
                let n11 = self.mesh.n[v3];
                let ns = (n00 + n10 + n01 + n11) * 0.25;
                n.face_forward_n(ns)
            } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
                -n
            } else {
                n
            };
            return DirectionCone::from_angle(n.into());
        }

        // Compute bilinear patch normal n00 at (0, 0).
        let n00 = (p10 - p00).cross(p01 - p00).normalize();
        let n00 = if !self.mesh.n.is_empty()
        {
            n00.face_forward_n(self.mesh.n[v0])
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            -n00
        } else {
            n00
        };

        // Compute bilinear patch normals at n10, n01, n11
        let n10 = (p11 - p10).cross(p00 - p10).normalize();
        let n01 = (p00 - p01).cross(p11 - p01).normalize();
        let n11 = (p01 - p11).cross(p10 - p11).normalize();
        let (n10, n01, n11) = if !self.mesh.n.is_empty()
        {
            (
                n10.face_forward_n(self.mesh.n[v1]),
                n01.face_forward_n(self.mesh.n[v2]),
                n11.face_forward_n(self.mesh.n[v3]),
            )
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            (-n10, -n01, -n11)
        } else {
            (n10, n01, n11)
        };

        // Compute average normal and return normal bounds for patch;
        // this is not an exact bounds, but is practically sufficient.
        let n = (n00 + n10 + n01 + n11).normalize();
        let cos_theta = [n.dot(n00), n.dot(n10), n.dot(n01), n.dot(n11)].into_iter().min_by(|a, b| a.partial_cmp(b).expect("Unexpected NaN")).unwrap();
        DirectionCone::new(n.into(), Float::clamp(cos_theta, -1.0, 1.0))
    }

    fn intersect(&self, ray: &crate::ray::Ray, t_max: Float) -> Option<super::ShapeIntersection> {
        todo!()
    }

    fn intersect_predicate(&self, ray: &crate::ray::Ray, t_max: Float) -> bool {
        todo!()
    }

    fn area(&self) -> Float {
        self.area
    }

    fn sample(&self, u: crate::vecmath::Point2f) -> Option<super::ShapeSample> {
        todo!()
    }

    fn pdf(&self, interaction: &crate::interaction::Interaction) -> Float {
        todo!()
    }

    fn sample_with_context(&self, ctx: &super::ShapeSampleContext, u: crate::vecmath::Point2f) -> Option<super::ShapeSample> {
        todo!()
    }

    fn pdf_with_context(&self, ctx: &super::ShapeSampleContext, wi: Vector3f) -> Float {
        todo!()
    }
}