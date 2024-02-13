use std::sync::Arc;

use crate::math::lerp;
use crate::vecmath::point::Point3;
use crate::vecmath::{Length, Point3f, Vector3f};
use crate::Float;
use crate::vecmath::vector::Vector3;
use crate::vecmath::normalize::Normalize;

use super::mesh::BilinearPatchMesh;


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