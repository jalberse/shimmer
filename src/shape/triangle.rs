use std::{mem::size_of, sync::Arc};

use crate::{
    bounding_box::Bounds3f,
    direction_cone::DirectionCone,
    float::gamma,
    math::DifferenceOfProducts,
    ray::Ray,
    vecmath::{
        spherical::spherical_triangle_area, vector::Vector3, Length, Normalize, Point3f, Tuple3,
        Vector3f,
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

    /// Intersect a triangle with a ray; useful to avoid needing a full TriangleMesh
    /// to test for intersections.
    pub fn intersect_triangle(
        ray: &Ray,
        t_max: Float,
        p0: Point3f,
        p1: Point3f,
        p2: Point3f,
    ) -> Option<TriangleIntersection> {
        // Return no intersection if the triangle is degenerate
        if (p2 - p0).cross(&(p1 - p0)).length_squared() == 0.0 {
            return None;
        }

        // We first transform the ray and points s.t. the ray's origin is at (0, 0, 0)
        // and pointing along the positive Z axis; this simplifies some operations,
        // since we can consider the xy projection of the ray and vertices.
        // This also makes it possible to have a watertight algorithm.

        // Transform triangle vertices to ray coordinate space

        // Translate vertices based on ray origin
        let p0t = p0 - Vector3f::from(ray.o);
        let p1t = p1 - Vector3f::from(ray.o);
        let p2t = p2 - Vector3f::from(ray.o);

        // Permute components of triangle vertices and ray direction
        let kz = ray.d.abs().max_component_index();
        let mut kx = kz + 1;
        if kx == 3 {
            kx = 0;
        }
        let mut ky = kx + 1;
        if ky == 3 {
            ky = 0;
        }
        let d = ray.d.permute((kx, ky, kz));
        let mut p0t = p0t.permute((kx, ky, kz));
        let mut p1t = p1t.permute((kx, ky, kz));
        let mut p2t = p2t.permute((kx, ky, kz));

        // Apply shear transformation to translated vertex positions
        let sx = -d.x / d.z;
        let sy = -d.y / d.z;
        let sz = 1.0 / d.z;
        p0t.x += sx * p0t.z;
        p0t.y += sy * p0t.z;
        p1t.x += sx * p1t.z;
        p1t.y += sy * p1t.z;
        p2t.x += sx * p2t.z;
        p2t.y += sy * p2t.z;

        // TODO Notice that the permutation and shear calculations only rely on the
        // ray and not the triangle; it may be worth caching those with the ray.

        // Compute edge function coefficients e0, e1, and e2
        let mut e0 = Float::difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
        let mut e1 = Float::difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
        let mut e2 = Float::difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

        // Fall back to double-precision test at triangle edges
        if size_of::<Float>() == size_of::<f32>() && (e0 == 0.0 || e1 == 0.0 || e2 == 0.0) {
            let p2txp1ty = p2t.x as f64 * p1t.y as f64;
            let p2typ1tx = p2t.y as f64 * p1t.x as f64;
            e0 = (p2typ1tx - p2txp1ty) as f32;
            let p0txp2ty = p0t.x as f64 * p2t.y as f64;
            let p0typ2tx = p0t.y as f64 * p2t.x as f64;
            e1 = (p0typ2tx - p0txp2ty) as f32;
            let p1txp0ty = p1t.x as f64 * p0t.y as f64;
            let p1typ0tx = p1t.y as f64 * p0t.x as f64;
            e2 = (p1typ0tx - p1txp0ty) as f32;
        }

        // Perform triangle edge and determinant tests
        if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
            return None;
        }
        let det = e0 + e1 + e2;
        if det == 0.0 {
            return None;
        }

        // Compute scaled hit distance to triangle and test against ray t range.
        p0t.z *= sz;
        p1t.z *= sz;
        p2t.z *= sz;
        let t_scaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
        if det < 0.0 && (t_scaled >= 0.0 || t_scaled < t_max * det) {
            return None;
        } else if det > 0.0 && (t_scaled <= 0.0 || t_scaled > t_max * det) {
            return None;
        }

        // Compute barycentric coordinates and t value for triangle intersection
        let inv_det = 1.0 / det;
        let b0 = e0 * inv_det;
        let b1 = e1 * inv_det;
        let b2 = e2 * inv_det;
        let t = t_scaled * inv_det;
        debug_assert!(!t.is_nan());

        // Ensure that the computed traingle t is conservatively greater than zero.
        // Compute delta_z term for triangel t error bounds.
        let max_zt = Vector3f::new(p0t.z, p1t.z, p2t.z)
            .abs()
            .max_component_value();
        let delta_z = gamma(3) * max_zt;

        // Compute delta_x and delta_y terms for triangle t error bounds
        let max_xt = Vector3f::new(p0t.x, p1t.x, p2t.x)
            .abs()
            .max_component_value();
        let max_yt = Vector3f::new(p0t.y, p1t.y, p2t.y)
            .abs()
            .max_component_value();
        let delta_x = gamma(5) * (max_xt + max_zt);
        let delta_y = gamma(5) * (max_yt + max_zt);

        // Compute delta_e term for triangle t error bounds
        let delta_e = 2.0 * (gamma(2) * max_xt * max_yt + delta_y * max_xt + delta_x * max_yt);

        // Compute delta_t term for triangle t error bounds and check t
        let max_e = Vector3f::new(e0, e1, e2).abs().max_component_value();
        let delta_t = 3.0
            * (gamma(3) * max_e * max_zt + delta_e * max_zt + delta_z * max_e)
            * Float::abs(inv_det);
        if t <= delta_t {
            return None;
        }

        Some(TriangleIntersection { b0, b1, b2, t })
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

pub struct TriangleIntersection {
    b0: Float,
    b1: Float,
    b2: Float,
    t: Float,
}
