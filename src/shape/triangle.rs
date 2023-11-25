use std::{mem::size_of, sync::Arc};

use crate::{
    bounding_box::Bounds3f,
    direction_cone::DirectionCone,
    float::gamma,
    interaction::{Interaction, SurfaceInteraction},
    math::{difference_of_products_float_vec, DifferenceOfProducts},
    ray::Ray,
    shape::ShapeIntersection,
    vecmath::{
        normal::Normal3, point::Point3fi, spherical::spherical_triangle_area, vector::Vector3,
        Length, Normal3f, Normalize, Point2f, Point3f, Tuple3, Vector2f, Vector3f,
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

    // TODO If/when we move to GPU, this will likely need to change to accomodate; see PBRT.
    fn interaction_from_intersection(
        &self,
        ti: TriangleIntersection,
        time: Float,
        wo: Vector3f,
    ) -> SurfaceInteraction {
        let (p0, p1, p2) = self.get_points();
        let v = self.mesh.vertex_indices[3 * self.tri_index as usize];

        // Compute triangle partial derivatives.
        // Compute deltas and matrix determinant for triangle partial derivatives.
        // Get triangle texture coordinates in uv array.
        let uv = if self.mesh.uv.is_empty() {
            [Point2f::ZERO, Point2f::X, Point2f::ONE]
        } else {
            [self.mesh.uv[v], self.mesh.uv[v + 1], self.mesh.uv[v + 2]]
        };
        let duv02 = uv[0] - uv[2];
        let duv12 = uv[1] - uv[2];
        let dp02 = p0 - p2;
        let dp12 = p1 - p2;
        let determinant = Float::difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);

        let degenerate_uv = determinant.abs() < 1e-9;
        let (dpdu, dpdv) = if !degenerate_uv {
            // Compute triangle dpdu and dpdv via matrix inversion
            let inv_det = 1.0 / determinant;
            let dpdu = difference_of_products_float_vec(duv12[1], dp02, duv02[1], dp12) * inv_det;
            let dpdv = difference_of_products_float_vec(duv02[0], dp12, duv12[0], dp02) * inv_det;
            (dpdu, dpdv)
        } else {
            (Vector3f::ZERO, Vector3f::ZERO)
        };
        // Handle degenerate triaangle uv parameterization or partial derivatives
        let (dpdu, dpdv) = if degenerate_uv || dpdu.cross(&dpdv).length_squared() == 0.0 {
            let mut ng = (p2 - p0).cross(&(p1 - p0));
            if ng.length_squared() == 0.0 {
                // TODO This could be better if we made Vector3 more generic s.t. we could just say Vector3<f64>
                // and call its cross implementation.
                // Retry with double precision
                let v1 = p2 - p0;
                let v2 = p1 - p0;
                ng = Vector3f {
                    x: f64::difference_of_products(
                        v1.y() as f64,
                        v2.z() as f64,
                        v1.z() as f64,
                        v2.y() as f64,
                    ) as Float,
                    y: f64::difference_of_products(
                        v1.z() as f64,
                        v2.x() as f64,
                        v1.x() as f64,
                        v2.z() as f64,
                    ) as Float,
                    z: f64::difference_of_products(
                        v1.x() as f64,
                        v2.y() as f64,
                        v1.y() as f64,
                        v2.x() as f64,
                    ) as Float,
                };
                debug_assert_ne!(ng.length_squared(), 0.0);
            }
            ng.normalize().coordinate_system()
        } else {
            (dpdu, dpdv)
        };

        // Interpolate uv parametric coordiantes and hit point.
        let p_hit: Point3f = ti.b0 * p0 + ti.b1 * Vector3f::from(p1) + ti.b2 * Vector3f::from(p2);
        let uv_hit: Point2f =
            ti.b0 * uv[0] + ti.b1 * Vector2f::from(uv[1]) + ti.b2 * Vector2f::from(uv[2]);

        let flip_normal = self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness;

        // Compute error bounds for triangle intersection
        let p_abs_sum = (ti.b0 * p0).abs()
            + Vector3f::from((ti.b1 * p1).abs())
            + Vector3f::from((ti.b2 * p2).abs());
        let p_error = gamma(7) * Vector3f::from(p_abs_sum);

        let mut isect = SurfaceInteraction::new(
            Point3fi::from_value_and_error(p_hit, p_error),
            uv_hit,
            wo,
            dpdu,
            dpdv,
            Normal3f::ZERO,
            Normal3f::ZERO,
            time,
            flip_normal,
        );

        isect.face_index = if self.mesh.face_indices.is_empty() {
            0
        } else {
            self.mesh.face_indices[self.tri_index as usize] as i32
        };

        // Set final surface normal and shading geometry for triangle
        // Override seruface normal in isect for triangle
        isect.shading.n = dp02.cross(&dp12).normalize().into();
        isect.interaction.n = isect.shading.n;
        if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            isect.shading.n = -isect.shading.n;
            isect.interaction.n = -isect.interaction.n;
        }

        if !self.mesh.n.is_empty() || !self.mesh.s.is_empty() {
            // Initialize triangle shading geometry
            let ns = if self.mesh.n.is_empty() {
                isect.interaction.n
            } else {
                let n = ti.b0 * self.mesh.n[v]
                    + ti.b1 * self.mesh.n[v + 1]
                    + ti.b2 * self.mesh.n[v + 2];
                if n.length_squared() > 0.0 {
                    n.normalize()
                } else {
                    isect.interaction.n
                }
            };

            // Compute shading tangent ss for triangle
            let ss = if self.mesh.s.is_empty() {
                isect.dpdu
            } else {
                let s = ti.b0 * self.mesh.s[v]
                    + ti.b1 * self.mesh.s[v + 1]
                    + ti.b2 * self.mesh.s[v + 2];
                if s.length_squared() == 0.0 {
                    isect.dpdu
                } else {
                    s
                }
            };

            // Compute shading bitangent ts for triangle and adjust ss
            let ts = ns.cross(&ss);
            let (ss, ts) = if ts.length_squared() > 0.0 {
                (ts.cross_normal(&ns), ts)
            } else {
                Vector3f::from(ns).coordinate_system()
            };

            // Compute dndu and dndv for triangle shading geometry
            let (dndu, dndv) = if self.mesh.n.is_empty() {
                (Normal3f::ZERO, Normal3f::ZERO)
            } else {
                // Compute deltas for triangle partial derivatives of normal
                let duv02 = uv[0] - uv[2];
                let duv12 = uv[1] - uv[2];

                let determinant =
                    Float::difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);
                let degenerate_uv = determinant.abs() < 1e-9;
                if degenerate_uv {
                    // We can still compute dndu and dndv, with respect to the
                    // same arbitrary coordinate system we use to compute dpdu
                    // and dpdv when this happens. It's important to do this
                    // (rather than giving up) so that ray differentials for
                    // rays reflected from triangles with degenerate
                    // parameterizations are still reasonable.
                    let dn = Vector3f::from(self.mesh.n[v + 2] - self.mesh.n[v])
                        .cross(&Vector3f::from(self.mesh.n[v + 1] - self.mesh.n[v]));
                    if dn.length_squared() == 0.0 {
                        (Normal3f::ZERO, Normal3f::ZERO)
                    } else {
                        let (dnu, dnv) = dn.coordinate_system();
                        (dnu.into(), dnv.into())
                    }
                } else {
                    let inv_det = 1.0 / determinant;
                    let dn1 = self.mesh.n[v] - self.mesh.n[v + 2];
                    let dn2 = self.mesh.n[v + 1] - self.mesh.n[v + 2];
                    (
                        (difference_of_products_float_vec(
                            duv12[1],
                            dn1.into(),
                            duv02[1],
                            dn2.into(),
                        ) * inv_det)
                            .into(),
                        (difference_of_products_float_vec(
                            duv02[0],
                            dn2.into(),
                            duv12[0],
                            dn1.into(),
                        ) * inv_det)
                            .into(),
                    )
                }
            };

            isect.set_shading_geometry(ns, ss, ts, dndu, dndv, true)
        }

        isect
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
        let (p0, p1, p2) = self.get_points();
        let tri_isect = Triangle::intersect_triangle(ray, t_max, p0, p1, p2)?;
        let t_hit = tri_isect.t;
        let intr = self.interaction_from_intersection(tri_isect, ray.time, -ray.d);
        Some(ShapeIntersection { intr, t_hit })
    }

    fn intersect_predicate(&self, ray: &crate::ray::Ray, t_max: Float) -> bool {
        let (p0, p1, p2) = self.get_points();
        let tri_isect = Triangle::intersect_triangle(ray, t_max, p0, p1, p2);
        tri_isect.is_some()
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
    // Barycentric coordinates of the intersection
    b0: Float,
    b1: Float,
    b2: Float,
    /// The t value along the ray the intersection occured
    t: Float,
}