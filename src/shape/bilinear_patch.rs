use std::sync::Arc;

use crate::bounding_box::Bounds3f;
use crate::direction_cone::DirectionCone;
use crate::float::gamma;
use crate::interaction::{Interaction, SurfaceInteraction};
use crate::math::{lerp, quadratic};
use crate::ray::Ray;
use crate::sampling::{bilinear_pdf, sample_bilinear};
use crate::shape::ShapeSample;
use crate::square_matrix::{Determinant, SquareMatrix};
use crate::transform::Transform;
use crate::vecmath::normal::Normal3;
use crate::vecmath::point::{Point3, Point3fi};
use crate::vecmath::{Length, Normal3f, Point2f, Point3f, Tuple2, Tuple3, Vector2f, Vector3f};
use crate::Float;
use crate::vecmath::vector::Vector3;
use crate::vecmath::normalize::Normalize;
use crate::math::DifferenceOfProducts;

use super::mesh::BilinearPatchMesh;
use super::{Shape, ShapeI, ShapeIntersection};


pub struct BilinearPatch{
    mesh: Arc<BilinearPatchMesh>,
    blp_index: usize,
    area: Float,
}

pub struct BilinearIntersection{
    pub uv: Point2f,
    pub t: Float,
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

    fn intersect_blp(ray: &Ray, t_max: Float, p00: Point3f, p10: Point3f, p01: Point3f, p11: Point3f) -> Option<BilinearIntersection>
    {
        // Find quadratic coefficients for distance from ray to u iso-lines
        let a = (p10 - p00).cross(p01 - p11).dot(ray.d);
        let c = (p00 - ray.o).cross(ray.d).dot(p01 - p00);
        let b = (p10 - ray.o).cross(ray.d).dot(p11 - p10) - (a + c);

        // Solve quadratic for bilinear patch u intersection
        let us = quadratic(a, b, c);
        if us.is_none()
        {
            return None;
        }
        let (u1, u2) = us.unwrap();

        // Find epsilon eps to ensure that candidate t is greater than zero
        let eps = gamma(10) * (
            ray.o.abs().max_component_value() + ray.d.abs().max_component_value() +
            p00.abs().max_component_value() + p10.abs().max_component_value() +
            p01.abs().max_component_value() + p11.abs().max_component_value()
        );

        // Compute v and t for the first u intersection
        let mut t = t_max;
        let mut u = 0.0;
        let mut v = 0.0;
        if 0.0 <= u1 && u1 <= 1.0
        {
            // Precompute common terms for v and t computation
            let uo: Point3f = lerp::<Vector3f>(u1, p00.into(), p10.into()).into();
            let ud: Vector3f = lerp::<Vector3f>(u1, p01.into(), p11.into()) - Vector3f::from(uo);
            let deltao = uo - ray.o;
            let perp = ray.d.cross(ud);
            let p2 = perp.length_squared();

            // Compute matrix determinants for v and t numerators
            let v1 = SquareMatrix::<3>::new(
                [[deltao.x, ray.d.x, perp.x],
                [deltao.y, ray.d.y, perp.y],
                [deltao.z, ray.d.z, perp.z]]
            ).determinant();
            let t1 = SquareMatrix::<3>::new(
                [[deltao.x, ud.x, perp.x],
                [deltao.y, ud.y, perp.y],
                [deltao.z, ud.z, perp.z]]
            ).determinant();

            // Set u, v, and t if intersection is valid
            if t1 > p2 * eps && 0.0 <= v1 && v1 <= p2
            {
                u = u1;
                v = v1 / p2;
                t = t1 / p2;
            }
        }

        // Compute v and t for second u intersection
        if 0.0 <= u2 && u2 <= 1.0 && u2 != u1
        {
            let uo: Point3f = lerp::<Vector3f>(u2, p00.into(), p10.into()).into();
            let ud = lerp::<Vector3f>(u2, p01.into(), p11.into()) - Vector3f::from(uo);
            let deltao = uo - ray.o;
            let perp = ray.d.cross(ud);
            let p2 = perp.length_squared();
            let v2 = SquareMatrix::<3>::new(
                [[deltao.x, ray.d.x, perp.x],
                [deltao.y, ray.d.y, perp.y],
                [deltao.z, ray.d.z, perp.z]]
            ).determinant();
            let mut t2 = SquareMatrix::<3>::new(
                [[deltao.x, ud.x, perp.x],
                [deltao.y, ud.y, perp.y],
                [deltao.z, ud.z, perp.z]]
            ).determinant();
            t2 /= p2;
            if 0.0 <= v2 && v2 <= p2 && t > t2 && t2 > eps 
            {
                t = t2;
                u = u2;
                v = v2 / p2;
            }
        }

        if t >= t_max
        {
            return None;
        }

        Some(BilinearIntersection{
            uv: Point2f::new(u, v),
            t,
        })
    }

    fn interaction_from_intersection(
        mesh: &Arc<BilinearPatchMesh>,
        blp_index: usize,
        uv: Point2f,
        time: Float,
        wo: Vector3f,
    ) -> SurfaceInteraction
    {
        // Compute bilienar patch point pt, dpdu, and dpdv for (u, v)
        // Get bilinear patch vertices in p00, p01, p10, p11
        let (p00, p10, p01, p11) = BilinearPatch::get_points(mesh, blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(mesh, blp_index);

        let p: Point3f = lerp(uv[0], 
            lerp::<Vector3f>(uv[1], p00.into(), p01.into()),
            lerp::<Vector3f>(uv[1], p10.into(), p11.into())
        ).into();
        let mut dpdu = lerp::<Vector3f>(uv[1], p10.into(), p11.into())
            - lerp::<Vector3f>(uv[1], p00.into(), p01.into());
        let mut dpdv = lerp::<Vector3f>(uv[0], p01.into(), p11.into())
            - lerp::<Vector3f>(uv[0], p00.into(), p10.into());

        // Compute (s, t) texture coordinates at bilinear patch (u, v)
        let mut st = uv;
        let mut duds = 1.0;
        let mut dudt = 0.0;
        let mut dvds = 0.0;
        let mut dvdt = 1.0;
        if !mesh.uv.is_empty()
        {
            let uv00 = mesh.uv[v0];
            let uv10 = mesh.uv[v1];
            let uv01 = mesh.uv[v2];
            let uv11 = mesh.uv[v3];
            st = lerp::<Vector2f>(
                uv[0],
                lerp::<Vector2f>(uv[1], uv00.into(), uv01.into()),
                lerp::<Vector2f>(uv[1], uv10.into(), uv11.into())
            ).into();

            // Update bilinear patch dpdu and dpdv accounting for (s, t)
            // Compute partial derivatives of (u, v) w.r.t. (s, t)
            let dstdu = lerp::<Vector2f>(
                uv[1],
                uv10.into(),
                uv11.into()
            ) - lerp::<Vector2f>(
                uv[1],
                uv00.into(),
                uv01.into()
            );
            let dstdv = lerp::<Vector2f>(
                uv[0],
                uv01.into(),
                uv11.into()
            ) - lerp::<Vector2f>(
                uv[0],
                uv00.into(),
                uv10.into()
            );
            duds = if Float::abs(dstdu[0]) < 1e-8
            {
                0.0
            } else {
                1.0 / dstdu[0]
            };
            dvds = if Float::abs(dstdv[0]) < 1e-8
            {
                0.0
            } else {
                1.0 / dstdv[0]
            };
            dudt = if Float::abs(dstdu[1]) < 1e-8
            {
                0.0
            } else {
                1.0 / dstdu[1]
            };
            dvdt = if Float::abs(dstdv[1]) < 1e-8
            {
                0.0
            } else {
                1.0 / dstdv[1]
            };

            // Compute partial derivatives of pt w.r.t. (s, t)
            let dpds = dpdu * duds + dpdv * dvds;
            let mut dpdt = dpdu * dudt + dpdv * dvdt;

            // Set dpdu and dpdv to updated partial derivatives
            if dpds.cross(dpdt) != Vector3f::ZERO
            {
                if dpdu.cross(dpdv).dot(dpds.cross(dpdt)) < 0.0
                {
                    dpdt = -dpdt;
                }
                debug_assert!(dpdu.cross(dpdv).normalize().dot(dpds.cross(dpdt).normalize()) > -1e3);
                dpdu = dpds;
                dpdv = dpdt;
            }
        }
        
        // Find partial derivatives dndu and dndv for bilinear patch
        let d2pduu = Vector3f::ZERO;
        let d2pdvv = Vector3f::ZERO;
        let d2pduv = (p00 - p01) + (p11 - p10);

        // Compute coefficients for fundamental forms
        let e1 = dpdu.dot(dpdu);
        let f1 = dpdu.dot(dpdv);
        let g1 = dpdv.dot(dpdv);
        let n = dpdu.cross(dpdv).normalize();
        let e2 = n.dot(d2pduu);
        let f2 = n.dot(d2pduv);
        let g2 = n.dot(d2pdvv);

        // Compute dndu and dndv from fundamental form coefficients
        let egf2 = Float::difference_of_products(e1, g1, f1, f1);
        let inv_egf2 = if egf2 != 0.0
        {
            1.0 / egf2
        } else {
            0.0
        };
        let mut dndu: Normal3f = ((f1 * f2 - e2 * g1) * inv_egf2 * dpdu + (e2 * f1 - f2 * e1) * inv_egf2 * dpdv).into();
        let mut dndv: Normal3f = ((g2 * f1 - f2 * g1) * inv_egf2 * dpdu + (f2 * f1 - g2 * e1) * inv_egf2 * dpdv).into();

        // Update dndu and dndv to account for (s, t) parameterization
        let dnds = dndu * duds + dndv * dvds;
        let dndt = dndu * dudt + dndv * dvdt;

        dndu = dnds;
        dndv = dndt;

        // Initialize bilinear patch intersection point error
        let p_abs_sum = p00.abs() + p01.abs().into() + p10.abs().into() + p11.abs().into();
        let p_error = gamma(6) * Vector3f::from(p_abs_sum);

        // Initialize surface interaction for bilinear patch intersection
        let face_index = if mesh.face_indices.is_empty()
        {
            0
        } else {
            mesh.face_indices[blp_index]
        };

        let flip_normal = mesh.reverse_orientation ^ mesh.transform_swaps_handedness;
        let mut isect = SurfaceInteraction::new_with_face_index(
            Point3fi::from_value_and_error(p, p_error),
            st,
            wo,
            dpdu,
            dpdv,
            dndu,
            dndv,
            time,
            flip_normal,
            face_index as i32,
        );

        // Compute bilinear patch shading normal if necessary
        if !mesh.n.is_empty()
        {
            let n00 = mesh.n[v0];
            let n10 = mesh.n[v1];
            let n01 = mesh.n[v2];
            let n11 = mesh.n[v3];
            let mut ns = lerp(uv[0], lerp(uv[1], n00, n01), lerp(uv[1], n10, n11));
            
            if ns.length_squared() > 0.0 
            {
                ns = ns.normalize();
                // Set shading geometry for bilinear patch intersection
                let mut dndu = lerp(uv[1], n10, n11) - lerp(uv[1], n00, n01);
                let mut dndv = lerp(uv[0], n01, n11) - lerp(uv[0], n00, n10);
                // Update dndu and dndv to account for (s, t) parameterization
                let dnds = dndu * duds + dndv * dvds;
                let dndt = dndu * dudt + dndv * dvdt;
                dndu = dnds;
                dndv = dndt;

                let r = Transform::rotate_from_to(&isect.interaction.n.into(), &ns.into());
                isect.set_shading_geometry(ns, r.apply(&dpdu), r.apply(&dpdv), dndu, dndv, true);
            }
        }

        isect
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
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let intersection = BilinearPatch::intersect_blp(ray, t_max, p00, p10, p01, p11);
        if intersection.is_none()
        {
            return None;
        }
        let intersection = intersection.unwrap();
        let interaction = BilinearPatch::interaction_from_intersection(&self.mesh, self.blp_index, intersection.uv, ray.time, -ray.d);
        Some(ShapeIntersection{
            intr: interaction,
            t_hit: intersection.t,
        })
    }

    fn intersect_predicate(&self, ray: &crate::ray::Ray, t_max: Float) -> bool {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let intersection = BilinearPatch::intersect_blp(ray, t_max, p00, p10, p01, p11);
        intersection.is_some()
    }

    fn area(&self) -> Float {
        self.area
    }

    fn sample(&self, u: crate::vecmath::Point2f) -> Option<super::ShapeSample> {
        let (p00, p10, p01, p11) = BilinearPatch::get_points(&self.mesh, self.blp_index);
        let (v0, v1, v2, v3) = BilinearPatch::get_vertex_indices(&self.mesh, self.blp_index);

        // TODO Handle if we use an image distribution for emission
        // Sample bilinear patch parametric (u, v) coordinates
        let (uv, pdf) = if BilinearPatch::is_rectangle(&self.mesh, self.blp_index)
        {
            (u, 1.0)
        } else {
            let w = [
                (p10 - p00).cross(p01 - p00).length(),
                (p10 - p00).cross(p11 - p10).length(),
                (p01 - p00).cross(p11 - p01).length(),
                (p11 - p10).cross(p11 - p01).length(),
            ];

            let uv = sample_bilinear(u, &w);
            let pdf = bilinear_pdf(uv, &w);
            (uv, pdf)
        };

        // Compute bilinear patch geometric quantities at the sampled uv
        let pu0: Point3f = lerp::<Vector3f>(uv[0], p00.into(), p10.into()).into();
        let pu1: Point3f = lerp::<Vector3f>(uv[1], p10.into(), p11.into()).into();
        let p: Point3f = lerp::<Vector3f>(uv[0], pu0.into(), pu1.into()).into();
        let dpdu  = pu1 - pu0;
        let dpdv = lerp::<Vector3f>(uv[0], p01.into(), p11.into()) - lerp::<Vector3f>(uv[0], p00.into(), p10.into());

        if dpdu.length_squared() == 0.0 || dpdv.length_squared() == 0.0
        {
            return None;
        }

        let mut st = uv;
        if !self.mesh.uv.is_empty()
        {
            // Compute texture coordinates for bilinear patch intersection point
            let uv00 = self.mesh.uv[v0];
            let uv10 = self.mesh.uv[v1];
            let uv01 = self.mesh.uv[v2];
            let uv11 = self.mesh.uv[v3];
            st = lerp::<Vector2f>(
                uv[0],
                lerp::<Vector2f>(uv[1], uv00.into(), uv01.into()),
                lerp::<Vector2f>(uv[1], uv10.into(), uv11.into())
            ).into();
        }

        // Compute surface normal for sampled bilinear patch uv
        let mut n: Normal3f = dpdu.cross(dpdv).normalize().into();

        // Flip normal if necessary
        if !self.mesh.n.is_empty()
        {
            let n00 = self.mesh.n[v0];
            let n10 = self.mesh.n[v1];
            let n01 = self.mesh.n[v2];
            let n11 = self.mesh.n[v3];
            let ns = lerp(uv[0], lerp(uv[1], n00, n01), lerp(uv[1], n10, n11));
            n = n.face_forward(ns);
        } else if self.mesh.reverse_orientation ^ self.mesh.transform_swaps_handedness {
            n = -n;
        }

        // Compute p_error for sampled bilinear patch uv
        let p_abs_sum = p00.abs() + p01.abs().into() + p10.abs().into() + p11.abs().into();
        let p_error = gamma(6) * Vector3f::from(p_abs_sum);

        Some(ShapeSample{
            intr: Interaction::new(
                Point3fi::from_value_and_error(p, p_error),
                n,
                st,
                Default::default(),
                Default::default(),
            ),
            pdf: pdf / dpdu.cross(dpdv).length(),
        })
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