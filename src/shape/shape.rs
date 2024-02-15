use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;
use ply_rs::ply;

use crate::{
    bounding_box::Bounds3f, direction_cone::DirectionCone, file::resolve_filename, interaction::{Interaction, SurfaceInteraction}, loading::{paramdict::ParameterDictionary, parser_target::FileLoc}, options::Options, ray::Ray, shape::{mesh::{BilinearPatchMesh, TriQuadMesh}, TriangleMesh}, texture::FloatTexture, transform::Transform, vecmath::{point::Point3fi, Normal3f, Point2f, Point3f, Vector3f}, Float
};

use super::{sphere::Sphere, BilinearPatch, Triangle};

/// The Shape interface provides basic geometric properties of the primitive,
/// such as its surface area and its ray intersection routine. The non-geometric
/// data such as material properties are handled by the PrimitiveI interface.
pub trait ShapeI {
    /// Spatial extent of the shape
    fn bounds(&self) -> Bounds3f;

    /// Bounds the range of surface normals
    fn normal_bounds(&self) -> DirectionCone;

    /// Finds the first ray-shape intersection from (0, t_max), or None if there is no intersection.
    /// The rays are passed in rendering space, so shapes are responsible for transforming them
    /// to object space if needed for intersection tests. the intersection information returned should
    /// be in rendering space.
    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection>;

    /// Detects IF an intersection occurs, but does not return information about the intersection.
    /// This saves computation if only a true/false intersection is needed.
    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool;

    /// Surface area of the shape in rendering space.
    fn area(&self) -> Float;

    /// Samples a point on the surface of the shape.
    /// This is useful for using shapes as emitters.
    fn sample(&self, u: Point2f) -> Option<ShapeSample>;

    /// Probability density for sampling the specified point on the shape
    /// that corresponds to the given interaction.
    /// The interaction should be on the surface of the shape; the caller must ensure this.
    fn pdf(&self, interaction: &Interaction) -> Float;

    /// Like sample(), but takes a reference point from which the shape is being viewed.
    /// This is useful for lighting, since the caller can pass in the point
    /// to be lit and allow shape implementations to ensure they only sample the portion
    /// of the shape that is potentially visible from that point.
    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample>;

    /// Returns the shape's probability of sampling a point on the light such that the incident
    /// direction at the reference point is wi. The density should be with respect to the solid
    /// angle at the reference point. This should only be called for a direction that is known to
    /// intersect with the shape from the reference point.
    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vector3f) -> Float;
}

// TODO consider using enum_dispatch or equivalent; I'm doing this explicitly because I
// am doing write-ups comparing Rust and C++, so "raw rust" can be helpful.
#[derive(Debug, Clone)]
pub enum Shape {
    Sphere(Sphere),
    Triangle(Triangle),
    BilinearPatch(BilinearPatch),
}

impl Shape {
    pub fn create(
        name: &str,
        render_from_object: &Transform,
        object_from_render: &Transform,
        reverse_orientation: bool,
        parameters: &mut ParameterDictionary,
        _float_textures: &HashMap<String, Arc<FloatTexture>>, // TODO will be used in future
        loc: &FileLoc,
        options: &Options,
    ) -> Vec<Arc<Shape>> {
        let shapes = match name {
            "sphere" => {
                let sphere = Sphere::create(
                    render_from_object,
                    object_from_render,
                    reverse_orientation,
                    parameters,
                    loc,
                );
                vec![Arc::new(Shape::Sphere(sphere))]
            }
            "trianglemesh" => {
                let trianglemesh = Arc::new(Triangle::create_mesh(
                    render_from_object,
                    reverse_orientation,
                    parameters,
                    loc,
                ));
                Triangle::create_triangles(trianglemesh)
            }
            "plymesh" => {
                let filename = parameters.get_one_string("filename", "".to_string());
                let filename = resolve_filename(options, &filename);
                let ply_mesh = TriQuadMesh::read_ply(&filename);

                // TODO Handle displacement texture.

                let mut tri_quad_shapes = Vec::new();
                if !ply_mesh.tri_indices.is_empty()
                {
                    // TODO We could try to share these lists instead of cloning them.
                    let mesh = Arc::new(TriangleMesh::new(
                        render_from_object,
                        reverse_orientation,
                        ply_mesh.tri_indices.into_iter().map(|x| x as usize).collect_vec(),
                        ply_mesh.p.clone(),
                        Vec::new(),
                        ply_mesh.n.clone(),
                        ply_mesh.uv.clone(),
                        ply_mesh.face_indices.clone().into_iter().map(|x| x as usize).collect_vec(),
                    ));
                    tri_quad_shapes = Triangle::create_triangles(mesh);
                }

                if !ply_mesh.quad_indices.is_empty()
                {
                    let quad_mesh = Arc::new(BilinearPatchMesh::new(
                        render_from_object,
                        reverse_orientation,
                        ply_mesh.quad_indices.into_iter().map(|x| x as usize).collect_vec(),
                        ply_mesh.p,
                        ply_mesh.n,
                        ply_mesh.uv,
                        ply_mesh.face_indices.into_iter().map(|x| x as usize).collect_vec(),
                    ));
                    let patches = BilinearPatch::create_patches(quad_mesh);
                    tri_quad_shapes.extend(patches);
                }
                tri_quad_shapes
            }
            _ => {
                panic!("Unknown Shape {}", name);
            }
        };

        shapes
    }
}

impl ShapeI for Shape {
    fn bounds(&self) -> Bounds3f {
        match self {
            Shape::Sphere(s) => s.bounds(),
            Shape::Triangle(t) => t.bounds(),
            Shape::BilinearPatch(s) => s.bounds(),
            
        }
    }

    fn normal_bounds(&self) -> DirectionCone {
        match self {
            Shape::Sphere(s) => s.normal_bounds(),
            Shape::Triangle(t) => t.normal_bounds(),
            Shape::BilinearPatch(s) => s.normal_bounds(),
        }
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        match self {
            Shape::Sphere(s) => s.intersect(ray, t_max),
            Shape::Triangle(t) => t.intersect(ray, t_max),
            Shape::BilinearPatch(s) => s.intersect(ray, t_max),
        }
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        match self {
            Shape::Sphere(s) => s.intersect_predicate(ray, t_max),
            Shape::Triangle(t) => t.intersect_predicate(ray, t_max),
            Shape::BilinearPatch(s) => s.intersect_predicate(ray, t_max),
        }
    }

    fn area(&self) -> Float {
        match self {
            Shape::Sphere(s) => s.area(),
            Shape::Triangle(t) => t.area(),
            Shape::BilinearPatch(s) => s.area(),
        }
    }

    fn sample(&self, u: Point2f) -> Option<ShapeSample> {
        match self {
            Shape::Sphere(s) => s.sample(u),
            Shape::Triangle(t) => t.sample(u),
            Shape::BilinearPatch(s) => s.sample(u),
        }
    }

    fn pdf(&self, interaction: &Interaction) -> Float {
        match self {
            Shape::Sphere(s) => s.pdf(interaction),
            Shape::Triangle(t) => t.pdf(interaction),
            Shape::BilinearPatch(s) => s.pdf(interaction),
        }
    }

    fn sample_with_context(&self, ctx: &ShapeSampleContext, u: Point2f) -> Option<ShapeSample> {
        match self {
            Shape::Sphere(s) => s.sample_with_context(ctx, u),
            Shape::Triangle(t) => t.sample_with_context(ctx, u),
            Shape::BilinearPatch(s) => s.sample_with_context(ctx, u),
        }
    }

    fn pdf_with_context(&self, ctx: &ShapeSampleContext, wi: Vector3f) -> Float {
        match self {
            Shape::Sphere(s) => s.pdf_with_context(ctx, wi),
            Shape::Triangle(t) => t.pdf_with_context(ctx, wi),
            Shape::BilinearPatch(s) => s.pdf_with_context(ctx, wi),
        }
    }
}

pub struct ShapeIntersection {
    pub intr: SurfaceInteraction,
    pub t_hit: Float,
}

pub struct QuadricIntersection {
    pub t_hit: Float,
    pub p_obj: Point3f,
    pub phi: Float,
}

pub struct ShapeSample {
    /// Interaction corresponding to the sampled point on the surface
    pub intr: Interaction,
    /// PDF for that sample w.r.t. the surface area of the shape
    pub pdf: Float,
}

pub struct ShapeSampleContext {
    /// Reference point
    pub pi: Point3fi,
    /// Geometric normal. For participating media, zero.
    pub n: Normal3f,
    /// Shading normal. For participating media, zero.
    pub ns: Normal3f,
    pub time: Float,
}

impl ShapeSampleContext {
    pub fn new(pi: Point3fi, n: Normal3f, ns: Normal3f, time: Float) -> ShapeSampleContext {
        ShapeSampleContext { pi, n, ns, time }
    }

    pub fn from_surface_interaction(si: &SurfaceInteraction) -> ShapeSampleContext {
        ShapeSampleContext {
            pi: si.interaction.pi,
            n: si.interaction.n,
            ns: si.shading.n,
            time: si.interaction.time,
        }
    }

    // TODO From medium interaction.

    /// Provides the point for scenarios where numerical error is not important.
    pub fn p(&self) -> Point3f {
        self.pi.into()
    }

    pub fn offset_ray_origin(&self, w: Vector3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n, w)
    }

    pub fn offset_ray_origin_pt(&self, pt: Point3f) -> Point3f {
        self.offset_ray_origin(pt - self.p())
    }

    pub fn spawn_ray(&self, _w: Vector3f) -> Ray {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ray::Ray,
        shape::{shape::ShapeI, sphere::Sphere},
        transform::Transform,
        vecmath::{Point3f, Tuple3, Vector3f},
        Float,
    };

    #[test]
    fn sphere_basic() {
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            1.0,
            -1.0,
            1.0,
            360.0,
        );
        let ray = Ray::new(Point3f::new(0.0, 0.0, -2.0), Vector3f::Z, None);
        assert!(sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(ray.o, -ray.d, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 1.0001, -2.0), Vector3f::Z, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));
    }

    #[test]
    fn sphere_partial_basic() {
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            1.0,
            -0.5,
            0.5,
            360.0,
        );
        let ray = Ray::new(Point3f::new(0.0, -2.0, 0.0), Vector3f::Y, None);
        assert!(sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(ray.o, -ray.d, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 0.0, 0.5001), Vector3f::Y, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));

        let ray = Ray::new(Point3f::new(0.0, 0.0, -0.5001), Vector3f::Y, None);
        assert!(!sphere.intersect_predicate(&ray, Float::INFINITY));
    }
}
