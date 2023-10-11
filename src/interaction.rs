use crate::{
    light::Light,
    material::Material,
    vecmath::{point::Point3fi, Normal3f, Point2f, Vector3f},
    Float,
};

pub struct Interaction {
    pub pi: Point3fi,
    pub time: Float,
    pub wo: Vector3f,
    pub n: Normal3f,
    pub uv: Point2f,
    // TODO Medium, MediumInterface. But omitting for now for simplicity.
}

pub struct SurfaceInteraction {
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shading: SurfaceInteractionShading,
    pub face_index: i32,
    pub material: Material,
    pub dpdx: Vector3f,
    pub area_light: Light,
    pub dpdy: Vector3f,
    pub dudx: Float,
    pub dvdx: Float,
    pub dudy: Float,
    pub dvdy: Float,
}

pub struct SurfaceInteractionShading {
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}
