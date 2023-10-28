use crate::{
    light::Light,
    material::Material,
    vecmath::{point::Point3fi, vector::Vector3, Normal3f, Normalize, Point2f, Point3f, Vector3f},
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

impl Interaction {
    pub fn new(pi: Point3fi, n: Normal3f, uv: Point2f, wo: Vector3f, time: Float) -> Interaction {
        Interaction {
            pi,
            time,
            wo,
            n,
            uv,
        }
    }
}

pub struct SurfaceInteraction {
    pub interaction: Interaction,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shading: SurfaceInteractionShading,
    pub face_index: i32,
    pub material: Option<Material>,
    pub area_light: Option<Light>,
    pub dpdx: Option<Vector3f>,
    pub dpdy: Option<Vector3f>,
    pub dudx: Float,
    pub dvdx: Float,
    pub dudy: Float,
    pub dvdy: Float,
}

impl SurfaceInteraction {
    pub fn new(
        pi: Point3fi,
        uv: Point2f,
        wo: Vector3f,
        dpdu: Vector3f,
        dpdv: Vector3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: Float,
        flip_normal: bool,
    ) -> SurfaceInteraction {
        let normal_sign = if flip_normal { -1.0 } else { 1.0 };
        let normal = normal_sign * Normal3f::from(dpdu.cross(&dpdv).normalize());
        let interaction = Interaction::new(pi, normal, uv, wo, time);
        SurfaceInteraction {
            interaction,
            dpdu,
            dpdv,
            dndu,
            dndv,
            shading: SurfaceInteractionShading {
                n: normal,
                dpdu,
                dpdv,
                dndu,
                dndv,
            },
            face_index: 0,
            material: None,
            area_light: None,
            dpdx: None,
            dpdy: None,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }

    pub fn p(&self) -> Point3f {
        self.interaction.pi.into()
    }
}

pub struct SurfaceInteractionShading {
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}
