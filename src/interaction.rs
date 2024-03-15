use std::sync::Arc;

use log::warn;
use rand::rngs::SmallRng;

use crate::{
    bsdf::BSDF,
    bxdf::{self, BxDFFLags, DiffuseBxDF},
    camera::{Camera, CameraI},
    light::{Light, LightI},
    material::{Material, MaterialEvalContext, MaterialI, UniversalTextureEvaluator},
    math::{sqr, DifferenceOfProducts},
    options::Options,
    ray::{AuxiliaryRays, Ray, RayDifferential},
    sampler::{Sampler, SamplerI},
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    vecmath::{
        normal::Normal3, point::Point3fi, vector::Vector3, Length, Normal3f, Normalize, Point2f,
        Point3f, Vector3f,
    },
    Float,
};

#[derive(Debug, Clone)]
pub struct Interaction {
    /// The point the interaction is at
    pub pi: Point3fi,
    /// The time of the interaction
    pub time: Float,
    /// For interactions that lie along a ray, the negative ray direction is stored in wo.
    /// wo is used to match the notation for the outgoing direction when computing lighting at points.
    /// For interactions where the notion of an outgoing direction doesn't apply, wo is (0, 0, 0).
    pub wo: Vector3f,
    // For interactions on surfaces, the surface normal at the point
    pub n: Normal3f,
    /// The UV of the interaction, if applicable; (0, 0) otherwise.
    pub uv: Point2f,
    // TODO Medium, MediumInterface. But omitting for now for simplicity.
}

impl Default for Interaction {
    fn default() -> Self {
        Self {
            pi: Default::default(),
            time: Default::default(),
            wo: Default::default(),
            n: Default::default(),
            uv: Default::default(),
        }
    }
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

    pub fn p(&self) -> Point3f {
        self.pi.into()
    }

    pub fn offset_ray_origin(&self, w: Vector3f) -> Point3f {
        Ray::offset_ray_origin(self.pi, self.n, w)
    }

    pub fn spawn_ray(&self, d: Vector3f) -> RayDifferential {
        // TODO get_medium() here.
        let ray = Ray::new(self.offset_ray_origin(d), d, None);
        RayDifferential {
            ray,
            auxiliary: None,
        }
    }

    pub fn spawn_ray_to_interaction(&self, it: &Interaction) -> Ray {
        let mut r = Ray::spawn_ray_to_both_offset(self.pi, self.n, self.time, it.pi, it.n);
        // TODO set medium
        r
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceInteraction {
    pub interaction: Interaction,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
    pub shading: SurfaceInteractionShading,
    pub face_index: i32,
    pub material: Option<Arc<Material>>,
    // TODO consider using just an Option<Light> if I make Light copyable.
    // Would require moving to cacheing DenselySampledSpectrum in Lights to avoid
    // a large/impossible copy.
    pub area_light: Option<Arc<Light>>,
    pub dpdx: Vector3f,
    pub dpdy: Vector3f,
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
        let normal = normal_sign * Normal3f::from(dpdu.cross(dpdv).normalize());
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
            dpdx: Vector3f::ZERO,
            dpdy: Vector3f::ZERO,
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
        }
    }

    pub fn new_with_face_index(
        pi: Point3fi,
        uv: Point2f,
        wo: Vector3f,
        dpdu: Vector3f,
        dpdv: Vector3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: Float,
        flip_normal: bool,
        face_index: i32,
    ) -> SurfaceInteraction
    {
        let mut isect = SurfaceInteraction::new(
            pi, uv, wo, dpdu, dpdv, dndu, dndv, time, flip_normal
        );
        isect.face_index = face_index;
        isect
    }

    pub fn p(&self) -> Point3f {
        self.interaction.p()
    }

    // TODO We'll also want to add the Medium information here, once we implement participating media. pg 398.
    pub fn set_intersection_properties(
        &mut self,
        mtl: &Arc<Material>,
        area_light: &Option<Arc<Light>>,
    ) {
        self.area_light = area_light.clone();
        self.material = Some(mtl.clone());
    }

    // TODO Can this and compute_differentials() be made const?
    //   Although I suppose that surface interactions are short-lived, so perhaps it's okay
    //   to leave them mut.
    pub fn get_bsdf(
        &mut self,
        ray: &RayDifferential,
        lambda: &mut SampledWavelengths,
        camera: &Camera,
        sampler: &mut Sampler,
        options: &Options,
        rng: &mut SmallRng,
    ) -> Option<BSDF> {
        // Estimate (u, v) and position differentials at intersection point
        self.compute_differentials(ray, camera, sampler.samples_per_pixel(), options);


        // This should occur only at a non-scattering interface between two types of participating media.
        if self.material.is_none() {
            return None;
        }
        let mut material = self.material.as_ref().unwrap().clone();

        // Resolve mixed materials
        let mut is_mixed = match material.as_ref() {
            Material::Mix(_) => true,
            _ => false,
        };

        let material_eval_context = MaterialEvalContext::from(&*self);
        while is_mixed
        {
            match material.as_ref() {
                Material::Mix(m) => {
                    material = m.choose_material(&UniversalTextureEvaluator {}, &material_eval_context, rng);
                },
                _ => is_mixed = false,
            };
        }

        let material = match material.as_ref()
        {
            Material::Mix(m) => panic!("Material::Mix should have been resolved by now!"),
            Material::Single(m) => m,
        };

        let displacement = material.get_displacement();
        let normal_map = material.get_normal_map();
        if displacement.is_some() || normal_map.is_some() {
            // TODO handle shading using normal or bump map - we just won't do anything right now...
            warn!("Normal and displacement maps not fully implemented yet!");
        }

        let bsdf = material.get_bsdf(
            &UniversalTextureEvaluator {},
            &material_eval_context,
            lambda,
        );

        let bsdf = if options.force_diffuse {
            // TODO PBRT also checks if bsdf is null. PBRT does this alot with its
            //   tagged pointers. We might want to wrap things in Option<> if this is possible.

            // Override bsdf with the diffuse equivalent
            let r = bsdf.rho_hd(
                self.interaction.wo,
                &[sampler.get_1d()],
                &[sampler.get_2d()],
            );
            BSDF::new(
                self.shading.n,
                self.shading.dpdu,
                bxdf::BxDF::Diffuse(DiffuseBxDF::new(r)),
            )
        } else {
            bsdf
        };

        Some(bsdf)
    }

    pub fn compute_differentials(
        &mut self,
        ray: &RayDifferential,
        camera: &Camera,
        samples_per_pixel: i32,
        options: &Options,
    ) {
        if options.disable_texture_filtering {
            self.dudx = 0.0;
            self.dudy = 0.0;
            self.dvdx = 0.0;
            self.dvdy = 0.0;
            self.dpdx = Vector3f::ZERO;
            self.dpdy = Vector3f::ZERO;
            return;
        }

        if ray.auxiliary.as_ref().is_some_and(|aux| {
            self.interaction.n.dot_vector(aux.rx_direction) != 0.0
                && self.interaction.n.dot_vector(aux.ry_direction) != 0.0
        }) {
            let aux = ray.auxiliary.as_ref().unwrap();
            // Estimate screen-space change in pt using ray differentials
            // Compute auxiliary intersection points with plane, px and py
            let d = -self.interaction.n.dot_vector(self.p().into());
            let tx = (-self.interaction.n.dot_vector(aux.rx_origin.into()) - d)
                / self.interaction.n.dot_vector(aux.rx_direction);
            debug_assert!(tx.is_finite() && !tx.is_nan());
            let px = aux.rx_origin + tx * aux.rx_direction;
            let ty = (-self.interaction.n.dot_vector(aux.ry_origin.into()) - d)
                / self.interaction.n.dot_vector(aux.ry_direction);
            debug_assert!(ty.is_finite() && !ty.is_nan());
            let py = aux.ry_origin + ty * aux.ry_direction;

            self.dpdx = px - self.p();
            self.dpdy = py - self.p();
        } else {
            // Approximate screen-based change in pt based on camera projection
            (self.dpdx, self.dpdy) = camera.approximate_dp_dxy(
                self.p(),
                self.interaction.n,
                self.interaction.time,
                samples_per_pixel,
                options,
            );
        }

        // Estimate screen-space changes in (u, v).
        let ata00 = self.dpdu.dot(self.dpdu);
        let ata01 = self.dpdu.dot(self.dpdv);
        let ata11 = self.dpdv.dot(self.dpdv);
        let inv_det = 1.0 / Float::difference_of_products(ata00, ata11, ata01, ata01);
        let inv_det = if inv_det.is_finite() { inv_det } else { 0.0 };

        let atb0x = self.dpdu.dot(self.dpdx);
        let atb1x = self.dpdv.dot(self.dpdx);
        let atb0y = self.dpdu.dot(self.dpdy);
        let atb1y = self.dpdv.dot(self.dpdy);

        // Compute u and v derivatives wrt x and y
        self.dudx = Float::difference_of_products(ata11, atb0x, ata01, atb1x) * inv_det;
        self.dvdx = Float::difference_of_products(ata00, atb1x, ata01, atb0x) * inv_det;
        self.dudy = Float::difference_of_products(ata11, atb0y, ata01, atb1y) * inv_det;
        self.dvdy = Float::difference_of_products(ata00, atb1y, ata01, atb0y) * inv_det;

        // Clamp derivatives to reasonable values
        self.dudx = if self.dudx.is_finite() {
            Float::clamp(self.dudx, -1e8, 1e8)
        } else {
            0.0
        };
        self.dvdx = if self.dvdx.is_finite() {
            Float::clamp(self.dvdx, -1e8, 1e8)
        } else {
            0.0
        };
        self.dudy = if self.dudy.is_finite() {
            Float::clamp(self.dudy, -1e8, 1e8)
        } else {
            0.0
        };
        self.dvdy = if self.dvdy.is_finite() {
            Float::clamp(self.dvdy, -1e8, 1e8)
        } else {
            0.0
        };
    }

    /// Computes the emitted radiance at a surface point intersected by a ray.
    pub fn le(&self, w: Vector3f, lambda: &SampledWavelengths) -> SampledSpectrum {
        if let Some(area_light) = &self.area_light {
            area_light
                .as_ref()
                .l(self.p(), self.interaction.n, self.interaction.uv, w, lambda)
        } else {
            SampledSpectrum::from_const(0.0)
        }
    }

    pub fn set_shading_geometry(
        &mut self,
        ns: Normal3f,
        dpdus: Vector3f,
        dpdvs: Vector3f,
        dndus: Normal3f,
        dndvs: Normal3f,
        orientation_is_authoritative: bool,
    ) {
        self.shading.n = ns;
        debug_assert_ne!(self.shading.n, Normal3f::ZERO);
        if orientation_is_authoritative {
            self.interaction.n = self.interaction.n.face_forward(self.shading.n);
        } else {
            self.shading.n = self.shading.n.face_forward(self.interaction.n);
        }

        self.shading.dpdu = dpdus;
        self.shading.dpdv = dpdvs;
        self.shading.dndu = dndus;
        self.shading.dndv = dndvs;
        while self.shading.dpdu.length_squared() > 1e16 || self.shading.dpdv.length_squared() > 1e16
        {
            self.shading.dpdu /= 1e8;
            self.shading.dpdv /= 1e8;
        }
    }

    /// When a ray crosses to a new media that does not scatter light, spawn a ray
    /// in the same direction as the original ray, and set up any ray differentials
    /// so they continue in a straight line across the boundary.
    pub fn skip_intersection(&mut self, ray: &mut RayDifferential, t: Float) {
        let mut new_ray = self.interaction.spawn_ray(ray.ray.d);
        new_ray.auxiliary = if let Some(aux) = &ray.auxiliary {
            let rx_origin = aux.rx_origin + t * aux.rx_direction;
            let ry_origin = aux.ry_origin + t * aux.ry_direction;
            let rx_direction = aux.rx_direction;
            let ry_direction = aux.ry_direction;
            Some(AuxiliaryRays {
                rx_origin,
                rx_direction,
                ry_origin,
                ry_direction,
            })
        } else {
            None
        };
        *ray = new_ray
    }

    // This is useful for ray differentials for specular reflection and transmission,
    // which lets us have ray differentials after the first bounce.
    // Introduced by Igehy "Tracing ray differentials" (1999).
    // This is not used by the SimplePathIntegrator, but can be used
    // by more complex integrators.
    pub fn spawn_ray_with_differentials(
        &self,
        ray_i: &RayDifferential,
        wi: Vector3f,
        flags: BxDFFLags,
        eta: Float,
    ) -> RayDifferential
    {
        let mut rd = self.interaction.spawn_ray(wi);
        if let Some(aux) = &ray_i.auxiliary
        {
            // Compute ray differentials for specular reflection or transmission
            // Compute common factors for specular ray differentials.
            let mut n = self.shading.n;
            let mut dndx = self.shading.dndu * self.dudx + self.shading.dndv * self.dvdx;
            let mut dndy = self.shading.dndu * self.dudy + self.shading.dndv * self.dvdy;
            let dwodx = -aux.rx_direction - self.interaction.wo;
            let dwody = -aux.ry_direction - self.interaction.wo;

            rd.auxiliary = if flags == BxDFFLags::SPECULAR_REFLECTION {
                // Initialize origins of specular differential rays
                let rx_origin = self.interaction.p() + self.dpdx;
                let ry_origin = self.interaction.p() + self.dpdy;

                // Compute differential reflected directions
                let dwo_dotn_dx = dwodx.dot_normal(n) + self.interaction.wo.dot_normal(dndx);
                let dwo_dotn_dy = dwody.dot_normal(n) + self.interaction.wo.dot_normal(dndy);
                let rx_direction = wi - dwodx + 2.0 * (self.interaction.wo.dot_normal(n) * dndx + dwo_dotn_dx * n);
                let ry_direction = wi - dwody + 2.0 * (self.interaction.wo.dot_normal(n) * dndy + dwo_dotn_dy * n);
                Some(AuxiliaryRays
                {
                    rx_origin,
                    rx_direction,
                    ry_origin,
                    ry_direction,
                })
            } else if flags == BxDFFLags::SPECULAR_TRANSMISSION {
                // Initialize origins of specular differential rays
                let rx_origin = self.interaction.p() + self.dpdx;
                let ry_origin = self.interaction.p() + self.dpdy;

                // Compute differential transmitted directions
                // Find oriented surface normal for transmission
                if self.interaction.wo.dot_normal(n) < 0.0
                {
                    n = -n;
                    dndx = -dndx;
                    dndy = -dndy;
                }

                // Compute partial derivatives of mu
                let dwo_dotn_dx = dwodx.dot_normal(n) + self.interaction.wo.dot_normal(dndx);
                let dwo_dotn_dy = dwody.dot_normal(n) + self.interaction.wo.dot_normal(dndy);
                let mu = self.interaction.wo.dot_normal(n) / eta - wi.abs_dot_normal(n);
                let dmudx = dwo_dotn_dx * (1.0 / eta + 1.0 / sqr(eta) * self.interaction.wo.dot_normal(n) / wi.dot_normal(n));
                let dmudy = dwo_dotn_dy * (1.0 / eta + 1.0 / sqr(eta) * self.interaction.wo.dot_normal(n) / wi.dot_normal(n));

                let rx_direction = wi - eta * dwodx + Vector3f::from(mu * dndx + dmudx * n);
                let ry_direction = wi - eta * dwody + Vector3f::from(mu * dndy + dmudy * n);

                Some(AuxiliaryRays{
                    rx_origin,
                    rx_direction,
                    ry_origin,
                    ry_direction,
                })
            } else {
                None
            };
        }

        // Squash any potentially troublesome differentials
        if rd.auxiliary.as_ref().is_some_and(|aux| {
            aux.rx_direction.length_squared() > 1e16 || aux.ry_direction.length_squared() > 1e16
            || Vector3f::from(aux.rx_origin).length_squared() > 1e16 || Vector3f::from(aux.ry_origin).length_squared() > 1e16
        }) {
            rd.auxiliary = None;
        }

        rd
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceInteractionShading {
    pub n: Normal3f,
    pub dpdu: Vector3f,
    pub dpdv: Vector3f,
    pub dndu: Normal3f,
    pub dndv: Normal3f,
}
