// TODO implement depth-of-field
// TODO perspective camera
// TODO circular camera (maybe, low priority)
// TODO place Camera types into a Camera enum that impl CameraI (typical pattern).

use crate::{
    bounding_box::Bounds2f,
    film::{Film, FilmI},
    image::ImageMetadata,
    math::lerp,
    medium::Medium,
    options::{Options, RenderingCoordinateSystem},
    ray::{AuxiliaryRays, Ray, RayDifferential, RayI},
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    transform::Transform,
    vecmath::{normal::Normal3, Normal3f, Normalize, Point2f, Point3f, Tuple3, Vector3f},
    Float,
};

/// Interface that all different kinds of cameras must implement.
pub trait CameraI {
    /// Computes the ray corresponding to a given image sample
    fn generate_ray(&self, sample: &CameraSample, lambda: &SampledWavelengths)
        -> Option<CameraRay>;

    /// Like generate_ray(), but also computes the corresponding rays
    /// for pixels shifted one picel in the x and y directions on the film plane.
    /// This is useful for anti-aliasing.
    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential>;

    fn get_film(&mut self) -> &mut Film;

    /// Maps a uniform random sample u [0, 1) to a time when the camera shutter is open.
    fn sample_time(&self, u: Float) -> Float;

    fn init_metadata(&self, metadata: &mut ImageMetadata);

    fn get_camera_transform(&self) -> &CameraTransform;

    /// Returns an appoximation for (dpdx, dpdy) for a point in the scene.
    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vector3f, Vector3f);
}

pub enum Camera {
    Orthographic(OrthographicCamera),
}

impl CameraI for Camera {
    fn generate_ray(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        match self {
            Camera::Orthographic(c) => c.generate_ray(sample, lambda),
        }
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        match self {
            Camera::Orthographic(c) => c.generate_ray_differential(sample, lambda),
        }
    }

    fn get_film(&mut self) -> &mut Film {
        match self {
            Camera::Orthographic(c) => c.get_film(),
        }
    }

    fn sample_time(&self, u: Float) -> Float {
        match self {
            Camera::Orthographic(c) => c.sample_time(u),
        }
    }

    fn init_metadata(&self, metadata: &mut ImageMetadata) {
        match self {
            Camera::Orthographic(c) => c.init_metadata(metadata),
        }
    }

    fn get_camera_transform(&self) -> &CameraTransform {
        match self {
            Camera::Orthographic(c) => c.get_camera_transform(),
        }
    }

    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vector3f, Vector3f) {
        match self {
            Camera::Orthographic(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel, options),
        }
    }
}

/// Shared implementation details for different kinds of cameras.
struct CameraBase {
    camera_transform: CameraTransform,
    /// The time of the shutter opening
    shutter_open: Float,
    /// The time of the shutter closing
    shutter_close: Float,
    film: Film,
    /// The scattering medium that the camera lies in, if any.
    medium: Option<Medium>,
    min_pos_differential_x: Vector3f,
    min_pos_differential_y: Vector3f,
    min_dir_differential_x: Vector3f,
    min_dir_differential_y: Vector3f,
}

impl CameraBase {
    pub fn get_film(&self) -> &Film {
        &self.film
    }

    pub fn init_metadata(&self, metadata: &mut ImageMetadata) {
        metadata.camera_from_world = Some(
            *self
                .camera_transform
                .camera_from_world(self.shutter_open)
                .get_matrix(),
        );
    }

    // TODO rather than _v variants etc, should probably just specify that these are Trasnformable via generic constraint.
    // I think I already have that available in Transform now anyways?

    pub fn render_from_camera_v(&self, v: &Vector3f) -> Vector3f {
        self.camera_transform.render_from_camera_v(v)
    }

    pub fn render_from_camera_n(&self, n: &Normal3f) -> Normal3f {
        self.camera_transform.render_from_camera_n(n)
    }

    pub fn render_from_camera_p(&self, p: &Point3f) -> Point3f {
        self.camera_transform.render_from_camera_p(p)
    }

    pub fn render_from_camera_r(&self, r: &Ray) -> Ray {
        self.camera_transform.render_from_camera_r(r)
    }

    pub fn render_from_camera_rd(&self, r: &RayDifferential) -> RayDifferential {
        self.camera_transform.render_from_camera_rd(r)
    }

    pub fn camera_from_render_p(&self, p: &Point3f, _time: Float) -> Point3f {
        self.camera_transform.camera_from_render_p(p, _time)
    }

    pub fn camera_from_render_v(&self, v: &Vector3f, _time: Float) -> Vector3f {
        self.camera_transform.camera_from_render_v(v, _time)
    }

    pub fn camera_from_render_n(&self, n: Normal3f, _time: Float) -> Normal3f {
        self.camera_transform.camera_from_render_n(&n, _time)
    }

    pub fn camera_from_render_r(&self, r: &Ray, _time: Float) -> Ray {
        self.camera_transform.camera_from_render_r(r, _time)
    }

    pub fn camera_from_render_rd(&self, r: &RayDifferential, _time: Float) -> RayDifferential {
        self.camera_transform.camera_from_render_rd(r, _time)
    }

    /// u - the fraction between the shutter open and shutter close.
    pub fn sample_time(&self, u: Float) -> Float {
        lerp(u, &self.shutter_open, &self.shutter_close)
    }

    pub fn generate_ray_differential<T: CameraI>(
        &self,
        camera: &T,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        // Generate the base/central ray.
        let base_camera_ray = camera.generate_ray(sample, lambda)?;

        // Find a camera ray after shifting one pixel in the x direction; may need to
        // try in the negative x direction as well. Just store the origin and direction
        // since we don't need the time etc.
        let rx = [0.05, -0.05]
            .iter()
            .map(|eps| -> (Float, CameraSample) {
                // Shift the camera sample a bit, keep track of the epsilon.
                let mut sshift = sample.clone();
                sshift.p_film.x += eps;
                (*eps, sshift)
            })
            .find_map(|(eps, sshift)| -> Option<(Point3f, Vector3f)> {
                // Check if we can generate a ray for the shifted sample, and return the
                // origin and direction if so.
                if let Some(rx) = camera.generate_ray(&sshift, lambda) {
                    Some((
                        base_camera_ray.ray.o + (rx.ray.o - base_camera_ray.ray.o) / eps,
                        base_camera_ray.ray.d + (rx.ray.d - base_camera_ray.ray.d) / eps,
                    ))
                } else {
                    None
                }
            });

        // The same for the y direction
        let ry = [0.05, -0.05]
            .iter()
            .map(|eps| -> (Float, CameraSample) {
                let mut sshift = sample.clone();
                sshift.p_film.y += eps;
                (*eps, sshift)
            })
            .find_map(|(eps, sshift)| -> Option<(Point3f, Vector3f)> {
                if let Some(ry) = camera.generate_ray(&sshift, lambda) {
                    Some((
                        base_camera_ray.ray.o + (ry.ray.o - base_camera_ray.ray.o) / eps,
                        base_camera_ray.ray.d + (ry.ray.d - base_camera_ray.ray.d) / eps,
                    ))
                } else {
                    None
                }
            });

        let aux = if let (Some(rx), Some(ry)) = (rx, ry) {
            Some(AuxiliaryRays::new(rx.0, rx.1, ry.0, ry.1))
        } else {
            None
        };

        let ray_differential = RayDifferential::new(base_camera_ray.ray, aux);
        Some(CameraRayDifferential::new_with_weight(
            ray_differential,
            base_camera_ray.weight,
        ))
    }

    pub fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vector3f, Vector3f) {
        let p_camera = self.camera_from_render_p(&p, time);

        // Compute tangent plane equation for ray differential intersections
        let down_z_from_camera =
            Transform::rotate_from_to(&Vector3f::from(p_camera).normalize(), &Vector3f::Z);
        let p_down_z = down_z_from_camera.apply(&p_camera);
        let n_down_z = down_z_from_camera.apply(&self.camera_from_render_n(n, time));
        let d = n_down_z.z * p_down_z.z;

        // Find intersection points for approximated camera differential rays
        let x_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_x,
            Vector3f::Z + self.min_dir_differential_x,
            None,
        );
        let tx = -(n_down_z.dot_vector(&x_ray.o.into()) - d) / n_down_z.dot_vector(&x_ray.d);
        let y_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_y,
            Vector3f::Z + self.min_dir_differential_y,
            None,
        );

        let ty = -(n_down_z.dot_vector(&y_ray.o.into()) - d) / n_down_z.dot_vector(&y_ray.d);
        let px = x_ray.get(tx);
        let py = y_ray.get(ty);

        let spp_scale = if options.disable_pixel_jitter {
            1.0
        } else {
            Float::max(0.125, 1.0 / (samples_per_pixel as Float).sqrt())
        };

        let dpdx =
            spp_scale * self.render_from_camera_v(&down_z_from_camera.apply_inv(&(px - p_down_z)));
        let dpdy =
            spp_scale * self.render_from_camera_v(&down_z_from_camera.apply_inv(&(py - p_down_z)));
        (dpdx, dpdy)
    }
}

pub struct CameraRay {
    ray: Ray,
    weight: SampledSpectrum,
}

impl CameraRay {
    pub fn new(ray: Ray) -> CameraRay {
        CameraRay {
            ray,
            weight: SampledSpectrum::from_const(1.0),
        }
    }
}

pub struct CameraRayDifferential {
    pub ray: RayDifferential,
    pub weight: SampledSpectrum,
}

impl CameraRayDifferential {
    pub fn new(ray: RayDifferential) -> CameraRayDifferential {
        CameraRayDifferential {
            ray,
            weight: SampledSpectrum::from_const(1.0),
        }
    }

    pub fn new_with_weight(ray: RayDifferential, weight: SampledSpectrum) -> CameraRayDifferential {
        CameraRayDifferential { ray, weight }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CameraSample {
    pub p_film: Point2f,
    pub p_lens: Point2f,
    pub time: Float,
    pub filter_weight: Float,
}

pub struct CameraTransform {
    // TODO render_from_camera should be an AnimatedTransform when that's implemented,
    //  along with any associated changes that entails.
    render_from_camera: Transform,
    world_from_render: Transform,
}

impl CameraTransform {
    pub fn new(world_from_camera: &Transform, options: &Options) -> CameraTransform {
        // TODO would need to update this for AnimatedTransform
        let world_from_render = match options.rendering_coord_system {
            RenderingCoordinateSystem::Camera => *world_from_camera,
            RenderingCoordinateSystem::CameraWorld => {
                let p_camera = world_from_camera.apply(&Point3f::ZERO);
                Transform::translate(p_camera.into())
            }
            RenderingCoordinateSystem::World => Transform::default(),
        };
        let render_from_world = world_from_render.inverse();
        let render_from_camera = render_from_world * world_from_camera;
        CameraTransform {
            render_from_camera,
            world_from_render,
        }
    }

    pub fn camera_from_render_has_scale(&self) -> bool {
        self.render_from_camera.has_scale()
    }

    pub fn render_from_camera_p(&self, p: &Point3f) -> Point3f {
        self.render_from_camera.apply(p)
    }
    pub fn render_from_camera_v(&self, v: &Vector3f) -> Vector3f {
        self.render_from_camera.apply(v)
    }
    pub fn render_from_camera_n(&self, n: &Normal3f) -> Normal3f {
        self.render_from_camera.apply(n)
    }
    pub fn render_from_camera_r(&self, r: &Ray) -> Ray {
        self.render_from_camera.apply(r)
    }
    pub fn render_from_camera_rd(&self, r: &RayDifferential) -> RayDifferential {
        self.render_from_camera.apply(r)
    }

    pub fn camera_from_render_p(&self, p: &Point3f, _time: Float) -> Point3f {
        self.render_from_camera.apply_inv(p)
    }
    pub fn camera_from_render_v(&self, v: &Vector3f, _time: Float) -> Vector3f {
        self.render_from_camera.apply_inv(v)
    }
    pub fn camera_from_render_n(&self, n: &Normal3f, _time: Float) -> Normal3f {
        self.render_from_camera.apply_inv(n)
    }
    pub fn camera_from_render_r(&self, r: &Ray, _time: Float) -> Ray {
        self.render_from_camera.apply_inv(r)
    }
    pub fn camera_from_render_rd(&self, r: &RayDifferential, _time: Float) -> RayDifferential {
        self.render_from_camera.apply_inv(r)
    }

    pub fn render_from_world_p(&self, p: &Point3f) -> Point3f {
        self.world_from_render.apply_inv(p)
    }

    pub fn render_from_world(&self) -> Transform {
        Transform::inverse(&self.world_from_render)
    }
    pub fn camera_from_render(&self, _time: Float) -> Transform {
        Transform::inverse(&self.render_from_camera)
    }
    pub fn camera_from_world(&self, _time: Float) -> Transform {
        Transform::inverse(&(self.world_from_render * self.render_from_camera))
    }
    pub fn render_from_camera(&self) -> &Transform {
        &self.render_from_camera
    }
    pub fn world_from_render(&self) -> &Transform {
        &self.world_from_render
    }
}

/// Serves to provide shared functionality across orthographic and perspective cameras
struct ProjectiveCameraBase {
    camera_base: CameraBase,
    screen_from_camera: Transform,
    camera_from_raster: Transform,
    raster_from_screen: Transform,
    screen_from_raster: Transform,
    lens_radius: Float,
    focal_distance: Float,
}

impl ProjectiveCameraBase {
    pub fn new(
        camera_transform: CameraTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Film,
        medium: Option<Medium>,
        lens_radius: Float,
        focal_distance: Float,
        screen_from_camera: Transform,
        screen_window: Bounds2f,
    ) -> ProjectiveCameraBase {
        let camera_base = CameraBase {
            camera_transform,
            shutter_open,
            shutter_close,
            film,
            medium,
            // These differentials can be set by the calling code. TODO - Can we improve this?
            min_pos_differential_x: Default::default(),
            min_pos_differential_y: Default::default(),
            min_dir_differential_x: Default::default(),
            min_dir_differential_y: Default::default(),
        };
        let ndc_from_screen = Transform::scale(
            1.0 / (screen_window.max.x - screen_window.min.x),
            1.0 / (screen_window.max.y - screen_window.min.y),
            1.0,
        ) * Transform::translate(Vector3f::new(
            -screen_window.min.x,
            -screen_window.max.y,
            1.0,
        ));
        let raster_from_ndc = Transform::scale(
            camera_base.film.full_resolution().x as Float,
            -camera_base.film.full_resolution().y as Float,
            1.0,
        );
        let raster_from_screen = raster_from_ndc * ndc_from_screen;
        let screen_from_raster = raster_from_screen.inverse();
        let camera_from_raster = screen_from_camera.inverse() * screen_from_raster;
        ProjectiveCameraBase {
            camera_base,
            screen_from_camera,
            camera_from_raster,
            raster_from_screen,
            screen_from_raster,
            lens_radius,
            focal_distance,
        }
    }

    pub fn init_metadata(&self, metadata: &mut ImageMetadata) {
        self.camera_base.init_metadata(metadata);
        if let Some(camera_from_world) = metadata.camera_from_world {
            metadata.ndc_from_world = Some(
                Transform::translate(Vector3f::new(0.5, 0.5, 0.5)).get_matrix()
                    * Transform::scale(0.5, 0.5, 0.5).get_matrix()
                    * self.screen_from_camera.get_matrix()
                    * camera_from_world,
            );
        }
    }
}

pub struct OrthographicCamera {
    projective_base: ProjectiveCameraBase,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
}

impl OrthographicCamera {
    pub fn new(
        camera_transform: CameraTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Film,
        medium: Option<Medium>,
        lens_radius: Float,
        focal_distance: Float,
        screen_window: Bounds2f,
    ) -> OrthographicCamera {
        let screen_from_camera = Transform::orthographic(0.0, 1.0);

        let mut projective_base = ProjectiveCameraBase::new(
            camera_transform,
            shutter_open,
            shutter_close,
            film,
            medium,
            lens_radius,
            focal_distance,
            screen_from_camera,
            screen_window,
        );
        let dx_camera = projective_base.camera_from_raster.apply(&Vector3f::X);
        let dy_camera = projective_base.camera_from_raster.apply(&Vector3f::Y);

        // TODO I don't love having to make projective_base mutable and initializing these values
        // here; can we re-work the initialization to keep things const?
        projective_base.camera_base.min_dir_differential_x = Vector3f::ZERO;
        projective_base.camera_base.min_dir_differential_y = Vector3f::ZERO;
        projective_base.camera_base.min_pos_differential_x = dx_camera;
        projective_base.camera_base.min_pos_differential_y = dy_camera;

        OrthographicCamera {
            projective_base,
            dx_camera,
            dy_camera,
        }
    }
}

impl CameraI for OrthographicCamera {
    fn generate_ray(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective_base.camera_from_raster.apply(&p_film);

        let ray = Ray::new_with_time(
            p_camera,
            Vector3f::Z,
            self.projective_base.camera_base.sample_time(sample.time),
            self.projective_base.camera_base.medium,
        );

        // TODO Adjust for depth-of-field here

        Some(CameraRay::new(
            self.projective_base.camera_base.render_from_camera_r(&ray),
        ))
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective_base.camera_from_raster.apply(&p_film);

        let ray = Ray::new_with_time(
            p_camera,
            Vector3f::Z,
            self.projective_base.camera_base.sample_time(sample.time),
            self.projective_base.camera_base.medium,
        );

        // TODO Adjust for depth-of-field here (and in aux ray calculation)

        let aux_rays =
            AuxiliaryRays::new(ray.o + self.dx_camera, ray.d, ray.o + self.dy_camera, ray.d);

        let rd = RayDifferential::new(ray, Some(aux_rays));

        Some(CameraRayDifferential::new(rd))
    }

    fn get_film(&mut self) -> &mut Film {
        &mut self.projective_base.camera_base.film
    }

    fn sample_time(&self, u: Float) -> Float {
        self.projective_base.camera_base.sample_time(u)
    }

    fn init_metadata(&self, metadata: &mut ImageMetadata) {
        self.projective_base.camera_base.init_metadata(metadata);
        self.projective_base.init_metadata(metadata);
    }

    fn get_camera_transform(&self) -> &CameraTransform {
        &self.projective_base.camera_base.camera_transform
    }

    fn approximate_dp_dxy(
        &self,
        p: Point3f,
        n: Normal3f,
        time: Float,
        samples_per_pixel: i32,
        options: &Options,
    ) -> (Vector3f, Vector3f) {
        self.projective_base
            .camera_base
            .approximate_dp_dxy(p, n, time, samples_per_pixel, options)
    }
}
