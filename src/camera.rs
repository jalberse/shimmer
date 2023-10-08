use crate::{
    bounding_box::Bounds2f,
    film::Film,
    image_metadata::ImageMetadata,
    math::lerp,
    medium::Medium,
    options::{Options, RenderingCoordinateSystem},
    ray::{AuxiliaryRays, Ray, RayDifferential},
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    transform::Transform,
    vecmath::{Normal3f, Point2f, Point3f, Tuple3, Vector3f},
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
        lamda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential>;

    fn film(&self) -> &Film;

    /// Maps a uniform random sample u [0, 1) to a time when the camera shutter is open.
    fn sample_time(&self, u: Float) -> Float;

    fn init_metadata(&self, metadata: &mut ImageMetadata);

    fn get_camera_transform(&self) -> &CameraTransform;
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
    medium: Medium,
}

impl CameraBase {
    pub fn new(
        camera_transform: CameraTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Film,
        medium: Medium,
    ) -> CameraBase {
        CameraBase {
            camera_transform,
            shutter_open,
            shutter_close,
            film,
            medium,
        }
    }

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

    // TODO need transforms for rays and ray differentials when those are implemented.

    pub fn render_from_camera_v(&self, v: &Vector3f, _time: Float) -> Vector3f {
        self.camera_transform.render_from_camera_v(v, _time)
    }

    pub fn render_from_camera_n(&self, n: &Normal3f, _time: Float) -> Normal3f {
        self.camera_transform.render_from_camera_n(n, _time)
    }

    pub fn render_from_camera_p(&self, p: &Point3f, _time: Float) -> Point3f {
        self.camera_transform.render_from_camera_p(p, _time)
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
        Some(CameraRayDifferential::new(
            ray_differential,
            base_camera_ray.weight,
        ))
    }
}

pub struct CameraRay {
    ray: Ray,
    weight: SampledSpectrum,
}

pub struct CameraRayDifferential {
    ray: RayDifferential,
    weight: SampledSpectrum,
}

impl CameraRayDifferential {
    pub fn new(ray: RayDifferential, weight: SampledSpectrum) -> CameraRayDifferential {
        CameraRayDifferential { ray, weight }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CameraSample {
    p_film: Point2f,
    p_lens: Point2f,
    time: Float,
    filter_weight: Float,
}

pub struct CameraTransform {
    // TODO render_from_camera should be an AnimatedTransform when that's implemented,
    //  along with any associated changes that entails.
    render_from_camera: Transform,
    world_from_render: Transform,
}

impl CameraTransform {
    pub fn new(world_from_camera: &Transform, options: Options) -> CameraTransform {
        // TODO would need to update this for AnimatedTransform
        let world_from_render = match options.rendering_coord_system {
            RenderingCoordinateSystem::Camera => *world_from_camera,
            RenderingCoordinateSystem::CameraWorld => {
                let p_camera = world_from_camera.apply_p(&Point3f::ZERO);
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

    pub fn render_from_camera_p(&self, p: &Point3f, _time: Float) -> Point3f {
        self.render_from_camera.apply_p(p)
    }

    pub fn camera_from_render_p(&self, p: &Point3f, _time: Float) -> Point3f {
        self.render_from_camera.apply_p_inv(p)
    }

    pub fn render_from_world_p(&self, p: &Point3f) -> Point3f {
        self.world_from_render.apply_p_inv(p)
    }

    pub fn camera_from_render_has_scale(&self) -> bool {
        self.render_from_camera.has_scale()
    }

    pub fn render_from_camera_v(&self, v: &Vector3f, _time: Float) -> Vector3f {
        self.render_from_camera.apply_v(v)
    }

    pub fn camera_from_render_v(&self, v: &Vector3f, _time: Float) -> Vector3f {
        self.render_from_camera.apply_v_inv(v)
    }

    pub fn render_from_camera_n(&self, n: &Normal3f, _time: Float) -> Normal3f {
        self.render_from_camera.apply_n(n)
    }

    pub fn camera_from_render_n(&self, n: &Normal3f, _time: Float) -> Normal3f {
        self.render_from_camera.apply_n_inv(n)
    }

    // TODO Similar for ray and ray differentials. Requires implementing transforms on rays.
    // That requires an Interval implementation, and a Point3<Interval> impl.

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
        medium: Medium,
        lens_radius: Float,
        focal_distance: Float,
        screen_from_camera: Transform,
        screen_window: Bounds2f,
    ) -> ProjectiveCameraBase {
        let camera_base = CameraBase::new(
            camera_transform,
            shutter_open,
            shutter_close,
            film.clone(),
            medium,
        );
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
            film.full_resolution().x as Float,
            -film.full_resolution().y as Float,
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
        medium: Medium,
        lens_radius: Float,
        focal_distance: Float,
        screen_window: Bounds2f,
    ) -> OrthographicCamera {
        let screen_from_camera = Transform::orthographic(0.0, 1.0);
        let projective_base = ProjectiveCameraBase::new(
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
        let dx_camera = projective_base.camera_from_raster.apply_v(&Vector3f::X);
        let dy_camera = projective_base.camera_from_raster.apply_v(&Vector3f::Y);

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
        todo!()
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lamda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        todo!()
    }

    fn film(&self) -> &Film {
        &self.projective_base.camera_base.film
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
}
