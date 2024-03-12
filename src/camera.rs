// TODO implement depth-of-field
// TODO perspective camera
// TODO circular camera (maybe, low priority)
// TODO place Camera types into a Camera enum that impl CameraI (typical pattern).

use log::warn;

use crate::{
    bounding_box::Bounds2f,
    film::{Film, FilmI},
    filter::FilterI,
    frame::Frame,
    image::ImageMetadata,
    loading::{paramdict::ParameterDictionary, parser_target::FileLoc},
    math::{lerp, radians},
    media::Medium,
    options::{Options, RenderingCoordinateSystem},
    ray::{AuxiliaryRays, Ray, RayDifferential, RayI},
    sampling::sample_uniform_disk_concentric,
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    transform::{InverseTransformI, InverseTransformRayI, Transform, TransformI, TransformRayI},
    vecmath::{
        normal::Normal3, Length, Normal3f, Normalize, Point2f, Point3f, Tuple2, Tuple3, Vector3f
    },
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

    fn get_film_const(&self) -> &Film;

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

#[derive(Debug, Clone)]
pub enum Camera {
    Orthographic(OrthographicCamera),
    Perspective(PerspectiveCamera),
}

impl Camera {
    pub fn create(
        name: &str,
        parameters: &mut ParameterDictionary,
        medium: Option<Medium>,
        camera_transform: CameraTransform,
        film: Film,
        options: &Options,
        loc: &FileLoc,
    ) -> Camera {
        match name {
            "perspective" => Camera::Perspective(PerspectiveCamera::create(
                parameters,
                camera_transform,
                film,
                medium,
                options,
                loc,
            )),
            "orthographic" => Camera::Orthographic(OrthographicCamera::create(
                parameters,
                camera_transform,
                film,
                medium,
                options,
                loc,
            )),
            _ => panic!("Camera type \"{}\" unknown.", name),
        }
    }
}

impl CameraI for Camera {
    fn generate_ray(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        match self {
            Camera::Orthographic(c) => c.generate_ray(sample, lambda),
            Camera::Perspective(c) => c.generate_ray(sample, lambda),
        }
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        match self {
            Camera::Orthographic(c) => c.generate_ray_differential(sample, lambda),
            Camera::Perspective(c) => c.generate_ray_differential(sample, lambda),
        }
    }

    fn get_film(&mut self) -> &mut Film {
        match self {
            Camera::Orthographic(c) => c.get_film(),
            Camera::Perspective(c) => c.get_film(),
        }
    }

    fn get_film_const(&self) -> &Film {
        match self {
            Camera::Orthographic(c) => c.get_film_const(),
            Camera::Perspective(c) => c.get_film_const(),
        }
    }

    fn sample_time(&self, u: Float) -> Float {
        match self {
            Camera::Orthographic(c) => c.sample_time(u),
            Camera::Perspective(c) => c.sample_time(u),
        }
    }

    fn init_metadata(&self, metadata: &mut ImageMetadata) {
        match self {
            Camera::Orthographic(c) => c.init_metadata(metadata),
            Camera::Perspective(c) => c.init_metadata(metadata),
        }
    }

    fn get_camera_transform(&self) -> &CameraTransform {
        match self {
            Camera::Orthographic(c) => c.get_camera_transform(),
            Camera::Perspective(c) => c.get_camera_transform(),
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
            Camera::Perspective(c) => c.approximate_dp_dxy(p, n, time, samples_per_pixel, options),
        }
    }
}

/// Shared implementation details for different kinds of cameras.
#[derive(Debug, Clone)]
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
        lerp(u, self.shutter_open, self.shutter_close)
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
        let p_down_z = down_z_from_camera.apply(p_camera);
        // TODO Finding cases where n_down_z.z is 0, which propagates down to dpdx and dpdy being NaN.
        let n_down_z = down_z_from_camera.apply(self.camera_from_render_n(n, time));
        let d = n_down_z.z * p_down_z.z;

        // Find intersection points for approximated camera differential rays
        let x_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_x,
            Vector3f::Z + self.min_dir_differential_x,
            None,
        );
        let tx = -(n_down_z.dot_vector(x_ray.o.into()) - d) / n_down_z.dot_vector(x_ray.d);
        let y_ray = Ray::new(
            Point3f::ZERO + self.min_pos_differential_y,
            Vector3f::Z + self.min_dir_differential_y,
            None,
        );

        let ty = -(n_down_z.dot_vector(y_ray.o.into()) - d) / n_down_z.dot_vector(y_ray.d);
        let px = x_ray.get(tx);
        let py = y_ray.get(ty);

        let spp_scale = if options.disable_pixel_jitter {
            1.0
        } else {
            Float::max(0.125, 1.0 / (samples_per_pixel as Float).sqrt())
        };

        let dpdx =
            spp_scale * self.render_from_camera_v(&down_z_from_camera.apply_inverse(px - p_down_z));
        let dpdy =
            spp_scale * self.render_from_camera_v(&down_z_from_camera.apply_inverse(py - p_down_z));
        (dpdx, dpdy)
    }

    pub fn find_minimum_differentials<T>(&mut self, camera: &T)
    where
        T: CameraI,
    {
        self.min_pos_differential_x =
            Vector3f::new(Float::INFINITY, Float::INFINITY, Float::INFINITY);
        self.min_pos_differential_y =
            Vector3f::new(Float::INFINITY, Float::INFINITY, Float::INFINITY);
        self.min_dir_differential_x =
            Vector3f::new(Float::INFINITY, Float::INFINITY, Float::INFINITY);
        self.min_dir_differential_y =
            Vector3f::new(Float::INFINITY, Float::INFINITY, Float::INFINITY);

        let mut sample = CameraSample {
            p_film: Default::default(),
            p_lens: Point2f::new(0.5, 0.5),
            time: 0.5,
            filter_weight: 1.0,
        };
        let lambda = SampledWavelengths::sample_visible(0.5);

        let n = 512;
        for i in 0..n {
            sample.p_film.x =
                i as Float / (n - 1) as Float * self.film.full_resolution().x as Float;
            sample.p_film.y =
                i as Float / (n - 1) as Float * self.film.full_resolution().y as Float;

            let crd = camera.generate_ray_differential(&sample, &lambda);

            if crd.is_none() {
                continue;
            }
            let mut crd = crd.unwrap();

            let ray = &mut crd.ray;

            let dox = self.camera_from_render_v(
                &(ray.auxiliary.as_ref().unwrap().rx_origin - ray.ray.o),
                ray.ray.time,
            );
            if dox.length() < self.min_pos_differential_x.length() {
                self.min_pos_differential_x = dox;
            }
            let doy = self.camera_from_render_v(
                &(ray.auxiliary.as_ref().unwrap().ry_origin - ray.ray.o),
                ray.ray.time,
            );
            if doy.length() < self.min_pos_differential_y.length() {
                self.min_pos_differential_y = doy;
            }

            ray.ray.d = ray.ray.d.normalize();
            ray.auxiliary.as_mut().unwrap().rx_direction =
                ray.auxiliary.as_ref().unwrap().rx_direction.normalize();
            ray.auxiliary.as_mut().unwrap().ry_direction =
                ray.auxiliary.as_ref().unwrap().ry_direction.normalize();

            let f = Frame::from_z(ray.ray.d);
            let df = f.to_local_v(&ray.ray.d); // Should be (0, 0, 1)
            let dxf = f
                .to_local_v(&ray.auxiliary.as_ref().unwrap().rx_direction)
                .normalize();
            let dyf = f
                .to_local_v(&ray.auxiliary.as_ref().unwrap().ry_direction)
                .normalize();

            if (dxf - df).length() < self.min_dir_differential_x.length() {
                self.min_dir_differential_x = dxf - df;
            }
            if (dyf - df).length() < self.min_dir_differential_y.length() {
                self.min_dir_differential_y = dyf - df;
            }
        }
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
    /// The point on the film to which the generated ray should carry radiance
    pub p_film: Point2f,
    /// The point on the lens through which the ray passes, for cameras which include the notion of lenses.
    pub p_lens: Point2f,
    /// The time at which the ray should sample the scene
    pub time: Float,
    /// Additional scale factor for when the ray for this camera sample is stored by the film; it accounts
    /// for the reconstruction filter used to filter image samples at each pixel.
    pub filter_weight: Float,
}

impl Default for CameraSample {
    fn default() -> Self {
        Self {
            p_film: Default::default(),
            p_lens: Default::default(),
            time: 0.0,
            filter_weight: 1.0,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CameraTransform {
    // TODO render_from_camera should be an AnimatedTransform when that's implemented,
    //  along with any associated changes that entails (notably, handling the time)
    render_from_camera: Transform,
    world_from_render: Transform,
}

impl Default for CameraTransform {
    fn default() -> Self {
        Self {
            render_from_camera: Transform::default(),
            world_from_render: Transform::default(),
        }
    }
}

impl CameraTransform {
    pub fn new(world_from_camera: &Transform, options: &Options) -> CameraTransform {
        // TODO would need to update this for AnimatedTransform
        let world_from_render = match options.rendering_coord_system {
            RenderingCoordinateSystem::Camera => *world_from_camera,
            RenderingCoordinateSystem::CameraWorld => {
                let p_camera = world_from_camera.apply(Point3f::ZERO);
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
        self.render_from_camera.apply(*p)
    }
    pub fn render_from_camera_v(&self, v: &Vector3f) -> Vector3f {
        self.render_from_camera.apply(*v)
    }
    pub fn render_from_camera_n(&self, n: &Normal3f) -> Normal3f {
        self.render_from_camera.apply(*n)
    }
    pub fn render_from_camera_r(&self, r: &Ray) -> Ray {
        self.render_from_camera.apply_ray(*r, None)
    }
    pub fn render_from_camera_rd(&self, r: &RayDifferential) -> RayDifferential {
        self.render_from_camera.apply_ray(*r, None)
    }

    pub fn camera_from_render_p(&self, p: &Point3f, _time: Float) -> Point3f {
        self.render_from_camera.apply_inverse(*p)
    }
    pub fn camera_from_render_v(&self, v: &Vector3f, _time: Float) -> Vector3f {
        self.render_from_camera.apply_inverse(*v)
    }
    pub fn camera_from_render_n(&self, n: &Normal3f, _time: Float) -> Normal3f {
        self.render_from_camera.apply_inverse(*n)
    }
    pub fn camera_from_render_r(&self, r: &Ray, _time: Float) -> Ray {
        self.render_from_camera.apply_ray_inverse(*r, None)
    }
    pub fn camera_from_render_rd(&self, r: &RayDifferential, _time: Float) -> RayDifferential {
        self.render_from_camera.apply_ray_inverse(*r, None)
    }

    pub fn render_from_world_p(&self, p: &Point3f) -> Point3f {
        self.world_from_render.apply_inverse(*p)
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
#[derive(Debug, Clone)]
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
        camera_base_parameters: CameraBaseParameters,
        lens_radius: Float,
        focal_distance: Float,
        screen_from_camera: Transform,
        screen_window: Bounds2f,
    ) -> ProjectiveCameraBase {
        let camera_base = CameraBase {
            camera_transform: camera_base_parameters.camera_transform,
            shutter_open: camera_base_parameters.shutter_open,
            shutter_close: camera_base_parameters.shutter_close,
            film: camera_base_parameters.film,
            medium: camera_base_parameters.medium,
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
            0.0,
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

#[derive(Debug, Clone)]
pub struct OrthographicCamera {
    projective_base: ProjectiveCameraBase,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
}

impl OrthographicCamera {
    pub fn create(
        parameters: &mut ParameterDictionary,
        camera_transform: CameraTransform,
        film: Film,
        medium: Option<Medium>,
        options: &Options,
        loc: &FileLoc,
    ) -> OrthographicCamera {
        let camera_base_paramters =
            CameraBaseParameters::new(camera_transform, film, medium, parameters, loc);

        let lens_radius = parameters.get_one_float("lensradius", 0.0);
        let focal_distance = parameters.get_one_float("focaldistance", 1e6);
        let frame = parameters.get_one_float(
            "frameaspectratio",
            camera_base_paramters.film.full_resolution().x as Float
                / camera_base_paramters.film.full_resolution().y as Float,
        );

        let mut screen = if frame > 1.0 {
            Bounds2f::new(Point2f::new(-frame, -1.0), Point2f::new(frame, 1.0))
        } else {
            Bounds2f::new(
                Point2f::new(-1.0, -1.0 / frame),
                Point2f::new(1.0, 1.0 / frame),
            )
        };
        let sw = parameters.get_float_array("screenwindow");
        if !sw.is_empty() {
            if options.fullscreen {
                warn!("screenwindow is ignored in fullscreen mode");
            } else {
                if sw.len() == 4 {
                    screen = Bounds2f::new(Point2f::new(sw[0], sw[2]), Point2f::new(sw[1], sw[3]));
                } else {
                    warn!(
                        "{} Expected four values for \"screenwindow\" parameter. Got {}.",
                        loc,
                        sw.len()
                    );
                }
            }
        }

        OrthographicCamera::new(camera_base_paramters, lens_radius, focal_distance, screen)
    }

    pub fn new(
        camera_base_parameters: CameraBaseParameters,
        lens_radius: Float,
        focal_distance: Float,
        screen_window: Bounds2f,
    ) -> OrthographicCamera {
        let screen_from_camera = Transform::orthographic(0.0, 1.0);

        let mut projective_base = ProjectiveCameraBase::new(
            camera_base_parameters,
            lens_radius,
            focal_distance,
            screen_from_camera,
            screen_window,
        );
        let dx_camera = projective_base.camera_from_raster.apply(Vector3f::X);
        let dy_camera = projective_base.camera_from_raster.apply(Vector3f::Y);

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
        let p_camera = self.projective_base.camera_from_raster.apply(p_film);

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
        let p_camera = self.projective_base.camera_from_raster.apply(p_film);

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

    fn get_film_const(&self) -> &Film {
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

#[derive(Debug, Clone)]
pub struct PerspectiveCamera {
    projective_base: ProjectiveCameraBase,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
    _cos_total_width: Float, // TODO Will be used in the future
    _area: Float,            // TODO Will be used in the future
}

impl PerspectiveCamera {
    pub fn create(
        parameters: &mut ParameterDictionary,
        camera_transform: CameraTransform,
        film: Film,
        medium: Option<Medium>,
        options: &Options,
        loc: &FileLoc,
    ) -> PerspectiveCamera {
        let camera_base_paramters =
            CameraBaseParameters::new(camera_transform, film, medium, parameters, loc);

        let lens_radius = parameters.get_one_float("lensradius", 0.0);
        let focal_distance = parameters.get_one_float("focaldistance", 1e6);
        let frame = parameters.get_one_float(
            "frameaspectratio",
            camera_base_paramters.film.full_resolution().x as Float
                / camera_base_paramters.film.full_resolution().y as Float,
        );
        let mut screen = if frame > 1.0 {
            Bounds2f::new(Point2f::new(-frame, -1.0), Point2f::new(frame, 1.0))
        } else {
            Bounds2f::new(
                Point2f::new(-1.0, -1.0 / frame),
                Point2f::new(1.0, 1.0 / frame),
            )
        };

        let sw = parameters.get_float_array("screenwindow");
        if !sw.is_empty() {
            if options.fullscreen {
                warn!("screenwindow is ignored in fullscreen mode");
            } else {
                if sw.len() == 4 {
                    screen = Bounds2f::new(Point2f::new(sw[0], sw[2]), Point2f::new(sw[1], sw[3]));
                } else {
                    warn!(
                        "Expeced four values for \"screenwindow\" parameter. Got {}.",
                        sw.len()
                    );
                }
            }
        }

        let fov = parameters.get_one_float("fov", 90.0);
        PerspectiveCamera::new(
            camera_base_paramters,
            fov,
            screen,
            lens_radius,
            focal_distance,
        )
    }

    pub fn new(
        camera_base_parameters: CameraBaseParameters,
        fov: Float,
        screen_window: Bounds2f,
        lens_radius: Float,
        focal_distance: Float,
    ) -> PerspectiveCamera {
        // TODO must calculate screen_from_camera
        let screen_from_camera = Transform::perspective(fov, 1e-2, 1000.0);
        let mut projective_base = ProjectiveCameraBase::new(
            camera_base_parameters,
            lens_radius,
            focal_distance,
            screen_from_camera,
            screen_window,
        );
        // Compute differential changfes in origin for perspective camera rays;
        // their origins are unchanges and the ray differentials differ only in their direction
        // for perspective cameras. Compute the change in position on the near perspective plane
        // in camera space wrt shifts in pixel locations.
        let dx_camera = projective_base.camera_from_raster.apply(Point3f::X)
            - projective_base.camera_from_raster.apply(Point3f::ZERO);
        let dy_camera = projective_base.camera_from_raster.apply(Point3f::Y)
            - projective_base.camera_from_raster.apply(Point3f::ZERO);

        // Compute cos_total_width for perspective camera, the cosine of the maximum angle of the FOV.
        // This is used in a few places, such as for culling points outside the FOV quickly.
        let radius = Point2f::from(projective_base.camera_base.film.get_filter().radius());
        let p_corner = Point3f::new(-radius.x, -radius.y, 0.0);
        let w_corner_camera =
            Vector3f::from(projective_base.camera_from_raster.apply(p_corner)).normalize();
        let cos_total_width = w_corner_camera.z;
        debug_assert!(0.9999 * cos_total_width <= Float::cos(radians(fov / 2.0)));

        // Compute image plane area at z == 1.0 for perspective
        let res = projective_base.camera_base.film.full_resolution();
        let mut p_min = projective_base.camera_from_raster.apply(Point3f::ZERO);
        let mut p_max = projective_base.camera_from_raster.apply(Point3f::new(
            res.x as Float,
            res.y as Float,
            0.0,
        ));
        p_min /= p_min.z;
        p_max /= p_max.z;
        let area = Float::abs((p_max.x - p_min.x) * (p_max.y - p_min.y));

        let camera = PerspectiveCamera {
            projective_base: projective_base.clone(),
            dx_camera,
            dy_camera,
            _cos_total_width: cos_total_width,
            _area: area,
        };

        projective_base
            .camera_base
            .find_minimum_differentials(&camera);

        // I dislike having to make the PerspectiveCamera twice, but we need to create it in order
        // to (conveniently) use it to find its minimum differentials, and then
        // make a new camera with those minimum differntials set. There are other approaches, but this is simple,
        // if a bit inefficient.
        // TODO - This approach could be improved.
        PerspectiveCamera {
            projective_base,
            dx_camera,
            dy_camera,
            _cos_total_width: cos_total_width,
            _area: area,
        }
    }
}

impl CameraI for PerspectiveCamera {
    fn generate_ray(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRay> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective_base.camera_from_raster.apply(p_film);

        let mut ray = Ray::new_with_time(
            Point3f::ZERO,
            Vector3f::from(p_camera).normalize(),
            self.sample_time(sample.time),
            self.projective_base.camera_base.medium,
        );

        // Modify ray for depth of field
        if self.projective_base.lens_radius > 0.0 {
            // Sample point on lens
            let p_lens =
                self.projective_base.lens_radius * sample_uniform_disk_concentric(sample.p_lens);

            // Compute point on plane of focus
            let ft = self.projective_base.focal_distance / ray.d.z;
            let p_focus = ray.get(ft);

            // Update ray for effect of lens
            ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            ray.d = (p_focus - ray.o).normalize();
        }

        Some(CameraRay {
            ray: self.projective_base.camera_base.render_from_camera_r(&ray),
            weight: SampledSpectrum::from_const(1.0),
        })
    }

    fn generate_ray_differential(
        &self,
        sample: &CameraSample,
        _lambda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential> {
        let p_film = Point3f::new(sample.p_film.x, sample.p_film.y, 0.0);
        let p_camera = self.projective_base.camera_from_raster.apply(p_film);

        let dir = Vector3f::from(p_camera).normalize();
        let mut base_ray = Ray::new_with_time(
            Point3f::ZERO,
            dir,
            self.sample_time(sample.time),
            self.projective_base.camera_base.medium,
        );

        // Modify base ray for depth of field
        if self.projective_base.lens_radius > 0.0 {
            // Sample point on lens
            let p_lens =
                self.projective_base.lens_radius * sample_uniform_disk_concentric(sample.p_lens);

            // Compute point on plane of focus
            let ft = self.projective_base.focal_distance / base_ray.d.z;
            let p_focus = base_ray.get(ft);

            // Update ray for effect of lens
            base_ray.o = Point3f::new(p_lens.x, p_lens.y, 0.0);
            base_ray.d = (p_focus - base_ray.o).normalize();
        }

        // Compute auxilliary rays
        let aux_rays = if self.projective_base.lens_radius > 0.0 {
            // Compute _PerspectiveCamera_ ray differentials accounting for lens
            // Sample point on lens
            let p_lens =
                self.projective_base.lens_radius * sample_uniform_disk_concentric(sample.p_lens);

            let dx = Vector3f::from(p_camera + self.dx_camera).normalize();
            let ft = self.projective_base.focal_distance / dx.z;
            let p_focus = Point3f::ZERO + ft * dx;
            let rx_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            let rx_direction = (p_focus - rx_origin).normalize();

            let dy = Vector3f::from(p_camera + self.dy_camera).normalize();
            let ft = self.projective_base.focal_distance / dy.z;
            let p_focus = Point3f::ZERO + ft * dy;
            let ry_origin = Point3f::new(p_lens.x, p_lens.y, 0.0);
            let ry_direction = (p_focus - ry_origin).normalize();

            AuxiliaryRays {
                rx_origin,
                rx_direction,
                ry_origin,
                ry_direction,
            }
        } else {
            let rx_origin = base_ray.o;
            let ry_origin = base_ray.o;
            let rx_direction = (Vector3f::from(p_camera) + self.dx_camera).normalize();
            let ry_direction = (Vector3f::from(p_camera) + self.dy_camera).normalize();
            AuxiliaryRays {
                rx_origin,
                rx_direction,
                ry_origin,
                ry_direction,
            }
        };
        Some(CameraRayDifferential::new(
            self.projective_base
                .camera_base
                .render_from_camera_rd(&RayDifferential {
                    ray: base_ray,
                    auxiliary: Some(aux_rays),
                }),
        ))
    }

    fn get_film(&mut self) -> &mut Film {
        &mut self.projective_base.camera_base.film
    }

    fn get_film_const(&self) -> &Film {
        self.projective_base.camera_base.get_film()
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

pub struct CameraBaseParameters {
    pub camera_transform: CameraTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Film,
    pub medium: Option<Medium>,
}

impl CameraBaseParameters {
    pub fn new(
        camera_transform: CameraTransform,
        film: Film,
        medium: Option<Medium>,
        parameters: &mut ParameterDictionary,
        loc: &FileLoc,
    ) -> CameraBaseParameters {
        let mut shutter_open = parameters.get_one_float("shutteropen", 0.0);
        let mut shutter_close = parameters.get_one_float("shutterclose", 1.0);
        if shutter_close < shutter_open {
            warn!(
                "{} Shutter close time [{}] < shutter open [{}]. Swapping them.",
                loc, shutter_close, shutter_open
            );
            std::mem::swap(&mut shutter_close, &mut shutter_open);
        }
        CameraBaseParameters {
            camera_transform: camera_transform,
            shutter_open,
            shutter_close,
            film,
            medium,
        }
    }
}
