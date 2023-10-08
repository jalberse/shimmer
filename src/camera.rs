use crate::{
    film::Film,
    image_metadata::ImageMetadata,
    options::{Options, RenderingCoordinateSystem},
    ray::{Ray, RayDifferential},
    spectra::{sampled_spectrum::SampledSpectrum, sampled_wavelengths::SampledWavelengths},
    transform::Transform,
    vecmath::{normal::Normal3, Normal3f, Point2f, Point3f, Vector3f},
    Float,
};

pub trait CameraI {
    /// Computes the ray corresponding to a given image sample
    fn generate_ray(&self, sample: CameraSample, lambda: &SampledWavelengths) -> Option<CameraRay>;

    /// Like generate_ray(), but also computes the corresponding rays
    /// for pixels shifted one picel in the x and y directions on the film plane.
    /// This is useful for anti-aliasing.
    fn generate_ray_differential(
        &self,
        sample: CameraSample,
        lamda: &SampledWavelengths,
    ) -> Option<CameraRayDifferential>;

    fn film(&self) -> &Film;

    /// Maps a uniform random sample u [0, 1) to a time when the camera shutter is open.
    fn sample_time(&self, u: Float) -> Float;

    fn init_metadata(&self, metadata: &mut ImageMetadata);

    fn get_camera_transform(&self) -> &CameraTransform;
}

pub struct CameraRay {
    ray: Ray,
    weight: SampledSpectrum,
}

pub struct CameraRayDifferential {
    ray: RayDifferential,
    weight: SampledSpectrum,
}

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

    pub fn render_from_camera_n(&self, n: &Normal3f, _time: Float) -> Normal3f {
        self.render_from_camera.apply_n(n)
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
