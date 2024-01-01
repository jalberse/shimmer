use crate::bounding_box::{Bounds2f, Bounds2i};

pub enum RenderingCoordinateSystem {
    Camera,
    CameraWorld,
    World,
}

pub struct Options {
    pub rendering_coord_system: RenderingCoordinateSystem,
    pub disable_texture_filtering: bool,
    pub disable_pixel_jitter: bool,
    pub disable_wavelength_jitter: bool,
    pub force_diffuse: bool,
    pub image_file: String,
    pub quick_render: bool,
    pub pixel_bounds: Option<Bounds2i>,
    pub crop_window: Option<Bounds2f>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rendering_coord_system: RenderingCoordinateSystem::CameraWorld,
            disable_texture_filtering: true, // TODO change back to false, see #41.
            disable_pixel_jitter: false,
            disable_wavelength_jitter: false,
            force_diffuse: false,
            image_file: "".to_string(),
            quick_render: false,
            pixel_bounds: None,
            crop_window: None,
        }
    }
}
