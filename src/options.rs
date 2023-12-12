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
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rendering_coord_system: RenderingCoordinateSystem::CameraWorld,
            disable_texture_filtering: false,
            disable_pixel_jitter: false,
            disable_wavelength_jitter: false,
            force_diffuse: false,
            image_file: "".to_string(),
        }
    }
}
