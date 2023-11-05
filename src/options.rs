pub enum RenderingCoordinateSystem {
    Camera,
    CameraWorld,
    World,
}

pub struct Options {
    pub rendering_coord_system: RenderingCoordinateSystem,
    pub disable_texture_filtering: bool,
    pub disable_pixel_jitter: bool,
    pub force_diffuse: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            rendering_coord_system: RenderingCoordinateSystem::CameraWorld,
            disable_texture_filtering: false,
            disable_pixel_jitter: false,
            force_diffuse: false,
        }
    }
}
