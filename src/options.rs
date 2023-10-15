pub enum RenderingCoordinateSystem {
    Camera,
    CameraWorld,
    World,
}

pub struct Options {
    pub rendering_coord_system: RenderingCoordinateSystem,
}
