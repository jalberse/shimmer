#[cfg(use_f64)]
pub type Vec2f = glam::DVec2;
#[cfg(not(use_f64))]
pub type Vec2f = glam::Vec2;
#[cfg(use_f64)]
pub type Vec3f = glam::DVec3;
#[cfg(not(use_f64))]
pub type Vec3f = glam::Vec3;
