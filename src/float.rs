#[cfg(use_f64)]
pub type Float = f64;

#[cfg(not(use_f64))]
pub type Float = f32;
