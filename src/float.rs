#[cfg(use_f64)]
type Float = f64;

#[cfg(not(use_f64))]
type Float = f32;
