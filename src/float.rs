#[cfg(use_f64)]
pub type Float = f64;

#[cfg(not(use_f64))]
pub type Float = f32;

#[cfg(use_f64)]
pub type FloatAsBits = u64;
#[cfg(not(use_f64))]
pub type FloatAsBits = u32;

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *f32* to *u32* (or *u64* to *f64* if use_f64 enabled)
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
#[cfg(not(use_f64))]
pub fn float_to_bits(f: Float) -> FloatAsBits {
    // uint64_t ui;
    // memcpy(&ui, &f, sizeof(double));
    // return ui;
    let rui: FloatAsBits;
    unsafe {
        let ui: FloatAsBits = std::mem::transmute_copy(&f);
        rui = ui;
    }
    rui
}

/// Use **unsafe**
/// [std::mem::transmute_copy][transmute_copy]
/// to convert *u32* to *f32* (or *u64* to *f64* if use_f64 enabled)
///
/// [transmute_copy]: https://doc.rust-lang.org/std/mem/fn.transmute_copy.html
#[cfg(not(use_f64))]
pub fn bits_to_float(ui: FloatAsBits) -> Float {
    let rf: Float;
    unsafe {
        let f: Float = std::mem::transmute_copy(&ui);
        rf = f;
    }
    rf
}

/// Bump a floating-point value up to the next greater representable
/// floating-point value.
pub fn next_float_up(v: Float) -> Float {
    if v.is_infinite() && v > 0.0 {
        v
    } else {
        let new_v = if v == -0.0 { 0.0 } else { v };
        let mut ui: FloatAsBits = float_to_bits(new_v);
        if new_v >= 0.0 {
            ui += 1;
        } else {
            ui -= 1;
        }
        bits_to_float(ui)
    }
}

/// Bump a floating-point value down to the next smaller representable
/// floating-point value.
pub fn next_float_down(v: Float) -> Float {
    if v.is_infinite() && v < 0.0 {
        v
    } else {
        let new_v = if v == 0.0 { -0.0 } else { v };
        let mut ui: FloatAsBits = float_to_bits(new_v);
        if new_v > 0.0 {
            ui -= 1;
        } else {
            ui += 1;
        }
        bits_to_float(ui)
    }
}

// Note - to properly test,
// cargo test -F use_f64
// must be used in addition to cargo test with default features.
// This will test the f64 implementations.
// While this makes the default cargo test less comprehensive, it means
// that the project does not need to recompile with the f64 feature enabled
// while testing. That's a trade-off that we find acceptable.
// Comprehensive testing processes should build and test for all features.
//
// TODO We could implement separate versions of tested functions for f64 and f32:
//     next_float_up_f64(f: f64) -> f64
//     next_float_up_f32(f: f32) -> f32
//     ... etc
//   and have our base functions (e.g. next_float_up()) call the appropriate one
//   depending on cfg(use_f64). That way, the test module could test all those functions
//   regardless of if the feature is enabled or not.
//   The compiler would likely ellide the indirection.
//   Maybe that's worth doing, but for now this is sufficient.
#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::float::float_to_bits;

    use super::bits_to_float;
    use super::next_float_down;
    use super::next_float_up;
    use super::Float;
    use super::FloatAsBits;

    use float_next_after::NextAfter;

    #[test]
    fn next_up_down_float() {
        assert!(next_float_up(-0.0) > 0.0);
        assert!(next_float_down(0.0) < 0.0);

        assert_eq!(next_float_up(Float::INFINITY), Float::INFINITY);
        assert!(next_float_down(Float::INFINITY) < Float::INFINITY);

        assert_eq!(next_float_down(Float::NEG_INFINITY), Float::NEG_INFINITY);
        assert!(next_float_up(Float::NEG_INFINITY) > Float::NEG_INFINITY);

        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let f: Float = rng.gen();
            if f.is_infinite() {
                continue;
            }

            // Note that we use float_next_after to check against.
            // Why not just use it, instead of implementing our own?
            // Partially because we're trying to stick to PBRT's implementation,
            // and partially because we're developing with an eye to making code
            // portable to the GPU, where using this crate might preclude that.
            assert_eq!(f.next_after(Float::INFINITY), next_float_up(f));
            assert_eq!(f.next_after(Float::NEG_INFINITY), next_float_down(f));
        }
    }

    #[test]
    fn float_bits() {
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let ui: FloatAsBits = rng.gen();
            let f: Float = bits_to_float(ui);
            if f.is_nan() {
                continue;
            }
            assert_eq!(ui, float_to_bits(f));
        }
    }
}
