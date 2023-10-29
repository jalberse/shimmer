use crate::{
    float::{next_float_down, Float, PI_F},
    math::{lerp, safe_sqrt, INV_2PI},
    vecmath::{Point2f, Vector3f},
};

// See PBRT v4 2.14
pub fn balance_heuristic(nf: u8, f_pdf: Float, ng: u8, g_pdf: Float) -> Float {
    (nf as Float * f_pdf) / (nf as Float * f_pdf + ng as Float * g_pdf)
}

// See PBRT v4 2.15
pub fn power_heuristic(nf: u8, f_pdf: Float, ng: u8, g_pdf: Float) -> Float {
    let f = nf as Float * f_pdf;
    let g = ng as Float * g_pdf;
    (f * f) / (f * f + g * g)
}

// Takes a not-necessarily normalized set of nonnegative weights, a uniform random sample u,
// and returns the index of one of the weights with probability propotional to its weight.
// If weights is empty, None is returned.
// If pmf is provided, it will be populated with the value of the pmf for the sample.
// If u_remapped is provided, it will be populated with a new uniform random sample derived from u.
pub fn sample_discrete(
    weights: &[Float],
    u: Float,
    pmf: Option<&mut Float>,
    u_remapped: Option<&mut Float>,
) -> Option<usize> {
    if weights.is_empty() {
        if let Some(pmf) = pmf {
            *pmf = 0.0;
        }
        return None;
    }

    let sum_weights: Float = weights.iter().sum();

    // Compute rescaled u' sample.
    let up = u * sum_weights;
    let up = if up == sum_weights {
        next_float_down(up)
    } else {
        up
    };

    // Find offset in weights corresponding to u'
    let mut offset = 0;
    let mut sum: Float = 0.0;
    while sum + weights[offset] <= up {
        sum += weights[offset];
        offset += 1;
        debug_assert!(offset < weights.len());
    }

    // Compute PMF and remapped u value if requested.
    if let Some(pmf) = pmf {
        *pmf = weights[offset] / sum_weights;
    }
    if let Some(u_remapped) = u_remapped {
        // The difference betweent the sum (the start of the bracket for the offset)
        // and u is itself a new uniform random value that can be remapped to between 0 and 1 here.
        *u_remapped = Float::min((up - sum) / weights[offset], 1.0 - Float::EPSILON);
    }

    Some(offset)
}

pub fn linear_pdf(x: Float, a: Float, b: Float) -> Float {
    debug_assert!(a >= 0.0 && b >= 0.0);
    if x < 0.0 || x > 1.0 {
        0.0
    } else {
        2.0 * lerp(x, &a, &b) / (a + b)
    }
}

pub fn sample_linear(u: Float, a: Float, b: Float) -> Float {
    debug_assert!(a >= 0.0 && b >= 0.0);
    if u == 0.0 && a == 0.0 {
        return 0.0;
    }
    let x = u * (a + b) / (a + Float::sqrt(lerp(u, &(a * a), &(b * b))));
    Float::min(x, 1.0 - Float::EPSILON)
}

pub fn invert_linear_sample(x: Float, a: Float, b: Float) -> Float {
    x * (a * (2.0 - x) + b * x) / (a + b)
}

pub fn sample_visible_wavelengths(u: Float) -> Float {
    538.0 - 138.888889 * Float::atanh(0.85691062 - 1.82750197 * u)
}

pub fn visible_wavelengths_pdf(lambda: Float) -> Float {
    if lambda < 360.0 || lambda > 830.0 {
        return 0.0;
    }
    let x = Float::cosh(0.0072 * (lambda - 538.0));
    0.0039398042 / (x * x)
}

pub fn sample_uniform_sphere(u: Point2f) -> Vector3f {
    let z = 1.0 - 2.0 * u[0];
    let r = safe_sqrt(1.0 - z * z);
    let phi = 2.0 * PI_F * u[1];
    Vector3f {
        x: r * Float::cos(phi),
        y: r * Float::sin(phi),
        z,
    }
}

pub fn sample_uniform_hemisphere(u: Point2f) -> Vector3f {
    let z = u[0];
    let r = safe_sqrt(1.0 - z * z);
    let phi = 2.0 * PI_F * u[1];
    Vector3f {
        x: r * Float::cos(phi),
        y: r * Float::sin(phi),
        z: z,
    }
}

pub fn uniform_hemisphere_pdf() -> Float {
    INV_2PI
}

// TODO get_camera_sample() pg 516; need to implement the Sampler interface/enum first.

#[cfg(test)]
mod tests {
    use crate::sampling::visible_wavelengths_pdf;

    use super::sample_discrete;
    use super::Float;

    #[test]
    fn visible_wavelength_pdf_outside_range() {
        assert_eq!(0.0, visible_wavelengths_pdf(359.9));
        assert_eq!(0.0, visible_wavelengths_pdf(830.1));
    }

    #[test]
    fn sample_discrete_basics() {
        let mut pdf: Float = 0.0;

        assert_eq!(
            Some(0),
            sample_discrete(&[5.0], 0.251, Some(&mut pdf), None)
        );
        assert_eq!(1.0, pdf);

        assert_eq!(
            Some(0),
            sample_discrete(&[0.5, 0.5], 0.0, Some(&mut pdf), None)
        );
        assert_eq!(0.5, pdf);

        assert_eq!(
            Some(0),
            sample_discrete(&[0.5, 0.5], 0.499, Some(&mut pdf), None)
        );
        assert_eq!(0.5, pdf);

        let mut u_remapped: Float = 0.0;
        assert_eq!(
            Some(1),
            sample_discrete(&[0.5, 0.5], 0.5, Some(&mut pdf), Some(&mut u_remapped))
        );
        assert_eq!(0.5, pdf);
        assert_eq!(0.0, u_remapped);
    }
}
