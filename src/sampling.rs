use crate::{
    camera::CameraSample,
    filter::{Filter, FilterI},
    float::{next_float_down, Float, PI_F},
    math::{lerp, safe_sqrt, DifferenceOfProducts, INV_2PI, INV_PI, PI_OVER_2, PI_OVER_4},
    options::Options,
    sampler::SamplerI,
    vecmath::{
        vector::Vector3, Length, Normalize, Point2f, Point2i, Point3f, Tuple2, Vector2f, Vector3f,
    },
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

pub fn sample_cosine_hemisphere(u: Point2f) -> Vector3f {
    let d = sample_uniform_disk_concentric(u);
    let z = safe_sqrt(1.0 - d.x * d.x - d.y * d.y);
    Vector3f {
        x: d.x,
        y: d.y,
        z: z,
    }
}

pub fn cosine_hemisphere_pdf(cos_theta: Float) -> Float {
    cos_theta * INV_PI
}

pub fn sample_uniform_disk_concentric(u: Point2f) -> Point2f {
    // map u to [-1, 1]^2 and handle degeneracy at origin
    let u_offset = 2.0 * u - Vector2f::ONE;
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Point2f::ZERO;
    }
    let (theta, r) = if u_offset.x.abs() > u_offset.y.abs() {
        (u_offset.x, PI_OVER_4 * (u_offset.y / u_offset.x))
    } else {
        (
            u_offset.y,
            PI_OVER_2 - PI_OVER_4 * (u_offset.x / u_offset.y),
        )
    };
    r * Point2f::new(Float::cos(theta), Float::sin(theta))
}

pub fn get_camera_sample<T: SamplerI>(
    sampler: &mut T,
    p_pixel: Point2i,
    filter: &Filter,
    options: &Options,
) -> CameraSample {
    let filter_sample = filter.sample(sampler.get_pixel_2d());
    if options.disable_pixel_jitter {
        CameraSample {
            p_film: Point2f::from(p_pixel) + Vector2f::new(0.5, 0.5),
            p_lens: Point2f::new(0.5, 0.5),
            time: 0.5,
            filter_weight: 1.0,
        }
    } else {
        CameraSample {
            p_film: Point2f::from(p_pixel)
                + Vector2f::from(filter_sample.p)
                + Vector2f::new(0.5, 0.5),
            p_lens: sampler.get_2d(),
            time: sampler.get_1d(),
            filter_weight: filter_sample.weight,
        }
    }
}

pub fn sample_uniform_triangle(u: Point2f) -> (Float, Float, Float) {
    let (b0, b1) = if u[0] < u[1] {
        let b0 = u[0] / 2.0;
        let b1 = u[1] - b0;
        (b0, b1)
    } else {
        let b1 = u[1] / 2.0;
        let b0 = u[0] - b1;
        (b0, b1)
    };
    (b0, b1, 1.0 - b1 - b0)
}

pub fn sample_bilinear(u: Point2f, w: &[Float]) -> Point2f {
    debug_assert_eq!(4, w.len());
    // Sample y for bilinear marginal distribution
    let y = sample_linear(u[1], w[0] + w[1], w[2] + w[3]);
    // Sample x for bilinear conditional distribution
    let x = sample_linear(u[0], lerp(y, &w[0], &w[2]), lerp(y, &w[1], &w[3]));
    Point2f { x, y }
}

pub fn bilinear_pdf(p: Point2f, w: &[Float]) -> Float {
    debug_assert_eq!(4, w.len());
    if p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 {
        return 0.0;
    }
    if w[0] + w[1] + w[2] + w[3] == 0.0 {
        return 1.0;
    }
    4.0 * ((1.0 - p[0]) * (1.0 - p[1]) * w[0]
        + p[0] * (1.0 - p[1]) * w[1]
        + (1.0 - p[0]) * p[1] * w[2]
        + p[0] * p[1] * w[3])
        / (w[0] + w[1] + w[2] + w[3])
}

/// Takes three triangle vertices v, a reference point p, and a uniform sample u.
/// Returns the sampled point vertices and the PDF.
pub fn sample_spherical_triangle(v: &[Point3f; 3], p: Point3f, u: Point2f) -> ([Float; 3], Float) {
    // Compute vectors a, b, and c to spherical triangle vertices
    let a = v[0] - p;
    let b = v[1] - p;
    let c = v[2] - p;
    debug_assert!(a.length_squared() > 0.0);
    debug_assert!(b.length_squared() > 0.0);
    debug_assert!(c.length_squared() > 0.0);

    let a = a.normalize();
    let b = b.normalize();
    let c = c.normalize();

    // Compute normalized cross products of all direction pairs.
    let n_ab = a.cross(&b);
    let n_bc = b.cross(&c);
    let n_ca = c.cross(&a);
    if n_ab.length_squared() == 0.0 || n_bc.length_squared() == 0.0 || n_ca.length_squared() == 0.0
    {
        // TODO Consider using an Option return type instead.
        return ([0.0, 0.0, 0.0], 0.0);
    }

    let n_ab = n_ab.normalize();
    let n_bc = n_bc.normalize();
    let n_ca = n_ca.normalize();

    // Find angles alpha, beta, and gamma at spherical triangle vertices
    let alpha = n_ab.angle_between(&-n_ca);
    let beta = n_bc.angle_between(&-n_ab);
    let gamma = n_ca.angle_between(&-n_bc);

    // Uniformly sample triangle area A to compute A'.
    let a_pi = alpha + beta + gamma;
    let ap_pi = lerp(u[0], &PI_F, &a_pi);
    let area = a_pi - PI_F;
    let pdf = if area <= 0.0 { 0.0 } else { 1.0 / area };

    // Find cos(beta) for point along b for sampled area.
    let cos_alpha = Float::cos(alpha);
    let sin_alpha = Float::sin(alpha);
    let sin_phi = Float::sin(ap_pi) * cos_alpha - Float::cos(ap_pi) * sin_alpha;
    let cos_phi = Float::cos(ap_pi) * cos_alpha + Float::sin(ap_pi) * sin_alpha;
    let k1 = cos_phi + cos_alpha;
    let k2 = sin_phi - sin_alpha * a.dot(&b);
    let cos_bp = (k2 + (Float::difference_of_products(k2, cos_phi, k1, sin_phi)) * cos_alpha)
        / (Float::sum_of_products(k2, sin_phi, k1, cos_phi) * sin_alpha);
    // Happens if the triangle basically covers the entire hemisphere.
    // We currently depend on calling code to detect this case, which is unfortunate.
    debug_assert!(!cos_bp.is_nan());
    let cos_bp = cos_bp.clamp(-1.0, 1.0);

    // Sample c' along the arc between b' and a.
    let sin_bp = safe_sqrt(1.0 - cos_bp * cos_bp);
    let cp = cos_bp * a + sin_bp * c.gram_schmidt(&a).normalize();

    // Compute sampled spherical triangle direction and return barycentrics.
    let cos_theta = 1.0 - u[1] * (1.0 - cp.dot(&b));
    let sin_theta = safe_sqrt(1.0 - cos_theta * cos_theta);
    let w = cos_theta * b + sin_theta * cp.gram_schmidt(&b).normalize();

    // Find barycentric coordinates for sampled direction w.
    let e1 = v[1] - v[0];
    let e2 = v[2] - v[0];
    let s1 = w.cross(&e2);
    let divisor = e1.dot(&e1);

    // TODO Can we have some debug assert_rare()? It could be a good crate.
    if divisor == 0.0 {
        // This happens with triangles that cover (nearly) the whole hemisphere.
        return ([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], pdf);
    }

    let inv_divisor = 1.0 / divisor;
    let s = p - v[0];
    let b1 = s.dot(&s1) * inv_divisor;
    let b2 = w.dot(&s.cross(&e1)) * inv_divisor;

    // Return clamped barycentrics for sampled direction
    let b1 = b1.clamp(0.0, 1.0);
    let b2 = b2.clamp(0.0, 1.0);
    let (b1, b2) = if b1 + b2 > 1.0 {
        (b1 / b1 + b2, b2 / b1 + b2)
    } else {
        (b1, b2)
    };
    ([1.0 - b1 - b2, b1, b2], pdf)
}

pub fn invert_spherical_triangle_sample(v: &[Point3f; 3], p: Point3f, w: Vector3f) -> Point2f {
    // Compute vectors a, b, and c to spherical triangle vertices
    let a = v[0] - p;
    let b = v[1] - p;
    let c = v[2] - p;
    debug_assert!(a.length_squared() > 0.0);
    debug_assert!(b.length_squared() > 0.0);
    debug_assert!(c.length_squared() > 0.0);

    let a = a.normalize();
    let b = b.normalize();
    let c = c.normalize();

    // Compute normalized cross products of all direction pairs.
    let n_ab = a.cross(&b);
    let n_bc = b.cross(&c);
    let n_ca = c.cross(&a);
    if n_ab.length_squared() == 0.0 || n_bc.length_squared() == 0.0 || n_ca.length_squared() == 0.0
    {
        // TODO Consider using an Option return type instead.
        return Point2f::ZERO;
    }

    let n_ab = n_ab.normalize();
    let n_bc = n_bc.normalize();
    let n_ca = n_ca.normalize();

    // Find angles alpha, beta, and gamma at spherical triangle vertices
    let alpha = n_ab.angle_between(&-n_ca);
    let beta = n_bc.angle_between(&-n_ab);
    let gamma = n_ca.angle_between(&-n_bc);

    // Find vertex c' along ac arc for w
    let cp = b.cross(&w).cross(&c.cross(&a)).normalize();
    let cp = if cp.dot(&(a + c)) < 0.0 { -cp } else { cp };

    // Invert uniform area sampling to find u0
    // 0.1 degrees
    let u0 = if a.dot(&cp) > 0.99999847691 {
        0.0
    } else {
        // Compute area a' of subtriangle
        let n_cpb = cp.cross(&b);
        let n_acp = a.cross(&cp);
        // TODO check rare
        if n_cpb.length_squared() == 0.0 || n_acp.length_squared() == 0.0 {
            return Point2f::new(0.5, 0.5);
        }
        let n_cpb = n_cpb.normalize();
        let n_acp = n_acp.normalize();
        let ap = alpha + n_ab.angle_between(&n_cpb) + n_acp.angle_between(&-n_cpb) - PI_F;

        // Compute sample u0 that gives area a'.
        let area = alpha + beta + gamma - PI_F;
        ap / area
    };

    // Invert arc sampling to find u1 and return result.
    let u1 = (1.0 - w.dot(&b)) / (1.0 - cp.dot(&b));
    Point2f::new(u0.clamp(0.0, 1.0), u1.clamp(0.0, 1.0))
}

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
