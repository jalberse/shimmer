use crate::{
    float::PI_F,
    math::{lerp, safe_sqrt, sqr},
    sampling::{sample_uniform_disk_concentric, sample_uniform_disk_polar},
    spectra::{sampled_spectrum::SampledSpectrum, NUM_SPECTRUM_SAMPLES},
    vecmath::{
        normal::Normal3,
        spherical::{abs_cos_theta, cos2_theta, cos_phi, sin_phi, tan2_theta},
        vector::Vector3,
        Length, Normal3f, Normalize, Point2f, Tuple3, Vector2f, Vector3f,
    },
    Float,
};
use num::complex::Complex;

#[inline]
pub fn reflect(wo: Vector3f, n: Normal3f) -> Vector3f {
    -wo + 2.0 * wo.dot_normal(n) * Vector3f::from(n)
}

// PAPERDOC - Easy example of where Option is useful, compared to returning a boolean.
// Option ensures both cases are handled, where callers might ignore the boolean.
/// Returns (wt, etap) where wt is the refracted direction and etap is the adjusted relative IOR.
/// wi: Incident direction
/// n: Surface normal in same hemisphere as wi
/// eta: Relative index of refraction
#[inline]
pub fn refract(wi: Vector3f, mut n: Normal3f, mut eta: Float) -> Option<(Vector3f, Float)> {
    let mut cos_theta_i = n.dot_vector(wi);
    // Potentially flip interface orientation for Snell's law
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
        n = -n;
    }

    let sin2_theta_i = Float::max(0.0, 1.0 - sqr(cos_theta_i));
    let sin2_theta_t = sin2_theta_i / sqr(eta);

    // Handle total internal reflection case
    if sin2_theta_t >= 1.0 {
        return None;
    }

    let cos_theta_t = Float::sqrt(1.0 - sin2_theta_t);

    let wt = -wi / eta + (cos_theta_i / eta - cos_theta_t) * Vector3f::from(n);
    let etap = eta;
    Some((wt, etap))
}

/// cos_theta_i: Cosine of the angle between the incident direction and the normal.
/// eta: Relative index of refraction.
/// output: Unpolorized fresnel reflection of a dielectric interface
#[inline]
pub fn fresnel_dielectric(cos_theta_i: Float, mut eta: Float) -> Float {
    let mut cos_theta_i = Float::clamp(cos_theta_i, -1.0, 1.0);

    // Potentially flip interface orientation
    if cos_theta_i < 0.0 {
        eta = 1.0 / eta;
        cos_theta_i = -cos_theta_i;
    }

    let sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i;
    let sin2_theta_t = sin2_theta_i / (eta * eta);
    if sin2_theta_t >= 1.0 {
        return 1.0;
    }

    let cos_theta_t = safe_sqrt(1.0 - sin2_theta_t);

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Used for conductors.
/// cos_theta_i: Cosine of the angle between the incident direction and the normal.
/// eta: The relative IOR. The real component describes the decrease in the speed of light,
/// while the imaginary component describes the decay of light as it travels deeper into the material.
/// output: Unpolorized fresnel reflection of the interface
#[inline]
pub fn fresnel_complex(cos_theta_i: Float, eta: Complex<Float>) -> Float {
    let cos_theta_i = Float::clamp(cos_theta_i, 0.0, 1.0);

    let sin2_theta_i = 1.0 - sqr(cos_theta_i);
    let sin2_theta_t = sin2_theta_i / sqr(eta);
    let cos_theta_t = Complex::sqrt(1.0 - sin2_theta_t);

    let r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
    let r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);

    (Complex::norm_sqr(&r_parl) + Complex::norm_sqr(&r_perp)) / 2.0
}

/// Wrapper around fresnel_complex() which takes spectrally varying complex IORs split into
/// the eta and k values.
#[inline]
pub fn fresnel_complex_spectral(
    cos_theta_i: Float,
    eta: SampledSpectrum,
    k: SampledSpectrum,
) -> SampledSpectrum {
    let mut s = [0.0; NUM_SPECTRUM_SAMPLES];
    for i in 0..NUM_SPECTRUM_SAMPLES {
        s[i] = fresnel_complex(cos_theta_i, Complex::new(eta[i], k[i]));
    }
    SampledSpectrum::new(s)
}

/// Encapsulates the microfacet distribution function according to the Trowbridge-Reitz model.
pub struct TrowbridgeReitzDistribution {
    alpha_x: Float,
    alpha_y: Float,
}

impl TrowbridgeReitzDistribution {
    pub fn new(ax: Float, ay: Float) -> Self {
        let d = Self {
            alpha_x: ax,
            alpha_y: ay,
        };
        if !d.effectively_smooth() {
            // If one direction has some roughness, then the other can't
            // have zero (or very low) roughness; the computation of |e| in
            // D() blows up in that case.
            let alpha_x = Float::max(d.alpha_x, 1e-4);
            let alpha_y = Float::max(d.alpha_y, 1e-4);
            Self { alpha_x, alpha_y }
        } else {
            d
        }
    }

    #[inline]
    pub fn effectively_smooth(&self) -> bool {
        self.alpha_x < 1e-3 && self.alpha_y < 1e-3
    }

    #[inline]
    pub fn d(&self, wm: Vector3f) -> Float {
        let tan2_theta = tan2_theta(wm);
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let cos4_theta = sqr(cos2_theta(wm));
        if cos4_theta < 1e-16 {
            return 0.0;
        }
        let e = tan2_theta * (sqr(cos_phi(wm) / self.alpha_x) + sqr(sin_phi(wm) / self.alpha_y));
        1.0 / (PI_F * self.alpha_x * self.alpha_y * cos4_theta * sqr(1.0 + e))
    }

    /// Masking function
    pub fn g1(&self, w: Vector3f) -> Float {
        1.0 / (1.0 + self.lambda(w))
    }

    pub fn lambda(&self, w: Vector3f) -> Float {
        let tan2_theta = tan2_theta(w);
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let alpha2 = sqr(cos_phi(w) * self.alpha_x) + sqr(sin_phi(w) * self.alpha_y);
        (-1.0 + Float::sqrt(1.0 + alpha2 * tan2_theta)) / 2.0
    }

    pub fn g(&self, wo: Vector3f, wi: Vector3f) -> Float {
        1.0 / (1.0 + self.lambda(wo) + self.lambda(wi))
    }

    pub fn d_w(&self, w: Vector3f, wm: Vector3f) -> Float {
        self.g1(w) / abs_cos_theta(w) * self.d(wm) * w.abs_dot(wm)
    }

    pub fn pdf(&self, w: Vector3f, wm: Vector3f) -> Float {
        self.d_w(w, wm)
    }

    pub fn sample_wm(&self, w: Vector3f, u: Point2f) -> Vector3f {
        // Transofrm w to hemispherical configuration
        let mut wh = Vector3f::new(self.alpha_x * w.x, self.alpha_y * w.y, w.z).normalize();
        if wh.z < 0.0 {
            wh = -wh;
        }

        // Find orthonormal basis for visible normal sampling
        let t1 = if wh.z < 0.99999 {
            Vector3f::Z.cross(wh).normalize()
        } else {
            Vector3f::X
        };
        let t2 = wh.cross(t1);

        // Generate uniformly distributed points on the unit disk
        let mut p = sample_uniform_disk_polar(u);

        // Warp hemispherical projection for visible normal sampling
        let h = Float::sqrt(1.0 - sqr(p.x));
        p.y = lerp((1.0 + wh.z) / 2.0, h, p.y);

        // Reproject to hemisphere and transform normal to ellipsoid configuration
        let pz = Float::sqrt(Float::max(0.0, 1.0 - Vector2f::from(p).length_squared()));
        let nh = p.x * t1 + p.y * t2 + pz * wh;
        Vector3f::new(
            self.alpha_x * nh.x,
            self.alpha_y * nh.y,
            Float::max(1e-6, nh.z),
        )
        .normalize()
    }

    pub fn roughness_to_alpha(roughness: Float) -> Float {
        Float::sqrt(roughness)
    }

    pub fn regularize(&mut self) {
        if self.alpha_x < 0.3 {
            self.alpha_x = Float::clamp(2.0 * self.alpha_x, 0.1, 0.3);
        }
        if self.alpha_y < 0.3 {
            self.alpha_y = Float::clamp(2.0 * self.alpha_y, 0.1, 0.3);
        }
    }
}

impl Default for TrowbridgeReitzDistribution {
    fn default() -> Self {
        Self {
            alpha_x: Default::default(),
            alpha_y: Default::default(),
        }
    }
}
