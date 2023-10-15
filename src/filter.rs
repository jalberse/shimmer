use crate::{
    math::lerp,
    vecmath::{Point2f, Tuple2, Vector2f},
    Float,
};

pub trait FilterI {
    fn radius(&self) -> Vector2f;

    fn evaluate(&self, p: Point2f) -> Float;

    fn integral(&self) -> Float;

    fn sample(&self, u: Point2f) -> FilterSample;
}

#[derive(Debug)]
pub enum Filter {
    BoxFilter(BoxFilter),
}

impl FilterI for Filter {
    fn radius(&self) -> Vector2f {
        match self {
            Filter::BoxFilter(f) => f.radius(),
        }
    }

    fn evaluate(&self, p: Point2f) -> Float {
        match self {
            Filter::BoxFilter(f) => f.evaluate(p),
        }
    }

    fn integral(&self) -> Float {
        match self {
            Filter::BoxFilter(f) => f.integral(),
        }
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        match self {
            Filter::BoxFilter(f) => f.sample(u),
        }
    }
}

/// Equivalent to not addressing filtering or reconstruction.
/// Other filters tend to be better; its simplicity is computationally efficient.
/// Equally weights all samples within a square region of an image.
#[derive(Debug)]
pub struct BoxFilter {
    radius: Vector2f,
}

impl BoxFilter {
    pub fn new(radius: Vector2f) -> BoxFilter {
        BoxFilter { radius }
    }
}

impl FilterI for BoxFilter {
    fn radius(&self) -> Vector2f {
        self.radius
    }

    fn evaluate(&self, p: Point2f) -> Float {
        if Float::abs(p.x) <= self.radius.x && Float::abs(p.y) <= self.radius.y {
            1.0
        } else {
            0.0
        }
    }

    fn integral(&self) -> Float {
        2.0 * self.radius.x * 2.0 * self.radius.y
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        let p = Point2f::new(
            lerp(u[0], &-self.radius.x, &self.radius.x),
            lerp(u[1], &-self.radius.y, &self.radius.y),
        );
        FilterSample { p, weight: 1.0 }
    }
}

pub struct FilterSample {
    p: Point2f,
    weight: Float,
}

// TODO FilterSampler, which is used in specific Filters. But don't implement until we need it.
