use crate::{
    vecmath::{Point2f, Vector2f},
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
    // TODO
    TODO,
}

impl FilterI for Filter {
    fn radius(&self) -> Vector2f {
        todo!()
    }

    fn evaluate(&self, p: Point2f) -> Float {
        todo!()
    }

    fn integral(&self) -> Float {
        todo!()
    }

    fn sample(&self, u: Point2f) -> FilterSample {
        todo!()
    }
}

pub struct FilterSample {
    // TODO
}
