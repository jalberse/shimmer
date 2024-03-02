use crate::{scattering::{henyey_greenstein, sample_henyey_greenstein}, vecmath::{vector::Vector3, Point2f, Vector3f}, Float};

// TODO Implement this; we currently are stubbing it out for usage within the Ray class.
#[derive(Debug, Copy, Clone)]
pub struct Medium {}


pub struct HGPhaseFunction {
    g: Float
}

impl HGPhaseFunction
{
    pub fn new(g: Float) -> HGPhaseFunction {
        HGPhaseFunction { g }
    }

    pub fn p(&self, wo: Vector3f, wi: Vector3f) -> Float
    {
        henyey_greenstein(wo.dot(wi), self.g)
    }

    pub fn pdf(&self, wo: Vector3f, wi: Vector3f) -> Float
    {
        self.p(wo, wi)
    }

    pub fn sample_p(&self, wo: Vector3f, u: Point2f) -> Option<PhaseFunctionSample>
    {
        let (pdf, wi) = sample_henyey_greenstein(wo, self.g, u);
        Some(PhaseFunctionSample { p: pdf, wi, pdf })
    }
}

pub struct PhaseFunctionSample
{
    pub p: Float,
    pub wi: Vector3f,
    pub pdf: Float
}