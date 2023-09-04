use std::ops::Div;

use super::{has_nan::HasNan, length::Length};

pub trait Normalize<T>: HasNan + Length<T>
where
    Self: Sized + Div<T, Output = Self>,
{
    fn normalize(self) -> Self {
        let len = self.length();
        debug_assert!(!self.has_nan());
        self / len
    }
}
