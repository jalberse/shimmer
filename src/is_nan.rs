use crate::float::Float;

pub trait IsNan {
    fn is_nan(self) -> bool;
}

impl IsNan for Float {
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl IsNan for i32 {
    fn is_nan(self) -> bool {
        false
    }
}
