use crate::Float;

struct CompensatedFloat {
    pub v: Float,
    pub err: Float,
}

impl CompensatedFloat {
    pub fn new(v: Float, err: Float) -> CompensatedFloat {
        CompensatedFloat { v, err }
    }
}

impl From<CompensatedFloat> for Float {
    fn from(value: CompensatedFloat) -> Self {
        value.v + value.err
    }
}
