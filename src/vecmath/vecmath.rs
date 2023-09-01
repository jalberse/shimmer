use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use glam::{IVec2, IVec3};

use crate::float::Float;
use crate::impl_unary_op_for_nt;
use crate::newtype_macros::{
    impl_binary_op_assign_for_nt_with_other, impl_binary_op_assign_trait_for_nt,
    impl_binary_op_for_nt_with_other, impl_binary_op_for_other_with_nt,
    impl_binary_op_trait_for_nt,
};
