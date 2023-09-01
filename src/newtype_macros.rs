//! Utility macros for newtype trait implementations.
//! For a type A defined via the newtype pattern by wrapping type B
//! (i.e. struct A(B)), if B satisfies some trait, we often want A to
//! satisfy the same trait for the same types by calling type B's implementation.
//! Rust provides no standard mechanism for doing this, so we provide these
//! macros for some common cases.
//!
//! For every macro, the underlying type of the newtype must implement
//! the trait already for the appropriate types, or else an error will occur.
//!
//! Note that these macros implement for `self` rather than `&self`. This is
//! how std::ops traits are defined. To implement an op for references to a type,
//! you'd need to implement the trait on the reference type itself i.e. &MyType.
//! That's really true for any implementation for std::ops, though, rather
//! than something unique to these macros.

/// Implements a unary operation (e.g. Neg) trait for the newtype,
/// where the LHS and the result are both of the newtype.
///
/// ```
/// # #[macro_use] extern crate shimmer; fn main() {
/// # use std::ops::Neg;
/// struct Years(i32);
/// impl_unary_op_for_nt!( impl Neg for Years { fn neg } );
/// let val = Years(30);
/// assert_eq!(-30, -val.0)
/// # }
/// ```
#[macro_export]
macro_rules! impl_unary_op_for_nt {
    (impl $trait_: ident for $type_: ident { fn $method: ident }) => {
        impl $trait_ for $type_ {
            type Output = $type_;
            #[inline]
            fn $method(self) -> $type_ {
                let $type_(a) = self;
                $type_(a.$method())
            }
        }
    };
}

/// Implements a binary operation (e.g. Add, Sub, Mul, Div...) trait for
/// the newtype where the LHS, RHS, and result are the newtype.
///
/// ```
/// # #[macro_use] extern crate shimmer; fn main() {
/// # use std::ops::Add;
/// struct Years(u32);
/// impl_binary_op_trait_for_nt!( impl Add for Years { fn add } );
/// let moms_age = Years(30);
/// let dads_age = Years(32);
/// let sum = moms_age + dads_age;
/// assert_eq!(62, sum.0)
/// # }
/// ```
///
/// The implementation expands to:
///
/// ```
/// # use std::ops::Add;
///
/// struct Years(u32);
/// impl Add<Years> for Years {
///     type Output = Years;
///     #[inline]
///     fn add(self, Years(b): Years) -> Years {
///         let Years(a) = self;
///         Self(a.add(b))
///     }
/// }
/// ```
///
/// The derive_more crate is another option other than
/// than calling this macro. But at the time of writing, rust-analyzer
/// does not expand that crate's procedural macros, which is why
/// this macro was created instead. The other macros in this module
/// are outside the scope of derive_more at the time of writing.
#[macro_export]
macro_rules! impl_binary_op_trait_for_nt {
    (impl $trait_: ident for $type_: ident { fn $method: ident }) => {
        impl $trait_<$type_> for $type_ {
            type Output = $type_;
            #[inline]
            fn $method(self, $type_(b): $type_) -> $type_ {
                let $type_(a) = self;
                $type_(a.$method(b))
            }
        }
    };
}
pub(crate) use impl_binary_op_trait_for_nt;

/// Implements a binary operation (e.g. Add, Sub, Mul, Div...) trait for
/// the newtype where the LHS and the result of the operation are the new type,
/// while the RHS is some other type. For example:
///
/// ```
/// # #[macro_use] extern crate shimmer; fn main() {
/// # use std::ops::Mul;
/// struct Money(f32);
/// impl_binary_op_for_nt_with_other!( impl Mul for Money with f32 { fn mul } );
/// let dollars = Money(10.0);
/// let result: Money = dollars * 10.0;
/// assert_eq!(100.0, result.0)
/// # }
/// ```
///
#[macro_export]
macro_rules! impl_binary_op_for_nt_with_other {
    (impl $trait_: ident for $new_type_: ident with $other_type_: ident { fn $method: ident }) => {
        impl $trait_<$other_type_> for $new_type_ {
            type Output = $new_type_;
            #[inline]
            fn $method(self, b: $other_type_) -> $new_type_ {
                let $new_type_(a) = self;
                $new_type_(a.$method(b))
            }
        }
    };
}
pub(crate) use impl_binary_op_for_nt_with_other;

/// See impl_binary_op_for_nt_with_other.
///
/// This macro serves the same purpose, but reversing the order of the operands' types.
macro_rules! impl_binary_op_for_other_with_nt {
    (impl $trait_: ident for $other_type_: ident with $new_type_: ident { fn $method: ident }) => {
        impl $trait_<$new_type_> for $other_type_ {
            type Output = $new_type_;
            #[inline]
            fn $method(self, $new_type_(b): $new_type_) -> $new_type_ {
                $new_type_(self.$method(b))
            }
        }
    };
}
pub(crate) use impl_binary_op_for_other_with_nt;

/// Implements an assignment operation trait (AddAssign, MulAssign, etc)
/// where both the LHS and the RHS of the equation are the newtype.
///
/// ```
/// # #[macro_use] extern crate shimmer; fn main() {
/// # use std::ops::AddAssign;
/// struct Years(u32);
/// impl_binary_op_assign_trait_for_nt!(impl AddAssign for Years { fn add_assign });
/// let mut my_age = Years(18);
/// let years_in_academia = Years(8);
/// my_age += years_in_academia;
/// assert_eq!(26, my_age.0)
/// # }
/// ```
#[macro_export]
macro_rules! impl_binary_op_assign_trait_for_nt {
    (impl $trait_: ident for $type_: ident { fn $method: ident }) => {
        impl $trait_<$type_> for $type_ {
            #[inline]
            fn $method(&mut self, $type_(b): $type_) {
                let $type_(a) = self;
                a.$method(b);
            }
        }
    };
}
pub(crate) use impl_binary_op_assign_trait_for_nt;

/// Implements an operation assignment trait (AddAssign, MulAssign, etc)
/// where the LHS of the equation is the newtype and the RHS of the equation is some other type.
///
/// ```
/// # #[macro_use] extern crate shimmer; fn main() {
/// # use std::ops::MulAssign;
/// #[derive(Debug, PartialEq)]
/// struct Money(f32);
/// impl_binary_op_assign_for_nt_with_other!(impl MulAssign for Money with f32 { fn mul_assign } );
/// let mut my_money = Money(-40.0);
/// my_money *= 2.0;
/// assert_eq!(Money(-80.0), my_money);
/// my_money *= 0.0;
/// assert_eq!(Money(0.0), my_money);
/// # }
/// ```
#[macro_export]
macro_rules! impl_binary_op_assign_for_nt_with_other {
    (impl $trait_: ident for $new_type_: ident with $other_type_: ident { fn $method: ident }) => {
        impl $trait_<$other_type_> for $new_type_ {
            #[inline]
            fn $method(&mut self, b: $other_type_) {
                let $new_type_(a) = self;
                a.$method(b);
            }
        }
    };
}
pub(crate) use impl_binary_op_assign_for_nt_with_other;
