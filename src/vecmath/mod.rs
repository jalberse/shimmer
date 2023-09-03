//! Why are there distinct types for **Vector** and **Point** and
//! **Normal**? Why not just use a unified vector class? Because they
//! are not the same, so let's capture that with our ~type system~.
//!
//! All **vectors** v in an affine space can be expressed as a linear
//! combination of the basis vectors:
//!
//! v = s_1 * v_1 + ... + s_n * v_n
//!
//! The scalars s_i are the representation of v with respect to the basis.
//!
//! Similarly, for all **points** p and origin p_0, there are unique scalars s_i
//! such that the point can be expressed in terms of the origin p_0 and
//! the basis vectors:
//!
//! p = p_0 + s_1 * v_1 + ... + s_n * v_n
//!
//! So, while vectors and points are both represented by 3 x, y, z values
//! in 3D space, they are distinct mathematical entities and not freely
//! interchangeable. Having separate types can therefore be more easily
//! understood and reasoned with, and allows us to handle them in different
//! manners as necessary (e.g points don't have a length).
//!
//! A **surface normal** (or a **normal**) is a vector perpendicular to a
//! surface at a particular position. Although normals are superficially
//! similar to vectors, it is important to distinguish between them
//! in the type system as well: because normals are defined in terms of
//! their relationship to a particular surface, they behave differently
//! than vectors in some situations, particularly when applying transformations.

// This module contains traits (Tuple, Length, Normalize, etc) which are created
// to allow us to share logic across our various types (Vectors, Normals, Points).
// However, we don't export the traits, and make their functions available within
// the public structs (calling the trait implementation). This way, users don't
// need to import the traits to use the structs.

// TODO Consider making the traits public.
//  I had previously thought we would keep them private and expose the API only through the structs,
//  so that users don't need to import the traits to use them.
//  Actually, I guess it can still make sense to keep the traits private and expose only within the structs.
//  Even if there's something like Lerp that I dont' want taking &self (since I think lerp(t, a, b) is cleaner than a.lerp(t, b),
//  we can still do that by adding it to the struct.)
//  I don't want callers of vecmath to implement on these traits necesssarily, so there's not a great reason to expose them.
// It's extra maintanence to ensure that we make a function in the struct to expose something in the trait,
// when really we should just design the trait that it's something we'd be happy exposing.

pub mod has_nan;
pub mod length;
pub mod normal;
pub mod normalize;
pub mod point;
pub mod tuple;
pub mod vector;

pub use normal::{Normal3f, Normal3i};
pub use point::{Point2f, Point2i, Point3f, Point3i};
pub use vector::{Vector2f, Vector2i, Vector3f, Vector3i};

// TODO Our impl_op_ex* methods are not all tested; test them.

// TODO Run a test coverage software and ensure full coverage. I know I'm missing a bit. https://lib.rs/crates/cargo-llvm-cov
//   Also hook that up into GitHub actions.

// TODO ensure we debug_assert() with has_nan() where appropriate.

// TODO also implement the list of functions on page 85
