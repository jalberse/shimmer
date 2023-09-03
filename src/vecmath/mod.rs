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

pub mod has_nan;
pub mod length;
pub mod normal;
pub mod normalize;
pub mod point;
pub mod tuple;
pub mod vector;

pub use has_nan::HasNan;
pub use length::Length;
pub use normal::{Normal3f, Normal3i};
pub use normalize::Normalize;
pub use point::{Point2f, Point2i, Point3f, Point3i};
pub use tuple::{Tuple2, Tuple3};
pub use vector::{Vector2f, Vector2i, Vector3f, Vector3i};

// TODO While I think we want to expose the traits as public so that users can import them and use them on the structs,
//   I think that we want to keep e.g. our helper cross() methods private to the vecmath module.
// I think we would do it by making some length_helpers() module or similar, and using that inside of vecmath.
//   But we would not make it a public module here.

// TODO Our impl_op_ex* methods are not all tested; test them.

// TODO Run a test coverage software and ensure full coverage. I know I'm missing a bit. https://lib.rs/crates/cargo-llvm-cov
//   Also hook that up into GitHub actions.

// TODO ensure we debug_assert() with has_nan() where appropriate.

// TODO also implement the list of functions on page 85
