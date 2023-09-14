//! This module contains the related concepts of a Tuple/Vector/Point/Normal.
//!
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
//!
//! Further, note that common behaviors are shared via traits in this module;
//! we rely on monomorphization on generic functions in order to do static dispatch.
//! It's not intended to use these traits as trait objects by e.g. creating a
//! Vec<dyn Tuple3<Float>> for runtime polymorphism across Normals and Vectors;
//! this would result in dynamic dispatch, which is not efficient.
//! It's unlikely one would need such runtime polymorphism on these particular types,
//! however, because of the discussion above, so we're okay with this.
//! If we ever *did* want runtime polymorphism on these tyeps, we should
//! use the enum_dispatch crate to generate an enum of implementing types and thus
//! use static dispatch.

pub mod has_nan;
pub mod length;
mod length_fns;
pub mod normal;
pub mod normalize;
pub mod point;
pub mod tuple;
mod tuple_fns;
pub mod vector;

pub use has_nan::HasNan;
pub use length::Length;
pub use normal::{Normal3f, Normal3i};
pub use normalize::Normalize;
pub use point::{Point2f, Point2i, Point3f, Point3i};
pub use tuple::{Tuple2, Tuple3};
pub use vector::{Vector2f, Vector2i, Vector3f, Vector3i};

// TODO Coordinate sys from vectors
// TODO Face forward functions.
// TODO ensure we debug_assert() with has_nan() where appropriate.
// TODO Improve testing coverage.
// TODO Consider some NormalizedVector, NormalizedNormal type or some other
// mechanism for enforcing that a vector must be normalized for an operation like angle_between.
// It's certainly possible with a type system but that's a lot of code to write.
// You would need the normalize() functions to return the Normalized* type,
// and then define operations on the Normalized* type as appropriate (where scaling
// would for example make it not-normalized).
// TODO We could share gram_schmidt() implementations for Vector types if we had
//  dot as a separate trait which both Vector2 and Vector3 become supertraits of.
//  This way we can place the trait constraints Dot + Mul + Sub onto a helper function
//  which all the various structs may call in their implementation.
// TODO we likely need a FMA function for Tuple, but let's hold off implementing it until we do need it
//   for something else. I've sent too long on vector math and not enough time on rendering.
// TODO Consider splitting e.g. Vector2 and Vector3 (and associated types imlementing them) into separate files.
// TODO There's still chunks of code that are repeated, mainly in impl_op_ex()'s.
//   I'd like to share them where possible, but that's not worth the effort at the time of writing.
