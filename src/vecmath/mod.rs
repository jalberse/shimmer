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

// TODO Consider specialization after Rust RFC 1210 is implemented, if ever.
//
// Consider: I think that the better way to represent e.g. Point2f vs Point2i,
//   is not to have Point2 be a trait. We can lets Point2<T> be a generic class,
//   with type aliases for 2f and 2i. Wherever we need different implementations for
//   i32 vs float, we can have an impl<i32> Point2<i32> {} block and a impl<Float> Point2<Float> {}
//   block, but shared implementations can remain in a generic impl<T> Point2<T> block.
// I think this would make the code cleaner.
// Something like this:
//
// struct TestType<T> {
//     t: T,
// }
//
// impl TestType<i32> {
//     pub fn new(v: i32) -> TestType<i32> {
//         TestType { t: v }
//     }
// }
//
// impl TestType<Float> {
//     pub fn new(v: Float) -> TestType<Float> {
//         TestType { t: v }
//     }
// }
//
// impl<T> TestType<T> {
//     pub fn new_generic(t: T) -> TestType<T> {
//         TestType { t }
//     }
// }
//
// Note that if the impl<T> block defined a new(), there would be a naming conflict since T is ALL T,
// including i32 or Float, so a TestType<i32> type would have two new() functions (but the i32
// and Float impls don't contradict, since an object can't be both a TestType<i32> and a TestType<Float>).
//
// BUT I'm not totally convinced of this approach. If we want to implement a function that takes a
// generic Point<T> under such an approach, I think we wouldn't be able to reference e.g. cross()
// which would be implemented for Point<i32> and Point<Float> but not all Point<T>, so the compiler
// wouldn't let us use a Point<T>::cross(). And you can't just implement Point<T>::cross() and let
// the compiler use the "more specific" i32 or Float definition under Stable rust.
//  We would need that calling code to be specialized as well, I think, which doesn't suit our need for generics.
// In which case, I think the Trait approach might actually be better. But it's something to consider.
// It turns out, I've run up against the Specialization RFC, which seeks to solve this problem:
// https://stackoverflow.com/questions/72913825/can-i-do-template-specialisation-with-concrete-types-in-rust
// https://rust-lang.github.io/rfcs/1210-impl-specialization.html
// https://github.com/rust-lang/rust/issues/31844
// So, no, we can't do this. The current Trait approach is idiomatic Rust.

// TODO Point3fi and Vector3fi
// TODO There's still chunks of code that are repeated, mainly in impl_op_ex()'s
// and across e.g. Point3* and Vector3* impls. I would like to go back and tighten everything up
// to re-use code more intelligently. This type rework can also touch on e.g. Interval not really
// being a reasonable candidate for PartialOrd, but needing to be so for TupleElement - could
// we rework it so the float interval types don't need that?
// Generally I'd like to revisit all these types and refactor into more sane generics.
// I do feel limited by the lack of specialization.
// But, I've actually made some things more generic in the process of implementing the 3fi variants,
// so I may find it easier to share functionality by following that approach.
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
