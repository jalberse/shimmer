// TODO We'll probably iumplement either Middle or EqualCounts as SplitMethod first because they're easy,
// but the others are more useful.
// The BVHAggregate is defined in a different module because it's useful to split them, but they will
//   also be in the Primitive enum and implement the PrimitiveI interface.
// PBRT constructs with pointers first and then converts to a vec-based implementation.
//   I could try to do the same I suppose, but I suspect we'll run into ownership issues.

use std::{io::Split, rc::Rc, sync::Mutex};

use itertools::Itertools;

use crate::{
    bounding_box::Bounds3f,
    primitive::{Primitive, PrimitiveI},
    ray::Ray,
    shape::ShapeIntersection,
    vecmath::Point3f,
    Float,
};

pub enum SplitMethod {
    // TODO Other split methods; Middle isn't very good, but simple for the first implementation.
    Middle,
}

struct BvhAggregate {
    max_prims_in_node: i32,
    primitives: Vec<Rc<Primitive>>,
    split_method: SplitMethod,
    nodes: Vec<LinearBvhNode>,
}

impl PrimitiveI for BvhAggregate {
    fn bounds(&self) -> Bounds3f {
        todo!()
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        todo!()
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        todo!()
    }
}

impl BvhAggregate {
    pub fn new(
        mut primitives: Vec<Rc<Primitive>>,
        max_prims_in_node: i32,
        split_method: SplitMethod,
    ) -> BvhAggregate {
        debug_assert!(!primitives.is_empty());

        // Build BVH from primitives
        // Initialize bvh_primitives array to store compact information about primitives while building.
        let bvh_primitives: Vec<BvhPrimitive> = primitives
            .iter()
            .enumerate()
            .map(|(i, &ref p)| BvhPrimitive {
                primitive_index: i,
                bounds: p.bounds(),
            })
            .collect_vec();

        // TODO For now, I will not use a specific allocator. We will likely want to go add
        // something like std::monotonic_buffer_resource later.

        // This will store the primitives ordered such that the primitives in each leaf node occupy a
        // contiguous range in the array for efficient access. It will be swapped with the original
        // primitives array after the tree is constructed.
        let mut ordered_primitives: Vec<Rc<Primitive>> = Vec::with_capacity(primitives.len());

        // TODO The counters will probably be an Arc mutex actually.

        // Keeps track of the number of nodes created; this makes it possible to allocate exactly
        // the right size Vec for the LinearBvhNodes list.
        let mut total_nodes = Mutex::new(0);

        // Build the BVH according to the selected split method
        let root = match split_method {
            // The default for all splitting methods but HLBVH will be to use build_recursive().
            _ => {
                // Keeps track of where the next free entry in ordered_primitives is, so that leaf nodes can claim
                // it as their starting primitive index, and reserve the next n slots for their n primitives.
                let mut ordered_prims_offset = Mutex::new(0);
                Self::build_recursive(
                    &bvh_primitives,
                    &mut total_nodes,
                    &mut ordered_prims_offset,
                    &mut ordered_primitives,
                )
            }
        };

        // Swap so that primtives stores the ordered primitives.
        std::mem::swap(&mut primitives, &mut ordered_primitives);

        // This can be a significant chunk of memory that we don't need anymore; free it.
        drop(bvh_primitives);

        let mut nodes: Vec<LinearBvhNode> = Vec::with_capacity(*total_nodes.lock().unwrap());

        let mut offset = 0;
        Self::flatten_bvh(&mut nodes, Some(Box::new(root)), &mut offset);
        debug_assert_eq!(*total_nodes.lock().unwrap(), offset);

        BvhAggregate {
            max_prims_in_node: i32::max(255, max_prims_in_node),
            primitives,
            split_method,
            nodes,
        }
    }

    fn build_recursive(
        bvh_primitives: &[BvhPrimitive],
        total_nodes: &mut Mutex<usize>,
        ordered_prims_offset: &mut Mutex<usize>,
        ordered_prims: &mut Vec<Rc<Primitive>>,
    ) -> BvhBuildNode {
        todo!()
    }

    // Performs a depth-first traversal and stores the nodes in memory in linear order.
    // offset tracks the current offset into the linear_nodes array.
    // node is the root of the tree; this function is recursive, so it is also the root of sub-trees
    // as this traverses down the tree.
    fn flatten_bvh(
        linear_nodes: &mut Vec<LinearBvhNode>,
        node: Option<Box<BvhBuildNode>>,
        offset: &mut usize,
    ) -> usize {
        let node = node.expect("flatten_bvh should be called with a valid root");

        // PAPERDOC This is an interesting comparison to the PBRT implementation.
        // The borrow checker would complain if we held both the linear_node as a mutable reference to
        // the element in the array, and passed the array recrusively. We instead need to initialize
        // the linear node locally, and set it after we're returned ownership of the vector.

        let linear_node_bounds = node.bounds;

        let node_offset = *offset;
        *offset += 1;

        let linear_node = if node.n_primitives > 0 {
            // Leaf node!
            debug_assert!(node.left_child.is_none() && node.right_child.is_none());
            debug_assert!(node.n_primitives < 65536);
            let linear_node_offset = LinearOffset::PrimitivesOffset(node.first_prim_offset);
            let linear_nod_n_primitives = node.n_primitives as u16;
            LinearBvhNode {
                bounds: linear_node_bounds,
                offset: linear_node_offset,
                n_primitives: linear_nod_n_primitives,
                axis: 0,
            }
        } else {
            // Create interior flattened BVH node.
            let linear_node_axis = node.split_axis as u8;
            let linear_node_n_primitives = 0;
            Self::flatten_bvh(linear_nodes, node.left_child, offset);
            let second_child_offset = Self::flatten_bvh(linear_nodes, node.right_child, offset);
            let linear_node_offset = LinearOffset::SecondChildOffset(second_child_offset);
            LinearBvhNode {
                bounds: linear_node_bounds,
                offset: linear_node_offset,
                n_primitives: linear_node_n_primitives,
                axis: linear_node_axis,
            }
        };
        linear_nodes[node_offset] = linear_node;

        node_offset
    }
}

/// Within a LinearBvhNode, stores either the offset to the primitives for the node,
/// or the offset to the second child.
pub enum LinearOffset {
    PrimitivesOffset(usize),
    SecondChildOffset(usize),
}

/// A BVH Node that is stored in a compact, linear representation of a BVH.
pub struct LinearBvhNode {
    bounds: Bounds3f,
    offset: LinearOffset,
    n_primitives: u16,
    axis: u8,
}

/// Stores the bounding box and its index in the primitives array; used while building the BVH.
struct BvhPrimitive {
    primitive_index: usize,
    bounds: Bounds3f,
}

impl BvhPrimitive {
    pub fn centroid(&self) -> Point3f {
        0.5 * self.bounds.min + (self.bounds.max * 0.5).into()
    }
}

// TODO consider making BvhBuildNode an enum with an InteriorBvhBuildNode and a LeafBvhBuildNode.
// That might be cleaner than checking for children, and possibly more space efficient.
/// Used for representing a BVH node during construction of the BVH.
struct BvhBuildNode {
    bounds: Bounds3f,
    // Nodes are leaves if both children are None.
    left_child: Option<Box<BvhBuildNode>>,
    right_child: Option<Box<BvhBuildNode>>,
    split_axis: u8,
    // The nodes in [first_prim_offset, first_prim_offset + n_primitives) are contained
    // within this node.
    first_prim_offset: usize,
    n_primitives: i32,
}

impl BvhBuildNode {
    pub fn leaf(first_prim_offset: usize, n_primitives: i32, bounds: Bounds3f) -> BvhBuildNode {
        BvhBuildNode {
            bounds,
            left_child: None,
            right_child: None,
            split_axis: 0,
            first_prim_offset,
            n_primitives,
        }
    }

    pub fn interior(
        split_axis: u8,
        left_child: Box<BvhBuildNode>,
        right_child: Option<Box<BvhBuildNode>>,
    ) -> BvhBuildNode {
        // Interior nodes always have at least one child.
        let bounds = if let Some(right) = &right_child {
            left_child.bounds.union(&right.bounds)
        } else {
            left_child.bounds
        };
        // The primitive offset and num_primitives are 0 because interior nodes do not contain primitives.
        BvhBuildNode {
            bounds,
            left_child: Some(left_child),
            right_child: right_child,
            split_axis,
            first_prim_offset: 0,
            n_primitives: 0,
        }
    }
}
