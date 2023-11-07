// TODO We'll probably iumplement either Middle or EqualCounts as SplitMethod first because they're easy,
// but the others are more useful.
// The BVHAggregate is defined in a different module because it's useful to split them, but they will
//   also be in the Primitive enum and implement the PrimitiveI interface.
// PBRT constructs with pointers first and then converts to a vec-based implementation.
//   I could try to do the same I suppose, but I suspect we'll run into ownership issues.

use std::{
    rc::Rc,
    sync::atomic::{AtomicUsize, Ordering},
};

use itertools::{partition, Itertools};

use crate::{
    bounding_box::Bounds3f,
    primitive::{Primitive, PrimitiveI},
    ray::Ray,
    shape::ShapeIntersection,
    vecmath::{Point3f, Tuple3, Vector3f},
    Float,
};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SplitMethod {
    // TODO Other split methods; Middle isn't very good, but simple for the first implementation.\
    // EqualCounts is the next simplest.
    Middle,
}

pub struct BvhAggregate {
    max_prims_in_node: usize,
    primitives: Vec<Rc<Primitive>>,
    split_method: SplitMethod,
    nodes: Vec<LinearBvhNode>,
}

impl PrimitiveI for BvhAggregate {
    fn bounds(&self) -> Bounds3f {
        // The root node's bounds cover the full extent.
        self.nodes[0].bounds
    }

    fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        if self.nodes.is_empty() {
            return None;
        }

        let inv_dir = Vector3f::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
        let dir_is_neg: [usize; 3] = [
            (inv_dir.x < 0.0) as usize,
            (inv_dir.y < 0.0) as usize,
            (inv_dir.z < 0.0) as usize,
        ];

        // Follow the ray through BVH nodes to find primitive intersections
        let mut si: Option<ShapeIntersection> = None;
        let mut t_max = t_max;

        let mut to_visit_offset = 0;
        let mut current_node_index = 0;
        // Acts as a stack storing the nodes that still need to be visited
        let mut nodes_to_visit = [0; 64];

        loop {
            let node = &self.nodes[current_node_index];
            // Check ray against BVH node
            if node
                .bounds
                .intersect_p_cached(ray.o, ray.d, t_max, inv_dir, dir_is_neg)
            {
                if node.n_primitives > 0 {
                    // Leaf node; intersect ray with primitives in the node
                    for i in 0..node.n_primitives {
                        // TODO Actually, given this use case, let's not store the offset
                        // in a variant. The runtime cost of the match while extremely small
                        // is probably less important than saving the byte or whatever.
                        let prim_si = self.primitives[node.primitive_offset + i as usize]
                            .as_ref()
                            .intersect(ray, t_max);
                        if let Some(prim_si) = prim_si {
                            // Don't break; there may be a closer intersection, enforced by t_max.
                            t_max = prim_si.t_hit;
                            si = Some(prim_si);
                        }
                    }
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                } else {
                    // Interior node; put far BVH node on nodes_to_visit stack,
                    // advance to the near node
                    if dir_is_neg[node.axis as usize] != 0 {
                        nodes_to_visit[to_visit_offset] = current_node_index + 1;
                        to_visit_offset += 1;
                        current_node_index = node.second_child_offset;
                    } else {
                        nodes_to_visit[to_visit_offset] = node.second_child_offset;
                        to_visit_offset += 1;
                        current_node_index += 1;
                    }
                }
            } else {
                if to_visit_offset == 0 {
                    break;
                }
                to_visit_offset -= 1;
                current_node_index = nodes_to_visit[to_visit_offset];
            }
        }

        si
    }

    fn intersect_predicate(&self, ray: &Ray, t_max: Float) -> bool {
        if self.nodes.is_empty() {
            return false;
        }

        let inv_dir = Vector3f::new(1.0 / ray.d.x, 1.0 / ray.d.y, 1.0 / ray.d.z);
        let dir_is_neg: [usize; 3] = [
            (inv_dir.x < 0.0) as usize,
            (inv_dir.y < 0.0) as usize,
            (inv_dir.z < 0.0) as usize,
        ];

        // Follow the ray through BVH nodes to find primitive intersections
        let mut to_visit_offset = 0;
        let mut current_node_index = 0;
        // Acts as a stack storing the nodes that still need to be visited
        let mut nodes_to_visit = [0; 64];

        loop {
            let node = &self.nodes[current_node_index];
            // Check ray against BVH node
            if node
                .bounds
                .intersect_p_cached(ray.o, ray.d, t_max, inv_dir, dir_is_neg)
            {
                if node.n_primitives > 0 {
                    // Leaf node; intersect ray with primitives in the node
                    for i in 0..node.n_primitives {
                        // TODO Actually, given this use case, let's not store the offset
                        // in a variant. The runtime cost of the match while extremely small
                        // is probably less important than saving the byte or whatever.
                        if self.primitives[node.primitive_offset + i as usize]
                            .as_ref()
                            .intersect_predicate(ray, t_max)
                        {
                            return true;
                        }
                    }
                    if to_visit_offset == 0 {
                        break;
                    }
                    to_visit_offset -= 1;
                    current_node_index = nodes_to_visit[to_visit_offset];
                } else {
                    // Interior node; put far BVH node on nodes_to_visit stack,
                    // advance to the near node
                    if dir_is_neg[node.axis as usize] != 0 {
                        nodes_to_visit[to_visit_offset] = current_node_index + 1;
                        to_visit_offset += 1;
                        current_node_index = node.second_child_offset;
                    } else {
                        nodes_to_visit[to_visit_offset] = node.second_child_offset;
                        to_visit_offset += 1;
                        current_node_index += 1;
                    }
                }
            } else {
                if to_visit_offset == 0 {
                    break;
                }
                to_visit_offset -= 1;
                current_node_index = nodes_to_visit[to_visit_offset];
            }
        }
        false
    }
}

impl BvhAggregate {
    pub fn new(
        mut primitives: Vec<Rc<Primitive>>,
        max_prims_in_node: usize,
        split_method: SplitMethod,
    ) -> BvhAggregate {
        debug_assert!(!primitives.is_empty());

        // Build BVH from primitives
        // Initialize bvh_primitives array to store compact information about primitives while building.
        let mut bvh_primitives: Vec<BvhPrimitive> = primitives
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

        // TODO Consider using MaybeUninit to avoid runtime costs of Option.
        // https://doc.rust-lang.org/std/mem/union.MaybeUninit.html#initializing-an-array-element-by-element
        // OR, we can reorganize this code to build ordered_primitives as we go (and use with_capacity(primitives.len()))
        // rather than initializing a pre-allocated vec.
        let mut ordered_primitives: Vec<Option<Rc<Primitive>>> = vec![None; primitives.len()];

        // Keeps track of the number of nodes created; this makes it possible to allocate exactly
        // the right size Vec for the LinearBvhNodes list.
        let total_nodes = AtomicUsize::new(0);

        // Build the BVH according to the selected split method
        let root = match split_method {
            // The default for all splitting methods but HLBVH will be to use build_recursive().
            _ => {
                // Keeps track of where the next free entry in ordered_primitives is, so that leaf nodes can claim
                // it as their starting primitive index, and reserve the next n slots for their n primitives.
                let ordered_prims_offset = AtomicUsize::new(0);
                Self::build_recursive(
                    &mut bvh_primitives,
                    &primitives,
                    &total_nodes,
                    &ordered_prims_offset,
                    &mut ordered_primitives,
                    split_method,
                )
            }
        };

        let mut ordered_primitives = ordered_primitives
            .into_iter()
            .map(|p| p.expect("Not all ordered primitives initialized in BVH construction!"))
            .collect_vec();

        // Swap so that primtives stores the ordered primitives.
        std::mem::swap(&mut primitives, &mut ordered_primitives);

        // This can be a significant chunk of memory that we don't need anymore; free it.
        drop(bvh_primitives);

        // TODO This has the same issue as the ordered_primitives above; we probably want a
        // similar solution to what we do there.
        let mut nodes: Vec<Option<LinearBvhNode>> = vec![None; total_nodes.load(Ordering::SeqCst)];

        let mut offset = 0;
        Self::flatten_bvh(&mut nodes, Some(Box::new(root)), &mut offset);
        debug_assert_eq!(total_nodes.load(Ordering::SeqCst), offset);

        let nodes = nodes
            .into_iter()
            .map(|n| n.expect("Not all nodes initialized in flatten_bvh()!"))
            .collect_vec();

        BvhAggregate {
            max_prims_in_node: usize::min(255, max_prims_in_node),
            primitives,
            split_method,
            nodes,
        }
    }

    /// Builds the BVH recursively; it is under the returned BvhBuildNode.
    ///
    /// bvh_primitives: The primitives that will be stored within the BVH under the returned BvhBuildNode.
    /// primitives: Stores the actual primitives; BvhPrimitive indexes into this.
    /// total_nodes: Tracks total number of nodes that are created.
    /// ordered_prims_offset: Tracks the current offset into ordered_prims, so that nodes
    ///   can reserve a chunk of it on their creation.
    /// ordered_prims: A vector with the same length as primitives; this function populates
    ///   ordered_prims with the primitives, but ordered such that each nodes' primitives are
    ///   contiguous, for efficient memory access. Passed in rather than constructed to enable
    ///   recursive behavior and to allow it to be initialized with the correct capacity from the start.
    /// split_metho: The algorithm by which we split the primitives
    fn build_recursive(
        bvh_primitives: &mut [BvhPrimitive],
        primitives: &Vec<Rc<Primitive>>,
        total_nodes: &AtomicUsize,
        ordered_prims_offset: &AtomicUsize,
        ordered_prims: &mut Vec<Option<Rc<Primitive>>>,
        split_method: SplitMethod,
    ) -> BvhBuildNode {
        debug_assert!(bvh_primitives.len() != 0);
        debug_assert!(ordered_prims.len() == primitives.len());

        let mut node = BvhBuildNode::default();

        total_nodes.fetch_add(1, Ordering::SeqCst);

        // Compute the bounds of all primitives in the primitive range.
        let bounds = bvh_primitives
            .iter()
            .fold(Bounds3f::new(Point3f::ZERO, Point3f::ZERO), |acc, p| {
                acc.union(&p.bounds)
            });

        if bounds.surface_area() == 0.0 || bvh_primitives.len() == 1 {
            // Create leaf BvhBuildNode
            let first_prim_offset =
                ordered_prims_offset.fetch_add(bvh_primitives.len(), Ordering::SeqCst);
            for i in 0..bvh_primitives.len() {
                let index = bvh_primitives[i].primitive_index;
                debug_assert!(ordered_prims[first_prim_offset + i].is_none());
                ordered_prims[first_prim_offset + i] = Some(primitives[index].clone());
            }
            node.init_leaf(first_prim_offset, bvh_primitives.len(), bounds);
            return node;
        }
        {
            // Compute bound of primitive centroids and choose split dimension to be the largest dimension of the bounds.
            let centroid_bounds = bvh_primitives
                .iter()
                .fold(Bounds3f::new(Point3f::ZERO, Point3f::ZERO), |acc, p| {
                    acc.union(&p.bounds)
                });
            let dim = centroid_bounds.max_dimension();

            if centroid_bounds.max[dim] == centroid_bounds.min[dim] {
                // Unusual edge case; create leaf BvhBuildNode
                let first_prim_offset =
                    ordered_prims_offset.fetch_add(bvh_primitives.len(), Ordering::SeqCst);
                for i in 0..bvh_primitives.len() {
                    let index = bvh_primitives[i].primitive_index;
                    debug_assert!(ordered_prims[first_prim_offset + i].is_none());
                    ordered_prims[first_prim_offset + i] = Some(primitives[index].clone());
                }
                node.init_leaf(first_prim_offset, bvh_primitives.len(), bounds);
                return node;
            } else {
                // Partition primitives based on the split method, and get the index of the split.
                let split_index = match split_method {
                    SplitMethod::Middle => {
                        let pmid = (centroid_bounds.min[dim] + centroid_bounds.max[dim]) / 2.0;
                        // TODO We want to partition bvh_primitives in place, which means we want
                        // a mutable slice, and we also want to use iter_mut().partition_in_place() which
                        // is nightly, or an equivalent...
                        let split_index =
                            partition(bvh_primitives.iter_mut(), |pi: &BvhPrimitive| {
                                pi.centroid()[dim] < pmid
                            });

                        if split_index == 0 || split_index == bvh_primitives.len() {
                            // If the primitives have large overlapping bounding boxes, this may
                            // fail to partition them; in which case, split by equal counts instead.
                            split_index
                            // TODO Actually implement the EqualCounts case and do that here instead
                            //  of returning split_index
                        } else {
                            split_index
                        }
                    }
                };

                // TODO This can be done in parallel, but let's do it sequentially for now.
                // C++ is quite happy to take disjoint subspans and operate on them in different threads,
                // which is fine if we know that they truly are disjoint.
                // Rust will likely complain if I try to do that here via some naive operations.
                // Luckily, Rayon would handle this for us.
                // https://doc.rust-lang.org/std/primitive.slice.html#method.split_at_mut

                let left_child = Box::new(Self::build_recursive(
                    &mut bvh_primitives[0..split_index],
                    primitives,
                    total_nodes,
                    ordered_prims_offset,
                    ordered_prims,
                    split_method,
                ));

                let right_child = Some(Box::new(Self::build_recursive(
                    &mut bvh_primitives[split_index..],
                    primitives,
                    total_nodes,
                    ordered_prims_offset,
                    ordered_prims,
                    split_method,
                )));

                node.init_interior(dim as u8, left_child, right_child)
            }
        }

        node
    }

    // Performs a depth-first traversal and stores the nodes in memory in linear order.
    // offset tracks the current offset into the linear_nodes array.
    // node is the root of the tree; this function is recursive, so it is also the root of sub-trees
    // as this traverses down the tree.
    fn flatten_bvh(
        linear_nodes: &mut Vec<Option<LinearBvhNode>>,
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
            let linear_nod_n_primitives = node.n_primitives as u16;
            LinearBvhNode {
                bounds: linear_node_bounds,
                primitive_offset: node.first_prim_offset,
                second_child_offset: 0,
                n_primitives: linear_nod_n_primitives,
                axis: 0,
            }
        } else {
            // Create interior flattened BVH node.
            let linear_node_axis = node.split_axis as u8;
            let linear_node_n_primitives = 0;
            Self::flatten_bvh(linear_nodes, node.left_child, offset);
            let second_child_offset = Self::flatten_bvh(linear_nodes, node.right_child, offset);
            LinearBvhNode {
                bounds: linear_node_bounds,
                primitive_offset: 0,
                second_child_offset,
                n_primitives: linear_node_n_primitives,
                axis: linear_node_axis,
            }
        };
        debug_assert!(linear_nodes[node_offset].is_none());
        linear_nodes[node_offset] = Some(linear_node);

        node_offset
    }
}

/// A BVH Node that is stored in a compact, linear representation of a BVH.
#[repr(align(32))]
#[derive(Debug, Copy, Clone)]
pub struct LinearBvhNode {
    bounds: Bounds3f,
    primitive_offset: usize,
    /// Only the second child offset is needed, as the first child
    /// immediately follows the parent in the flat vector.
    second_child_offset: usize,
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
    n_primitives: usize,
}

impl Default for BvhBuildNode {
    fn default() -> Self {
        Self {
            bounds: Default::default(),
            left_child: Default::default(),
            right_child: Default::default(),
            split_axis: Default::default(),
            first_prim_offset: Default::default(),
            n_primitives: Default::default(),
        }
    }
}

impl BvhBuildNode {
    pub fn init_leaf(&mut self, first_prim_offset: usize, n_primitives: usize, bounds: Bounds3f) {
        self.bounds = bounds;
        self.left_child = None;
        self.right_child = None;
        self.split_axis = 0;
        self.first_prim_offset = first_prim_offset;
        self.n_primitives = n_primitives;
    }

    pub fn init_interior(
        &mut self,
        split_axis: u8,
        left_child: Box<BvhBuildNode>,
        right_child: Option<Box<BvhBuildNode>>,
    ) {
        // Interior nodes always have at least one child.
        let bounds = if let Some(right) = &right_child {
            left_child.bounds.union(&right.bounds)
        } else {
            left_child.bounds
        };
        self.bounds = bounds;
        self.left_child = Some(left_child);
        self.right_child = right_child;
        self.split_axis = split_axis;
        // The primitive offset and num_primitives are 0 because interior nodes do not contain primitives.
        self.first_prim_offset = 0;
        self.n_primitives = 0;
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use float_cmp::assert_approx_eq;

    use crate::{
        aggregate::{BvhAggregate, SplitMethod},
        material::{DiffuseMaterial, Material},
        primitive::{Primitive, PrimitiveI, SimplePrimitive},
        ray::Ray,
        shape::{Shape, Sphere},
        spectra::{ConstantSpectrum, Spectrum},
        texture::SpectrumConstantTexture,
        transform::Transform,
        vecmath::{normal::Normal3, Normal3f, Point3f, Vector3f},
        Float,
    };

    #[test]
    fn single_primitive_bvh() {
        let radius = 1.0;
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            radius,
            -radius,
            radius,
            360.0,
        );
        let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
        let kd = crate::texture::SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
        let material = Rc::new(Material::Diffuse(DiffuseMaterial::new(kd)));
        let prim = Rc::new(Primitive::Simple(SimplePrimitive {
            shape: Shape::Sphere(sphere),
            material,
        }));
        let expected_bounds = prim.as_ref().bounds();
        let prims = vec![prim];
        let bvh = BvhAggregate::new(prims, 1, crate::aggregate::SplitMethod::Middle);
        // The BVH boudning box should match the bounding box of the primitive
        assert_eq!(expected_bounds, bvh.bounds());
        assert_eq!(1, bvh.max_prims_in_node);
        assert_eq!(SplitMethod::Middle, bvh.split_method);
    }

    #[test]
    fn single_primitive_bvh_intersetion() {
        let radius = 1.0;
        let sphere = Sphere::new(
            Transform::default(),
            Transform::default(),
            false,
            radius,
            -radius,
            radius,
            360.0,
        );
        let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
        let kd = crate::texture::SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
        let material = Rc::new(Material::Diffuse(DiffuseMaterial::new(kd)));
        let prim = Rc::new(Primitive::Simple(SimplePrimitive {
            shape: Shape::Sphere(sphere),
            material,
        }));
        let prims = vec![prim];
        let bvh = BvhAggregate::new(prims, 1, crate::aggregate::SplitMethod::Middle);

        let ray = Ray::new(Point3f::NEG_X * 5.0, Vector3f::X, None);
        let si = bvh.intersect(&ray, Float::INFINITY);
        assert!(si.is_some());
        let si = si.unwrap();
        // The normal should be in the negative X direction (we're hitting a sphere head-on in the positive X direction)
        assert_approx_eq!(Float, si.intr.shading.n.dot(&Normal3f::NEG_X), 1.0);
        // Ray started at -5, radius is 1, so it hits at 4.0.
        assert_approx_eq!(Float, si.t_hit, 4.0);
        // The hit should be at just about (-1, 0, 0)
        assert_approx_eq!(Float, si.intr.p().x, -1.0, epsilon = 0.000001);
        assert_approx_eq!(Float, si.intr.p().y, 0.0);
        assert_approx_eq!(Float, si.intr.p().y, 0.0);
    }

    #[test]
    fn set_of_spheres() {
        let mut prims: Vec<Rc<Primitive>> = Vec::new();
        for multiplier in [-3.5, 0.0, 5.0] {
            let radius = 1.0;

            let x_translate = Transform::translate(Vector3f::X * multiplier);
            let x_translate_inv = x_translate.inverse();

            let sphere = Sphere::new(
                x_translate,
                x_translate_inv,
                false,
                radius,
                -radius,
                radius,
                360.0,
            );
            let cs = Spectrum::Constant(ConstantSpectrum::new(0.5));
            let kd =
                crate::texture::SpectrumTexture::Constant(SpectrumConstantTexture { value: cs });
            let material = Rc::new(Material::Diffuse(DiffuseMaterial::new(kd)));
            let prim = Rc::new(Primitive::Simple(SimplePrimitive {
                shape: Shape::Sphere(sphere),
                material,
            }));
            prims.push(prim);
        }

        let bvh = BvhAggregate::new(prims, 1, crate::aggregate::SplitMethod::Middle);

        let ray = Ray::new(Point3f::NEG_X * 10.0, Vector3f::X, None);

        let si = bvh.intersect(&ray, Float::INFINITY);

        assert!(si.is_some());
        let si = si.unwrap();
        // The normal should be in the negative X direction (we're hitting a sphere head-on in the positive X direction)
        assert_approx_eq!(Float, si.intr.shading.n.dot(&Normal3f::NEG_X), 1.0);
        // Ray started at -10, radius is 1 pushing the closest sphere's position to -3.5 - 1.0 == -4.5, so it hits at 5.5.
        assert_approx_eq!(Float, si.t_hit, 5.5, epsilon = 0.00001);
        // The hit should be at just about (-4.5, 0, 0), as center is at (-3.5, 0, 0) and it has a radius of 1.
        assert_approx_eq!(Float, si.intr.p().x, -4.5, epsilon = 0.00001);
        assert_approx_eq!(Float, si.intr.p().y, 0.0);
        assert_approx_eq!(Float, si.intr.p().y, 0.0);

        let hit = bvh.intersect_predicate(&ray, Float::INFINITY);
        assert!(hit);

        // This ray should not intersect the spheres.
        let ray = Ray::new(
            Point3f::NEG_X * 10.0 + Vector3f::Z * 1.001,
            Vector3f::X,
            None,
        );
        let si = bvh.intersect(&ray, Float::INFINITY);
        assert!(si.is_none());
        let hit = bvh.intersect_predicate(&ray, Float::INFINITY);
        assert!(!hit);
    }
}