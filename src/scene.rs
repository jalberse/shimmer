use std::{
    collections::{HashMap, HashSet},
    ops::{Index, IndexMut},
    sync::Arc,
};

use crate::{
    camera::{Camera, CameraTransform},
    colorspace::RgbColorSpace,
    film::{Film, FilmI},
    filter::Filter,
    options::Options,
    paramdict::ParameterDictionary,
    parser::{FileLoc, ParsedParameterVector, ParserTarget},
    sampler::Sampler,
    square_matrix::SquareMatrix,
    transform::Transform,
    vecmath::{Point3f, Tuple3, Vector3f},
    Float,
};

use log::warn;
use string_interner::{symbol::SymbolU32, StringInterner};

pub struct BasicScene {
    pub integrator: SceneEntity,
    pub accelerator: SceneEntity,
    pub film_color_space: Arc<RgbColorSpace>,
    pub shapes: Vec<ShapeSceneEntity>,
    pub instances: Vec<InstanceSceneEntity>,
    pub instance_definitions: Vec<InstanceDefinitionSceneEntity>,
    camera: Camera,
    film: Film,
    sampler: Sampler,
    named_materials: Vec<(String, SceneEntity)>,
    materials: Vec<SceneEntity>,
    area_lights: Vec<SceneEntity>,
}

impl BasicScene {
    pub fn set_options(
        &mut self,
        mut filter: SceneEntity,
        mut film: SceneEntity,
        mut camera: CameraSceneEntity,
        mut sampler: SceneEntity,
        integ: SceneEntity,
        accel: SceneEntity,
        string_interner: &StringInterner,
        options: &Options,
    ) {
        self.film_color_space = film.parameters.color_space.clone();
        self.integrator = integ;
        self.accelerator = accel;

        let filt = Filter::create(
            &string_interner
                .resolve(filter.name)
                .expect("Unresolved name!"),
            &mut filter.parameters,
            &mut filter.loc,
        );

        let exposure_time = camera.base.parameters.get_one_float("shutterclose", 1.0)
            - camera.base.parameters.get_one_float("shutteropen", 0.0);

        if exposure_time <= 0.0 {
            panic!(
                "{} The specified camera shutter times imply the camera won't open.",
                camera.base.loc
            );
        }

        self.film = Film::create(
            &string_interner.resolve(film.name).unwrap(),
            &mut film.parameters,
            exposure_time,
            &camera.camera_transform,
            filt,
            &film.loc,
            options,
        );

        let res = self.film.full_resolution();
        self.sampler = Sampler::create(
            &string_interner.resolve(sampler.name).unwrap(),
            &mut sampler.parameters,
            res,
            options,
            &mut sampler.loc,
        );

        self.camera = Camera::create(
            &string_interner.resolve(camera.base.name).unwrap(),
            &mut camera.base.parameters,
            None,
            camera.camera_transform,
            self.film.clone(),
            options,
            &mut camera.base.loc,
        );
    }

    fn add_named_material(&mut self, name: &str, material: SceneEntity) {
        todo!()
    }

    fn add_material(&mut self, material: SceneEntity) {
        todo!()
    }

    fn add_medium(&mut self, medium: SceneEntity) {
        todo!()
    }

    fn add_float_texture(&mut self, name: &str, texture: TextureSceneEntity) {
        todo!()
    }

    fn add_spectrum_texture(&mut self, name: &str, texture: TextureSceneEntity) {
        todo!()
    }

    fn add_light(&mut self, light: LightSceneEntity, string_interner: &StringInterner) {
        todo!()
    }

    fn add_area_light(&mut self, light: SceneEntity) {
        todo!()
    }

    fn add_shapes(&mut self, shapes: &[ShapeSceneEntity]) {
        todo!()
    }

    // TODO add_animated_shapes().

    fn add_instance_definition(&mut self, instance: InstanceDefinitionSceneEntity) {
        todo!()
    }

    fn add_instance_uses(&mut self, instances: &[InstanceSceneEntity]) {
        todo!()
    }

    fn done(&mut self) {
        todo!()
    }
}

/// All objects in the scene are described by various *Entity classes.
/// The SceneEntity is the simplest; it records the name of the entity,
/// the file location of the of the associated statement in the scene description,
/// and any user-provided parameters.
/// It is used for the film, sampler, integrator, pixel filter, and accelerator,
/// and is also used as a "base" for some of the other scene entity types.
#[derive(Debug, Clone)]
pub struct SceneEntity {
    name: SymbolU32,
    loc: FileLoc,
    parameters: ParameterDictionary,
}

impl SceneEntity {
    pub fn new(
        name: &str,
        loc: FileLoc,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner,
    ) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            parameters,
        }
    }
}

pub enum MaterialRef {
    Index(i32),
    Name(String),
}

pub struct ShapeSceneEntity {
    base: SceneEntity,
    render_from_object: Arc<Transform>,
    object_from_render: Arc<Transform>,
    reverse_orientation: bool,
    material_ref: MaterialRef,
    light_index: i32,
    inside_medium: String,
    outside_medium: String,
}

#[derive(Debug, Clone)]
pub struct CameraSceneEntity {
    base: SceneEntity,
    camera_transform: CameraTransform,
    // TODO medium: String,
}

pub struct TransformedSceneEntity {
    base: SceneEntity,
    // TODO this may need to be an AnimatedTransform
    render_from_object: Transform,
}

impl TransformedSceneEntity {
    pub fn new(
        name: &str,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        render_from_object: Transform,
    ) -> Self {
        Self {
            base: SceneEntity::new(name, loc, parameters, string_interner),
            render_from_object,
        }
    }
}

pub type MediumSceneEntity = TransformedSceneEntity;
pub type TextureSceneEntity = TransformedSceneEntity;

pub struct LightSceneEntity {
    base: TransformedSceneEntity,
    medium: String,
}

impl LightSceneEntity {
    fn new(
        name: &str,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        render_from_light: Transform,
        medium: &str,
    ) -> LightSceneEntity {
        let base =
            TransformedSceneEntity::new(name, parameters, string_interner, loc, render_from_light);
        LightSceneEntity {
            base,
            medium: medium.to_owned(),
        }
    }
}

pub struct InstanceSceneEntity {
    name: SymbolU32,
    loc: FileLoc,
    render_from_instance: Transform,
    // TODO Possibly aniamted transform
}

pub struct InstanceDefinitionSceneEntity {
    // TODO we will need a stringinterner on this, same as in SceneEntity
    name: SymbolU32,
    loc: FileLoc,
    shapes: Vec<ShapeSceneEntity>,
    // TODO aniamted_shapes: Vec<AnimatedShapeSceneEntity>,
}

const MAX_TRANSFORMS: usize = 2;

#[derive(Debug, Copy, Clone, PartialEq)]
struct TransformSet {
    t: [Transform; MAX_TRANSFORMS],
}

impl TransformSet {
    fn inverse(ts: &TransformSet) -> TransformSet {
        let mut t_inv = TransformSet::default();
        for i in 0..MAX_TRANSFORMS {
            t_inv.t[i] = Transform::inverse(&ts.t[i]);
        }
        t_inv
    }

    fn is_animated(&self) -> bool {
        for i in 0..(MAX_TRANSFORMS - 1) {
            if self.t[i] != self.t[i + 1] {
                return true;
            }
        }
        false
    }
}

impl Default for TransformSet {
    fn default() -> Self {
        Self {
            t: Default::default(),
        }
    }
}

impl Index<usize> for TransformSet {
    type Output = Transform;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < MAX_TRANSFORMS);
        &self.t[index]
    }
}

impl IndexMut<usize> for TransformSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < MAX_TRANSFORMS);
        &mut self.t[index]
    }
}

struct GraphicsState {
    current_inside_medium: String,
    current_outside_medium: String,

    current_material_index: i32,
    current_named_material: String,

    area_light_name: String,
    area_light_params: ParameterDictionary,
    area_light_loc: FileLoc,

    shape_attributes: ParsedParameterVector,
    light_attributes: ParsedParameterVector,
    material_attributes: ParsedParameterVector,
    medium_attributes: ParsedParameterVector,
    texture_attributes: ParsedParameterVector,
    reverse_orientation: bool,
    color_space: Arc<RgbColorSpace>,
    ctm: TransformSet,
    active_transform_bits: u32,
    transform_start_time: Float,
    transform_end_time: Float,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            current_inside_medium: Default::default(),
            current_outside_medium: Default::default(),
            current_material_index: 0,
            current_named_material: Default::default(),
            area_light_name: Default::default(),
            area_light_params: Default::default(),
            area_light_loc: Default::default(),
            shape_attributes: Default::default(),
            light_attributes: Default::default(),
            material_attributes: Default::default(),
            medium_attributes: Default::default(),
            texture_attributes: Default::default(),
            reverse_orientation: Default::default(),
            color_space: RgbColorSpace::get_named(crate::colorspace::NamedColorSpace::SRGB).clone(),
            ctm: Default::default(),
            active_transform_bits: Default::default(),
            transform_start_time: 0.0,
            transform_end_time: 1.0,
        }
    }
}

impl GraphicsState {
    pub fn for_active_transforms(&mut self, func: impl Fn(&mut Transform)) {
        for i in 0..MAX_TRANSFORMS {
            if self.active_transform_bits & (1 << i) != 0 {
                func(&mut self.ctm[i]);
            }
        }
    }
}

enum BlockState {
    OptionsBlock,
    WorldBlock,
}

pub struct BasicSceneBuilder {
    scene: Box<BasicScene>,
    current_block: BlockState,
    graphics_state: GraphicsState,
    named_coordinate_systems: HashMap<String, TransformSet>,
    render_from_world: Transform,
    // TODO Transform cache - we will implement this in the future.
    pushed_graphics_states: Vec<GraphicsState>,
    // TODO Should the push_stack be an enum instead? Natural representation, each variant can store a FileLoc too.
    push_stack: Vec<(u8, FileLoc)>, // 'a' attribute, 'o' object
    // TODO instance definition stuff

    // Buffered for consistent ordering across runs
    shapes: Vec<ShapeSceneEntity>,
    instance_uses: Vec<InstanceSceneEntity>,

    named_material_names: HashSet<String>,
    medium_names: HashSet<String>,
    float_texture_names: HashSet<String>,
    spectrum_texture_names: HashSet<String>,
    instance_names: HashSet<String>,

    current_material_index: i32,
    current_light_index: i32,
    sampler: SceneEntity,
    film: SceneEntity,
    integrator: SceneEntity,
    filter: SceneEntity,
    accelerator: SceneEntity,
    camera: CameraSceneEntity,
}

impl BasicSceneBuilder {
    const START_TRANSFORM_BITS: u32 = 1 << 0;
    const END_TRANSFORM_BITS: u32 = 1 << 1;
    const ALL_TRANSFORM_BITS: u32 = (1 << MAX_TRANSFORMS) - 1;

    fn render_from_object(&self) -> Transform {
        // TODO Want to create a version for AnimatedTransform that uses both of ctm.
        self.render_from_world * self.graphics_state.ctm[0]
    }
}

impl ParserTarget for BasicSceneBuilder {
    fn shape(&mut self, name: &str, params: ParsedParameterVector, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn option(&mut self, name: &str, value: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn identity(&mut self, _loc: crate::parser::FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| *t = Transform::default());
    }

    fn translate(
        &mut self,
        dx: crate::Float,
        dy: crate::Float,
        dz: crate::Float,
        _loc: crate::parser::FileLoc,
    ) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = *t * Transform::translate(Vector3f::new(dx, dy, dz))
            });
    }

    fn scale(
        &mut self,
        sx: crate::Float,
        sy: crate::Float,
        sz: crate::Float,
        _loc: crate::parser::FileLoc,
    ) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| *t = *t * Transform::scale(sx, sy, sz));
    }

    fn rotate(
        &mut self,
        angle: crate::Float,
        ax: crate::Float,
        ay: crate::Float,
        az: crate::Float,
        _loc: crate::parser::FileLoc,
    ) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = *t * Transform::rotate(angle, &Vector3f::new(ax, ay, az))
            });
    }

    fn look_at(
        &mut self,
        ex: crate::Float,
        ey: crate::Float,
        ez: crate::Float,
        lx: crate::Float,
        ly: crate::Float,
        lz: crate::Float,
        ux: crate::Float,
        uy: crate::Float,
        uz: crate::Float,
        _loc: crate::parser::FileLoc,
    ) {
        let transform = Transform::look_at(
            &Point3f::new(ex, ey, ez),
            &Point3f::new(lx, ly, lz),
            &Vector3f::new(ux, uy, uz),
        );
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| *t = *t * transform);
    }

    fn transform(&mut self, transform: [crate::Float; 16], _loc: crate::parser::FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = Transform::transpose(&Transform::new_calc_inverse(SquareMatrix::<4>::from(
                    transform.as_slice(),
                )));
            })
    }

    fn concat_transform(&mut self, transform: [crate::Float; 16], _loc: crate::parser::FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = *t
                    * Transform::transpose(&Transform::new_calc_inverse(SquareMatrix::<4>::from(
                        transform.as_slice(),
                    )));
            })
    }

    fn coordinate_system(&mut self, name: &str, loc: crate::parser::FileLoc) {
        // TODO Normalize name to UTF-8.
        self.named_coordinate_systems
            .insert(name.to_owned(), self.graphics_state.ctm.clone());
    }

    fn coordinate_sys_transform(&mut self, name: &str, loc: crate::parser::FileLoc) {
        // TODO Normalize name to UTF-8.
        if let Some(ctm) = self.named_coordinate_systems.get(name) {
            self.graphics_state.ctm = ctm.clone();
        } else {
            warn!("{}: Couldn't find named coordinate system {}.", loc, name);
        }
    }

    fn active_transform_all(&mut self, _loc: crate::parser::FileLoc) {
        self.graphics_state.active_transform_bits = Self::ALL_TRANSFORM_BITS;
    }

    fn active_transform_end_time(&mut self, _loc: crate::parser::FileLoc) {
        self.graphics_state.active_transform_bits = Self::END_TRANSFORM_BITS;
    }

    fn active_transform_start_time(&mut self, _loc: crate::parser::FileLoc) {
        self.graphics_state.active_transform_bits = Self::START_TRANSFORM_BITS;
    }

    fn transform_times(
        &mut self,
        start: crate::Float,
        end: crate::Float,
        _loc: crate::parser::FileLoc,
    ) {
        // TODO verify options
        self.graphics_state.transform_start_time = start;
        self.graphics_state.transform_end_time = end;
    }

    fn color_space(
        &mut self,
        n: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn pixel_filter(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.filter = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn film(
        &mut self,
        film_type: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.film = SceneEntity::new(film_type, loc, dict, string_interner);
    }

    fn accelerator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.accelerator = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn integrator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.integrator = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn camera(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
        options: &Options,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());

        let camera_from_world = &self.graphics_state.ctm;
        let world_from_camera = TransformSet::inverse(&camera_from_world);
        // TODO Animated transform
        let camera_transform = CameraTransform::new(&world_from_camera[0], options);
        self.named_coordinate_systems
            .insert("camera".to_owned(), world_from_camera);

        self.render_from_world = camera_transform.render_from_world();

        self.camera = CameraSceneEntity {
            base: SceneEntity::new(name, loc, dict, string_interner),
            camera_transform,
        };
    }

    fn make_named_medium(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn medium_interface(
        &mut self,
        inside_name: &str,
        outside_name: &str,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn sampler(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.sampler = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn world_begin(
        &mut self,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
        options: &Options,
    ) {
        self.current_block = BlockState::WorldBlock;
        for i in 0..MAX_TRANSFORMS {
            self.graphics_state.ctm[i] = Transform::default();
        }
        self.graphics_state.active_transform_bits = Self::ALL_TRANSFORM_BITS;
        self.named_coordinate_systems
            .insert("world".to_owned(), self.graphics_state.ctm.clone());

        // Pass pre-world-begin entities to the scene
        self.scene.set_options(
            self.filter.clone(),
            self.film.clone(),
            self.camera.clone(),
            self.sampler.clone(),
            self.integrator.clone(),
            self.accelerator.clone(),
            string_interner,
            options,
        );
    }

    fn attribute_begin(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn attribute_end(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn attribute(
        &mut self,
        target: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn texture(
        &mut self,
        name: &str,
        texture_type: &str,
        tex_name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn material(&mut self, name: &str, params: ParsedParameterVector, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn make_named_material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn named_material(&mut self, name: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.light_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );
        self.scene.add_light(
            LightSceneEntity::new(
                name,
                dict,
                string_interner,
                loc,
                self.render_from_object(),
                &self.graphics_state.current_outside_medium,
            ),
            &string_interner,
        );
    }

    fn area_light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        // TODO Verify world
        self.graphics_state.area_light_name = name.to_owned();
        self.graphics_state.area_light_params = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.light_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );
        self.graphics_state.area_light_loc = loc;
    }

    fn reverse_orientation(&mut self, _loc: crate::parser::FileLoc) {
        // TODO verify world
        self.graphics_state.reverse_orientation = !self.graphics_state.reverse_orientation;
    }

    fn object_begin(&mut self, name: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn object_end(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn object_instance(&mut self, name: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn end_of_files(&mut self) {
        todo!()
    }
}
