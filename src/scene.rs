use std::{
    collections::{HashMap, HashSet},
    ops::{Index, IndexMut},
    path::Path,
    sync::{atomic, Arc},
};

use crate::{
    camera::{Camera, CameraTransform},
    color::LinearColorEncoding,
    colorspace::RgbColorSpace,
    film::{Film, FilmI},
    filter::Filter,
    image::Image,
    options::Options,
    paramdict::ParameterDictionary,
    parser::{FileLoc, ParsedParameterVector, ParserTarget},
    sampler::Sampler,
    square_matrix::SquareMatrix,
    transform::Transform,
    util::normalize_arg,
    vecmath::{Point3f, Tuple3, Vector3f},
    Float,
};

use log::warn;
use string_interner::{symbol::SymbolU32, StringInterner};

// TODO If/when we make this multi-threaded, most of these will be within a Mutex.
//      For now, code it sequentially.
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
    normal_maps: HashMap<String, Box<Image>>,
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
        // TODO, alright, we can just have a load_normal_map() fn that loads the image sequentially.
        //    We can eventually make it async, but for now, just do it sequentially.
        todo!()
    }

    // Returns the new material index
    fn add_material(&mut self, material: SceneEntity) -> i32 {
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

    /// Returns the new area light index.
    fn add_area_light(&mut self, light: SceneEntity) -> i32 {
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

    fn load_normal_map(&mut self, parameters: &mut ParameterDictionary) {
        let normal_map_filename = parameters.get_one_string("normalmap", "".to_string());
        let filename = Path::new(&normal_map_filename);
        if !filename.exists() {
            warn!("Normal map \"{}\" not found.", filename.display());
        }

        let image_and_metadata = Image::read(
            filename,
            Some(crate::color::ColorEncoding::Linear(LinearColorEncoding {})),
        );

        let image = image_and_metadata.image;
        let rgb_desc = image.get_channel_desc(&["R".to_owned(), "G".to_owned(), "B".to_owned()]);
        if rgb_desc.size() != 3 {
            panic!(
                "Normal map \"{}\" should have RGB channels.",
                filename.display()
            );
        }
        let image = Box::new(image);
        self.normal_maps.insert(normal_map_filename, image);
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

pub struct ShapeSceneEntity {
    base: SceneEntity,
    render_from_object: Arc<Transform>,
    object_from_render: Arc<Transform>,
    reverse_orientation: bool,
    // TODO It should be one of these two - enum?
    // It just makes instatiation a bit more complex
    material_index: i32,
    mateiral_name: String,
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

impl InstanceSceneEntity {
    pub fn new(
        name: &str,
        loc: FileLoc,
        string_interner: &mut StringInterner,
        render_from_instance: Transform,
    ) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            render_from_instance,
        }
    }
}

pub struct InstanceDefinitionSceneEntity {
    // TODO we will need a stringinterner on this, same as in SceneEntity
    name: SymbolU32,
    loc: FileLoc,
    shapes: Vec<ShapeSceneEntity>,
    // TODO aniamted_shapes: Vec<AnimatedShapeSceneEntity>,
}

impl InstanceDefinitionSceneEntity {
    pub fn new(name: &str, loc: FileLoc, string_interner: &mut StringInterner) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            shapes: Default::default(),
        }
    }
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

#[derive(Debug, Clone)]
struct GraphicsState {
    current_inside_medium: String,
    current_outside_medium: String,

    // TODO It's one or the other - use an enum.
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

#[derive(Debug, Copy, Clone, PartialEq)]
enum BlockState {
    OptionsBlock,
    WorldBlock,
}

struct ActiveInstanceDefinition {
    pub active_imports: atomic::AtomicI32,
    pub entity: InstanceDefinitionSceneEntity,
    pub parent: Option<Arc<ActiveInstanceDefinition>>,
}

impl ActiveInstanceDefinition {
    pub fn new(name: &str, loc: FileLoc, string_interner: &mut StringInterner) -> Self {
        Self {
            active_imports: atomic::AtomicI32::new(1),
            entity: InstanceDefinitionSceneEntity::new(name, loc, string_interner),
            parent: None,
        }
    }
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
    sampler: SceneEntity,
    film: SceneEntity,
    integrator: SceneEntity,
    filter: SceneEntity,
    accelerator: SceneEntity,
    camera: CameraSceneEntity,

    active_instance_definition: Option<ActiveInstanceDefinition>,
}

impl BasicSceneBuilder {
    const START_TRANSFORM_BITS: u32 = 1 << 0;
    const END_TRANSFORM_BITS: u32 = 1 << 1;
    const ALL_TRANSFORM_BITS: u32 = (1 << MAX_TRANSFORMS) - 1;

    fn render_from_object(&self) -> Transform {
        // TODO Want to create a version for AnimatedTransform that uses both of ctm.
        self.render_from_world * self.graphics_state.ctm[0]
    }

    fn ctm_is_animated(&self) -> bool {
        self.graphics_state.ctm.is_animated()
    }
}

impl ParserTarget for BasicSceneBuilder {
    fn shape(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        // TODO Verify world
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.shape_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        let area_light_index = if self.graphics_state.area_light_name.is_empty() {
            -1
        } else {
            if self.active_instance_definition.is_some() {
                warn!("{} Area lights not supported with object instancing", loc)
            }
            self.scene.add_area_light(SceneEntity::new(
                &self.graphics_state.area_light_name,
                self.graphics_state.area_light_loc.clone(),
                self.graphics_state.area_light_params.clone(),
                string_interner,
            ))
        };

        if self.ctm_is_animated() {
            todo!(); // Not yet implemented! We don't have animated transforms yet.
        } else {
            // TODO Use transform cache.
            let render_from_object = self.render_from_object();
            let object_from_render = Transform::inverse(&render_from_object);

            let entity = ShapeSceneEntity {
                base: SceneEntity::new(name, loc, dict, string_interner),
                render_from_object: Arc::new(render_from_object),
                object_from_render: Arc::new(object_from_render),
                reverse_orientation: self.graphics_state.reverse_orientation,
                material_index: self.graphics_state.current_material_index,
                mateiral_name: self.graphics_state.current_named_material.clone(),
                light_index: area_light_index,
                inside_medium: self.graphics_state.current_inside_medium.clone(),
                outside_medium: self.graphics_state.current_outside_medium.clone(),
            };
            if let Some(active_instance_definition) = &mut self.active_instance_definition {
                active_instance_definition.entity.shapes.push(entity)
            } else {
                self.shapes.push(entity)
            }
        }
    }

    fn option(
        &mut self,
        name: &str,
        value: &str,
        options: &mut Options,
        loc: crate::parser::FileLoc,
    ) {
        let name = normalize_arg(name);

        match name.as_str() {
            "disablepixeljitter" => match value {
                "true" => options.disable_pixel_jitter = true,
                "false" => options.disable_pixel_jitter = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "disabletexturefiltering" => match value {
                "true" => options.disable_texture_filtering = true,
                "false" => options.disable_texture_filtering = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "disablewavelengthjitter" => match value {
                "true" => options.disable_wavelength_jitter = true,
                "false" => options.disable_wavelength_jitter = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "displacementedgescale" => {
                options.displacement_edge_scale = value
                    .parse()
                    .unwrap_or_else(|_| panic!("{} Unable to parse option value {}", loc, value));
            }
            "msereferenceimage" => {
                if value.len() < 3 || !value.starts_with("\"") || !value.ends_with('\"') {
                    panic!("{} Expected quotes string for option value {}", loc, value);
                }
                options.mse_reference_image = value[1..value.len() - 1].to_owned();
            }
            "msereferenceout" => {
                if value.len() < 3 || !value.starts_with("\"") || !value.ends_with('\"') {
                    panic!("{} Expected quotes string for option value {}", loc, value);
                }
                options.mse_reference_output = value[1..value.len() - 1].to_owned();
            }
            "rendercoordsys" => {
                if value.len() < 3 || !value.starts_with("\"") || !value.ends_with("\"") {
                    panic!("{} Expected quotes string for option value {}", loc, value);
                }
                let render_coord_sys = value[1..value.len() - 1].to_owned();
                match render_coord_sys.as_str() {
                    "camera" => {
                        options.rendering_coord_system =
                            crate::options::RenderingCoordinateSystem::Camera
                    }
                    "cameraworld" => {
                        options.rendering_coord_system =
                            crate::options::RenderingCoordinateSystem::CameraWorld
                    }
                    "world" => {
                        options.rendering_coord_system =
                            crate::options::RenderingCoordinateSystem::World
                    }
                    _ => panic!("{} Unknown option value {}", loc, value),
                }
            }
            "seed" => {
                options.seed = value
                    .parse()
                    .unwrap_or_else(|_| panic!("{} Unable to parse option value {}", loc, value));
            }
            "forcediffuse" => match value {
                "true" => options.force_diffuse = true,
                "false" => options.force_diffuse = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "pixelstats" => match value {
                "true" => options.record_pixel_statistics = true,
                "false" => options.record_pixel_statistics = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "wavefront" => match value {
                "true" => options.wavefront = true,
                "false" => options.wavefront = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            _ => panic!("{} Unknown option {}", loc, name),
        }
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
        _params: ParsedParameterVector,
        _string_interner: &mut StringInterner,
        _loc: crate::parser::FileLoc,
    ) {
        let cs = RgbColorSpace::get_named(n.into());
        self.graphics_state.color_space = cs.clone();
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
        // TODO Verify world
        self.pushed_graphics_states
            .push(self.graphics_state.clone());
        self.push_stack.push(('a' as u8, loc));
    }

    fn attribute_end(&mut self, loc: crate::parser::FileLoc) {
        // TODO Verify world
        if self.push_stack.is_empty() {
            panic!("{} Unmatched attribute_end statement.", loc);
        }

        // Note: Must keep the following consistent with code in ObjectEnd.
        self.graphics_state = self.pushed_graphics_states.pop().unwrap();

        if self.push_stack.last().unwrap().0 == 'o' as u8 {
            panic!(
                "{} Masmatched nesting: open ObjectBegin from {} at attribute_end.",
                loc,
                self.push_stack.last().unwrap().1
            );
        } else {
            assert!(self.push_stack.last().unwrap().0 == 'a' as u8);
        }
    }

    fn attribute(
        &mut self,
        target: &str,
        mut attrib: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        let current_attributes = match target {
            "shape" => &mut self.graphics_state.shape_attributes,
            "light" => &mut self.graphics_state.light_attributes,
            "material" => &mut self.graphics_state.material_attributes,
            "medium" => &mut self.graphics_state.medium_attributes,
            "texture" => &mut self.graphics_state.texture_attributes,
            _ => panic!("{} Unknown attribute target {}", loc, target),
        };

        // We hold onto the curent color space and associate it with the parameters.
        for p in attrib.iter_mut() {
            p.may_be_unused = true;
            p.color_space = Some(self.graphics_state.color_space.clone());
            current_attributes.push(p.to_owned())
        }
    }

    fn texture(
        &mut self,
        name: &str,
        texture_type: &str,
        tex_name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        // TODO Normalize name to UTF8
        // TODO Verify world

        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.texture_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if texture_type != "float" && texture_type != "spectrum" {
            panic!("{} Texture type \"{}\" unknown.", loc, texture_type);
        }

        let names = if texture_type == "float" {
            &mut self.float_texture_names
        } else {
            &mut self.spectrum_texture_names
        };

        if names.insert(name.to_owned()) {
            match texture_type {
                "float" => {
                    self.scene.add_float_texture(
                        name,
                        TextureSceneEntity::new(
                            tex_name,
                            dict,
                            string_interner,
                            loc,
                            self.render_from_object(),
                        ),
                    );
                }
                "spectrum" => {
                    self.scene.add_spectrum_texture(
                        name,
                        TextureSceneEntity::new(
                            tex_name,
                            dict,
                            string_interner,
                            loc,
                            self.render_from_object(),
                        ),
                    );
                }
                _ => panic!("{} Unknown texture type {}", loc, texture_type),
            }
        } else {
            // TODO defer error instead
            panic!("{} Texture \"{}\" redefined.", loc, name);
        }
    }

    fn material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        // TODO Verify world
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.material_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        self.graphics_state.current_material_index =
            self.scene
                .add_material(SceneEntity::new(name, loc, dict, string_interner));
        self.graphics_state.current_named_material.clear();
    }

    fn make_named_material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: crate::parser::FileLoc,
    ) {
        // TODO Normalize name to UTF8
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.material_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if self.named_material_names.insert(name.to_owned()) {
            self.scene
                .add_named_material(name, SceneEntity::new(name, loc, dict, string_interner));
        } else {
            // TODO defer error instead
            panic!("{} Named material {} redefined.", loc, name);
        }
    }

    fn named_material(
        &mut self,
        name: &str,
        _params: ParsedParameterVector,
        _string_interner: &mut StringInterner,
        _loc: crate::parser::FileLoc,
    ) {
        // TODO Normalize name to UTF8
        // TODO Verify world
        self.graphics_state.current_named_material = name.to_owned();
        self.current_material_index = -1;
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

    fn object_begin(
        &mut self,
        name: &str,
        loc: crate::parser::FileLoc,
        string_interner: &mut StringInterner,
    ) {
        // TODO Verify world
        // TODO Normalize name to UTF8

        self.pushed_graphics_states
            .push(self.graphics_state.clone());

        if self.active_instance_definition.is_some() {
            panic!("{} ObjectBegin called inside of instance definition.", loc);
        }

        let inserted = self.instance_names.insert(name.to_owned());
        if !inserted {
            panic!(
                "{} ObjectBegin trying to redefine object instance {}.",
                loc, name
            );
        }

        self.active_instance_definition =
            Some(ActiveInstanceDefinition::new(name, loc, string_interner));
    }

    fn object_end(&mut self, loc: crate::parser::FileLoc) {
        // TODO Verify world
        if self.active_instance_definition.is_none() {
            panic!("{} ObjectEnd called outside of instance definition.", loc);
        }
        if self
            .active_instance_definition
            .as_ref()
            .unwrap()
            .parent
            .is_some()
        {
            panic!(
                "{} ObjectEnd called inside Import for instance definition.",
                loc
            );
        }

        // Note: Must keep the following consistent with AttributeEnd.
        if self.pushed_graphics_states.last().is_none() {
            panic!("{} Unmatched ObjectEnd statement.", loc);
        }
        self.graphics_state = self.pushed_graphics_states.pop().unwrap();

        if self.push_stack.last().unwrap().0 == 'a' as u8 {
            panic!(
                "{} Mismatched nesting: open AttributeBegin from {} at ObjectEnd.",
                loc,
                self.push_stack.last().unwrap().1
            );
        } else {
            assert!(self.push_stack.last().unwrap().0 == 'o' as u8);
        }
        self.push_stack.pop();

        let active_instance_definition = self.active_instance_definition.take().unwrap();
        active_instance_definition
            .active_imports
            .fetch_sub(1, atomic::Ordering::SeqCst);
        // TODO Technically, the value could change between fetch_sub() and load().
        // It would be better to do this entirely within one atomic operation.

        // Otherwise will be taken care of in MergeImported().
        if active_instance_definition
            .active_imports
            .load(atomic::Ordering::SeqCst)
            == 0
        {
            self.scene
                .add_instance_definition(active_instance_definition.entity);
        }

        self.active_instance_definition = None;
    }

    fn object_instance(
        &mut self,
        name: &str,
        loc: crate::parser::FileLoc,
        string_interner: &mut StringInterner,
    ) {
        // TODO Normalize name to UTF8
        // TODO Verify world

        if self.active_instance_definition.is_some() {
            panic!(
                "{} ObjectInstance called inside of instance definition.",
                loc
            );
        }

        let worlf_from_render = self.render_from_world.inverse();

        if self.ctm_is_animated() {
            todo!()
        }

        // TODO Use transformCache
        let render_from_instance = self.render_from_object() * worlf_from_render;

        let entity = InstanceSceneEntity::new(name, loc, string_interner, render_from_instance);
        self.instance_uses.push(entity);
    }

    fn end_of_files(&mut self) {
        if self.current_block != BlockState::WorldBlock {
            panic!("End of files before WorldBegin.");
        }

        // Ensure there are no pushed graphics states
        while !self.pushed_graphics_states.is_empty() {
            self.pushed_graphics_states.pop();
            panic!("Missing end to AttributeBegin.");
        }

        // TODO If we start deferring error handling rather than panicing, this would be a good spot
        //   to check for any deferred errors and report them and actually error out.
        //   (Deferred error handling would be good because it lets the user see all the issues
        //    so they can fix them before running the scene again.)

        if !self.shapes.is_empty() {
            self.scene.add_shapes(&self.shapes);
        }
        if !self.instance_uses.is_empty() {
            self.scene.add_instance_uses(&self.instance_uses);
        }

        self.scene.done();
    }
}
