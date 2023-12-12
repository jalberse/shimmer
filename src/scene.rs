use std::rc::Rc;

use crate::{
    camera::CameraTransform,
    colorspace::RgbColorSpace,
    film::Film,
    filter::Filter,
    paramdict::ParameterDictionary,
    parser::{FileLoc, ParsedParameterVector, ParserTarget},
    transform::Transform,
};

use string_interner::{symbol::SymbolU32, StringInterner};

pub struct BasicScene {
    pub integrator: SceneEntity,
    pub accelerator: SceneEntity,
    pub film_color_space: Rc<RgbColorSpace>,
    pub shapes: Vec<ShapeSceneEntity>,
    pub instances: Vec<InstanceSceneEntity>,
    pub instance_definitions: Vec<InstanceDefinitionSceneEntity>,
    film: Film,
}

impl BasicScene {
    pub fn new(
        filter: SceneEntity,
        film: SceneEntity,
        camera: CameraSceneEntity,
        sampler: SceneEntity,
        integ: SceneEntity,
        accel: SceneEntity,
        string_interner: StringInterner,
    ) -> BasicScene {
        let filt = Filter::create(
            &string_interner
                .resolve(filter.name)
                .expect("Unresolved name!"),
            &filter.parameters,
            &filter.loc,
        );

        let exposure_time = camera.base.parameters.get_one_float("shutterclose", 1.0)
            - camera.base.parameters.get_one_float("shutteropen", 0.0);

        if exposure_time <= 0.0 {
            panic!(
                "{} The specified camera shutter times imply the camera won't open.",
                camera.base.loc
            );
        }

        let concrete_film = Film::create(
            &string_interner.resolve(film.name).unwrap(),
            &film.parameters,
            exposure_time,
            &camera.camera_transform,
            filt,
            film.loc,
        );

        // TODO the rest of this
        BasicScene {
            integrator: integ,
            accelerator: accel,
            film_color_space: film.parameters.color_space.clone(),
            shapes: (),
            instances: (),
            instance_definitions: (),
            film: concrete_film,
        }
    }
}

/// All objects in the scene are described by various *Entity classes.
/// The SceneEntity is the simplest; it records the name of the entity,
/// the file location of the of the associated statement in the scene description,
/// and any user-provided parameters.
/// It is used for the film, sampler, integrator, pixel filter, and accelerator,
/// and is also used as a "base" for some of the other scene entity types.
pub struct SceneEntity {
    // TODO We will need to use a global StringInterner for this. We can use a Lazy one.
    name: SymbolU32,
    loc: FileLoc,
    parameters: ParameterDictionary,
}

pub enum MaterialRef {
    Index(i32),
    Name(String),
}

pub struct ShapeSceneEntity {
    base: SceneEntity,
    render_from_object: Rc<Transform>,
    object_from_render: Rc<Transform>,
    reverse_orientation: bool,
    material_ref: MaterialRef,
    light_index: i32,
    inside_medium: String,
    outside_medium: String,
}

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

pub struct BasicSceneBuilder {}

impl ParserTarget for BasicSceneBuilder {
    fn scale(
        &mut self,
        sx: crate::Float,
        sy: crate::Float,
        sz: crate::Float,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn shape(&mut self, name: &str, params: ParsedParameterVector, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn option(&mut self, name: &str, value: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn identity(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn translate(
        &mut self,
        dx: crate::Float,
        dy: crate::Float,
        dz: crate::Float,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn rotate(
        &mut self,
        angle: crate::Float,
        ax: crate::Float,
        ay: crate::Float,
        az: crate::Float,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
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
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn concat_transform(&mut self, transform: [crate::Float; 16], loc: crate::parser::FileLoc) {
        todo!()
    }

    fn transform(&mut self, transform: [crate::Float; 16], loc: crate::parser::FileLoc) {
        todo!()
    }

    fn coordinate_system(&mut self, name: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn coordinate_sys_transform(&mut self, name: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn active_transform_all(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn active_transform_end_time(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn active_transform_start_time(&mut self, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn transform_times(
        &mut self,
        start: crate::Float,
        end: crate::Float,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn color_space(&mut self, n: &str, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn pixel_filter(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn film(
        &mut self,
        film_type: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn accelerator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn integrator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn camera(&mut self, name: &str, params: ParsedParameterVector, loc: crate::parser::FileLoc) {
        todo!()
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

    fn sampler(&mut self, name: &str, params: ParsedParameterVector, loc: crate::parser::FileLoc) {
        todo!()
    }

    fn world_begin(&mut self, loc: crate::parser::FileLoc) {
        todo!()
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
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn area_light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        loc: crate::parser::FileLoc,
    ) {
        todo!()
    }

    fn reverse_orientation(&mut self, loc: crate::parser::FileLoc) {
        todo!()
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
