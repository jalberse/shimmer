use std::rc::Rc;

use crate::{
    colorspace::RgbColorSpace,
    paramdict::ParameterDictionary,
    parser::{FileLoc, ParsedParameterVector, ParserTarget},
};

use string_interner::symbol::SymbolUsize;

/// All objects in the scene are described by various *Entity classes.
/// The SceneEntity is the simplest; it records the name of the entity,
/// the file location of the of the associated statement in the scene description,
/// and any user-provided parameters.
/// It is used for the film, sampler, integrator, pixel filter, and accelerator,
/// and is also used as a "base" for some of the other scene entity types.
pub struct SceneEntity {
    name: SymbolUsize,
    loc: FileLoc,
    parameters: ParameterDictionary,
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
