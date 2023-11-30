// TODO The ParserTarget will be a Trait.
//  We'll just start with the BasicSceneBuilder to implement it.

// TODO Uhm, I think we coudl start with just the SceneBuilder and do parsing later.
//  We can call things directly.

// TODO Possibly use nom for parsing.

use std::fmt::Display;

use crate::{paramdict::ParsedParameter, Float};

use arrayvec::ArrayVec;

pub type ParsedParameterVector = ArrayVec<ParsedParameter, 8>;

/// Used for error reporting to convey error locations in scene description files.
pub struct FileLoc {
    filename: String,
    line: i32,
    column: i32,
}

impl Display for FileLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.filename, self.line, self.column)
    }
}

pub trait ParserTarget {
    fn scale(&mut self, sx: Float, sy: Float, sz: Float, loc: FileLoc);
    fn shape(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn option(&mut self, name: &str, value: &str, loc: FileLoc);
    fn identity(&mut self, loc: FileLoc);
    fn translate(&mut self, dx: Float, dy: Float, dz: Float, loc: FileLoc);
    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float, loc: FileLoc);
    fn look_at(
        &mut self,
        ex: Float,
        ey: Float,
        ez: Float,
        lx: Float,
        ly: Float,
        lz: Float,
        ux: Float,
        uy: Float,
        uz: Float,
        loc: FileLoc,
    );
    fn concat_transform(&mut self, transform: [Float; 16], loc: FileLoc);
    fn transform(&mut self, transform: [Float; 16], loc: FileLoc);
    fn coordinate_system(&mut self, name: &str, loc: FileLoc);
    fn coordinate_sys_transform(&mut self, name: &str, loc: FileLoc);
    fn active_transform_all(&mut self, loc: FileLoc);
    fn active_transform_end_time(&mut self, loc: FileLoc);
    fn active_transform_start_time(&mut self, loc: FileLoc);
    fn transform_times(&mut self, start: Float, end: Float, loc: FileLoc);
    fn color_space(&mut self, n: &str, loc: FileLoc);
    fn pixel_filter(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn film(&mut self, film_type: &str, params: ParsedParameterVector, loc: FileLoc);
    fn accelerator(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn integrator(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn camera(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn make_named_medium(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn medium_interface(&mut self, inside_name: &str, outside_name: &str, loc: FileLoc);
    fn sampler(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn world_begin(&mut self, loc: FileLoc);
    fn attribute_begin(&mut self, loc: FileLoc);
    fn attribute_end(&mut self, loc: FileLoc);
    fn attribute(&mut self, target: &str, params: ParsedParameterVector, loc: FileLoc);
    fn texture(
        &mut self,
        name: &str,
        texture_type: &str,
        tex_name: &str,
        params: ParsedParameterVector,
        loc: FileLoc,
    );
    fn material(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn make_named_material(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn named_material(&mut self, name: &str, loc: FileLoc);
    fn light_source(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn reverse_orientation(&mut self, loc: FileLoc);
    fn object_begin(&mut self, name: &str, loc: FileLoc);
    fn object_end(&mut self, loc: FileLoc);
    fn object_instance(&mut self, name: &str, loc: FileLoc);
    fn end_of_files(&mut self);
}
