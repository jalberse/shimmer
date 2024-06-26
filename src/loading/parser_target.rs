use std::{collections::HashMap, fmt::Display, sync::{Arc, Mutex}};

use crate::{color::ColorEncodingCache, loading::paramdict::ParsedParameter, mipmap::MIPMap, options::Options, spectra::Spectrum, texture::TexInfo, Float};

use arrayvec::ArrayVec;
use string_interner::StringInterner;

use super::param::ParamList;

pub type ParsedParameterVector = ArrayVec<ParsedParameter, 8>;

impl<'a> From<ParamList<'a>> for ParsedParameterVector {
    fn from(param_list: ParamList) -> Self {
        let mut params = ArrayVec::new();

        for param in param_list.0.into_iter() {
            // Ignore the string; it's just param.name.
            let param: ParsedParameter = param.1.into();
            params.push(param);
        }

        params
    }
}

/// Used for error reporting to convey error locations in scene description files.
#[derive(Debug, Clone)]
pub struct FileLoc {
    filename: String,
    line: i32,
    column: i32,
}

impl Default for FileLoc {
    fn default() -> Self {
        Self {
            filename: Default::default(),
            line: Default::default(),
            column: Default::default(),
        }
    }
}

impl Display for FileLoc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.filename, self.line, self.column)
    }
}

pub trait ParserTarget {
    fn shape(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    );
    fn option(&mut self, name: &str, value: &str, options: &mut Options, loc: FileLoc);
    fn identity(&mut self, loc: FileLoc);
    fn translate(&mut self, dx: Float, dy: Float, dz: Float, loc: FileLoc);
    fn scale(&mut self, sx: Float, sy: Float, sz: Float, loc: FileLoc);
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
    fn transform(&mut self, transform: [Float; 16], loc: FileLoc);
    fn concat_transform(&mut self, transform: [Float; 16], loc: FileLoc);
    fn coordinate_system(&mut self, name: &str, loc: FileLoc);
    fn coordinate_sys_transform(&mut self, name: &str, loc: FileLoc);
    fn active_transform_all(&mut self, loc: FileLoc);
    fn active_transform_end_time(&mut self, loc: FileLoc);
    fn active_transform_start_time(&mut self, loc: FileLoc);
    fn transform_times(&mut self, start: Float, end: Float, loc: FileLoc);
    fn color_space(&mut self, n: &str, loc: FileLoc);
    fn pixel_filter(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    );
    fn film(
        &mut self,
        film_type: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    );
    fn accelerator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    );
    fn integrator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    );
    fn camera(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
    );
    fn make_named_medium(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn medium_interface(&mut self, inside_name: &str, outside_name: &str, loc: FileLoc);
    fn sampler(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    );
    fn world_begin(
        &mut self,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
    );
    fn attribute_begin(&mut self, loc: FileLoc);
    fn attribute_end(&mut self, loc: FileLoc);
    fn attribute(&mut self, target: &str, params: ParsedParameterVector, loc: FileLoc);
    fn texture(
        &mut self,
        name: &str,
        texture_type: &str,
        tex_name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    );
    fn material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
    );
    fn make_named_material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
    );
    fn named_material(&mut self, name: &str, loc: FileLoc);
    fn light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    );
    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc);
    fn reverse_orientation(&mut self, loc: FileLoc);
    fn object_begin(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner);
    fn object_end(&mut self, loc: FileLoc);
    fn object_instance(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner);
    fn end_of_files(&mut self);
}
