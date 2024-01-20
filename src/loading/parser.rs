//! Directives parser.
// Adapted from pbrt4 crate under apache license.

use std::collections::HashMap;

use string_interner::StringInterner;

use crate::{
    loading::error::Error,
    loading::error::Result,
    loading::param::{Param, ParamList},
    loading::token::{Directive, Token},
    loading::tokenizer::Tokenizer,
    options::Options,
    Float,
};

use super::parser_target::ParserTarget;

pub fn parse_str<T: ParserTarget>(data: &str, target: &mut T, options: &mut Options) {
    let mut parsers = Vec::new();
    parsers.push(Parser::new(data));

    let mut string_interner = StringInterner::default();
    let mut cached_spectra = HashMap::new();

    while let Some(parser) = parsers.last_mut() {
        // Fetch next element
        let element = match parser.parse_next() {
            Ok(element) => element,
            Err(Error::EndOfFile) => {
                parsers.pop();
                continue;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        };

        // TODO I'm passing a default FileLoc here since we don't yet track it in the tokenizer.
        //  We'd like to do that at some point, and then pass it here. It will make error messages better.
        let loc = Default::default();
        match element {
            Element::Include(path) => todo!("Support include directive"),
            Element::Import(_) => todo!("Support import directive"),
            Element::Option(param) => target.option(param.name, param.value, options, loc),
            Element::Film { ty, params } => {
                target.film(ty, params.into(), &mut string_interner, loc)
            }
            Element::ColorSpace { ty } => target.color_space(ty, loc),
            Element::Camera { ty, params } => {
                target.camera(ty, params.into(), &mut string_interner, loc, options)
            }
            Element::Sampler { ty, params } => {
                target.sampler(ty, params.into(), &mut string_interner, loc)
            }
            Element::Integrator { ty, params } => {
                target.integrator(ty, params.into(), &mut string_interner, loc)
            }
            Element::Accelerator { ty, params } => {
                target.accelerator(ty, params.into(), &mut string_interner, loc)
            }
            Element::CoordinateSystem { name } => target.coordinate_system(name, loc),
            Element::CoordSysTransform { name } => target.coordinate_sys_transform(name, loc),
            Element::PixelFilter { name, params } => {
                target.pixel_filter(name, params.into(), &mut string_interner, loc)
            }
            Element::Identity => target.identity(loc),
            Element::Translate { v } => target.translate(v[0], v[1], v[2], loc),
            Element::Scale { v } => target.scale(v[0], v[1], v[2], loc),
            Element::Rotate { angle, v } => target.rotate(angle, v[0], v[1], v[2], loc),
            Element::LookAt { eye, look_at, up } => target.look_at(
                eye[0], eye[1], eye[2], look_at[0], look_at[1], look_at[2], up[0], up[1], up[2],
                loc,
            ),
            Element::Transform { m } => target.transform(m, loc),
            Element::ConcatTransform { m } => target.concat_transform(m, loc),
            Element::TransformTimes { start, end } => target.transform_times(start, end, loc),
            Element::ActiveTransform { ty } => match ty {
                "StartTime" => target.active_transform_start_time(loc),
                "EndTime" => target.active_transform_end_time(loc),
                "All" => target.active_transform_all(loc),
                _ => todo!("Unknown active transform type"),
            },
            Element::ReverseOrientation => target.reverse_orientation(loc),
            Element::WorldBegin => target.world_begin(&mut string_interner, loc, options),
            Element::AttributeBegin => target.attribute_begin(loc),
            Element::AttributeEnd => target.attribute_end(loc),
            Element::Attribute {
                attr_target,
                params,
            } => target.attribute(attr_target, params.into(), loc),
            Element::LightSource { ty, params } => target.light_source(
                ty,
                params.into(),
                &mut string_interner,
                loc,
                &mut cached_spectra,
                options,
            ),
            Element::AreaLightSource { ty, params } => {
                target.area_light_source(ty, params.into(), loc)
            }
            Element::Material { ty, params } => {
                target.material(ty, params.into(), &mut string_interner, loc)
            }
            Element::MakeNamedMaterial { name, params } => {
                target.make_named_material(name, params.into(), &mut string_interner, loc)
            }
            Element::NamedMaterial { name } => target.named_material(name, loc),
            Element::Texture {
                name,
                ty,
                class,
                params,
            } => target.texture(
                name,
                ty,
                class,
                params.into(),
                &mut string_interner,
                loc,
                options,
                &mut cached_spectra,
            ),
            Element::Shape { name, params } => {
                target.shape(name, params.into(), &mut string_interner, loc)
            }
            Element::ObjectBegin { name } => target.object_begin(name, loc, &mut string_interner),
            Element::ObjectEnd => target.object_end(loc),
            Element::ObjectInstance { name } => {
                target.object_instance(name, loc, &mut string_interner)
            }
            Element::MakeNamedMedium { name, params } => {
                target.make_named_medium(name, params.into(), loc)
            }
            Element::MediumInterface { interior, exterior } => {
                target.medium_interface(interior, exterior, loc)
            }
        }
    }

    // TODO By this point our BasicSceneBuilder should be done, maybe we need to call some method to finalze.
    // But then we can return the scene and stuff.
    todo!()
}

/// Parsed directive.
#[derive(Debug, PartialEq)]
pub enum Element<'a> {
    /// Include behaves similarly to the #include directive in C++: parsing of the current file is suspended,
    /// the specified file is parsed in its entirety, and only then does parsing of the current file resume.
    /// Its effect is equivalent to direct text substitution of the included file.
    Include(&'a str),
    Import(&'a str),
    Option(Param<'a>),
    Film {
        ty: &'a str,
        params: ParamList<'a>,
    },
    ColorSpace {
        ty: &'a str,
    },
    Camera {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Sampler {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Integrator {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Accelerator {
        ty: &'a str,
        params: ParamList<'a>,
    },
    CoordinateSystem {
        name: &'a str,
    },
    CoordSysTransform {
        name: &'a str,
    },
    PixelFilter {
        name: &'a str,
        params: ParamList<'a>,
    },
    Identity,
    /// `Translate x y z`
    Translate {
        v: [Float; 3],
    },
    /// `Scale x y z`
    Scale {
        v: [Float; 3],
    },
    /// `Rotate angle x y z`
    Rotate {
        angle: Float,
        v: [Float; 3],
    },
    /// `LookAt eye_x eye_y eye_z look_x look_y look_z up_x up_y up_z`
    LookAt {
        eye: [Float; 3],
        look_at: [Float; 3],
        up: [Float; 3],
    },
    /// `Transform m00 ... m33`
    Transform {
        m: [Float; 16],
    },
    /// `ConcatTransform m00 .. m33`
    ConcatTransform {
        m: [Float; 16],
    },
    /// `TransformTimes start end`.
    TransformTimes {
        start: Float,
        end: Float,
    },
    ActiveTransform {
        ty: &'a str,
    },
    /// `ReverseOrientation`.
    ReverseOrientation,
    /// `WorldBegin`
    WorldBegin,
    /// `AttributeBegin`
    AttributeBegin,
    /// `AttributeEnd`
    AttributeEnd,
    /// `Attribute "target" parameter-list`
    Attribute {
        attr_target: &'a str,
        params: ParamList<'a>,
    },
    LightSource {
        ty: &'a str,
        params: ParamList<'a>,
    },
    AreaLightSource {
        ty: &'a str,
        params: ParamList<'a>,
    },
    Material {
        ty: &'a str,
        params: ParamList<'a>,
    },
    MakeNamedMaterial {
        name: &'a str,
        params: ParamList<'a>,
    },
    NamedMaterial {
        name: &'a str,
    },
    /// `Texture "name" "type" "class" [ parameter-list ]`
    Texture {
        name: &'a str,
        ty: &'a str,
        class: &'a str,
        params: ParamList<'a>,
    },
    /// `Shape "name" parameter-list`
    Shape {
        name: &'a str,
        params: ParamList<'a>,
    },
    ObjectBegin {
        name: &'a str,
    },
    ObjectEnd,
    ObjectInstance {
        name: &'a str,
    },
    MakeNamedMedium {
        name: &'a str,
        params: ParamList<'a>,
    },
    MediumInterface {
        interior: &'a str,
        exterior: &'a str,
    },
}

pub struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
}

impl<'a> Parser<'a> {
    pub fn new(str: &'a str) -> Self {
        let tokenizer = Tokenizer::new(str);
        Self { tokenizer }
    }

    /// Parse next element.
    pub fn parse_next(&mut self) -> Result<Element<'a>> {
        let Some(next_token) = self.tokenizer.next() else {
            return Err(Error::EndOfFile);
        };

        // Check if token is directive
        let directive = next_token.directive().ok_or(Error::UnknownDirective)?;

        let element = match directive {
            Directive::Include => Element::Include(self.read_str()?),
            Directive::Import => Element::Import(self.read_str()?),
            Directive::Option => Element::Option(self.read_param()?),
            Directive::Film => Element::Film {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::ColorSpace => Element::ColorSpace {
                ty: self.read_str()?,
            },
            Directive::Camera => Element::Camera {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Sampler => Element::Sampler {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Integrator => Element::Integrator {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Accelerator => Element::Accelerator {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::CoordinateSystem => Element::CoordinateSystem {
                name: self.read_str()?,
            },
            Directive::CoordSysTransform => Element::CoordSysTransform {
                name: self.read_str()?,
            },
            Directive::PixelFilter => Element::PixelFilter {
                name: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Identity => Element::Identity,
            Directive::Translate => Element::Translate {
                v: self.read_point()?,
            },
            Directive::Scale => Element::Scale {
                v: self.read_point()?,
            },
            Directive::Rotate => Element::Rotate {
                angle: self.read_float()?,
                v: self.read_point()?,
            },
            Directive::LookAt => Element::LookAt {
                eye: self.read_point()?,
                look_at: self.read_point()?,
                up: self.read_point()?,
            },
            Directive::Transform => {
                // Skip [
                self.skip_brace()?;

                let elem = Element::Transform {
                    m: self.read_matrix()?,
                };

                // Skip ]
                self.skip_brace()?;

                elem
            }
            Directive::ConcatTransform => {
                // Skip [
                self.skip_brace()?;

                let elem = Element::ConcatTransform {
                    m: self.read_matrix()?,
                };

                // Skip ]
                self.skip_brace()?;

                elem
            }
            Directive::TransformTimes => Element::TransformTimes {
                start: self.read_float()?,
                end: self.read_float()?,
            },
            Directive::ActiveTransform => Element::ActiveTransform {
                ty: self.read_str()?,
            },
            Directive::ReverseOrientation => Element::ReverseOrientation,
            Directive::WorldBegin => Element::WorldBegin,
            Directive::AttributeBegin => Element::AttributeBegin,
            Directive::AttributeEnd => Element::AttributeEnd,
            Directive::Attribute => Element::Attribute {
                attr_target: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::LightSource => Element::LightSource {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::AreaLightSource => Element::AreaLightSource {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Material => Element::Material {
                ty: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::MakeNamedMaterial => Element::MakeNamedMaterial {
                name: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::NamedMaterial => Element::NamedMaterial {
                name: self.read_str()?,
            },
            Directive::Texture => Element::Texture {
                name: self.read_str()?,
                ty: self.read_str()?,
                class: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::Shape => Element::Shape {
                name: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::ObjectBegin => Element::ObjectBegin {
                name: self.read_str()?,
            },
            Directive::ObjectEnd => Element::ObjectEnd,
            Directive::ObjectInstance => Element::ObjectInstance {
                name: self.read_str()?,
            },
            Directive::MakeNamedMedium => Element::MakeNamedMedium {
                name: self.read_str()?,
                params: self.read_param_list()?,
            },
            Directive::MediumInterface => Element::MediumInterface {
                interior: self.read_str()?,
                exterior: self.read_str()?,
            },
        };

        Ok(element)
    }

    fn skip_brace(&mut self) -> Result<()> {
        let Some(token) = self.tokenizer.next() else {
            return Err(Error::UnexpectedToken);
        };

        let is_open = token.is_open_brace();
        let is_close = token.is_close_brace();

        // Not a brace
        if !is_open && !is_close {
            return Err(Error::UnexpectedToken);
        }

        Ok(())
    }

    /// Read next token or return [Error::UnexpectedEnd].
    fn read_token(&mut self) -> Result<Token<'a>> {
        match self.tokenizer.next() {
            Some(token) => {
                if !token.is_valid() {
                    return Err(Error::InvalidToken);
                }

                Ok(token)
            }
            None => Err(Error::NoToken),
        }
    }

    /// Read token as `Float`.
    fn read_float(&mut self) -> Result<Float> {
        let token = self.read_token()?;
        let parsed = token.parse::<Float>()?;
        Ok(parsed)
    }

    /// Read 3 floats.
    fn read_point(&mut self) -> Result<[Float; 3]> {
        let x = self.read_float()?;
        let y = self.read_float()?;
        let z = self.read_float()?;

        Ok([x, y, z])
    }

    /// Read 16 floats.
    fn read_matrix(&mut self) -> Result<[Float; 16]> {
        let mut m = [0.0; 16];
        for m in &mut m {
            *m = self.read_float()?;
        }
        Ok(m)
    }

    /// Read a quoted string.
    fn read_str(&mut self) -> Result<&'a str> {
        let token = self.read_token()?;
        token.unquote().ok_or(Error::InvalidString)
    }

    /// Parse a single option
    ///
    /// Valid inputs:
    /// - "integer indices" [ 0 1 2 0 2 3 ]
    /// - "float scale" [10]
    /// - "float iso" 150
    fn read_param(&mut self) -> Result<Param<'a>> {
        let type_and_name = self.read_str()?;

        let mut start = self.tokenizer.offset();
        let end;

        // Either [ or a single value.
        let value = self.read_token()?;

        if value.is_open_brace() {
            // Skip brace offset
            start = self.tokenizer.offset();

            // Read array of values
            loop {
                let value = self.read_token()?;

                if value.is_close_brace() {
                    end = self.tokenizer.offset() - 1;
                    break;
                }

                // Got directive without closing bracket token.
                if value.is_directive() {
                    return Err(Error::UnexpectedToken);
                }
            }
        } else {
            // Single value
            end = start + value.token_size() + 1;
        }

        let token = self.tokenizer.token(start, end);
        let param = Param::new(type_and_name, token.value())?;

        Ok(param)
    }

    #[inline]
    fn read_param_list(&mut self) -> Result<ParamList<'a>> {
        let mut list = ParamList::default();

        loop {
            match self.tokenizer.peek_token() {
                // Each parameter starts with a quoted string
                Some(token) if token.is_quote() => {
                    let param = self.read_param()?;
                    list.add(param)?;
                }
                // Other token, break loop
                Some(_) => break,
                // No more tokens
                None => break,
            }
        }

        Ok(list)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        colorspace::RgbColorSpace,
        loading::{
            param::ParamType,
            paramdict::{ParameterDictionary, ParsedParameter},
            parser_target::ParsedParameterVector,
        },
    };

    use super::*;

    #[test]
    fn parse_includes() {
        let mut parser = Parser::new(
            "
Include \"geometry/car.pbrt\"
Import \"geometry/bigcar.pbrt.gz\"
        ",
        );

        let element = parser.parse_next().unwrap();
        assert!(matches!(element, Element::Include("geometry/car.pbrt")));

        let element = parser.parse_next().unwrap();
        assert!(matches!(
            element,
            Element::Import("geometry/bigcar.pbrt.gz")
        ));
    }

    #[test]
    fn parse_scale_and_rotate() {
        let mut parser = Parser::new(
            "
Scale -1 1 1
Rotate 1 0 0 1
        ",
        );

        assert!(matches!(
            parser.parse_next().unwrap(),
            Element::Scale { .. }
        ));

        assert!(matches!(
            parser.parse_next().unwrap(),
            Element::Rotate { .. }
        ));
    }

    #[test]
    fn parse_look_at() {
        let mut parser = Parser::new(
            "
        LookAt 0.322839 0.0534825 0.504299
        -0.140808 -0.162727 -0.354936
        0.0355799 0.964444 -0.261882
        ",
        );

        let element = parser.parse_next().unwrap();
        assert!(matches!(element, Element::LookAt { .. }));
    }

    #[test]
    fn parse_option() {
        let mut parser = Parser::new(
            "
Option \"string filename\" [\"foo.exr\"]
Option \"string filename\" \"foo.exr\"
        ",
        );

        let expected = Param::new("string filename", "\"foo.exr\"").unwrap();

        assert_eq!(
            parser.parse_next().unwrap(),
            Element::Option(expected.clone())
        );

        assert_eq!(parser.parse_next().unwrap(), Element::Option(expected));
    }

    #[test]
    fn parse_film() {
        let mut parser = Parser::new(
            "
Film \"rgb\"
    \"string filename\" [ \"crown.exr\" ]
    \"integer yresolution\" [ 1400 ]
    \"integer xresolution\" [ 1000 ]
    \"float iso\" 150
    \"string sensor\" \"canon_eos_5d_mkiv\"
        ",
        );

        let elem = parser.parse_next().unwrap();

        match elem {
            Element::Film { ty, params } => {
                assert_eq!(ty, "rgb");
                assert_eq!(params.len(), 5);

                let param = params.get("filename").unwrap();
                assert_eq!(param.name, "filename");
                assert_eq!(param.ty, ParamType::String);

                let param = params.get("iso").unwrap();
                assert_eq!(param.name, "iso");
                assert_eq!(param.ty, ParamType::Float);

                // Test conversiont to ParsedParameter.
                let params: ParsedParameterVector = params.into();
                assert_eq!(params.len(), 5);
                let mut dict = ParameterDictionary::new(
                    params,
                    RgbColorSpace::get_named(crate::colorspace::NamedColorSpace::SRGB).clone(),
                );
                assert_eq!(dict.params.len(), 5);
                let param = dict.get_one_string("filename", "".to_owned());
                assert_eq!(param, "crown.exr");
                let param = dict.get_one_int("yresolution", 0);
                assert_eq!(param, 1400);
                let param = dict.get_one_int("xresolution", 0);
                assert_eq!(param, 1000);
                let param = dict.get_one_float("iso", 0.0);
                assert_eq!(param, 150.0);
                let param = dict.get_one_string("sensor", "".to_owned());
                assert_eq!(param, "canon_eos_5d_mkiv");
            }
            _ => panic!("Unexpected element type"),
        }
    }

    #[test]
    fn parse_film_no_params() {
        let mut parser = Parser::new(
            "
Film \"rgb\"
LookAt 0 5.5 24 0 11 -10 0 1 0
        ",
        );

        assert!(matches!(
            parser.parse_next().unwrap(),
            Element::Film { ty: "rgb", .. }
        ));

        assert!(matches!(
            parser.parse_next().unwrap(),
            Element::LookAt { .. }
        ));
    }

    #[test]
    fn parse_transform() {
        let mut parser = Parser::new("Transform [ 1 0 0 0 0 1 0 0 0 0 1 0 3 1 -4 1 ]");
        let next = parser.parse_next().unwrap();

        assert!(matches!(next, Element::Transform { .. }));
    }

    #[test]
    fn parse_concat_transform() {
        let mut parser = Parser::new("ConcatTransform [ 1 0 0 0 0 1 0 0 0 0 1 0 3 1 -4 1 ]");
        let next = parser.parse_next().unwrap();

        assert!(matches!(next, Element::ConcatTransform { .. }));
    }

    #[test]
    fn param_to_parsed_param_int() {
        // Note that Param doesn't store the [ and ] characters; it's stripped in tokenizing.
        let param = Param::new("integer indices", "0 1 2 0 2 3").unwrap();
        let param: ParsedParameter = param.into();

        assert_eq!(param.name, "indices");
        assert_eq!(param.param_type, "integer");
        assert_eq!(param.looked_up, false);
        assert_eq!(param.color_space, None);
        assert!(param.bools.is_empty());
        assert!(param.strings.is_empty());
        assert!(param.floats.is_empty());
        assert_eq!(param.ints, vec![0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn param_to_parsed_param_float() {
        // Note that Param doesn't store the [ and ] characters; it's stripped in tokenizing.
        let param = Param::new("float values", "0.0 10.0 2 0.1 2.5 3.4").unwrap();
        let param: ParsedParameter = param.into();

        assert_eq!(param.name, "values");
        assert_eq!(param.param_type, "float");
        assert_eq!(param.looked_up, false);
        assert_eq!(param.color_space, None);
        assert!(param.bools.is_empty());
        assert!(param.strings.is_empty());
        assert_eq!(param.floats, vec![0.0, 10.0, 2.0, 0.1, 2.5, 3.4]);
        assert!(param.ints.is_empty());
    }

    #[test]
    fn param_to_parsed_param_bool() {
        let param = Param::new("bool values", "true false true").unwrap();
        let param: ParsedParameter = param.into();

        assert_eq!(param.name, "values");
        assert_eq!(param.param_type, "bool");
        assert_eq!(param.looked_up, false);
        assert_eq!(param.color_space, None);
        assert_eq!(param.bools, vec![true, false, true]);
        assert!(param.strings.is_empty());
        assert!(param.floats.is_empty());
        assert!(param.ints.is_empty());
    }

    #[test]
    fn param_to_parsed_param_string() {
        let param = Param::new("string values", "\"foo\" \"bar\" \"baz\"").unwrap();
        let param: ParsedParameter = param.into();

        assert_eq!(param.name, "values");
        assert_eq!(param.param_type, "string");
        assert_eq!(param.looked_up, false);
        assert_eq!(param.color_space, None);
        assert!(param.bools.is_empty());
        assert_eq!(param.strings, vec!["foo", "bar", "baz"]);
        assert!(param.floats.is_empty());
        assert!(param.ints.is_empty());
    }
}
