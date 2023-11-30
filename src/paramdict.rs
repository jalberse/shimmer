use std::{fmt::Display, rc::Rc};

use crate::{
    colorspace::RgbColorSpace,
    parser::{FileLoc, ParsedParameterVector},
    vecmath::{Normal3f, Point2f, Point3f, Tuple2, Tuple3, Vector2f, Vector3f},
    Float,
};

/// Seal this trait so that it remains internal.
pub trait ParameterType: sealed::Sealed {
    const TYPE_NAME: &'static str;
    const N_PER_ITEM: i32;
    type ConvertType;
    type ReturnType;

    fn convert(v: &[Self::ConvertType], loc: &FileLoc) -> Self::ReturnType;
    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType>;
}

// Parameter types; a ParameterDictionary can hold these types of parameters.
// This should all implement ParameterType.
// PAPERDOC This is an interesting construct; if Rust could support enum values as const generics,
// we could implement some ParameterType<const T: ParameterTypeEnum>, and define the associated types
// and constants in the impl block for each variant. That's similar to what PBRT does, loosely,
// within C++'s system, though that requires some templating which is less clean.
// This Rust approach of using a trait is quite simple comparatively; the sealed construct
// is jarring if you haven't seen it before, but it's a simple extension of the trait system
// to just ensure that outside callers can't implement the trait themselves.
struct BooleanParam;
struct FloatParam;
struct IntegerParam;
struct Point2fParam;
struct Vector2fParam;
struct Point3fParam;
struct Vector3fParam;
struct Normal3fParam;
struct SpectrumParam;
struct StringParam;
struct TextureParam;

impl ParameterType for BooleanParam {
    const TYPE_NAME: &'static str = "bool";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = bool;
    type ReturnType = bool;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0]
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.bools
    }
}

impl ParameterType for FloatParam {
    const TYPE_NAME: &'static str = "float";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = Float;
    type ReturnType = Float;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0]
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for IntegerParam {
    const TYPE_NAME: &'static str = "integer";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = i32;
    type ReturnType = i32;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0]
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.ints
    }
}

impl ParameterType for Point2fParam {
    const TYPE_NAME: &'static str = "point2";

    const N_PER_ITEM: i32 = 2;

    type ConvertType = Float;
    type ReturnType = Point2f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Point2f::new(v[0], v[1])
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Vector2fParam {
    const TYPE_NAME: &'static str = "vector2";

    const N_PER_ITEM: i32 = 2;

    type ConvertType = Float;
    type ReturnType = Vector2f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Vector2f::new(v[0], v[1])
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Point3fParam {
    const TYPE_NAME: &'static str = "point3";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Point3f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Point3f::new(v[0], v[1], v[2])
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Vector3fParam {
    const TYPE_NAME: &'static str = "vector3";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Vector3f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Vector3f::new(v[0], v[1], v[2])
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for Normal3fParam {
    const TYPE_NAME: &'static str = "normal";

    const N_PER_ITEM: i32 = 3;

    type ConvertType = Float;
    type ReturnType = Normal3f;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        Normal3f::new(v[0], v[1], v[2])
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.floats
    }
}

impl ParameterType for StringParam {
    const TYPE_NAME: &'static str = "string";

    const N_PER_ITEM: i32 = 1;

    type ConvertType = String;
    type ReturnType = String;

    fn convert(v: &[Self::ConvertType], _loc: &FileLoc) -> Self::ReturnType {
        v[0].clone()
    }

    fn get_values<'a>(param: &'a ParsedParameter) -> &'a Vec<Self::ConvertType> {
        &param.strings
    }
}

pub struct ParsedParameter {
    /// The name of the parameter, e.g. "radius"
    name: String,
    /// The type of the parameter; e.g. "float"
    param_type: String,
    /// The location in the file
    loc: FileLoc,
    // These store the parameter values
    floats: Vec<Float>,
    ints: Vec<i32>,
    strings: Vec<String>,
    bools: Vec<bool>,
    /// Used for code relating to extracting parameter values; used for error handling.
    looked_up: bool,
    color_space: Option<Box<RgbColorSpace>>,
    may_be_unused: bool,
}

enum SpectrumType {
    Illuminant,
    Albedo,
    Unbounded,
}

impl Display for SpectrumType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpectrumType::Illuminant => write!(f, "Illuminant"),
            SpectrumType::Albedo => write!(f, "Albedo"),
            SpectrumType::Unbounded => write!(f, "Unbounded"),
        }
    }
}

/// Most of the scene entity objects store lists of associated parameters from the scene description file. While the
/// ParsedParameter is a convenient representation for the parser to generate, it does not provide capabilities
/// for checking the validity of parameters or for easily extracting parameter values. To that end,
/// ParameterDictionary adds both semantics and convenience to vectors of ParsedParameters.
/// Thus, it is the class that is used for SceneEntity::parameters.
pub struct ParameterDictionary {
    params: ParsedParameterVector,
    color_space: Rc<RgbColorSpace>,
    n_owned_params: i32,
}

impl ParameterDictionary {
    /// The RGBColorSpace defines the color space of any RGB-valued parameters.
    pub fn new(
        params: ParsedParameterVector,
        color_space: Rc<RgbColorSpace>,
    ) -> ParameterDictionary {
        let n_owned_params = params.len() as i32;
        // TODO PBRT reverses params; why?
        let d = ParameterDictionary {
            params,
            color_space,
            n_owned_params,
        };
        d.check_parameter_types();
        d
    }

    pub fn check_parameter_types(&self) {
        for p in &self.params {
            match p.param_type.as_str() {
                BooleanParam::TYPE_NAME => {
                    if p.bools.is_empty() {
                        panic!("No boolean values provided for boolean-valued parameter!");
                    }
                }
                FloatParam::TYPE_NAME
                | IntegerParam::TYPE_NAME
                | Point2fParam::TYPE_NAME
                | Vector2fParam::TYPE_NAME
                | Point3fParam::TYPE_NAME
                | Vector3fParam::TYPE_NAME
                | Normal3fParam::TYPE_NAME
                | "rgb"
                | "blackbody" => {
                    if p.ints.is_empty() && p.floats.is_empty() {
                        panic!(
                            "{} Non-numeric values provided for numeric-valued parameter!",
                            p.loc
                        );
                    }
                }
                StringParam::TYPE_NAME | "texture" => {
                    if p.strings.is_empty() {
                        panic!(
                            "{} Non-string values provided for string-valued parameter!",
                            p.loc
                        )
                    }
                }
                "spectrum" => {
                    if p.strings.is_empty() && p.ints.is_empty() && p.floats.is_empty() {
                        panic!("{} Expecting string or numeric-valued parameter for spectrum parameter.", p.loc)
                    }
                }
                _ => {
                    panic!("{} Unknown parameter type!", p.loc)
                }
            }
        }
    }

    fn lookup_single<P: ParameterType>(
        &mut self,
        name: &str,
        default_value: P::ReturnType,
    ) -> P::ReturnType {
        for p in &mut self.params {
            if p.name != name || p.param_type != P::TYPE_NAME {
                continue;
            }

            p.looked_up = true;
            let values = P::get_values(p);

            if values.is_empty() {
                panic!("{} No values provided for parameter {}", p.loc, p.name);
            }
            if values.len() != P::N_PER_ITEM as usize {
                panic!(
                    "{} Expected {} values for parameter {}",
                    p.loc,
                    P::N_PER_ITEM,
                    p.name
                );
            }

            return P::convert(values.as_slice(), &p.loc);
        }
        default_value
    }
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::BooleanParam {}
    impl Sealed for super::FloatParam {}
    impl Sealed for super::IntegerParam {}
    impl Sealed for super::Point2fParam {}
    impl Sealed for super::Vector2fParam {}
    impl Sealed for super::Point3fParam {}
    impl Sealed for super::Vector3fParam {}
    impl Sealed for super::Normal3fParam {}
    impl Sealed for super::SpectrumParam {}
    impl Sealed for super::StringParam {}
    impl Sealed for super::TextureParam {}
}
