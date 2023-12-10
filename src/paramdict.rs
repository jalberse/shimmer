use std::{collections::HashMap, fmt::Display, rc::Rc, sync::Arc};

use log::warn;

use crate::{
    color::RGB,
    colorspace::RgbColorSpace,
    options::Options,
    parser::{FileLoc, ParsedParameterVector},
    spectra::{
        named_spectrum,
        spectrum::{RgbAlbedoSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum},
        BlackbodySpectrum, NamedSpectrum, PiecewiseLinearSpectrum, Spectrum,
    },
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
    color_space: Option<Rc<RgbColorSpace>>,
    may_be_unused: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum SpectrumType {
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

    pub fn get_one_float(&mut self, name: &str, default_value: Float) -> Float {
        self.lookup_single::<FloatParam>(name, default_value)
    }

    pub fn get_one_int(&mut self, name: &str, default_value: i32) -> i32 {
        self.lookup_single::<IntegerParam>(name, default_value)
    }

    pub fn get_one_bool(&mut self, name: &str, default_value: bool) -> bool {
        self.lookup_single::<BooleanParam>(name, default_value)
    }

    pub fn get_one_point2f(&mut self, name: &str, default_value: Point2f) -> Point2f {
        self.lookup_single::<Point2fParam>(name, default_value)
    }

    pub fn get_one_vector2f(&mut self, name: &str, default_value: Vector2f) -> Vector2f {
        self.lookup_single::<Vector2fParam>(name, default_value)
    }

    pub fn get_one_vector3f(&mut self, name: &str, default_value: Vector3f) -> Vector3f {
        self.lookup_single::<Vector3fParam>(name, default_value)
    }

    pub fn get_one_normal3f(&mut self, name: &str, default_value: Normal3f) -> Normal3f {
        self.lookup_single::<Normal3fParam>(name, default_value)
    }

    pub fn get_one_string(&mut self, name: &str, default_value: String) -> String {
        self.lookup_single::<StringParam>(name, default_value)
    }

    pub fn get_one_spectrum(
        &mut self,
        name: &str,
        default_value: Spectrum,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> Arc<Spectrum> {
        let p = self.params.iter_mut().find(|p| p.name == name);
        if let Some(p) = p {
            let s = Self::extract_spectrum_array(
                p,
                spectrum_type,
                self.color_space.clone(),
                cached_spectra,
            );
            if !s.is_empty() {
                if s.len() > 1 {
                    panic!(
                        "{} More than one value provided for parameter {}",
                        p.loc, p.name
                    );
                }
                return s.into_iter().nth(0).expect("Expected non-empty vector");
            }
            Arc::new(default_value)
        } else {
            Arc::new(default_value)
        }
    }

    fn extract_spectrum_array(
        param: &mut ParsedParameter,
        spectrum_type: SpectrumType,
        color_space: Rc<RgbColorSpace>,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
    ) -> Vec<Arc<Spectrum>> {
        if param.param_type == "rgb" {
            // TODO We could also handle "color" in this block with an upgrade option, but
            //  I don't intend to use old PBRT scene files for now.

            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                3,
                |v: &[Float], loc: &FileLoc| -> Arc<Spectrum> {
                    let rgb = RGB::new(v[0], v[1], v[2]);
                    let cs = if let Some(cs) = &param.color_space {
                        cs.clone()
                    } else {
                        color_space.clone()
                    };
                    if rgb.r < 0.0 || rgb.g < 0.0 || rgb.b < 0.0 {
                        panic!(
                            "{} RGB Parameter {} has negative component",
                            loc, param.name
                        );
                    }
                    match spectrum_type {
                        SpectrumType::Illuminant => {
                            return Arc::new(Spectrum::RgbIlluminantSpectrum(
                                RgbIlluminantSpectrum::new(&cs, &rgb),
                            ));
                        }
                        SpectrumType::Albedo => {
                            if rgb.r > 1.0 || rgb.g > 1.0 || rgb.b > 1.0 {
                                panic!(
                                    "{} RGB Parameter {} has component value > 1.0",
                                    loc, param.name
                                );
                            }
                            Arc::new(Spectrum::RgbAlbedoSpectrum(RgbAlbedoSpectrum::new(
                                &cs, &rgb,
                            )))
                        }
                        SpectrumType::Unbounded => Arc::new(Spectrum::RgbUnboundedSpectrum(
                            RgbUnboundedSpectrum::new(&cs, &rgb),
                        )),
                    }
                },
            );
        } else if param.param_type == "blackbody" {
            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                1,
                |v: &[Float], _loc: &FileLoc| -> Arc<Spectrum> {
                    return Arc::new(Spectrum::Blackbody(BlackbodySpectrum::new(v[0])));
                },
            );
        } else if param.param_type == "spectrum" && !param.floats.is_empty() {
            if param.floats.len() % 2 != 0 {
                panic!(
                    "{} Found odd number of values for {}",
                    param.loc, param.name
                );
            }
            let n_samples = param.floats.len() / 2;
            if n_samples == 1 {
                warn!("{} {} Specified spectrum is only non-zero at a single wavelength; probably unintended", param.loc, param.name);
            }
            return Self::return_array(
                param.floats.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                param.floats.len() as i32,
                |v: &[Float], _loc: &FileLoc| -> Arc<Spectrum> {
                    let mut lambda = vec![0.0; n_samples];
                    let mut value = vec![0.0; n_samples];
                    for i in 0..n_samples {
                        if i > 0 && v[2 * i] <= lambda[i - 1] {
                            panic!("{} Spectrum description invalid: at {}'th entry, wavelengths aren't increasing: {} >= {}", param.loc, i - 1, lambda[i -1], v[2 * i]);
                        }
                        lambda[i] = v[2 * i];
                        value[i] = v[2 * i + 1];
                    }
                    return Arc::new(Spectrum::PiecewiseLinear(PiecewiseLinearSpectrum::new(
                        lambda.as_slice(),
                        value.as_slice(),
                    )));
                },
            );
        } else if param.param_type == "spectrum" && !param.strings.is_empty() {
            return Self::return_array(
                param.strings.as_slice(),
                &param.loc,
                &param.name,
                &mut param.looked_up,
                1,
                |s: &[String], loc: &FileLoc| -> Arc<Spectrum> {
                    let named_spectrum = NamedSpectrum::from_str(&s[0]);
                    if let Some(named_spectrum) = named_spectrum {
                        return Spectrum::get_named_spectrum(named_spectrum);
                    }

                    let spd = Spectrum::read_from_file(&s[0], cached_spectra);
                    if spd.is_none() {
                        panic!("{} Unable to read/invalid spectrum file {}", &s[0], loc);
                    }
                    return spd.unwrap();
                },
            );
        } else {
            return Vec::new();
        }
    }

    fn return_array<ValueType, ReturnType, C>(
        values: &[ValueType],
        loc: &FileLoc,
        name: &str,
        looked_up: &mut bool,
        n_per_item: i32,
        mut convert: C,
    ) -> Vec<ReturnType>
    where
        C: FnMut(&[ValueType], &FileLoc) -> ReturnType,
    {
        if values.is_empty() {
            panic!("{} No values provided for {}", loc, name);
        }
        if values.len() % n_per_item as usize != 0 {
            panic!(
                "{} Number of values provided for {} is not a multiple of {}",
                loc, name, n_per_item
            );
        }

        *looked_up = true;
        let n = values.len() / n_per_item as usize;

        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v[i] = convert(&values[n_per_item as usize * i..], &loc);
        }
        v
    }

    fn lookup_array<P: ParameterType>(&mut self, name: &str) -> Vec<P::ReturnType> {
        for p in &mut self.params {
            if p.name == name || p.param_type == P::TYPE_NAME {
                let mut looked_up = p.looked_up;
                let to_return = Self::return_array(
                    P::get_values(p),
                    &p.loc,
                    &p.name,
                    &mut looked_up,
                    P::N_PER_ITEM,
                    P::convert,
                );
                p.looked_up = looked_up;
                return to_return;
            }
        }
        Vec::new()
    }

    fn get_float_array(&mut self, name: &str) -> Vec<Float> {
        self.lookup_array::<FloatParam>(name)
    }

    fn get_int_array(&mut self, name: &str) -> Vec<i32> {
        self.lookup_array::<IntegerParam>(name)
    }

    fn get_bool_array(&mut self, name: &str) -> Vec<bool> {
        self.lookup_array::<BooleanParam>(name)
    }

    fn get_point2f_array(&mut self, name: &str) -> Vec<Point2f> {
        self.lookup_array::<Point2fParam>(name)
    }

    fn get_vector2f_array(&mut self, name: &str) -> Vec<Vector2f> {
        self.lookup_array::<Vector2fParam>(name)
    }

    fn get_point3f_array(&mut self, name: &str) -> Vec<Point3f> {
        self.lookup_array::<Point3fParam>(name)
    }

    fn get_vector3f_array(&mut self, name: &str) -> Vec<Vector3f> {
        self.lookup_array::<Vector3fParam>(name)
    }

    fn get_normal3f_array(&mut self, name: &str) -> Vec<Normal3f> {
        self.lookup_array::<Normal3fParam>(name)
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