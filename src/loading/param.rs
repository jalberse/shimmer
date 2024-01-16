// Adapted from pbrt4 crate under apache license.

use std::{
    collections::HashMap,
    num::{ParseFloatError, ParseIntError},
    result,
    str::{FromStr, ParseBoolError},
};

use crate::loading::error::{Error, Result};

/// Parameter type.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ParamType {
    Boolean,
    Float,
    Integer,
    Point2,
    Point3,
    Vector2,
    Vector3,
    Normal3,
    Spectrum,
    Rgb,
    Blackbody,
    String,
    Texture,
}

impl FromStr for ParamType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let ty = match s {
            "bool" => ParamType::Boolean,
            "integer" => ParamType::Integer,
            "float" => ParamType::Float,
            "point2" => ParamType::Point2,
            "vector2" => ParamType::Vector2,
            "point3" => ParamType::Point3,
            "vector3" => ParamType::Vector3,
            "normal3" => ParamType::Normal3,
            "spectrum" => ParamType::Spectrum,
            "rgb" => ParamType::Rgb,
            "blackbody" => ParamType::Blackbody,
            "string" => ParamType::String,
            "texture" => ParamType::Texture,
            _ => return Err(Error::InvalidParamType),
        };

        Ok(ty)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Spectrum {
    //  "rgb L" [ r g b ]
    Rgb([f32; 3]),
    // "blackbody L" 3000
    Blackbody(i32),
}

/// Represents a single parameter.
/// Similar to ParsedParameter, but stores the parameter's values as a string.
/// ParsedParameter stores the values as a vector of the appropriate type,
/// and provides some further utilities.
#[derive(Debug, PartialEq, Clone)]
pub struct Param<'a> {
    /// The name of the parameter, e.g. "radius"
    pub name: &'a str,
    /// The type of the parameter; e.g. "float"
    pub ty: ParamType,
    /// One or more values; not yet parsed.
    pub value: &'a str,
}

impl<'a> Param<'a> {
    pub fn new(type_and_name: &'a str, value: &'a str) -> Result<Self> {
        // Param name is "type name"
        let mut split = type_and_name.split_whitespace();

        let ty_name = split.next().ok_or(Error::InvalidParamName)?;
        let ty = ParamType::from_str(ty_name)?;

        let name = split.next().ok_or(Error::InvalidParamName)?;

        Ok(Self { name, ty, value })
    }

    pub fn items<T: FromStr>(
        &self,
    ) -> impl Iterator<Item = result::Result<T, <T as FromStr>::Err>> + 'a {
        self.value.split_whitespace().map(|str| T::from_str(str))
    }

    pub fn rgb(&self) -> Result<[f32; 3]> {
        let mut iter = self.items::<f32>();

        let r = iter.next().ok_or(Error::MissingRequiredParameter)??;
        let g = iter.next().ok_or(Error::MissingRequiredParameter)??;
        let b = iter.next().ok_or(Error::MissingRequiredParameter)??;

        Ok([r, g, b])
    }

    pub fn single<T: FromStr>(&self) -> result::Result<T, <T as FromStr>::Err> {
        T::from_str(self.value)
    }

    pub fn vec<T: FromStr>(&self) -> result::Result<Vec<T>, <T as FromStr>::Err> {
        self.items()
            .collect::<result::Result<Vec<T>, <T as FromStr>::Err>>()
    }

    pub fn spectrum(&self) -> Result<Spectrum> {
        let res = match self.ty {
            ParamType::Rgb => Spectrum::Rgb(self.rgb()?),
            ParamType::Blackbody => Spectrum::Blackbody(self.single()?),
            _ => return Err(Error::InvalidObjectType),
        };

        Ok(res)
    }
}

/// Parameters collection.
#[derive(Default, Debug, PartialEq, Clone)]
pub struct ParamList<'a>(pub HashMap<&'a str, Param<'a>>);

impl<'a> ParamList<'a> {
    /// Add a new parameter to the list.
    pub fn add(&mut self, param: Param<'a>) -> Result<()> {
        if self.0.insert(param.name, param).is_some() {
            return Err(Error::DuplicatedParamName);
        }

        Ok(())
    }

    /// Get parameter by name.
    pub fn get(&self, name: &str) -> Option<&Param<'a>> {
        self.0.get(name)
    }

    /// Return the number of parameters.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the list is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn vec<T: FromStr>(&self, name: &str) -> result::Result<Option<Vec<T>>, <T as FromStr>::Err> {
        let res = match self.get(name).map(|param| param.vec()) {
            Some(v) => Some(v?),
            None => None,
        };

        Ok(res)
    }

    pub fn floats(&self, name: &str) -> result::Result<Option<Vec<f32>>, ParseFloatError> {
        self.vec(name)
    }

    pub fn integers(&self, name: &str) -> result::Result<Option<Vec<i32>>, ParseIntError> {
        self.vec(name)
    }

    fn single<T: FromStr>(&self, name: &str, default: T) -> result::Result<T, <T as FromStr>::Err> {
        self.get(name)
            .map(|p| p.single::<T>())
            .unwrap_or(Ok(default))
    }

    /// Get a float value by name.
    ///
    /// If there is no parameter with name `name`, a `default` value will
    /// be returned.
    ///
    /// If there is a value and it's not possible to parse it into float,
    /// an error will be returned.
    pub fn float(&self, name: &str, default: f32) -> result::Result<f32, ParseFloatError> {
        self.single(name, default)
    }

    pub fn integer(&self, name: &str, default: i32) -> result::Result<i32, ParseIntError> {
        self.single(name, default)
    }

    pub fn boolean(&self, name: &str, default: bool) -> result::Result<bool, ParseBoolError> {
        self.single(name, default)
    }

    pub fn string(&self, name: &str) -> Option<&str> {
        self.get(name).map(|v| v.value)
    }

    pub fn extend(&mut self, other: &ParamList<'a>) {
        for (k, v) in &other.0 {
            self.0.insert(k, v.clone());
        }
    }
}

// TODO Expand these tests to also convert to our ParsedParameter type, and verify we parse the values correctly.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_param_type() {
        assert_eq!(ParamType::from_str("bool").ok(), Some(ParamType::Boolean));

        assert_eq!(ParamType::from_str("float").ok(), Some(ParamType::Float));
        assert_eq!(
            ParamType::from_str("integer").ok(),
            Some(ParamType::Integer)
        );

        assert_eq!(ParamType::from_str("point2").ok(), Some(ParamType::Point2));
        assert_eq!(ParamType::from_str("point3").ok(), Some(ParamType::Point3));

        assert_eq!(ParamType::from_str("rgb").ok(), Some(ParamType::Rgb));
    }

    #[test]
    fn add_dup_param() {
        let mut list = ParamList::default();

        let param = Param::new("bool dup_name", "true").unwrap();
        list.add(param.clone()).unwrap();

        assert!(matches!(list.add(param), Err(Error::DuplicatedParamName)));
    }

    #[test]
    fn as_ints() {
        let param = Param::new("integer test", "-1 0 1").unwrap();

        assert_eq!(param.vec::<i32>().unwrap(), vec![-1, 0, 1]);
    }

    #[test]
    fn parse_blackbody() -> Result<()> {
        let param = Param::new("blackbody I", "5500")?;

        let i = param.spectrum().unwrap();

        assert!(matches!(i, Spectrum::Blackbody(5500)));
        Ok(())
    }

    #[test]
    fn parse_rgb() -> Result<()> {
        let param = Param::new("rgb L", "7 0 7")?;
        let i = param.spectrum().unwrap();

        assert!(matches!(i, Spectrum::Rgb(_)));
        Ok(())
    }
}
