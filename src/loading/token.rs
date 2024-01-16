// Adapted from pbrt4 crate under apache license.

use std::fmt;
use std::str::FromStr;

use super::error::Error;

#[derive(Debug, PartialEq, Eq)]
pub struct Token<'a> {
    str: &'a str,
    // TODO We'd like to store a FileLoc here.
}

impl<'a> Token<'a> {
    pub fn new(str: &'a str) -> Self {
        Token { str }
    }

    /// Returns the number of chars in the token.
    pub fn token_size(&self) -> usize {
        self.str.len()
    }

    /// Returns token's value.
    pub fn value(&self) -> &'a str {
        self.str.trim()
    }

    /// Parse token.
    pub fn parse<F: FromStr>(&self) -> Result<F, <F as FromStr>::Err> {
        self.str.parse::<F>()
    }

    /// Returns `true` if token is a quoted string.
    pub fn is_quote(&self) -> bool {
        self.str.len() >= 2 && self.str.starts_with('\"') && self.str.ends_with('\"')
    }

    /// Returns `true` when token is directive.
    pub fn is_directive(&self) -> bool {
        Directive::from_str(self.str).is_ok()
    }

    /// Get directive.
    pub fn directive(&self) -> Option<Directive> {
        Directive::from_str(self.str).ok()
    }

    /// Whether token is `[`.
    pub fn is_open_brace(&self) -> bool {
        self.str == "["
    }

    /// Whether token is `]`
    pub fn is_close_brace(&self) -> bool {
        self.str == "]"
    }

    /// Return a string without quotes or `None` if the string is not a quoted string.
    pub fn unquote(&self) -> Option<&'a str> {
        if self.is_quote() {
            let len = self.str.len();
            Some(&self.str[1..len - 1])
        } else {
            None
        }
    }

    /// Check whether token is valid.
    pub fn is_valid(&self) -> bool {
        // Empty tokens are not allowed, something wrong with tokenizer
        if self.str.is_empty() {
            return false;
        }

        // Validate quoted string

        let starts_with_quote = self.str.starts_with('\"');
        let ends_with_quote = self.str.ends_with('\"');

        if starts_with_quote || ends_with_quote {
            // Should both start and end with "
            if starts_with_quote != ends_with_quote {
                return false;
            }

            if self.str.len() < 2 {
                return false;
            }
        }

        // No spaces unless its a quotes string
        if !starts_with_quote && self.str.contains(' ') {
            return false;
        }

        true
    }
}

/// Type of pbrt directive if [Token] is directive.
#[derive(Debug, PartialEq)]
pub enum Directive {
    Identity,
    Translate,
    Scale,
    Rotate,
    LookAt,
    CoordinateSystem,
    CoordSysTransform,
    Transform,
    ConcatTransform,
    TransformTimes,
    ActiveTransform,

    Include,
    Import,

    Option,

    Camera,
    Sampler,
    ColorSpace,
    Film,
    Integrator,
    Accelerator,
    PixelFilter,

    MakeNamedMedium,
    MediumInterface,

    WorldBegin,

    AttributeBegin,
    AttributeEnd,
    Attribute,

    Shape,
    ReverseOrientation,
    ObjectBegin,
    ObjectEnd,
    ObjectInstance,

    LightSource,
    AreaLightSource,

    Material,
    Texture,
    MakeNamedMaterial,
    NamedMaterial,
}

impl fmt::Display for Directive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl FromStr for Directive {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let e = match s {
            "Identity" => Directive::Identity,
            "Translate" => Directive::Translate,
            "Scale" => Directive::Scale,
            "Rotate" => Directive::Rotate,
            "LookAt" => Directive::LookAt,
            "CoordinateSystem" => Directive::CoordinateSystem,
            "CoordSysTransform" => Directive::CoordSysTransform,
            "Transform" => Directive::Transform,
            "ConcatTransform" => Directive::ConcatTransform,
            "TransformTimes" => Directive::TransformTimes,
            "ActiveTransform" => Directive::ActiveTransform,
            "Include" => Directive::Include,
            "Import" => Directive::Import,
            "Option" => Directive::Option,
            "Camera" => Directive::Camera,
            "Sampler" => Directive::Sampler,
            "ColorSpace" => Directive::ColorSpace,
            "Film" => Directive::Film,
            "Integrator" => Directive::Integrator,
            "Accelerator" => Directive::Accelerator,
            "MakeNamedMedium" => Directive::MakeNamedMedium,
            "MediumInterface" => Directive::MediumInterface,
            "WorldBegin" => Directive::WorldBegin,
            "AttributeBegin" => Directive::AttributeBegin,
            "AttributeEnd" => Directive::AttributeEnd,
            "Attribute" => Directive::Attribute,
            "Shape" => Directive::Shape,
            "ReverseOrientation" => Directive::ReverseOrientation,
            "ObjectBegin" => Directive::ObjectBegin,
            "ObjectEnd" => Directive::ObjectEnd,
            "ObjectInstance" => Directive::ObjectInstance,
            "LightSource" => Directive::LightSource,
            "AreaLightSource" => Directive::AreaLightSource,
            "Material" => Directive::Material,
            "Texture" => Directive::Texture,
            "MakeNamedMaterial" => Directive::MakeNamedMaterial,
            "NamedMaterial" => Directive::NamedMaterial,
            "PixelFilter" => Directive::PixelFilter,
            _ => return Err(Error::UnknownDirective),
        };

        Ok(e)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::loading::token::Directive;

    use super::Token;

    #[test]
    fn is_quote() {
        assert!(Token::new("\"foo\"").is_quote());
        assert!(Token::new("\"\"").is_quote());

        assert!(!Token::new("").is_quote());
        assert!(!Token::new("\"").is_quote());
        assert!(!Token::new("\"abc").is_quote());
        assert!(!Token::new("abc\"").is_quote());
    }

    #[test]
    fn unquote_str() {
        assert_eq!(Token::new("\"foo\"").unquote(), Some("foo"));
        assert_eq!(Token::new("\"\"").unquote(), Some(""));

        assert_eq!(Token::new("").unquote(), None);
        assert_eq!(Token::new("\"").unquote(), None);
        assert_eq!(Token::new("\"abc").unquote(), None);
        assert_eq!(Token::new("abc\"").unquote(), None);
    }

    #[test]
    fn parse() {
        assert_eq!(Token::new("32").parse(), Ok(32_u32));

        assert!(Token::new("").parse::<u32>().is_err());
        assert!(Token::new("-").parse::<u32>().is_err());
    }

    #[test]
    fn is_valid_token() {
        assert!(Token::new("bar").is_valid());
        assert!(Token::new("\"foo\"").is_valid());
        assert!(Token::new("\"foo bar\"").is_valid());

        assert!(!Token::new("").is_valid());
        assert!(!Token::new("\"").is_valid());
        assert!(!Token::new("\"foo").is_valid());
        assert!(!Token::new("bar\"").is_valid());
        assert!(!Token::new("foo bar").is_valid());
    }

    #[test]
    fn parse_directive() {
        assert_eq!(Directive::from_str("Shape").unwrap(), Directive::Shape);
        assert_eq!(
            Directive::from_str("Identity").unwrap(),
            Directive::Identity
        );
        assert_eq!(
            Directive::from_str("Material").unwrap(),
            Directive::Material
        );
        assert_eq!(Directive::from_str("Texture").unwrap(), Directive::Texture);
        assert_eq!(
            Directive::from_str("WorldBegin").unwrap(),
            Directive::WorldBegin
        );
        assert_eq!(
            Directive::from_str("AttributeBegin").unwrap(),
            Directive::AttributeBegin
        );
        assert_eq!(
            Directive::from_str("AttributeEnd").unwrap(),
            Directive::AttributeEnd
        );
    }

    #[test]
    fn token_is_directive() {
        assert!(Token::new("LookAt").is_directive());
        assert!(Token::new("Rotate").is_directive());
        assert!(Token::new("Scale").is_directive());

        assert!(!Token::new("\"Scale\"").is_directive());
        assert!(!Token::new("").is_directive());
        assert!(!Token::new("123").is_directive());
    }
}
