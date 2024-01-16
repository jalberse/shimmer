// Adapted from pbrt4 crate under apache license.

use std::{
    io,
    num::{ParseFloatError, ParseIntError},
    str::ParseBoolError,
};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    /// No more tokens.
    #[error("No tokens")]
    EndOfFile,

    /// Expected a token, but received `None`.
    #[error("Token expected, got end of stream")]
    NoToken,

    #[error("Failed to read file")]
    Io(#[from] io::Error),

    /// Token didn't pass basic validation checks.
    #[error("Invalid token")]
    InvalidToken,

    /// Failed to parse string to float.
    #[error("Unable to parse float")]
    ParseFloat(#[from] ParseFloatError),

    /// Failed to parse string to integer.
    #[error("Unable to parse number")]
    ParseInt(#[from] ParseIntError),

    /// Failed to parse boolean.
    #[error("Unable to parse bool")]
    ParseBool(#[from] ParseBoolError),

    /// Unable to cast from slice to array.
    #[error("Unexpected number of arguments in array")]
    ParseSlice,

    /// Directive is unknown.
    #[error("Unsupported directive")]
    UnknownDirective,

    #[error("Expected string token")]
    InvalidString,

    /// Failed to parse option's `[ value ]`
    #[error("Unable to parse option value")]
    InvalidOptionValue,

    #[error("Unsupported coordinate system")]
    UnknownCoordinateSystem,

    #[error("Invalid parameter name")]
    InvalidParamName,

    /// Unsupported parameter type.
    #[error("Parameter type is invalid")]
    InvalidParamType,

    #[error("Found duplicated parameter")]
    DuplicatedParamName,

    #[error("Duplicated WorldBegin statement")]
    WorldAlreadyStarted,

    #[error("Element is not allowed")]
    ElementNotAllowed,

    #[error("Too many AttributeEnd")]
    TooManyEndAttributes,

    #[error("Attempt to restore CoordSysTransform matrix with invalid name")]
    InvalidMatrixName,

    #[error("Invalid camera type")]
    InvalidCameraType,

    #[error("Unknown object type")]
    InvalidObjectType,

    #[error("Unexpted token received")]
    UnexpectedToken,

    #[error("Required param is missing")]
    MissingRequiredParameter,

    #[error("Nested object attributes are not allowed")]
    NestedObjects,

    #[error("Not found")]
    NotFound,
}
