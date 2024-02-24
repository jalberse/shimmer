
use std::path::Path;

use crate::{color::ColorEncodingPtr, image::{Image, WrapMode}};


pub struct MIPMap
{
    // TODO
}

impl MIPMap
{
    pub fn create_from_file(
        filename: &str,
        options: &MIPMapFilterOptions,
        wrap_mode: WrapMode,
        encoding: ColorEncodingPtr,
    ) -> MIPMap
    {
        // TODO So, I guess ColorEncoding can take an Option<ColorEncodingPtr> instead?
        todo!()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MIPMapFilterOptions
{
    // TODO
}