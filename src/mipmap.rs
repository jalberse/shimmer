
use std::{path::Path, sync::Arc};

use ordered_float::OrderedFloat;

use crate::{color::ColorEncodingPtr, colorspace::RgbColorSpace, image::{Image, WrapMode}, options::Options, vecmath::{Point2i, Tuple2}, Float};


pub struct MIPMap
{
    pyramid: Vec<Image>,
    color_space: Arc<RgbColorSpace>,
    wrap_mode: WrapMode,
    options: MIPMapFilterOptions,
}

impl MIPMap
{
    pub fn new(
        image: Image,
        color_space: Arc<RgbColorSpace>,
        wrap_mode: WrapMode,
        filter_options: MIPMapFilterOptions,
        options: &Options,
    ) -> MIPMap
    {
        let mut pyramid = Image::generate_pyramid(image, wrap_mode.into());
        if options.disable_image_textures
        {
            let top = pyramid.pop().unwrap();
            pyramid.clear();
            pyramid.push(top);
        }
        MIPMap
        {
            pyramid,
            color_space,
            wrap_mode,
            options: filter_options,   
        }
    }

    pub fn create_from_file(
        filename: &str,
        filter_options: MIPMapFilterOptions,
        wrap_mode: WrapMode,
        encoding: ColorEncodingPtr,
        options: &Options,
    ) -> MIPMap
    {
        let image_and_metadata = Image::read(Path::new(filename), Some(encoding));

        let mut image = image_and_metadata.image;
        if image.n_channels() != 1
        {
            let rgba_desc = image.get_channel_desc(&["R", "G", "B", "A"]);
            let rgb_desc = image.get_channel_desc(&["R", "G", "B"]);
            if let Some(rgba_desc) = rgba_desc
            {
                // Is alpha all ones?
                let mut all_one = true;
                for y in 0..image.resolution().y
                {
                    for x in 0..image.resolution().x
                    {
                        if image.get_channels_from_desc(Point2i::new(x, y), &rgba_desc)[3] != 1.0
                        {
                            all_one = false;
                        }
                    }
                }
                if all_one
                {
                    if let Some(rgb_desc) = rgb_desc
                    {
                        image = image.select_channels(&rgb_desc)
                    }
                    else {
                        panic!("{}: Expected RGB channels", filename);
                    }
                }
                else
                {
                    image = image.select_channels(&rgba_desc);
                }
            }
            else if let Some(rgb_desc) = rgb_desc
            {
                image = image.select_channels(&rgb_desc);
            }
            else
            {
                panic!("{}: Image doesn't have RGB channels", filename);
            }
        }

        let color_space = image_and_metadata.metadata.color_space.expect("Expected color space");

        MIPMap::new(
            image,
            color_space,
            wrap_mode,
            filter_options,
            options,
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FilterFunction
{
    Point,
    Bilinear,
    Trilinear,
    EWA,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct MIPMapFilterOptions
{
    filter: FilterFunction,
    // Use an OrderedFloat because this is used as a key in a hashmap cache
    max_anisotropy: OrderedFloat<Float>,
}

impl Default for MIPMapFilterOptions
{
    fn default() -> Self
    {
        MIPMapFilterOptions {
            filter: FilterFunction::EWA,
            max_anisotropy: OrderedFloat(8.0),
        }
    }
}