use arrayvec::ArrayVec;
use core::fmt;
use half::f16;
use log::Metadata;
use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufWriter, Write},
    ops::{Index, IndexMut},
    path::Path,
    sync::Arc,
};

use crate::{
    bounding_box::Bounds2i,
    color::{ColorEncoding, ColorEncodingI, SRgbColorEncoding, RGB},
    colorspace::RgbColorSpace,
    float::Float,
    math::lerp,
    square_matrix::SquareMatrix,
    util::has_extension,
    vec2d::Vec2d,
    vecmath::{Point2f, Point2i, Tuple2},
};

#[cfg(target_endian = "little")]
const HOST_LITTLE_ENDIAN: bool = true;

#[cfg(target_endian = "big")]
const HOST_LITTLE_ENDIAN: bool = false;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum PixelFormat {
    U256,
    Half,
    Float,
}

impl PixelFormat {
    pub fn is_8bit(&self) -> bool {
        *self == PixelFormat::U256
    }

    pub fn is_16bit(&self) -> bool {
        *self == PixelFormat::Half
    }

    pub fn is_32bit(&self) -> bool {
        *self == PixelFormat::Float
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelFormat::U256 => write!(f, "U256"),
            PixelFormat::Half => write!(f, "Half"),
            PixelFormat::Float => write!(f, "Float"),
        }
    }
}

impl PixelFormat {
    pub fn texel_bytes(&self) -> i32 {
        match self {
            PixelFormat::U256 => 1,
            PixelFormat::Half => 2,
            PixelFormat::Float => 4,
        }
    }
}

pub struct ResampleWeight {
    first_pixel: i32,
    weight: [Float; 4],
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum WrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
}

impl WrapMode {
    pub fn parse_wrap_mode(str: &str) -> Option<Self> {
        if str.eq(WRAP_MODE_CLAMP_STR) {
            Some(WrapMode::Clamp)
        } else if str.eq(WRAP_MODE_REPEAT_STR) {
            Some(WrapMode::Repeat)
        } else if str.eq(WRAP_MODE_BLACK_STR) {
            Some(WrapMode::Black)
        } else if str.eq(WRAP_MODE_OCTAHEDRAL_SPHERE_STR) {
            Some(WrapMode::OctahedralSphere)
        } else {
            None
        }
    }
}

/// Dictates the wrap mode for an image in both the horizontal and
/// vertical directions, so that they can have independent wrap modes.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WrapMode2D {
    wrap: [WrapMode; 2],
}

static WRAP_MODE_REPEAT_STR: &str = "repeat";
static WRAP_MODE_CLAMP_STR: &str = "clamp";
static WRAP_MODE_BLACK_STR: &str = "black";
static WRAP_MODE_OCTAHEDRAL_SPHERE_STR: &str = "octahedralsphere";

impl WrapMode2D {
    pub fn new(x: WrapMode, y: WrapMode) -> Self {
        Self { wrap: [x, y] }
    }
}

impl From<WrapMode> for WrapMode2D {
    fn from(value: WrapMode) -> Self {
        WrapMode2D {
            wrap: [value, value],
        }
    }
}

impl fmt::Display for WrapMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WrapMode::Black => write!(f, "{}", WRAP_MODE_BLACK_STR),
            WrapMode::Clamp => write!(f, "{}", WRAP_MODE_CLAMP_STR),
            WrapMode::Repeat => write!(f, "{}", WRAP_MODE_REPEAT_STR),
            WrapMode::OctahedralSphere => write!(f, "{}", WRAP_MODE_OCTAHEDRAL_SPHERE_STR),
        }
    }
}

/// Remaps the given pixel coordinate pp in place given the resolution and wrap mode of an image.
pub fn remap_pixel_coords(p: &mut Point2i, resolution: Point2i, wrap_mode: WrapMode2D) -> bool {
    if wrap_mode.wrap[0] == WrapMode::OctahedralSphere {
        // For Octahedral sphere, we assume both directions have the same wrap mode.
        debug_assert!(wrap_mode.wrap[1] == WrapMode::OctahedralSphere);
        if p[0] < 0 {
            p[0] = -p[0]; // Mirror across u = 0
            p[1] = resolution[1] - 1 - p[1]; // Mirror across v = 0.5
        } else if p[0] >= resolution[0] {
            p[0] = 2 * resolution[0] - 1 - p[0]; // Mirror across u = 1
            p[1] = resolution[1] - 1 - p[1]; // Mirror across v = 0.5
        }

        if p[1] < 0 {
            p[0] = resolution[0] - 1 - p[0]; // Mirror across u = 0.5
            p[1] = -p[1]; // Mirror across v = 0
        } else if p[1] >= resolution[1] {
            p[0] = resolution[0] - 1 - p[0]; // Mirror across u = 0.5
            p[1] = 2 * resolution[1] - 1 - p[1]; // Mirror across v = 1
        }

        if resolution[0] == 1 {
            p[0] = 0;
        }
        if resolution[1] == 1 {
            p[1] = 0;
        }

        return true;
    }

    for c in 0..2 {
        if p[c] >= 0 && p[c] < resolution[c] {
            // In bounds
            continue;
        }

        match wrap_mode.wrap[c] {
            WrapMode::Black => return false,
            WrapMode::Clamp => p[c] = p[c].clamp(0, resolution[c] - 1),
            WrapMode::Repeat => p[c] = p[c] % resolution[c],
            WrapMode::OctahedralSphere => panic!("Octahedral Sphere should be handled prior!"),
        }
    }
    true
}

// TODO Note we had a different .rs file with image_metadata.
//   I think we can just delete that, replacing it with this.
pub struct ImageMetadata {
    pub render_time_seconds: Option<Float>,
    pub camera_from_world: Option<SquareMatrix<4>>,
    pub ndc_from_world: Option<SquareMatrix<4>>,
    pub pixel_bounds: Option<Bounds2i>,
    pub full_resolution: Option<Point2i>,
    pub samples_per_pixel: Option<i32>,
    pub mse: Option<Float>,
    pub color_space: Option<Arc<RgbColorSpace>>,
    pub strings: HashMap<String, String>,
    pub string_ves: HashMap<String, Vec<String>>,
}

impl Default for ImageMetadata {
    fn default() -> Self {
        Self {
            render_time_seconds: None,
            camera_from_world: None,
            ndc_from_world: None,
            pixel_bounds: None,
            full_resolution: None,
            samples_per_pixel: None,
            mse: None,
            color_space: None,
            strings: Default::default(),
            string_ves: Default::default(),
        }
    }
}

pub struct ImageAndMetadata {
    pub image: Image,
    pub metadata: ImageMetadata,
}

// PAPERDOC - PBRT rolls its own `InlinedVector` class for a vector that can grow up to N in size.
// In Rust, it's trivial for me to find the arrayvec crate and add it to my project.
// Obviously libraries exist for C++, but they tend to be more tedious to add to the project -
// I think that people often roll their own if there's not something in boost, unless it's truly a large
// library/dependency. This can lead to more bugs and less time doing useful development.
// I have no data to back this up. This is vibes only for now.

pub struct ImageChannelDesc {
    offset: ArrayVec<i32, 4>,
}

impl Default for ImageChannelDesc {
    fn default() -> Self {
        Self {
            offset: Default::default(),
        }
    }
}

impl ImageChannelDesc {
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    pub fn size(&self) -> usize {
        self.offset.len()
    }

    pub fn is_identity(&self) -> bool {
        for i in 0..self.offset.len() {
            if self.offset[i] != i as i32 {
                return false;
            }
        }
        true
    }
}

pub struct ImageChannelValues {
    pub values: ArrayVec<Float, 4>,
}

impl Default for ImageChannelValues {
    fn default() -> Self {
        Self {
            values: Default::default(),
        }
    }
}

impl ImageChannelValues {
    pub fn new(size: usize) -> Self {
        Self {
            values: ArrayVec::new(),
        }
    }

    pub fn max_value(&self) -> Float {
        *self
            .values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).expect("Unexpected NaN!"))
            .expect("Tried to find the max value of an empty vector")
    }

    pub fn average(&self) -> Float {
        self.values.iter().sum::<Float>() / self.values.len() as Float
    }
}

impl Index<usize> for ImageChannelValues {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for ImageChannelValues {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl From<ImageChannelValues> for Float {
    fn from(value: ImageChannelValues) -> Self {
        debug_assert!(value.values.len() == 1);
        value[0]
    }
}

impl From<ImageChannelValues> for [Float; 3] {
    fn from(value: ImageChannelValues) -> Self {
        debug_assert!(value.values.len() == 3);
        [value[0], value[1], value[2]]
    }
}

/// Stores a 2D array of pixel values where each pixel stores
/// a fixed unmber of scalar-valued channels (e.g. RGB is three channels).
pub struct Image {
    format: PixelFormat,
    resolution: Point2i,
    channel_names: Vec<String>,
    /// For images with fixed-precision (non-floating-point) pixel values, a ColorEncoding is specified.
    /// This is for e.g. PNG images; floating point formats like EXR will not use this.
    color_encoding: Option<ColorEncoding>,
    // Which one of p8, p16, or p32 is used depends on the PixelFormat.
    p8: Vec<u8>,
    p16: Vec<f16>,
    p32: Vec<f32>,
}

impl Image {
    pub fn new(
        format: PixelFormat,
        resolution: Point2i,
        channels: &[String],
        encoding: Option<ColorEncoding>,
    ) -> Image {
        // TODO We could improve this API by splitting into fixed-point and floating-point versions,
        // with only the former taking an encoding. That would remove the need for checks...

        let mut image = Image {
            format,
            resolution,
            channel_names: channels.to_vec(),
            color_encoding: encoding,
            p8: Default::default(),
            p16: Default::default(),
            p32: Default::default(),
        };

        if format.is_8bit() {
            image.p8.resize(
                image.n_channels() * resolution[0] as usize * resolution[1] as usize,
                0,
            );
            debug_assert!(image.color_encoding.is_some());
        } else if format.is_16bit() {
            image.p16.resize(
                image.n_channels() * resolution[0] as usize * resolution[1] as usize,
                f16::from_f32(0.0),
            );
            debug_assert!(image.color_encoding.is_none());
        } else if format.is_32bit() {
            image.p32.resize(
                image.n_channels() * resolution[0] as usize * resolution[1] as usize,
                0.0,
            );
            debug_assert!(image.color_encoding.is_none());
        } else {
            panic!("Unsupported image format in Image::new()")
        }
        image
    }

    pub fn new_p8(
        p8: Vec<u8>,
        resolution: Point2i,
        channels: &[String],
        encoding: ColorEncoding,
    ) -> Image {
        debug_assert!(p8.len() as i32 == channels.len() as i32 * resolution[0] * resolution[1]);
        Image {
            format: PixelFormat::U256,
            resolution,
            channel_names: channels.to_vec(),
            color_encoding: Some(encoding),
            p8,
            p16: Default::default(),
            p32: Default::default(),
        }
    }

    pub fn new_p16(p16: Vec<f16>, resolution: Point2i, channels: &[String]) -> Image {
        debug_assert!(p16.len() as i32 == channels.len() as i32 * resolution[0] * resolution[1]);
        Image {
            format: PixelFormat::Half,
            resolution,
            channel_names: channels.to_vec(),
            color_encoding: None,
            p8: Default::default(),
            p16,
            p32: Default::default(),
        }
    }

    pub fn new_p32(p32: Vec<f32>, resolution: Point2i, channels: &[String]) -> Image {
        debug_assert!(p32.len() as i32 == channels.len() as i32 * resolution[0] * resolution[1]);
        Image {
            format: PixelFormat::Half,
            resolution,
            channel_names: channels.to_vec(),
            color_encoding: None,
            p8: Default::default(),
            p16: Default::default(),
            p32,
        }
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn resolution(&self) -> Point2i {
        self.resolution
    }

    pub fn n_channels(&self) -> usize {
        self.channel_names.len()
    }

    pub fn channel_names(&self) -> Vec<String> {
        self.channel_names.clone()
    }

    pub fn encoding(&self) -> &Option<ColorEncoding> {
        &self.color_encoding
    }

    pub fn is_empty(&self) -> bool {
        self.resolution.x > 0 && self.resolution.y < 0
    }

    pub fn bytes_used(&self) -> usize {
        self.p8.len() + 2 * self.p16.len() + 4 * self.p32.len()
    }

    /// Returns the offset intot he pixel value array for given integer
    /// pixel coordinates.
    pub fn pixel_offset(&self, p: Point2i) -> usize {
        debug_assert!(Bounds2i::new(Point2i { x: 0, y: 0 }, self.resolution).inside_exclusive(&p));
        (self.n_channels() as i32 * (p.y * self.resolution.x + p.x)) as usize
    }

    pub fn get_channel(&self, p: Point2i, c: usize) -> Float {
        self.get_channel_wrapped(p, c, WrapMode::Clamp.into())
    }

    /// Returns the floating point value for a single image channel,
    /// taking care of both addressing pixels and converting the in-memory value to a float.
    /// It is the caller's responsibility to track what is being stored in each channel.
    pub fn get_channel_wrapped(&self, p: Point2i, c: usize, wrap_mode: WrapMode2D) -> Float {
        // Remap provided pixel coordinates before reading the channel
        let mut p = p;
        if !remap_pixel_coords(&mut p, self.resolution, wrap_mode) {
            // If WrapMode::black is used and the coordinates are out of bounds, return 0.
            return 0.0;
        }

        match self.format {
            PixelFormat::U256 => {
                let mut r = [0.0];
                self.color_encoding
                    .as_ref()
                    .expect("Non-floating point images need encoding")
                    .to_linear(&[self.p8[self.pixel_offset(p) + c]], &mut r);
                return r[0];
            }
            PixelFormat::Half => {
                return self.p16[self.pixel_offset(p) + c].into();
            }
            PixelFormat::Float => return self.p32[self.pixel_offset(p) + c],
        }
    }

    pub fn get_channels(&self, p: Point2i) -> ImageChannelValues {
        self.get_channels_wrapped(p, WrapMode::Clamp.into())
    }

    pub fn get_channels_wrapped(&self, p: Point2i, wrap_mode: WrapMode2D) -> ImageChannelValues {
        let mut cv = ImageChannelValues::default();
        let mut p = p;
        if !remap_pixel_coords(&mut p, self.resolution, wrap_mode) {
            return cv;
        }
        let pixel_offset = self.pixel_offset(p);
        match self.format {
            PixelFormat::U256 => self
                .color_encoding
                .as_ref()
                .expect("Fixed point images should have an encoding")
                .to_linear(
                    &self.p8[pixel_offset..pixel_offset + self.n_channels()],
                    &mut cv.values,
                ),
            PixelFormat::Half => {
                for i in 0..self.n_channels() {
                    cv[i] = self.p16[pixel_offset + i].to_f32();
                }
            }
            PixelFormat::Float => {
                for i in 0..self.n_channels() {
                    cv[i] = self.p32[pixel_offset + i];
                }
            }
        }
        cv
    }

    pub fn get_channels_from_desc(
        &self,
        p: Point2i,
        desc: &ImageChannelDesc,
    ) -> ImageChannelValues {
        self.get_channels_from_desc_wrapped(p, desc, WrapMode::Clamp.into())
    }

    pub fn get_channels_from_desc_wrapped(
        &self,
        p: Point2i,
        desc: &ImageChannelDesc,
        wrap_mode: WrapMode2D,
    ) -> ImageChannelValues {
        let mut cv = ImageChannelValues::new(desc.offset.len());
        let mut p = p;
        if !remap_pixel_coords(&mut p, self.resolution, wrap_mode) {
            return cv;
        }

        let pixel_offset = self.pixel_offset(p);
        match self.format {
            PixelFormat::U256 => {
                for i in 0..desc.offset.len() {
                    let index = pixel_offset + desc.offset[i] as usize;
                    self.color_encoding
                        .as_ref()
                        .expect("Expected color encoding")
                        .to_linear(&self.p8[index..index + 1], &mut cv.values[i..i + 1]);
                }
            }
            PixelFormat::Half => {
                for i in 0..desc.offset.len() {
                    let index = pixel_offset + desc.offset[i] as usize;
                    cv[i] = self.p16[index].to_f32();
                }
            }
            PixelFormat::Float => {
                for i in 0..desc.offset.len() {
                    let index = pixel_offset + desc.offset[i] as usize;
                    cv[i] = self.p32[index];
                }
            }
        }

        cv
    }

    pub fn get_channel_desc(&self, requested_channels: &[String]) -> ImageChannelDesc {
        let mut offset = ArrayVec::<i32, 4>::new();
        for i in 0..requested_channels.len() {
            let mut j = 0;
            while j < self.channel_names.len() {
                if requested_channels[i] == self.channel_names[j] {
                    offset[i] = j as i32;
                    break;
                }
                j += 1;
            }
            if j == self.channel_names.len() {
                return ImageChannelDesc::default();
            }
        }
        ImageChannelDesc { offset }
    }

    pub fn all_channels_desc(&self) -> ImageChannelDesc {
        let mut offset = ArrayVec::<i32, 4>::new();
        for i in 0..self.n_channels() {
            offset[i] = i as i32;
        }
        ImageChannelDesc { offset }
    }

    pub fn lookup_nearest_channel(&self, p: Point2f, c: usize) -> Float {
        self.lookup_nearest_channel_wrapped(p, c, WrapMode::Clamp.into())
    }

    /// Returns the specified channel for a pizel sample nearest the provided coordinate w.r.t. [0,1]^2.
    pub fn lookup_nearest_channel_wrapped(
        &self,
        p: Point2f,
        c: usize,
        wrap_mode: WrapMode2D,
    ) -> Float {
        let pi = Point2i::new(
            (p.x * self.resolution.x as Float) as i32,
            (p.y * self.resolution.y as Float) as i32,
        );
        self.get_channel_wrapped(pi, c, wrap_mode)
    }

    pub fn bilerp_channel(&self, p: Point2f, c: usize) -> Float {
        self.bilerp_channel_wrapped(p, c, WrapMode::Clamp.into())
    }

    /// Uses bilinear interpolation between four image pixels to compute the channel value,
    /// equivalent to filtering witha  pixel-wide triangle filter.
    pub fn bilerp_channel_wrapped(&self, p: Point2f, c: usize, wrap_mode: WrapMode2D) -> Float {
        // Compute discrete pixel coordinates and offsetrs for p
        let x = p[0] * self.resolution.x as Float - 0.5;
        let y = p[1] * self.resolution.y as Float - 0.5;
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let dx = x - xi as Float;
        let dy = y - yi as Float;

        // Load pixel channel values and return bilinearly interpolated value
        let v: [Float; 4] = [
            self.get_channel_wrapped(Point2i { x: xi, y: yi }, c, wrap_mode),
            self.get_channel_wrapped(Point2i { x: xi + 1, y: yi }, c, wrap_mode),
            self.get_channel_wrapped(Point2i { x: xi, y: yi + 1 }, c, wrap_mode),
            self.get_channel_wrapped(
                Point2i {
                    x: xi + 1,
                    y: yi + 1,
                },
                c,
                wrap_mode,
            ),
        ];
        (1.0 - dx) * (1.0 - dy) * v[0]
            + dx * (1.0 - dy) * v[1]
            + (1.0 - dx) * dy * v[2]
            + dx * dy * v[3]
    }

    pub fn set_channel(&mut self, p: Point2i, c: usize, value: Float) {
        let value = if value.is_nan() { 0.0 } else { value };

        let index = self.pixel_offset(p) + c;
        match self.format {
            PixelFormat::U256 => self
                .color_encoding
                .as_ref()
                .expect("Non-floating-point images need encoding")
                .from_linear(&[value], &mut self.p8[index..index + 1]),
            PixelFormat::Half => self.p16[index] = f16::from_f32(value),
            PixelFormat::Float => self.p32[index] = value,
        }
    }

    pub fn set_channels(&mut self, p: Point2i, values: &ImageChannelValues) {
        debug_assert!(values.values.len() == self.n_channels());
        for i in 0..values.values.len() {
            self.set_channel(p, i, values[i]);
        }
    }

    pub fn set_channels_slice(&mut self, p: Point2i, values: &[Float]) {
        debug_assert!(values.len() == self.n_channels());
        for i in 0..values.len() {
            self.set_channel(p, i, values[i]);
        }
    }

    pub fn read(name: &str, encoding: Option<ColorEncoding>) -> ImageAndMetadata {
        // TODO Should return an IO Result instead likely.

        // TODO Other file extension types.
        if has_extension(name, "png") {
            return Self::read_png(name, encoding);
        } else {
            panic!("Unsupported file extension for {}", name);
        }
    }

    // TODO Test.
    fn read_png(name: &str, encoding: Option<ColorEncoding>) -> ImageAndMetadata {
        let encoding = if let Some(encoding) = encoding {
            encoding
        } else {
            ColorEncoding::SRGB(SRgbColorEncoding {})
        };

        let mut decoder = png::Decoder::new(File::open(Path::new(name)).unwrap());
        decoder.set_transformations(png::Transformations::IDENTITY);
        let mut reader = decoder.read_info().unwrap();

        let mut img_data = vec![0; reader.output_buffer_size()];
        // Get the metadata info, and read the raw bytes into the buffer.
        let info = reader.next_frame(&mut img_data).unwrap();

        // Transform the raw bytes into an Image accounting for the format encoded in the header.
        let image = match info.color_type {
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => match info.bit_depth {
                png::BitDepth::Eight => Image::new_p8(
                    img_data,
                    Point2i::new(info.width as i32, info.height as i32),
                    &["Y".to_owned()],
                    encoding,
                ),
                png::BitDepth::Sixteen => {
                    let mut image = Image::new(
                        PixelFormat::Half,
                        Point2i::new(info.width as i32, info.height as i32),
                        &["Y".to_owned()],
                        None,
                    );
                    for y in 0..info.height {
                        for x in 0..info.width {
                            let v = f16::from_le_bytes(
                                img_data[(2 * (y * info.width + x)) as usize
                                    ..(2 * (y * info.width + x) + 2) as usize]
                                    .try_into()
                                    .unwrap(),
                            );
                            let v: Float = v.into();
                            let v = encoding.to_float_linear(v);
                            image.set_channel(Point2i::new(x as i32, y as i32), 0, v);
                        }
                    }
                    image
                }
                _ => panic!("Unsupported bit depth"),
            },
            png::ColorType::Rgb | png::ColorType::Rgba => {
                let has_alpha = info.color_type == png::ColorType::Rgba;
                match info.bit_depth {
                    png::BitDepth::Eight => match has_alpha {
                        true => Image::new_p8(
                            img_data,
                            Point2i::new(info.width as i32, info.height as i32),
                            &[
                                "R".to_owned(),
                                "G".to_owned(),
                                "B".to_owned(),
                                "A".to_owned(),
                            ],
                            encoding,
                        ),
                        false => Image::new_p8(
                            img_data,
                            Point2i::new(info.width as i32, info.height as i32),
                            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
                            encoding,
                        ),
                    },
                    png::BitDepth::Sixteen => match has_alpha {
                        true => {
                            let mut image = Image::new(
                                PixelFormat::Half,
                                Point2i::new(info.width as i32, info.height as i32),
                                &[
                                    "R".to_owned(),
                                    "G".to_owned(),
                                    "B".to_owned(),
                                    "A".to_owned(),
                                ],
                                None,
                            );
                            let mut idx = 0;
                            for y in 0..info.height {
                                for x in 0..info.width {
                                    let r = f16::from_le_bytes(
                                        img_data[idx..idx + 2].try_into().unwrap(),
                                    );
                                    let g = f16::from_le_bytes(
                                        img_data[idx + 2..idx + 4].try_into().unwrap(),
                                    );
                                    let b = f16::from_le_bytes(
                                        img_data[idx + 4..idx + 6].try_into().unwrap(),
                                    );
                                    let a = f16::from_le_bytes(
                                        img_data[idx + 6..idx + 8].try_into().unwrap(),
                                    );
                                    let rgba = [r, g, b, a];
                                    for c in 0..4 {
                                        let cv = encoding.to_float_linear(rgba[c].into());
                                        image.set_channel(
                                            Point2i::new(x as i32, y as i32),
                                            c,
                                            cv.into(),
                                        );
                                    }
                                    idx += 8;
                                }
                            }
                            image
                        }
                        false => {
                            let mut image = Image::new(
                                PixelFormat::Half,
                                Point2i::new(info.width as i32, info.height as i32),
                                &["R".to_owned(), "G".to_owned(), "B".to_owned()],
                                None,
                            );
                            let mut idx = 0;
                            for y in 0..info.height {
                                for x in 0..info.width {
                                    let r = f16::from_le_bytes(
                                        img_data[idx..idx + 2].try_into().unwrap(),
                                    );
                                    let g = f16::from_le_bytes(
                                        img_data[idx + 2..idx + 4].try_into().unwrap(),
                                    );
                                    let b = f16::from_le_bytes(
                                        img_data[idx + 4..idx + 6].try_into().unwrap(),
                                    );
                                    let rgb = [r, g, b];
                                    for c in 0..3 {
                                        let cv = encoding.to_float_linear(rgb[c].into());
                                        image.set_channel(
                                            Point2i::new(x as i32, y as i32),
                                            c,
                                            cv.into(),
                                        );
                                    }
                                    idx += 6;
                                }
                            }
                            image
                        }
                    },
                    _ => panic!("Unsupported bit depth"),
                }
            }
            png::ColorType::Indexed => panic!("Indexed PNGs are not supported!"),
        };

        let metadata = match info.color_type {
            png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => ImageMetadata::default(),
            png::ColorType::Rgb | png::ColorType::Rgba => {
                let mut metadata = ImageMetadata::default();
                metadata.color_space = Some(
                    RgbColorSpace::get_named(crate::colorspace::NamedColorSpace::SRGB).clone(),
                );
                metadata
            }
            png::ColorType::Indexed => panic!("Unspported indexed PNGs!"),
        };

        ImageAndMetadata { image, metadata }
    }

    pub fn write(&self, filename: &str, metadata: &ImageMetadata) -> std::io::Result<()> {
        // TODO There's additional logic we'll need here with other filetypes, but
        // this should be "OK" for writing PFM files.

        // TODO Add exr support.

        if has_extension(filename, "pfm") {
            return self.write_pfm(filename, metadata);
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid file extension!",
            ));
        }
    }

    // TODO write_exr().

    /// Writes the PFM file format.
    /// https://netpbm.sourceforge.net/doc/pfm.html
    fn write_pfm(&self, filename: &str, metadata: &ImageMetadata) -> std::io::Result<()> {
        let file = File::create(filename)?;

        if self.n_channels() != 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Only 3-channel images are supported by PFM",
            ));
        }

        let mut buf = BufWriter::new(file);

        write!(buf, "PF\n")?;
        write!(buf, "{} {}\n", self.resolution.x, self.resolution.y)?;
        // Write the scale, which encodes endianness.
        let scale = if HOST_LITTLE_ENDIAN { -1.0 } else { 1.0 };
        write!(buf, "{}\n", scale)?;

        let mut scanline: Vec<f32> = vec![0.0; 3 * self.resolution.x as usize];

        // Write the data from bottom left to upper right.
        // The raster is a sequence of pixels, packed one after another, with no
        // delimiters of any kind. They are grouped by row, with the pixels in each
        // row ordered left to right and the rows ordered bottom to top.
        for y in (0..self.resolution.y).rev() {
            for x in 0..self.resolution.x {
                for c in 0..3 {
                    // Cast to f32 in case Float is typedefed as f64.
                    scanline[(3 * x + c) as usize] =
                        self.get_channel(Point2i { x, y }, c as usize) as f32;
                }
            }
            let scan_bytes = unsafe {
                std::slice::from_raw_parts(
                    scanline.as_ptr() as *const u8,
                    scanline.len() * std::mem::size_of::<f32>(),
                )
            };
            buf.write(scan_bytes)?;
        }

        buf.flush()?;

        Ok(())
    }
}

impl Default for Image {
    fn default() -> Self {
        Self {
            format: PixelFormat::U256,
            resolution: Point2i { x: 0, y: 0 },
            channel_names: Default::default(),
            color_encoding: None,
            p8: Default::default(),
            p16: Default::default(),
            p32: Default::default(),
        }
    }
}

// TODO Image tests; can copy PBRT's tests.
#[cfg(test)]
mod tests {
    use crate::{
        color::{ColorEncoding, LinearColorEncoding},
        vecmath::{Point2i, Tuple2},
    };

    use super::Image;

    #[test]
    fn image_basics() {
        let encoding = ColorEncoding::Linear(LinearColorEncoding {});
        let y8 = Image::new(
            super::PixelFormat::U256,
            Point2i::new(4, 8),
            &["Y".to_owned()],
            Some(encoding.clone()),
        );
        assert_eq!(y8.n_channels(), 1);
        assert_eq!(
            y8.bytes_used(),
            (y8.resolution()[0] * y8.resolution()[1]) as usize
        );

        let y16 = Image::new(
            super::PixelFormat::Half,
            Point2i::new(4, 8),
            &["Y".to_owned()],
            None,
        );
        assert_eq!(y16.n_channels(), 1);
        assert_eq!(
            y16.bytes_used(),
            (2 * y16.resolution()[0] * y16.resolution()[1]) as usize
        );

        let y32 = Image::new(
            super::PixelFormat::Float,
            Point2i::new(4, 8),
            &["Y".to_owned()],
            None,
        );
        assert_eq!(y32.n_channels(), 1);
        assert_eq!(
            y32.bytes_used(),
            (4 * y32.resolution()[0] * y32.resolution()[1]) as usize
        );

        let rgb8 = Image::new(
            crate::image::PixelFormat::U256,
            Point2i { x: 4, y: 8 },
            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
            Some(encoding),
        );
        assert_eq!(rgb8.n_channels(), 3);
        assert_eq!(
            rgb8.bytes_used(),
            (3 * rgb8.resolution()[0] * rgb8.resolution()[1]) as usize
        );

        let rgb16 = Image::new(
            crate::image::PixelFormat::Half,
            Point2i { x: 4, y: 8 },
            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
            None,
        );
        assert_eq!(rgb16.n_channels(), 3);
        assert_eq!(
            rgb16.bytes_used(),
            (2 * 3 * rgb16.resolution()[0] * rgb16.resolution()[1]) as usize
        );

        let rgb32 = Image::new(
            crate::image::PixelFormat::Float,
            Point2i { x: 4, y: 8 },
            &["R".to_owned(), "G".to_owned(), "B".to_owned()],
            None,
        );
        assert_eq!(rgb32.n_channels(), 3);
        assert_eq!(
            rgb32.bytes_used(),
            (4 * 3 * rgb32.resolution()[0] * rgb32.resolution()[1]) as usize
        );
    }
}
