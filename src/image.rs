use arrayvec::ArrayVec;
use core::fmt;
use half::f16;
use std::{collections::HashMap, fs::File, io::{BufWriter, Write}, ops::{Index, IndexMut}, path::Path, sync::Arc
};

use crate::{
    bounding_box::Bounds2i, color::{ColorEncoding, ColorEncodingI, ColorEncodingPtr}, colorspace::RgbColorSpace, float::Float, math::windowed_sinc, square_matrix::SquareMatrix, tile::Tile, vecmath::{Point2f, Point2i, Tuple2}
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ResampleWeight {
    first_pixel: i32,
    weight: [Float; 4],
}

impl Default for ResampleWeight
{
    fn default() -> Self {
        Self { first_pixel: Default::default(), weight: Default::default() }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone)]
pub struct Image {
    format: PixelFormat,
    resolution: Point2i,
    channel_names: Vec<String>,
    /// For images with fixed-precision (non-floating-point) pixel values, a ColorEncoding is specified.
    /// This is for e.g. PNG images; floating point formats like EXR will not use this.
    color_encoding: Option<ColorEncodingPtr>,
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
        encoding: Option<ColorEncodingPtr>,
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
        } else if format.is_32bit() {
            image.p32.resize(
                image.n_channels() * resolution[0] as usize * resolution[1] as usize,
                0.0,
            );
        } else {
            panic!("Unsupported image format in Image::new()")
        }
        image
    }

    pub fn new_p8(
        p8: Vec<u8>,
        resolution: Point2i,
        channels: &[String],
        encoding: ColorEncodingPtr,
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

    pub fn encoding(&self) -> &Option<ColorEncodingPtr> {
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
                    .expect("Non-floating point images need encoding").0
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
                .expect("Fixed point images should have an encoding").0
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
                        .expect("Expected color encoding").0
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

    pub fn get_channel_desc(&self, requested_channels: &[&str]) -> Option<ImageChannelDesc> {
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
                return None;
            }
        }
        Some(ImageChannelDesc { offset })
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
                .expect("Non-floating-point images need encoding").0
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

    pub fn select_channels(&self, desc: &ImageChannelDesc) -> Image 
    {
        let desc_channel_names = desc.offset.iter().map(|i| self.channel_names[*i as usize].clone()).collect::<Vec<String>>();

        let mut image = Image::new(self.format, self.resolution, &desc_channel_names, self.color_encoding.clone());
        // TODO Test
        for y in 0..self.resolution.y
        {
            for x in 0..self.resolution.x
            {
                let p = Point2i::new(x, y);
                let values = self.get_channels_from_desc(p, desc);
                image.set_channels(p, &values);
            }
        }
        image
    }


    // TODO I probably want to test these. Can go ahead and write test cases and just load/output images to disk in the test case
    //   so I can look at them and make sure they're reasonable. I can delete them after (or tag so it's not normally run).

    pub fn generate_pyramid(mut image: Image, wrap_mode: WrapMode2D) -> Vec<Image>
    {
        let orig_format = image.format;
        let n_channels = image.n_channels();
        let orig_encoding = image.color_encoding.clone();

        // Prepare image for building pyramid
        let mut image = if !(image.resolution[0] as u32).is_power_of_two() || !(image.resolution[1] as u32).is_power_of_two()
        {
            image.float_resize_up(Point2i::new(
                (image.resolution[0] as u32).next_power_of_two() as i32,
                (image.resolution[1] as u32).next_power_of_two() as i32,
            ), wrap_mode)
        } else if !image.format.is_32bit() {
            image.convert_to_format(PixelFormat::Float)
        } else {
            image
        };
        assert!(image.format.is_32bit());

        // TODO Can take log2 more efficiently.
        let n_levels = 1 + ((i32::max(image.resolution[0], image.resolution[1]) as Float).log2() as i32);

        let mut pyramid = Vec::with_capacity(n_levels as usize);
        for i in 0..(n_levels - 1)
        {
            // Initialize i + 1 level from ith level, and copy the ith into the pyramid
            pyramid.push(Image::new(orig_format, image.resolution, &image.channel_names, orig_encoding.clone()));

            let next_resolution = Point2i::new(
                i32::max(1, (image.resolution[0] + 1) / 2),
                i32::max(1, (image.resolution[1] + 1) / 2),
            );
            let mut next_image = Image::new(image.format, next_resolution, &image.channel_names, orig_encoding.clone());

            // Compute offsets from pixels to the 4 pixels used for downsampling
            let mut src_deltas = [
                0,
                n_channels,
                n_channels * image.resolution[0] as usize,
                n_channels * (image.resolution[0] as usize + 1),
            ];

            if image.resolution[0] == 1
            {
                src_deltas[1] = 0;
                src_deltas[3] -= n_channels;
            }
            if image.resolution[1] == 1
            {
                src_deltas[2] = 0;
                src_deltas[3] -= n_channels * image.resolution[0] as usize;
            }

            // TODO We'd like to parallelize this with into_par_iter(), but we would need to reconfigure as p32/pyramid can't be borrowed mutably in
            //   a parallel iterator. Likely we'd want to build a new p32 in parallel then copy it to new_image?, or wrap next_image (and pyramid?) in Arc<Mutex<..>>.
            //   Maybe try the latter.

            // Downsample image to create next level and update pyramid
            (0..next_resolution[1]).into_iter().for_each(|y| {
                // Loop over pixels in scanline y and downsample for the next pyramid level
                let mut src_offset = image.pixel_offset(Point2i::new(0, 2 * y));
                let mut next_offset = next_image.pixel_offset(Point2i::new(0, y));
                for _x in 0..next_resolution[0]
                {
                    for _c in 0..n_channels
                    {
                        next_image.p32[next_offset] = 0.25 * (
                            image.p32[src_offset] +
                            image.p32[src_offset + src_deltas[1]] +
                            image.p32[src_offset + src_deltas[2]] +
                            image.p32[src_offset + src_deltas[3]]
                        );

                        src_offset += 1;
                        next_offset += 1;
                    }
                    src_offset += n_channels
                }

                // Copy two scanlines from image output to its pyramid level
                let y_start = 2 * y;
                let y_end = i32::min(2 * y + 2, image.resolution[1]);
                let offset = image.pixel_offset(Point2i::new(0, y_start));
                let count = (y_end - y_start) * n_channels as i32 * image.resolution[0];
                pyramid[i as usize].copy_rect_in(
                    &Bounds2i::new(Point2i::new(0, y_start), Point2i::new(image.resolution[0], y_end)),
                    &image.p32[offset..offset + count as usize],
                );
            });

            image = next_image;
        }

        // Initialize top level of pyramid and return it
        assert!(image.resolution[0] == 1 && image.resolution[1] == 1);
        pyramid.push(Image::new(orig_format, Point2i::new(1, 1), &image.channel_names, orig_encoding));
        pyramid[n_levels as usize - 1].copy_rect_in(
            &Bounds2i::new(Point2i::new(0, 0), Point2i::new(1, 1)),
            &image.p32[0..n_channels],
        );
        pyramid
    }

    // Apply op to the extent in the image
    fn for_extent<F>(&mut self, extent: &Bounds2i, wrap_mode: WrapMode2D, mut op: F)
    where
        F: FnMut(&mut Self, usize)
    {
        assert!(extent.min.x < extent.max.x);
        assert!(extent.min.y < extent.max.y);

        let nx = extent.max[0] - extent.min[0];
        let nc = self.n_channels();
        let intersection = extent.intersect(&Bounds2i::new(Point2i::ZERO, self.resolution));
        if intersection.is_some_and(|i| i == *extent)
        {
            // All in bounds
            for y in extent.min[1]..extent.max[1]
            {
                let mut offset = self.pixel_offset(Point2i::new(extent.min[0], y));
                for _x in 0..nx
                {
                    for _c in 0..nc
                    {
                        op(self, offset);
                        offset += 1;
                    }
                }
            }
        } else {
            for y in extent.min[1]..extent.max[1]
            {
                for x in 0..nx
                {
                    let mut p = Point2i::new(extent.min[0] + x, y);
                    // TODO This will fail on Black wrap mode
                    assert!(remap_pixel_coords(&mut p, self.resolution(), wrap_mode));
                    let mut offset = self.pixel_offset(p);
                    for _c in 0..nc 
                    {
                        op(self, offset);
                        offset += 1;
                    }
                }
            }
        }
    }

    pub fn copy_rect_in(&mut self, extent: &Bounds2i, buf: &[f32]) 
    {
        let mut buf_offset = 0;
        assert!(buf.len() >= extent.area() as usize * self.n_channels() as usize);
        match self.format
        {
            PixelFormat::U256 => 
            {
                if extent.intersect(&Bounds2i::new(Point2i::ZERO, self.resolution)).is_some_and(|i| i == *extent) 
                {
                    // All in bounds
                    let count = self.n_channels() * (extent.max.x - extent.min.x) as usize;
                    for y in extent.min.y..extent.max.y
                    {
                        // Convert scanlines all at once (unless we're using double precision)
                        let offset = self.pixel_offset(Point2i::new(extent.min.x, y));
                        #[cfg(use_f64)]
                        {
                            for i in 0..count
                            {
                                self.color_encoding.as_ref().expect("Expected color encoding").0.from_linear(
                                    &buf[buff_offset..buf_offset + 1],
                                    &mut self.p8[offset + i..offset + i + 1],
                                );
                                buf_offset += 1;
                            }
                        }
                        #[cfg(not(use_f64))]
                        {
                            self.color_encoding.as_ref().expect("Expected color encoding").0.from_linear(
                                &buf[buf_offset..buf_offset + count],
                                &mut self.p8[offset..offset + count],
                            );
                            buf_offset += count;
                        }
                    } 
                } else {
                    self.for_extent(extent, WrapMode::Clamp.into(), 
                        |image, offset: usize| {
                                image.color_encoding.as_ref().expect("Expected color encoding").0.from_linear(
                                &buf[buf_offset..buf_offset + 1], 
                                &mut image.p8[offset..offset + 1]);
                                buf_offset += 1;
                            })
                }
            },
            PixelFormat::Half => 
            {
                // PAPERDOC: Useful for obviating interprocedural conflicts - pass a reference to self to the closure.
                self.for_extent(
                    extent,
                    WrapMode::Clamp.into(),
                    |image, offset: usize| {
                        image.p16[offset] = f16::from_f32(buf[buf_offset]);
                        buf_offset += 1;
                    })
            },
            PixelFormat::Float => {
                self.for_extent(
                    extent,
                    WrapMode::Clamp.into(),
                    |image, offset: usize| {
                        image.p32[offset] = buf[buf_offset];
                        buf_offset += 1;
                    })
            
            },
        }
    }

    fn copy_rect_out(&mut self, extent: &Bounds2i, buf: &mut [f32], wrap_mode: WrapMode2D)
    {
        assert!(buf.len() >= extent.area() as usize * self.n_channels() as usize);

        let mut buf_offset = 0;
        match self.format
        {
            PixelFormat::U256 => 
            {
                if extent.intersect(&Bounds2i::new(Point2i::ZERO, self.resolution)).is_some_and(|i| i == *extent)
                {
                    // All in bounds
                    let count = self.n_channels() * (extent.max.x - extent.min.x) as usize;
                    for y in extent.min.y..extent.max.y
                    {
                        // Convert scanlines all at once, unless we're using doubles
                        let offset = self.pixel_offset(Point2i::new(extent.min.x, y));
                        #[cfg(use_f64)]
                        {
                            for i in 0..count
                            {
                                self.color_encoding.as_ref().expect("Expected color encoding").0.to_linear(
                                    &self.p8[offset + i..offset + i + 1],
                                    &mut buf[buf_offset..buf_offset + 1],
                                );
                                buf_offset += 1;
                            }
                        }
                        #[cfg(not(use_f64))]
                        {
                            self.color_encoding.as_ref().expect("Expected color encoding").0.to_linear(
                                &self.p8[offset..offset + count],
                                &mut buf[buf_offset..buf_offset + count],
                            );
                            buf_offset += count;
                        }
                    }
                } else {
                    self.for_extent(extent, wrap_mode, |image, offset: usize| {
                        image.color_encoding.as_ref().expect("Expected color encoding").0.to_linear(
                            &image.p8[offset..offset + 1],
                            &mut buf[buf_offset..buf_offset + 1],
                        );
                        buf_offset += 1;
                    })
                }
            },
            PixelFormat::Half => {
                self.for_extent(extent, wrap_mode, |image, offset| {
                    buf[buf_offset] = image.p16[offset].to_f32();
                    buf_offset += 1;
                })
            },
            PixelFormat::Float => {
                self.for_extent(extent, wrap_mode, |image, offset| {
                    buf[buf_offset] = image.p32[offset];
                    buf_offset += 1;
                })
            },
        }
    }

    pub fn convert_to_format(&self, format: PixelFormat) -> Image 
    {
        if self.format == format
        {
            return self.clone();
        }

        let mut new_image = Image::new(
            format,
            self.resolution,
            &self.channel_names,
            self.color_encoding.clone(),
        );
        for y in 0..self.resolution.y
        {
            for x in 0..self.resolution.x
            {
                for c in 0..self.n_channels()
                {
                    let v = self.get_channel(Point2i::new(x, y), c);
                    new_image.set_channel(Point2i::new(x, y), c, v);
                }
            }
        }
        new_image
    }

    pub fn float_resize_up(&mut self, new_res: Point2i, wrap_mode: WrapMode2D) -> Image
    {
        assert!(new_res.x > self.resolution.x);
        assert!(new_res.y > self.resolution.y);

        let mut resampled_image = Image::new(PixelFormat::Float, new_res, &self.channel_names, None);

        // Compute x and y resampling weights for image resizing
        let x_weights = self.resample_weights(self.resolution[0] as usize, new_res[0] as usize);
        let y_weights = self.resample_weights(self.resolution[1] as usize, new_res[1] as usize);
        
        // TODO We want to parallelize this. Either map to  some output and copy to resampled_image after, or wrap resampled_image in an Arc<Mutex<..>>.
        //   I'm going to test for correctness first, though.

        // Resample image, working in tiles.
        let tiles = Tile::tile(Bounds2i::new(Point2i::ZERO, new_res), 8, 8);
        tiles.into_iter().for_each(|tile|
        {
            let in_extent = Bounds2i::new(
                Point2i::new(
                    x_weights[tile.bounds.min.x as usize].first_pixel,
                    y_weights[tile.bounds.min.y as usize].first_pixel,
                ),
                Point2i::new(
                    x_weights[tile.bounds.max.x as usize - 1].first_pixel + 4,
                    y_weights[tile.bounds.max.y as usize - 1].first_pixel + 4,
                ),
            );

            // Get the input data from the starting image, into the in_buf.
            let mut in_buf = vec![0.0; in_extent.area() as usize * self.n_channels()];
            self.copy_rect_out(&in_extent, &mut in_buf, wrap_mode);

            let nx_out = tile.bounds.max.x - tile.bounds.min.x;
            let ny_out = tile.bounds.max.y - tile.bounds.min.y;
            let nx_in = in_extent.max.x - in_extent.min.x;
            let ny_in = in_extent.max.y - in_extent.min.y;

            let mut x_buf = vec![0.0; (ny_in * nx_out) as usize * self.n_channels()];
            let mut x_buf_offset = 0;
            for y_out in in_extent.min.y..in_extent.max.y
            {
                for x_out in tile.bounds.min.x..tile.bounds.max.x
                {
                    // Resample image pixel at (x_out, y_out)
                    debug_assert!(x_out >= 0 && x_out < x_weights.len() as i32);
                    let rsw = &x_weights[x_out as usize];
                    // Compute in_offset into in_buf for (x_out, y_out) w.r.t. in_buf
                    let x_in = rsw.first_pixel - in_extent.min.x;
                    debug_assert!(x_in >=0);
                    debug_assert!(x_in + 3 < nx_in);
                    let y_in = y_out - in_extent.min.y;
                    let mut in_offset = self.n_channels() * (x_in + y_in * nx_in) as usize;
                    debug_assert!(in_offset + 3 * self.n_channels() < in_buf.len());

                    for _c in 0..self.n_channels()
                    {
                        x_buf[x_buf_offset] = rsw.weight[0] * in_buf[in_offset] +
                                                rsw.weight[1] * in_buf[in_offset + self.n_channels()] +
                                                rsw.weight[2] * in_buf[in_offset + 2 * self.n_channels()] +
                                                rsw.weight[3] * in_buf[in_offset + 3 * self.n_channels()];
                        x_buf_offset += 1;
                        in_offset += 1;
                    }
                }
            }

            // Resize image in the y dimension
            let mut out_buf = vec![0.0; (nx_out * ny_out) as usize * self.n_channels()];
            for x in 0..nx_out
            {
                for y in 0..ny_out
                {
                    let y_out = y + tile.bounds[0][1];
                    debug_assert!(y_out >= 0);
                    debug_assert!(y_out < y_weights.len() as i32);
                    let rsw = &y_weights[y_out as usize];

                    debug_assert!(rsw.first_pixel - in_extent[0][1] >= 0);
                    let mut x_buf_offset = self.n_channels() * (x + nx_out * (rsw.first_pixel - in_extent[0][1])) as usize;
                    debug_assert!(x_buf_offset >= 0);
                    let step = self.n_channels() * nx_out as usize;
                    debug_assert!(x_buf_offset + 3 * step < x_buf.len());

                    let mut out_offset = self.n_channels() * (x + y * nx_out) as usize;
                    for _c in 0..self.n_channels()
                    {
                        out_buf[out_offset] = Float::max(0.0,
                            rsw.weight[0] * x_buf[x_buf_offset] +
                            rsw.weight[1] * x_buf[x_buf_offset + step] +
                            rsw.weight[2] * x_buf[x_buf_offset + 2 * step] +
                            rsw.weight[3] * x_buf[x_buf_offset + 3 * step]);

                        out_offset += 1;
                        x_buf_offset += 1;
                    }
                }
            }

            // Copy resampled image pixels out to the resampled_image.
            resampled_image.copy_rect_in(&tile.bounds, &out_buf);
        });

        resampled_image
    }

    fn resample_weights(&self, old_res: usize, new_res: usize) -> Vec<ResampleWeight>
    {
        assert!(old_res < new_res);
        let mut wt = vec![ResampleWeight::default(); new_res as usize];
        let filter_radius = 2.0;
        let tau = 2.0;
        for i in 0..new_res
        {
            // Compute image resampling weights for ith pixel
            let center = (i as Float + 0.5) * old_res as Float / new_res as Float;
            wt[i].first_pixel = ((center - filter_radius + 0.5).floor() as i32).max(0);
            for j in 0..4{
                let pos: Float = wt[i].first_pixel as Float + 0.5;
                wt[i].weight[j] = windowed_sinc(pos - center, filter_radius, tau);
            }

            // Normalize filter weights for pixel resampling
            let inv_sum_wts =
                1.0 / (wt[i].weight[0] + wt[i].weight[1] + wt[i].weight[2] + wt[i].weight[3]);
            for j in 0..4
            {
                wt[i].weight[j] *= inv_sum_wts;
            }
        }
        wt
    }

    pub fn read(path: &Path, encoding: Option<ColorEncodingPtr>) -> ImageAndMetadata {
        // TODO Should return an IO Result instead likely.

        // TODO Other file extension types.
        if path.extension().unwrap().eq("png") {
            return Self::read_png(path, encoding);
        } else {
            panic!("Unsupported file extension for {}", path.to_str().unwrap());
        }
    }

    fn read_png(path: &Path, encoding: Option<ColorEncodingPtr>) -> ImageAndMetadata {
        let encoding = if let Some(encoding) = encoding {
            encoding
        } else {
            ColorEncoding::get("srgb", None).clone()
        };

        let mut decoder = png::Decoder::new(File::open(path).unwrap());
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
                            let v = encoding.0.to_float_linear(v);
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
                                        let cv = encoding.0.to_float_linear(rgba[c].into());
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
                                        let cv = encoding.0.to_float_linear(rgb[c].into());
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
            png::ColorType::Indexed => panic!("Unsupported indexed PNGs!"),
        };

        ImageAndMetadata { image, metadata }
    }

    pub fn write(&self, path: &Path, metadata: &ImageMetadata) -> std::io::Result<()> {
        // TODO There's additional logic we'll need here with other filetypes, but
        // this should be "OK" for writing PFM files.

        // TODO Add exr support.

        if path.extension().unwrap().eq("pfm") {
            return self.write_pfm(path, metadata);
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
    fn write_pfm(&self, path: &Path, metadata: &ImageMetadata) -> std::io::Result<()> {
        let file = File::create(path)?;

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        color::{ColorEncoding, ColorEncodingPtr, LinearColorEncoding},
        vecmath::{Point2i, Tuple2},
    };

    use super::Image;

    // Note: This is commented out because I've just texted by visually verifying the output images.
    // Ideally we'd mock the input image and test it properly.
    // #[test]
    // fn image_pyramid()
    // {
    //     let image = Image::read(
    //         path::Path::new("./pyramid_test.png"),
    //         Some(ColorEncoding::get("linear", None))).image;
    //     let pyramid = Image::generate_pyramid(image, WrapMode::Clamp.into());
    //     // Write out the pyramid images as PFMs so we can look at them
    //     for (i, level) in pyramid.iter().enumerate()
    //     {
    //         level.write(Path::new(&format!("./pyramid_level_{}.pfm", i)), &Default::default()).unwrap();
    //     }
    // }

    #[test]
    fn image_basics() {
        let encoding = ColorEncodingPtr(Arc::new(ColorEncoding::Linear(LinearColorEncoding {})));
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
