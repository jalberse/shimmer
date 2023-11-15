use core::fmt;
use std::io::{self, Write};

use crate::{
    bounding_box::Bounds2i, color::RGB, float::Float, math::lerp, vec2d::Vec2d, vecmath::Point2i,
};

// TODO We currently implement Image as a simple PPM file, which really isn't the best for accurate colors
//  given only 8 bits for each channel. But it's simple, which is why I'm using it during development.
//  We should later follow PBRT's image architecture to support OpenEXR.
// TODO We also only support writing to stdout right now, so filenames aren't really used.
#[derive(Debug)]
pub struct SimpleImage {
    pub data: Vec2d<RGB>,
}

impl SimpleImage {
    pub fn new(bounds: Bounds2i) -> SimpleImage {
        let data = Vec2d::from_bounds(bounds);
        SimpleImage { data }
    }

    pub fn write(&self) {
        let stdout = io::stdout();
        let mut buf_writer = io::BufWriter::new(stdout);
        write!(
            buf_writer,
            "P3\n{} {}\n255\n",
            self.data.extent().width(),
            self.data.extent().height()
        )
        .expect("Unable to write header!");

        for y in (0..self.data.extent().height()).rev() {
            for x in 0..self.data.extent().width() {
                let color = self.data.get_xy(x, y);
                let r = lerp(color.r, &0.0, &255.0) as i32;
                let g = lerp(color.g, &0.0, &255.0) as i32;
                let b = lerp(color.b, &0.0, &255.0) as i32;
                write!(buf_writer, "{} {} {}\n", r, g, b).expect("Unable to write color!");
            }
        }

        buf_writer.flush().unwrap();
    }
}

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
