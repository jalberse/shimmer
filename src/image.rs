use std::io::{self, Write};

use crate::{bounding_box::Bounds2i, color::RGB, math::lerp, vec2d::Vec2d};

// TODO We currently implement Image as a simple PPM file, which really isn't the best for accurate colors
//  given only 8 bits for each channel. But it's simple, which is why I'm using it during development.
//  We should later follow PBRT's image architecture to support OpenEXR.
// TODO We also only support writing to stdout right now, so filenames aren't really used.
#[derive(Debug)]
pub struct Image {
    pub data: Vec2d<RGB>,
}

impl Image {
    pub fn new(bounds: Bounds2i) -> Image {
        let data = Vec2d::from_bounds(bounds);
        Image { data }
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
