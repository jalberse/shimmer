use crate::{bounding_box::Bounds2i, vecmath::Point2i};


pub struct Tile {
    pub bounds: Bounds2i,
}

impl Tile {
    /// Returns a list of Tiles covering the original bounds.
    ///
    /// The tiles are returned in a flattened Vec in row-major order.
    /// If the image cannot be perfectly divided by the tile width or height,
    /// then smaller tiles are created to fill the remainder of the image width or height.
    /// It's recommended to pick a tiling size that fits into the original bounds well.
    /// Note that 8x8 is a reasonable tile size and 8 evenly divides common resolution
    /// sizes like 1920, 1080, 720, etc.
    ///
    /// * `orig_bounds` - The original bounds of to divide into tiles.
    /// * `tile_width` - Width of each tile.
    /// * `tile_height` - Height of each tile.
    pub fn tile(orig_bounds: Bounds2i, tile_width: i32, tile_height: i32) -> Vec<Tile> {
        let image_width = orig_bounds.width();
        let image_height = orig_bounds.height();
        let num_horizontal_tiles = image_width / tile_width;
        let remainder_horizontal_pixels = image_width % tile_width;
        let num_vertical_tiles = image_height / tile_height;
        let remainder_vertical_pixels = image_height % tile_height;

        let mut tiles = Vec::with_capacity((num_horizontal_tiles * num_vertical_tiles) as usize);

        for tile_y in 0..num_vertical_tiles {
            for tile_x in 0..num_horizontal_tiles {
                let tile_start_x = orig_bounds.min.x + tile_x * tile_width;
                let tile_start_y = orig_bounds.min.y + tile_y * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i {
                            x: tile_start_x,
                            y: tile_start_y,
                        },
                        Point2i {
                            x: tile_start_x + tile_width,
                            y: tile_start_y + tile_height,
                        },
                    ),
                });
            }
            // Add the rightmost row if necessary
            if remainder_horizontal_pixels > 0 {
                let tile_start_x = orig_bounds.min.x + num_horizontal_tiles * tile_width;
                let tile_start_y = orig_bounds.min.y + tile_y * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i {
                            x: tile_start_x,
                            y: tile_start_y,
                        },
                        Point2i {
                            x: tile_start_x + remainder_horizontal_pixels,
                            y: tile_start_y + tile_height,
                        },
                    ),
                });
            }
        }
        // Add the bottom row if necessary
        if remainder_vertical_pixels > 0 {
            for tile_x in 0..num_horizontal_tiles {
                let tile_start_x = orig_bounds.min.x + tile_x * tile_width;
                let tile_start_y = orig_bounds.min.y + num_vertical_tiles * tile_height;
                tiles.push(Tile {
                    bounds: Bounds2i::new(
                        Point2i {
                            x: tile_start_x,
                            y: tile_start_y,
                        },
                        Point2i {
                            x: tile_start_x + tile_width,
                            y: tile_start_y + remainder_vertical_pixels,
                        },
                    ),
                });
            }
        }
        // Add the bottom-most, right-most Tile if necessary
        if remainder_horizontal_pixels > 0 && remainder_vertical_pixels > 0 {
            let tile_start_x = orig_bounds.min.x + num_horizontal_tiles * tile_width;
            let tile_start_y = orig_bounds.min.y + num_vertical_tiles * tile_height;
            tiles.push(Tile {
                bounds: Bounds2i::new(
                    Point2i {
                        x: tile_start_x,
                        y: tile_start_y,
                    },
                    Point2i {
                        x: tile_start_x + remainder_horizontal_pixels,
                        y: tile_start_y + remainder_vertical_pixels,
                    },
                ),
            });
        }

        tiles
    }
}
