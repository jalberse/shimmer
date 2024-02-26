
use std::{ops::{Add, Div, Mul}, path::Path, sync::Arc};

use ordered_float::OrderedFloat;

use crate::{color::{ColorEncodingPtr, RGB}, colorspace::RgbColorSpace, image::{Image, WrapMode}, math::{lerp, safe_sqrt, sqr}, options::Options, vecmath::{Length, Point2f, Point2i, Tuple2, Vector2f}, Float};

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

    pub fn get_color_space(&self) -> Arc<RgbColorSpace>
    {
        self.color_space.clone()
    }

    pub fn levels(&self) -> usize{
        self.pyramid.len()
    }

    pub fn level_resolution(&self, level: usize) -> Point2i{
        self.pyramid[level].resolution()
    }

    pub fn filter<T: TexelType>(&self, st: Point2f, mut dst0: Vector2f, mut dst1: Vector2f) -> T
    {
        match self.options.filter
        {
            FilterFunction::EWA => 
            {
                // Compute EWA ellipse axes
                if dst0.length_squared() < dst1.length_squared()
                {
                    std::mem::swap(&mut dst0, &mut dst1);
                }
                let longer_vec_length = dst0.length();
                let mut shorter_vec_length = dst1.length();

                // Clamp ellipse vector ratio if too large
                if shorter_vec_length * self.options.max_anisotropy.0 < longer_vec_length && shorter_vec_length > 0.0
                {
                    let scale = longer_vec_length / (shorter_vec_length * self.options.max_anisotropy.0);
                    dst1 *= scale;
                    shorter_vec_length *= scale;
                }
                if shorter_vec_length == 0.0
                {
                    return T::bilerp(self, 0, st);
                }

                // Choose level of detail for EWA lookup and perform EWA filtering
                let lod = Float::max(0.0, self.levels() as Float - 1.0 + shorter_vec_length.log2());
                let ilod = lod.floor() as usize;

                lerp::<T>(
                    lod - ilod as Float,
                    T::ewa(self, ilod, st, dst0, dst1),
                    T::ewa(self, ilod + 1, st, dst0, dst1),
                )
            },
            _ => {
                // Handle shared non-EWA filter code
                let width: Float = 2.0 * [dst0[0].abs(), dst0[1].abs(), dst1[0].abs(), dst1[1].abs()].into_iter().reduce(|acc, e| acc.max(e)).unwrap();

                // Compute MIP Map level for width and handle very wide filter
                let n_levels = self.levels();
                let level = n_levels as Float - 1.0 + width.max(1e-8).log2();
                if level >= n_levels as Float - 1.0
                {
                    return T::texel(self, n_levels - 1, Point2i::ZERO);
                }
                let i_level = i32::max(0, level.floor() as i32) as usize;

                match self.options.filter
                {
                    FilterFunction::Point => 
                    {
                        let resolution = self.level_resolution(i_level);
                        let sti = Point2i::new(
                            Float::round(st[0] * resolution.x as Float - 0.5) as i32,
                            Float::round(st[1] * resolution.y as Float - 0.5) as i32,
                        );
                        T::texel(self, i_level, sti)
                    },
                    FilterFunction::Bilinear => 
                    {
                        T::bilerp(self, i_level, st)
                    },
                    FilterFunction::Trilinear => 
                    {
                        if i_level == 0
                        {
                            T::bilerp(self, 0, st)
                        }
                        else
                        {
                            debug_assert!(level - i_level as Float <= 1.0);
                            lerp::<T>(level - i_level as Float, T::bilerp(self, i_level, st), T::bilerp(self, i_level + 1, st))
                        }
                    },
                    _ => unreachable!(),
                }
            }
        }
    }

    pub fn texel_rgb(&self, level: usize, st: Point2i) -> RGB
    {
        debug_assert!(level < self.pyramid.len());
        let nc = self.pyramid[level].n_channels();
        if nc == 3 || nc == 4
        {
            RGB::new(
                self.pyramid[level].get_channel_wrapped(st, 0, self.wrap_mode.into()),
                self.pyramid[level].get_channel_wrapped(st, 1, self.wrap_mode.into()),
                self.pyramid[level].get_channel_wrapped(st, 2, self.wrap_mode.into()),
            )
        } else {
            debug_assert!(nc == 1);
            let v = self.pyramid[level].get_channel_wrapped(st, 0, self.wrap_mode.into());
            RGB::new(v, v, v)
        }
    }

    pub fn texel_float(&self, level: usize, st: Point2i) -> Float
    {
        debug_assert!(level < self.pyramid.len());
        self.pyramid[level].get_channel_wrapped(st, 0, self.wrap_mode.into())
    }
}

pub trait TexelType : Add<Self, Output = Self> + Mul<Float, Output = Self> + Div<Float, Output = Self> + Sized + Default {
    fn texel(mipmap: &MIPMap, level: usize, st: Point2i) -> Self;

    fn bilerp(mipmap: &MIPMap, level: usize, st: Point2f) -> Self;

    fn ewa(mipmap: &MIPMap, level: usize, mut st: Point2f, mut dst0: Vector2f, mut dst1: Vector2f) -> Self
    {
        if level >= mipmap.levels()
        {
            return Self::texel(mipmap, mipmap.levels() - 1, Point2i::ZERO);
        }

        // Convert EWA coordinates to appropriate scale for level
        let level_res = mipmap.level_resolution(level);

        st[0] = st[0] * level_res.x as Float - 0.5;
        st[1] = st[1] * level_res.y as Float - 0.5;
        dst0[0] *= level_res.x as Float;
        dst0[1] *= level_res.y as Float;
        dst1[0] *= level_res.x as Float;
        dst1[1] *= level_res.y as Float;

        // Find ellipse coefficients that bound EWA filter region
        let mut a = sqr(dst0[1]) + sqr(dst1[1]) + 1.0;
        let mut b = -2.0 * (dst0[0] * dst0[1] + dst1[0] * dst1[1]);
        let mut c = sqr(dst0[0]) + sqr(dst1[0]) + 1.0;
        let inv_f = 1.0 / (a * c - sqr(b) * 0.25);
        a *= inv_f;
        b *= inv_f;
        c *= inv_f;

        // Compute the ellipse's (s, t) bounding box in texture space
        let det = -sqr(b) + 4.0 * a * c;
        let inv_det = 1.0 / det;
        let u_sqrt = safe_sqrt(det * c);
        let v_sqrt = safe_sqrt(a * det);
        let s0 = Float::ceil(st[0] - 2.0 * inv_det * u_sqrt) as i32;
        let s1 = Float::floor(st[0] + 2.0 * inv_det * u_sqrt) as i32;
        let t0 = Float::ceil(st[1] - 2.0 * inv_det * v_sqrt) as i32;
        let t1 = Float::floor(st[1] + 2.0 * inv_det * v_sqrt) as i32;

        // Scan over ellipse bound and evaluate quadratic equation to filter image
        let mut sum = Self::default();
        let mut sum_wts: Float = 0.0;

        for it in t0..=t1
        {
            let tt = it as Float - st[1];
            for is in s0..=s1
            {
                let ss = is as Float - st[0];
                // Compute squared radius and filter texel if inside ellipse
                let r2 = a * sqr(ss) + b * ss * tt + c * sqr(tt);
                if r2 < 1.0
                {
                    // TODO need the LUT.
                    let index = usize::min((r2 * MIP_FILTER_LUT_SIZE as Float) as usize, MIP_FILTER_LUT_SIZE - 1);
                    let weight = MIP_FILTER_LUT[index];
                    sum = sum + Self::texel(mipmap, level, Point2i::new(is, it)) * weight;
                    sum_wts += weight;
                }
            }
        }

        sum / sum_wts
    }
}

impl TexelType for Float{
    fn texel(mipmap: &MIPMap, level: usize, st: Point2i) -> Self{
        mipmap.texel_float(level, st)
    }

    fn bilerp(mipmap: &MIPMap, level: usize, st: Point2f) -> Self {
        match mipmap.pyramid[level].n_channels()
        {
            1 => mipmap.pyramid[level].bilerp_channel_wrapped(st, 0, mipmap.wrap_mode.into()),
            3 => mipmap.pyramid[level].bilerp(st, mipmap.wrap_mode.into()).average(),
            4 => mipmap.pyramid[level].bilerp_channel_wrapped(st, 3, mipmap.wrap_mode.into()), // Alpha
            _ => panic!("Unuespected number of image channels"),
        }
    }
}

impl TexelType for RGB{
    fn texel(mipmap: &MIPMap, level: usize, st: Point2i) -> Self{
        mipmap.texel_rgb(level, st)
    }

    fn bilerp(mipmap: &MIPMap, level: usize, st: Point2f) -> Self {
        let nc = mipmap.pyramid[level].n_channels();
        if nc == 3 || nc == 4
        {
            RGB::new(
                mipmap.pyramid[level].bilerp_channel_wrapped(st, 0, mipmap.wrap_mode.into()),
                mipmap.pyramid[level].bilerp_channel_wrapped(st, 1, mipmap.wrap_mode.into()),
                mipmap.pyramid[level].bilerp_channel_wrapped(st, 2, mipmap.wrap_mode.into()),
            )
        } else {
            debug_assert!(nc == 1);
            let v = mipmap.pyramid[level].bilerp_channel_wrapped(st, 0, mipmap.wrap_mode.into());
            RGB::new(v, v, v)
        }
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

const MIP_FILTER_LUT_SIZE: usize = 128;
const MIP_FILTER_LUT: [Float; MIP_FILTER_LUT_SIZE] = [
    0.864664733,
    0.849040031,
    0.83365953,
    0.818519294,
    0.80361563,
    0.788944781,
    0.774503231,
    0.760287285,
    0.746293485,
    0.732518315,
    0.718958378,
    0.705610275,
    0.692470789,
    0.679536581,
    0.666804492,
    0.654271305,
    0.641933978,
    0.629789352,
    0.617834508,
    0.606066525,
    0.594482362,
    0.583079159,
    0.571854174,
    0.560804546,
    0.549927592,
    0.539220572,
    0.528680861,
    0.518305838,
    0.50809288,
    0.498039544,
    0.488143265,
    0.478401601,
    0.468812168,
    0.45937258,
    0.450080454,
    0.440933526,
    0.431929469,
    0.423066139,
    0.414341331,
    0.405752778,
    0.397298455,
    0.388976216,
    0.380784035,
    0.372719884,
    0.364781618,
    0.356967449,
    0.34927541,
    0.341703475,
    0.334249914,
    0.32691282,
    0.319690347,
    0.312580705,
    0.305582166,
    0.298692942,
    0.291911423,
    0.285235822,
    0.278664529,
    0.272195935,
    0.265828371,
    0.259560347,
    0.253390193,
    0.247316495,
    0.241337672,
    0.235452279,
    0.229658857,
    0.223955944,
    0.21834214,
    0.212816045,
    0.207376286,
    0.202021524,
    0.196750447,
    0.191561714,
    0.186454013,
    0.181426153,
    0.176476851,
    0.171604887,
    0.166809067,
    0.162088141,
    0.157441005,
    0.152866468,
    0.148363426,
    0.143930718,
    0.139567271,
    0.135272011,
    0.131043866,
    0.126881793,
    0.122784719,
    0.11875169,
    0.114781633,
    0.11087364,
    0.107026696,
    0.103239879,
    0.0995122194,
    0.0958427936,
    0.0922307223,
    0.0886750817,
    0.0851749927,
    0.0817295909,
    0.0783380121,
    0.0749994367,
    0.0717130303,
    0.0684779733,
    0.0652934611,
    0.0621587038,
    0.0590728968,
    0.0560353249,
    0.0530452281,
    0.0501018465,
    0.0472044498,
    0.0443523228,
    0.0415447652,
    0.0387810767,
    0.0360605568,
    0.0333825648,
    0.0307464004,
    0.0281514227,
    0.0255970061,
    0.0230824798,
    0.0206072628,
    0.0181707144,
    0.0157722086,
    0.013411209,
    0.0110870898,
    0.0087992847,
    0.0065472275,
    0.00433036685,
    0.0021481365,
    0.
];