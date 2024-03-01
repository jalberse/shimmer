use std::{collections::HashMap, path::Path, sync::{Arc, Mutex}};

use spectrum::ConstantSpectrum;

use crate::{
    color::{ColorEncoding, ColorEncodingCache, ColorEncodingPtr, RGB}, file::resolve_filename, float::PI_F, image::WrapMode, interaction::{Interaction, SurfaceInteraction}, loading::{paramdict::{NamedTextures, ParameterDictionary, SpectrumType, TextureParameterDictionary}, parser_target::FileLoc}, math::{sqr, INV_2PI, INV_PI}, mipmap::{FilterFunction, MIPMap, MIPMapFilterOptions}, options::Options, spectra::{
        sampled_spectrum::SampledSpectrum,
        sampled_wavelengths::SampledWavelengths,
        spectrum::{self, RgbAlbedoSpectrum, RgbIlluminantSpectrum, RgbUnboundedSpectrum, SpectrumI},
        Spectrum,
    }, transform::Transform, vecmath::{normal::Normal3, spherical::spherical_theta, vector::Vector3, Normal3f, Normalize, Point2f, Point3f, Tuple2, Tuple3, Vector2f, Vector3f}, Float
};

pub trait FloatTextureI {
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float;
}

#[derive(Debug)]
pub struct ImageTextureBase
{
    mapping: TextureMapping2D,
    filename: String,
    scale: Float,
    invert: bool,
    mipmap: Arc<MIPMap>,
}

impl ImageTextureBase
{
    pub fn new(
        mapping: TextureMapping2D,
        filename: String,
        filter_options: MIPMapFilterOptions,
        wrap_mode: WrapMode,
        scale: Float,
        invert: bool,
        encoding: ColorEncodingPtr,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        options: &Options,
    ) -> ImageTextureBase
    {
        // Get MIPMap from texture cache if present
        let tex_info = TexInfo {
            filename: filename.clone(),
            filter_options: filter_options.clone(),
            wrap_mode: wrap_mode,
            encoding: encoding.clone(),
        };

        // PAPERDOC - Okay, here's a fun one - I forgot that texture_cache was already locked and tried to
        // lock it again to insert() within the match statement, leading to a deadlock.
        // Rust can't eliminate e.g. deadlocks, but these are easier to find.
        // We fix this just by moving the lock() call outside of the match statement to texture_cache_guard.
        // When that's dropped, we unlock the mutex.

        let mut texture_cache_guard = texture_cache.lock().unwrap();

        let mipmap = match texture_cache_guard.get(&tex_info) {
            Some(m) => m.clone(),
            None => {
                let m = Arc::new(MIPMap::create_from_file(
                    &filename,
                    filter_options,
                    wrap_mode,
                    encoding,
                    options,
                ));
                texture_cache_guard.insert(tex_info, m.clone());
                m
            }
        };

        ImageTextureBase
        {
            mapping,
            filename,
            scale,
            invert,
            mipmap,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TexInfo
{
    filename: String,
    filter_options: MIPMapFilterOptions,
    wrap_mode: WrapMode,
    encoding: ColorEncodingPtr,
}

#[derive(Debug)]
pub enum FloatTexture {
    Constant(FloatConstantTexture),
    Scaled(FloatScaledTexture),
    Mix(FloatMixTexture),
    DirectionMix(FloatDirectionMixTexture),
    Image(FloatImageTexture),
}

impl FloatTexture {
    pub fn create(
        name: &str,
        render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        loc: &FileLoc,
        textures: &NamedTextures,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> FloatTexture {
        let tex = match name {
            "constant" => {
                let t = FloatConstantTexture::create(render_from_texture, parameters, loc, textures);
                FloatTexture::Constant(t)
            }
            "scale" => {
                let t = FloatScaledTexture::create(render_from_texture, parameters, loc, textures);
                FloatTexture::Scaled(t)
            }
            "mix" => 
            {
                let t = FloatMixTexture::create(render_from_texture, parameters, loc, textures);
                FloatTexture::Mix(t)
            }
            "directionmix" => 
            {
                let t = FloatDirectionMixTexture::create(render_from_texture, parameters, loc, textures);
                FloatTexture::DirectionMix(t)
            }
            "imagemap" => {
                let t = FloatImageTexture::create(&render_from_texture, parameters, loc, options, texture_cache, gamma_encoding_cache);
                FloatTexture::Image(t)
            }
            _ => {
                panic!("Texture {} unknown", name);
            }
        };

        // TODO Track number of textures created for stats.
        // TODO Report unused paramters.

        tex
    }
}

impl FloatTextureI for FloatTexture {
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        match self {
            FloatTexture::Constant(t) => t.evaluate(ctx),
            FloatTexture::Scaled(t) => t.evaluate(ctx),
            FloatTexture::Mix(t) => t.evaluate(ctx),
            FloatTexture::DirectionMix(t) => t.evaluate(ctx),
            FloatTexture::Image(t) => t.evaluate(ctx),
        }
    }
}

#[derive(Debug)]
pub struct FloatConstantTexture {
    value: Float,
}

impl FloatConstantTexture {
    pub fn new(value: Float) -> Self {
        Self { value }
    }

    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        _loc: &FileLoc,
        _textures: &NamedTextures,
    ) -> FloatConstantTexture {
        let v = parameters.get_one_float("value", 1.0);
        FloatConstantTexture::new(v)
    }
}

impl FloatTextureI for FloatConstantTexture {
    fn evaluate(&self, _ctx: &TextureEvalContext) -> Float {
        self.value
    }
}

#[derive(Debug)]
pub struct FloatScaledTexture {
    tex: Arc<FloatTexture>,
    scale: Arc<FloatTexture>,
}

impl FloatScaledTexture {
    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        _loc: &FileLoc,
        textures: &NamedTextures,
    ) -> FloatScaledTexture {
        let tex = parameters.get_float_texture("tex", 1.0, textures);
        let scale = parameters.get_float_texture("scale", 1.0, textures);
        
        // TODO Handle if either is const? I think that can make it more efficient. See PBRT.
        
        FloatScaledTexture {
            tex,
            scale,
        }
    }
}

impl FloatTextureI for FloatScaledTexture {
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let sc = self.scale.evaluate(ctx);
        if sc == 0.0 {
            return 0.0;
        }
        self.tex.evaluate(ctx) * sc
    }
}

#[derive(Debug)]
pub struct FloatMixTexture 
{
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    amount: Arc<FloatTexture>,
}

impl FloatMixTexture
{
    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        _loc: &FileLoc,
        textures: &NamedTextures,
    ) -> FloatMixTexture {
        let tex1 = parameters.get_float_texture("tex1", 0.0, textures);
        let tex2 = parameters.get_float_texture("tex2", 1.0, textures);
        let amount = parameters.get_float_texture("amount", 0.5, textures);
        
        FloatMixTexture {
            tex1,
            tex2,
            amount,
        }
    }
}

impl FloatTextureI for FloatMixTexture
{
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let amt = self.amount.evaluate(ctx);
        let mut t1 = 0.0;
        let mut t2 = 0.0;
        if amt != 1.0 
        {
            t1 = self.tex1.evaluate(ctx);
        }
        if amt != 0.0
        {
            t2 = self.tex2.evaluate(ctx);
        }
        t1 * (1.0 - amt) + t2 * amt
    }
}

#[derive(Debug)]
pub struct FloatDirectionMixTexture 
{
    tex1: Arc<FloatTexture>,
    tex2: Arc<FloatTexture>,
    dir: Vector3f,
}

impl FloatDirectionMixTexture
{
    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        _loc: &FileLoc,
        textures: &NamedTextures,
    ) -> FloatDirectionMixTexture {
        let tex1 = parameters.get_float_texture("tex1", 0.0, textures);
        let tex2 = parameters.get_float_texture("tex2", 1.0, textures);
        let dir = parameters.get_one_vector3f("dir", Vector3f::new(0.0, 1.0, 0.0));
        
        FloatDirectionMixTexture {
            tex1,
            tex2,
            dir,
        }
    }
}

impl FloatTextureI for FloatDirectionMixTexture
{
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let amt = ctx.n.dot_vector(self.dir);
        let mut t1 = 0.0;
        let mut t2 = 0.0;
        if amt != 0.0 
        {
            t1 = self.tex1.evaluate(ctx);
        }
        if amt != 1.0
        {
            t2 = self.tex2.evaluate(ctx);
        }
        amt * t1 + (1.0 - amt) * t2
    }
}

#[derive(Debug)]
pub struct FloatImageTexture
{
    base: ImageTextureBase,
}

impl FloatImageTexture
{
    pub fn new(
        mapping: TextureMapping2D,
        filename: String,
        filter_options: MIPMapFilterOptions,
        wrap_mode: WrapMode,
        scale: Float,
        invert: bool,
        encoding: ColorEncodingPtr,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        options: &Options,
    ) -> FloatImageTexture
    {
        let base = ImageTextureBase::new(
            mapping,
            filename,
            filter_options,
            wrap_mode,
            scale,
            invert,
            encoding,
            texture_cache,
            options,
        );

        FloatImageTexture {
            base,
        }
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &mut TextureParameterDictionary,
        loc: &FileLoc,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> FloatImageTexture
    {
        let map = TextureMapping2D::create(&mut parameters.dict, render_from_texture, loc);

        let max_aniso = parameters.get_one_float("maxanisotropy", 8.0);
        let filter = parameters.get_one_string("filter", "bilinear".to_owned());
        
        let ff = FilterFunction::parse(&filter).expect("Unknown filter function");
        let filter_options = MIPMapFilterOptions::new(ff, max_aniso);

        let wrap = parameters.get_one_string("wrap", "repeat".to_owned());
        let wrap = WrapMode::parse(&wrap).expect("Unknown wrap mode");

        let scale = parameters.get_one_float("scale", 1.0);
        let invert = parameters.get_one_bool("invert", false);
        let filename = resolve_filename(options, &parameters.get_one_string("filename", "".to_owned()));

        let default_encoding = if Path::new(&filename).extension().expect("Expected extension") == "png"
        {
            "sRGB"
        } else {
            "linear"
        };
        let encoding_string = parameters.get_one_string("encoding", default_encoding.to_owned());

        let encoding = ColorEncoding::get(&encoding_string, Some(gamma_encoding_cache));

        FloatImageTexture::new(
            map,
            filename,
            filter_options,
            wrap,
            scale,
            invert,
            encoding,
            texture_cache,
            options,
        )
    }
}

impl FloatTextureI for FloatImageTexture
{
    fn evaluate(&self, ctx: &TextureEvalContext) -> Float {
        let mut c = self.base.mapping.map(ctx);
        // Texture coordinates are (0,0) in the lower left corner, but
        // image coordinates are (0,0) in the upper left.
        c.st[1] = 1.0 - c.st[1];

        let v = self.base.mipmap.filter::<Float>(c.st, Vector2f::new(c.dsdx, c.dtdx), Vector2f::new(c.dsdy, c.dtdy)) * self.base.scale;
        if self.base.invert { Float::max(0.0, 1.0 - v) } else { v }
    }
}

pub trait SpectrumTextureI {
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum;
}

#[derive(Debug)]
pub enum SpectrumTexture {
    Constant(SpectrumConstantTexture),
    Scaled(SpectrumScaledTexture),
    Mix(SpectrumMixTexture),
    DirectionMix(SpectrumDirectionMixTexture),
    Image(SpectrumImageTexture),
}

impl SpectrumTexture {
    pub fn create(
        name: &str,
        render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
        loc: &FileLoc,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> SpectrumTexture {
        let tex = match name {
            "constant" => {
                let t = SpectrumConstantTexture::create(
                    render_from_texture,
                    parameters,
                    spectrum_type,
                    cached_spectra,
                    loc,
                );
                SpectrumTexture::Constant(t)
            }
            "scale" => {
                let t = SpectrumScaledTexture::create(
                    render_from_texture,
                    parameters,
                    spectrum_type,
                    cached_spectra,
                    textures,
                    loc,
                );
                SpectrumTexture::Scaled(t)
            }
            "mix" => {
                let t = SpectrumMixTexture::create(
                    render_from_texture,
                    parameters,
                    spectrum_type,
                    cached_spectra,
                    textures,
                    loc,
                );
                SpectrumTexture::Mix(t)
            }
            "directionmix" => {
                let t = SpectrumDirectionMixTexture::create(
                    render_from_texture,
                    parameters,
                    spectrum_type,
                    cached_spectra,
                    textures,
                    loc,
                );
                SpectrumTexture::DirectionMix(t)
            }
            "imagemap" => {
                let t = SpectrumImageTexture::create(&render_from_texture, parameters, spectrum_type, options, texture_cache, gamma_encoding_cache, loc);
                SpectrumTexture::Image(t)
            }
            _ => {
                panic!("Texture {} unknown", name);
            }
        };

        // TODO Track number of textures created for stats.
        // TODO Report unused paramters.

        tex
    }
}

impl SpectrumTextureI for SpectrumTexture {
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        match self {
            SpectrumTexture::Constant(t) => t.evaluate(ctx, lambda),
            SpectrumTexture::Scaled(t) => t.evaluate(ctx, lambda),
            SpectrumTexture::Mix(t) => t.evaluate(ctx, lambda),
            SpectrumTexture::DirectionMix(t) => t.evaluate(ctx, lambda),
            SpectrumTexture::Image(t) => t.evaluate(ctx, lambda),
        }
    }
}

#[derive(Debug)]
pub struct SpectrumConstantTexture {
    pub value: Arc<Spectrum>,
}

impl SpectrumTextureI for SpectrumConstantTexture {
    fn evaluate(&self, _ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        self.value.sample(lambda)
    }
}

impl SpectrumConstantTexture {
    pub fn new(value: Arc<Spectrum>) -> Self {
        Self { value }
    }

    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        _loc: &FileLoc,
    ) -> SpectrumConstantTexture {
        let one = Spectrum::Constant(ConstantSpectrum::new(1.0));
        let c = parameters
            .get_one_spectrum("value", Some(Arc::new(one)), spectrum_type, cached_spectra)
            .unwrap();
        SpectrumConstantTexture::new(c)
    }
}

#[derive(Debug)]
pub struct SpectrumScaledTexture
{
    tex: Arc<SpectrumTexture>,
    scale: Arc<FloatTexture>,
}

impl SpectrumScaledTexture
{
    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
        _loc: &FileLoc,
    ) -> SpectrumScaledTexture {
        let one = ConstantSpectrum::new(1.0);
        let tex = parameters.get_spectrum_texture(
            "tex", 
            Some(Arc::new(Spectrum::Constant(one))),
            spectrum_type,
            cached_spectra,
            textures).expect("Expected default value");
        let scale = parameters.get_float_texture(
            "scale", 
            1.0, 
            textures);
        
        // TODO Handle if either is const? I think that can make it more efficient. See PBRT.
        
        SpectrumScaledTexture {
            tex,
            scale,
        }
    }
}

impl SpectrumTextureI for SpectrumScaledTexture {
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        let sc = self.scale.evaluate(ctx);
        if sc == 0.0 {
            return SampledSpectrum::from_const(0.0);
        }
        self.tex.evaluate(ctx, lambda) * sc
    }
}

#[derive(Debug)]
pub struct SpectrumMixTexture 
{
    tex1: Arc<SpectrumTexture>,
    tex2: Arc<SpectrumTexture>,
    amount: Arc<FloatTexture>,
}

impl SpectrumMixTexture
{
    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
        _loc: &FileLoc,
    ) -> SpectrumMixTexture {
        let zero = ConstantSpectrum::new(0.0);
        let one = ConstantSpectrum::new(1.0);
        let tex1 = parameters.get_spectrum_texture(
            "tex1", 
            Some(Arc::new(Spectrum::Constant(zero))),
            spectrum_type,
            cached_spectra,
            textures).expect("Expected default value");
        let tex2 = parameters.get_spectrum_texture(
            "tex2", 
            Some(Arc::new(Spectrum::Constant(one))),
            spectrum_type,
            cached_spectra,
            textures).expect("Expected default value");
        let amount = parameters.get_float_texture(
            "amount", 
            0.5, 
            textures);
        
        SpectrumMixTexture {
            tex1,
            tex2,
            amount,
        }
    }
}

impl SpectrumTextureI for SpectrumMixTexture
{
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        let amt = self.amount.evaluate(ctx);
        let mut t1 = SampledSpectrum::from_const(0.0);
        let mut t2 = SampledSpectrum::from_const(0.0);
        if amt != 1.0 
        {
            t1 = self.tex1.evaluate(ctx, lambda);
        }
        if amt != 0.0
        {
            t2 = self.tex2.evaluate(ctx, lambda);
        }
        t1 * (1.0 - amt) + t2 * amt
    }
}

#[derive(Debug)]
pub struct SpectrumDirectionMixTexture
{
    tex1: Arc<SpectrumTexture>,
    tex2: Arc<SpectrumTexture>,
    dir: Vector3f,
}

impl SpectrumDirectionMixTexture
{
    pub fn create(
        _render_from_texture: Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        textures: &NamedTextures,
        _loc: &FileLoc,
    ) -> SpectrumDirectionMixTexture {
        let zero = ConstantSpectrum::new(0.0);
        let one = ConstantSpectrum::new(1.0);
        let tex1 = parameters.get_spectrum_texture(
            "tex1", 
            Some(Arc::new(Spectrum::Constant(zero))),
            spectrum_type,
            cached_spectra,
            textures).expect("Expected default value");
        let tex2 = parameters.get_spectrum_texture(
            "tex2", 
            Some(Arc::new(Spectrum::Constant(one))),
            spectrum_type,
            cached_spectra,
            textures).expect("Expected default value");
        let dir = parameters.get_one_vector3f("dir", Vector3f::new(0.0, 1.0, 0.0));
        
        SpectrumDirectionMixTexture {
            tex1,
            tex2,
            dir,
        }
    }
}

#[derive(Debug)]
pub struct SpectrumImageTexture
{
    base: ImageTextureBase,
    spectrum_type: SpectrumType,
}

impl SpectrumImageTexture
{
    pub fn new(
        spectrum_type: SpectrumType,
        mapping: TextureMapping2D,
        filename: String,
        filter_options: MIPMapFilterOptions,
        wrap_mode: WrapMode,
        scale: Float,
        invert: bool,
        encoding: ColorEncodingPtr,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        options: &Options,
    ) -> SpectrumImageTexture
    {
        let base = ImageTextureBase::new(
            mapping,
            filename,
            filter_options,
            wrap_mode,
            scale,
            invert,
            encoding,
            texture_cache,
            options,
        );

        SpectrumImageTexture {
            base,
            spectrum_type,
        }
    }

    pub fn create(
        render_from_texture: &Transform,
        parameters: &mut TextureParameterDictionary,
        spectrum_type: SpectrumType,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
        loc: &FileLoc
    ) -> SpectrumImageTexture
    {
        let map = TextureMapping2D::create(&mut parameters.dict, render_from_texture, loc);

        let max_aniso = parameters.get_one_float("maxanisotropy", 8.0);
        let filter = parameters.get_one_string("filter", "bilinear".to_owned());

        let ff = FilterFunction::parse(&filter).expect("Unknown filter function");
        let filter_options = MIPMapFilterOptions::new(ff, max_aniso);

        let wrap = parameters.get_one_string("wrap", "repeat".to_owned());
        let wrap = WrapMode::parse(&wrap).expect("Unknown wrap mode");

        let scale = parameters.get_one_float("scale", 1.0);
        let invert = parameters.get_one_bool("invert", false);
        let filename = resolve_filename(options, &parameters.get_one_string("filename", "".to_owned()));
        
        let default_encoding = if Path::new(&filename).extension().expect("Expected extension") == "png"
        {
            "sRGB"
        } else {
            "linear"
        };
        let encoding_string = parameters.get_one_string("encoding", default_encoding.to_owned());
        let encoding = ColorEncoding::get(&encoding_string, Some(gamma_encoding_cache));

        SpectrumImageTexture::new(
            spectrum_type,
            map,
            filename,
            filter_options,
            wrap,
            scale,
            invert,
            encoding,
            texture_cache,
            options,
        )
    }
}

impl SpectrumTextureI for SpectrumImageTexture
{
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        // Apply texture mapping and flip t coordinate for image texture lookup (following typical conventions).
        let mut c = self.base.mapping.map(ctx);
        c.st[1] = 1.0 - c.st[1];

        let rgb = self.base.mipmap.filter::<RGB>(c.st, Vector2f::new(c.dsdx, c.dtdx), Vector2f::new(c.dsdy, c.dtdy)) * self.base.scale;
        let rgb = if self.base.invert { RGB::new(1.0, 1.0, 1.0) - rgb } else { rgb };
        let rgb = rgb.clamp_zero();

        if let Some(cs) = self.base.mipmap.get_color_space()
        {
            match self.spectrum_type
            {
                SpectrumType::Illuminant => {
                    RgbIlluminantSpectrum::new(&cs, &rgb).sample(lambda)
                },
                SpectrumType::Albedo => {
                    RgbAlbedoSpectrum::new(&cs, &rgb).sample(lambda)
                },
                SpectrumType::Unbounded => {
                    RgbUnboundedSpectrum::new(&cs, &rgb).sample(lambda)
                },
            }
        } else {
            // If no colorspace, then it should be a one-channel texture
            debug_assert!(rgb[0] == rgb[1] && rgb[1] == rgb[2]);
            SampledSpectrum::from_const(rgb[0])
        }
    }
}

impl SpectrumTextureI for SpectrumDirectionMixTexture
{
    fn evaluate(&self, ctx: &TextureEvalContext, lambda: &SampledWavelengths) -> SampledSpectrum {
        let amt = ctx.n.dot_vector(self.dir);
        let mut t1 = SampledSpectrum::from_const(0.0);
        let mut t2 = SampledSpectrum::from_const(0.0);
        if amt != 0.0 
        {
            t1 = self.tex1.evaluate(ctx, lambda);
        }
        if amt != 1.0
        {
            t2 = self.tex2.evaluate(ctx, lambda);
        }
        amt * t1 + (1.0 - amt) * t2
    }
}


/// Provides an interface for 2D texture coordinate generation.
pub trait TextureMapping2DI {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D;
}

// TODO let's also provide a 3D mapping equivalent. pg 654.

/// Provides 2D texture coordinate generation.
#[derive(Debug)]
pub enum TextureMapping2D {
    UV(UVMapping),
    Spherical(SphericalMapping),
    Cylindrical(CylindricalMapping),
    Planar(PlanarMapping),
}

impl TextureMapping2D
{
    pub fn create(
        parameters: &mut ParameterDictionary,
        render_from_texture: &Transform,
        loc: &FileLoc,
    ) -> TextureMapping2D
    {
        // TODO change the default to take &str...
        let ty = parameters.get_one_string("mapping", "uv".to_owned());
        match ty.as_str()
        {
            "uv" => {
                let su = parameters.get_one_float("uscale", 1.0);
                let sv = parameters.get_one_float("vscale", 1.0);
                let du = parameters.get_one_float("udelta", 0.0);
                let dv = parameters.get_one_float("vdelta", 0.0);
                TextureMapping2D::UV(UVMapping { su, sv, du, dv })
            },
            "spherical" => TextureMapping2D::Spherical(SphericalMapping {
                texture_from_render: render_from_texture.inverse(),
            }),
            "cylindrical" => TextureMapping2D::Cylindrical(CylindricalMapping{
                texture_from_render: render_from_texture.inverse(),
            }),
            "planaer" => TextureMapping2D::Planar(PlanarMapping {
                texture_from_render: render_from_texture.inverse(),
                vs: parameters.get_one_vector3f("v1", Vector3f::X),
                vt: parameters.get_one_vector3f("v2", Vector3f::Y),
                ds: parameters.get_one_float("udelta", 0.0),
                dt: parameters.get_one_float("vdelta", 0.0)
            }),
            _ => panic!("Unknown texture mapping type {}", ty),
        }
    }
}

impl TextureMapping2DI for TextureMapping2D {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        match self {
            TextureMapping2D::UV(m) => m.map(ctx),
            TextureMapping2D::Spherical(m) => m.map(ctx),
            TextureMapping2D::Cylindrical(m) => m.map(ctx),
            TextureMapping2D::Planar(m) => m.map(ctx),
        }
    }
}

/// Uses the (u, v) coordinates in the TextureEvalContext to compute the texture
/// coordinates, optionally scaling and offsetting their values in each dimension.
#[derive(Debug)]
pub struct UVMapping {
    /// Scale s from u
    su: Float,
    /// Scale s from v
    sv: Float,
    /// Offset u
    du: Float,
    /// Offset v
    dv: Float,
}

impl Default for UVMapping {
    fn default() -> Self {
        Self {
            su: 1.0,
            sv: 1.0,
            du: 0.0,
            dv: 0.0,
        }
    }
}

impl TextureMapping2DI for UVMapping {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        // Compute texture differentials for 2D (u, v) mapping
        let dsdx = self.su * ctx.dudx;
        let dsdy = self.su * ctx.dudy;
        let dtdx = self.sv * ctx.dvdx;
        let dtdy = self.sv * ctx.dvdy;

        let st = Point2f::new(self.su * ctx.uv[0] + self.du, self.sv * ctx.uv[1] + self.dv);
        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

#[derive(Debug)]
pub struct SphericalMapping
{
    texture_from_render: Transform,
}

impl TextureMapping2DI for SphericalMapping {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        let pt = self.texture_from_render.apply(&ctx.p);
        let x2y2 = sqr(pt.x) + sqr(pt.y);
        let sqrtx2y2 = x2y2.sqrt();
        let dsdp = Vector3f::new(-pt.y, pt.x, 0.0) / (2.0 * PI_F * x2y2);
        let dtdp = 
            1.0 / (PI_F * (x2y2 + sqr(pt.z))) *
            Vector3f::new(pt.x * pt.z / sqrtx2y2, pt.y * pt.z / sqrtx2y2, -sqrtx2y2);
        
        let dpdx = self.texture_from_render.apply(&ctx.dpdx);
        let dpdy = self.texture_from_render.apply(&ctx.dpdy);

        let dsdx = dsdp.dot(dpdx);
        let dsdy = dsdp.dot(dpdy);
        let dtdx = dtdp.dot(dpdx);
        let dtdy = dtdp.dot(dpdy);

        let vec = (pt - Point3f::ZERO).normalize();
        let st = Point2f::new(
            spherical_theta(vec) * INV_PI,
            spherical_theta(vec) * INV_2PI,
        );
        
        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

#[derive(Debug)]
pub struct CylindricalMapping
{
    texture_from_render: Transform,
}

impl TextureMapping2DI for CylindricalMapping {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        let pt = self.texture_from_render.apply(&ctx.p);
        let x2y2 = sqr(pt.x) + sqr(pt.y);
        let dsdp = Vector3f::new(-pt.y, pt.x, 0.0) / (2.0 * PI_F * x2y2);
        let dtdp = Vector3f::Z;
        let dpdx = self.texture_from_render.apply(&ctx.dpdx);
        let dpdy = self.texture_from_render.apply(&ctx.dpdy);
        let dsdx = dsdp.dot(dpdx);
        let dsdy = dsdp.dot(dpdy);
        let dtdx = dtdp.dot(dpdx);
        let dtdy = dtdp.dot(dpdy);

        let st = Point2f::new(
            PI_F + Float::atan2(pt.y, pt.x) * INV_2PI,
            pt.z,
        );

        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

#[derive(Debug)]
pub struct PlanarMapping
{
    texture_from_render: Transform,
    vs: Vector3f,
    vt: Vector3f,
    ds: Float,
    dt: Float,
}

impl TextureMapping2DI for PlanarMapping {
    fn map(&self, ctx: &TextureEvalContext) -> TexCoord2D {
        let vec: Vector3f = self.texture_from_render.apply(&ctx.p).into();
        let dpdx = self.texture_from_render.apply(&ctx.dpdx);
        let dpdy = self.texture_from_render.apply(&ctx.dpdy);
        let dsdx = self.vs.dot(dpdx);
        let dsdy = self.vs.dot(dpdy);
        let dtdx = self.vt.dot(dpdx);
        let dtdy = self.vt.dot(dpdy);

        let st = Point2f::new(
            self.ds + vec.dot(self.vs),
            self.dt + vec.dot(self.vt),
        );
        TexCoord2D {
            st,
            dsdx,
            dsdy,
            dtdx,
            dtdy,
        }
    }
}

/// Stores the (s, t) texture cordinates and estimates for the change in (s, t) w.r.t. pixel
/// x and y coordinates so that textures that using the mapping can determine the (s, t) sampling rate
/// and filter accordingly.
pub struct TexCoord2D {
    st: Point2f,
    dsdx: Float,
    dsdy: Float,
    dtdx: Float,
    dtdy: Float,
}

/// Stores relevant geometric information at the shading point for texture evaluation
pub struct TextureEvalContext {
    p: Point3f,
    dpdx: Vector3f,
    dpdy: Vector3f,
    n: Normal3f,
    uv: Point2f,
    dudx: Float,
    dudy: Float,
    dvdx: Float,
    dvdy: Float,
}

impl TextureEvalContext {
    pub fn new(
        p: Point3f,
        dpdx: Vector3f,
        dpdy: Vector3f,
        n: Normal3f,
        uv: Point2f,
        dudx: Float,
        dudy: Float,
        dvdx: Float,
        dvdy: Float,
    ) -> TextureEvalContext {
        TextureEvalContext {
            p,
            dpdx,
            dpdy,
            n,
            uv,
            dudx,
            dudy,
            dvdx,
            dvdy,
        }
    }
}

impl From<SurfaceInteraction> for TextureEvalContext {
    fn from(value: SurfaceInteraction) -> Self {
        Self {
            p: value.p(),
            dpdx: value.dpdx,
            dpdy: value.dpdy,
            n: value.interaction.n,
            uv: value.interaction.uv,
            dudx: value.dudx,
            dudy: value.dudy,
            dvdx: value.dvdx,
            dvdy: value.dvdy,
        }
    }
}

impl From<&SurfaceInteraction> for TextureEvalContext {
    fn from(value: &SurfaceInteraction) -> Self {
        Self {
            p: value.p(),
            dpdx: value.dpdx,
            dpdy: value.dpdy,
            n: value.interaction.n,
            uv: value.interaction.uv,
            dudx: value.dudx,
            dudy: value.dudy,
            dvdx: value.dvdx,
            dvdy: value.dvdy,
        }
    }
}

impl From<Interaction> for TextureEvalContext {
    fn from(value: Interaction) -> Self {
        Self {
            p: value.p(),
            dpdx: Default::default(),
            dpdy: Default::default(),
            n: Default::default(),
            uv: value.uv,
            dudx: Default::default(),
            dudy: Default::default(),
            dvdx: Default::default(),
            dvdy: Default::default(),
        }
    }
}
