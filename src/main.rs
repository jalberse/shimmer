use rand::{distributions::Uniform, prelude::Distribution, rngs::StdRng, SeedableRng};
use shimmer::{
    bounding_box::Bounds2i,
    color::RGB,
    colorspace::RgbColorSpace,
    film::{FilmI, PixelSensor, RgbFilm},
    filter::{BoxFilter, Filter},
    image_metadata::ImageMetadata,
    spectra::spectrum::RgbAlbedoSpectrum,
    spectra::{sampled_wavelengths::SampledWavelengths, spectrum::SpectrumI},
    vecmath::{Point2i, Tuple2, Vector2f},
    Float,
};

fn main() {
    let cs = RgbColorSpace::get_named(shimmer::colorspace::NamedColorSpace::SRGB);
    let sensor = PixelSensor::new(cs, &None, 1.0);
    let mut film = RgbFilm::new(
        Point2i::new(720, 720),
        Bounds2i::new(Point2i::ZERO, Point2i::new(720, 720)),
        Filter::BoxFilter(BoxFilter::new(Vector2f::ONE)),
        0.1,
        sensor,
        "NoFilename",
        cs.clone(),
        1.0,
        false,
    );
    let mut rng = StdRng::seed_from_u64(0);
    let between = Uniform::from(0.0..1.0);
    let num_samples = 1000;
    for x in film.pixel_bounds().min.x..film.pixel_bounds().max.x {
        for y in film.pixel_bounds().min.y..film.pixel_bounds().max.y {
            let r = x as Float / film.pixel_bounds().width() as Float;
            let b = y as Float / film.pixel_bounds().height() as Float;
            let g = 0.0;
            let rgb = RGB::new(r, g, b);
            let albedo_spectrum = RgbAlbedoSpectrum::new(&cs, &rgb);
            for _ in 0..num_samples {
                let p = Point2i::new(x, y);
                let u = between.sample(&mut rng);
                let sampled_wavelengths = SampledWavelengths::sample_uniform(u);
                let sampled_spectrum = albedo_spectrum.sample(&sampled_wavelengths);
                film.add_sample(&p, &sampled_spectrum, &sampled_wavelengths, &None, 1.0);
            }
        }
    }

    let splat_scale = 1.0;
    let metadata = ImageMetadata::new();
    film.write_image(&metadata, splat_scale);
}
