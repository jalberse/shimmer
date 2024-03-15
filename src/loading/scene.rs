use std::{
    collections::{HashMap, HashSet},
    ops::{Index, IndexMut},
    path::Path,
    sync::{atomic, Arc, Mutex},
};

use crate::{
    aggregate::{create_accelerator, BvhAggregate}, camera::{Camera, CameraI, CameraTransform}, color::{ColorEncoding, ColorEncodingCache, LinearColorEncoding}, colorspace::RgbColorSpace, file::resolve_filename, film::{Film, FilmI}, filter::Filter, image::Image, integrator::{create_integrator, Integrator}, light::Light, loading::{paramdict::{NamedTextures, ParameterDictionary, TextureParameterDictionary}, parser_target::{FileLoc, ParsedParameterVector, ParserTarget}}, material::Material, media::Medium, mipmap::MIPMap, options::Options, primitive::{GeometricPrimitive, Primitive, SimplePrimitive, TransformedPrimitive}, sampler::Sampler, shape::Shape, spectra::spectrum, square_matrix::SquareMatrix, texture::{FloatConstantTexture, FloatTexture, SpectrumTexture, TexInfo}, transform::Transform, util::normalize_arg, vecmath::{Point3f, Tuple3, Vector3f}, Float
};

use log::{trace, warn};
use spectrum::Spectrum;
use string_interner::{symbol::SymbolU32, StringInterner};

use super::paramdict::SpectrumType;

// TODO If/when we make this multi-threaded, most of these will be within a Mutex.
//      For now, code it sequentially.
pub struct BasicScene {
    pub integrator: Option<SceneEntity>,
    pub accelerator: Option<SceneEntity>,
    pub film_color_space: Option<Arc<RgbColorSpace>>,
    pub shapes: Vec<ShapeSceneEntity>,
    pub instances: Vec<InstanceSceneEntity>,
    pub instance_definitions: Vec<InstanceDefinitionSceneEntity>,
    camera: Option<Camera>,
    film: Option<Film>,
    sampler: Option<Sampler>,
    named_materials: Vec<(String, SceneEntity)>,
    materials: Vec<SceneEntity>,
    area_lights: Vec<SceneEntity>,
    normal_maps: HashMap<String, Arc<Image>>,
    serial_float_textures: Vec<(String, TextureSceneEntity)>,
    serial_spectrum_textures: Vec<(String, TextureSceneEntity)>,
    async_spectrum_textures: Vec<(String, TextureSceneEntity)>,
    loading_texture_filenames: HashSet<String>,

    // TODO When we switch to asynch, we will load all of these at once in parallel
    //   in create_textures(), create_lights() etc. For now, just load as we parse into this.
    textures: NamedTextures,
    lights: Vec<Arc<Light>>,
}

impl Default for BasicScene {
    fn default() -> Self {
        Self {
            integrator: Default::default(),
            accelerator: Default::default(),
            film_color_space: Default::default(),
            shapes: Default::default(),
            instances: Default::default(),
            instance_definitions: Default::default(),
            camera: Default::default(),
            film: Default::default(),
            sampler: Default::default(),
            named_materials: Default::default(),
            materials: Default::default(),
            area_lights: Default::default(),
            normal_maps: Default::default(),
            serial_float_textures: Default::default(),
            serial_spectrum_textures: Default::default(),
            async_spectrum_textures: Default::default(),
            loading_texture_filenames: Default::default(),
            textures: Default::default(),
            lights: Default::default(),
        }
    }
}

impl BasicScene {
    pub fn get_camera(&self) -> Option<Camera> {
        self.camera.clone()
    }

    pub fn get_film(&self) -> Option<Film> {
        self.film.clone()
    }

    pub fn get_sampler(&self) -> Option<Sampler> {
        self.sampler.clone()
    }

    pub fn set_options(
        &mut self,
        mut filter: SceneEntity,
        mut film: SceneEntity,
        mut camera: CameraSceneEntity,
        mut sampler: SceneEntity,
        integ: SceneEntity,
        accel: SceneEntity,
        string_interner: &StringInterner,
        options: &Options,
    ) {
        self.film_color_space = Some(film.parameters.color_space.clone());
        self.integrator = Some(integ);
        self.accelerator = Some(accel);

        let filt = Filter::create(
            &string_interner
                .resolve(filter.name)
                .expect("Unresolved name!"),
            &mut filter.parameters,
            &mut filter.loc,
        );

        let exposure_time = camera.base.parameters.get_one_float("shutterclose", 1.0)
            - camera.base.parameters.get_one_float("shutteropen", 0.0);

        if exposure_time <= 0.0 {
            panic!(
                "{} The specified camera shutter times imply the camera won't open.",
                camera.base.loc
            );
        }

        self.film = Some(Film::create(
            &string_interner.resolve(film.name).unwrap(),
            &mut film.parameters,
            exposure_time,
            &camera.camera_transform,
            filt,
            &film.loc,
            options,
        ));

        let res = self.film.as_ref().unwrap().full_resolution();
        self.sampler = Some(Sampler::create(
            &string_interner.resolve(sampler.name).unwrap(),
            &mut sampler.parameters,
            res,
            options,
            &mut sampler.loc,
        ));

        self.camera = Some(Camera::create(
            &string_interner.resolve(camera.base.name).unwrap(),
            &mut camera.base.parameters,
            None,
            camera.camera_transform,
            self.film.as_ref().unwrap().clone(),
            options,
            &mut camera.base.loc,
        ));
    }

    fn add_named_material(&mut self, name: &str, mut material: SceneEntity) {
        self.load_normal_map(&mut material.parameters);
        self.named_materials.push((name.to_owned(), material));
    }

    // Returns the new material index
    fn add_material(&mut self, mut material: SceneEntity) -> i32 {
        self.load_normal_map(&mut material.parameters);
        self.materials.push(material);
        (self.materials.len() - 1) as i32
    }

    #[allow(dead_code)] // TODO This fn
    fn add_medium(&mut self, _medium: SceneEntity) {
        todo!()
    }

    fn add_float_texture(
        &mut self,
        name: &str,
        mut texture: TextureSceneEntity,
        string_interner: &StringInterner,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) {
        // TODO Check if animated once we add animated transforms.

        if string_interner
            .resolve(texture.base.name)
            .expect("Unknown texture name")
            != "imagemap"
            && string_interner
                .resolve(texture.base.name)
                .expect("Unknown texture name")
                != "ptex"
        {
            self.serial_float_textures.push((name.to_owned(), texture));
            return;
        }

        let filename = texture
            .base
            .parameters
            .get_one_string("filename", "");

        let filename = resolve_filename(options, filename.as_str());
        if filename.is_empty() {
            panic!("{} No filename provided for texture.", texture.base.loc);
        }

        let path = Path::new(filename.as_str());
        if !path.exists() {
            panic!("Texture \"{}\" not found.", filename);
        }

        if self.loading_texture_filenames.contains(&filename) {
            self.serial_float_textures.push((name.to_owned(), texture));
            return;
        }

        self.loading_texture_filenames.insert(filename);

        // TODO Can make this async.
        let render_from_texture = texture.render_from_object;

        let mut tex_dict = TextureParameterDictionary::new(texture.base.parameters.clone());
        let float_texture = FloatTexture::create(
            string_interner
                .resolve(texture.base.name)
                .expect("Unknown symbol"),
            render_from_texture,
            &mut tex_dict,
            &texture.base.loc,
            &self.textures,
            options,
            texture_cache,
            gamma_encoding_cache,
        );

        self.textures
            .float_textures
            .insert(name.to_owned(), Arc::new(float_texture));
    }

    fn add_spectrum_texture(
        &mut self,
        name: &str,
        mut texture: TextureSceneEntity,
        string_interner: &StringInterner,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) {
        if string_interner.resolve(texture.base.name).unwrap() != "ptex"
            && string_interner.resolve(texture.base.name).unwrap() != "imagemap"
        {
            self.serial_spectrum_textures
                .push((name.to_owned(), texture));
            return;
        }

        let filename = texture
            .base
            .parameters
            .get_one_string("filename", "");
        let filename = resolve_filename(options, filename.as_str());

        if filename.is_empty() {
            panic!("{} No filename provided for texture.", texture.base.loc);
        }

        let path = Path::new(&filename);
        if !path.exists() {
            panic!("Texture \"{}\" not found.", filename);
        }

        if self.loading_texture_filenames.contains(&filename) {
            self.serial_spectrum_textures
                .push((name.to_owned(), texture));
            return;
        }
        self.loading_texture_filenames.insert(filename);

        self.async_spectrum_textures.push((name.to_owned(), texture.clone()));

        let render_from_texture = texture.render_from_object;
        // None for the textures, as with float textures.
        let mut text_dict = TextureParameterDictionary::new(texture.base.parameters.clone());
        // Only create Albedo for now; will get other two types created in create_textures().
        let spectrum_texture = SpectrumTexture::create(
            string_interner
                .resolve(texture.base.name)
                .expect("Unknown symbol"),
            render_from_texture,
            &mut text_dict,
            crate::loading::paramdict::SpectrumType::Albedo,
            cached_spectra,
            &self.textures,
            &texture.base.loc,
            options,
            texture_cache,
            gamma_encoding_cache,
        );
        self.textures
            .albedo_spectrum_textures
            .insert(name.to_owned(), Arc::new(spectrum_texture));
    }

    fn add_light(
        &mut self,
        light: &mut LightSceneEntity,
        string_interner: &StringInterner,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) {
        // TODO Get medium, when I add mediums - will replace none below.
        // TODO Check for animated light and warn, when I add animated transforms.

        // TODO Change to async, or otherwise parallelize (i.e. could place these
        //  params into a vec, and consume that vec in a par_iter() in create_lights).
        self.lights.push(Arc::new(Light::create(
            string_interner
                .resolve(light.base.base.name)
                .expect("Unknown symbol"),
            &mut light.base.base.parameters,
            light.base.render_from_object,
            self.get_camera().unwrap().get_camera_transform(),
            None,
            &light.base.base.loc,
            cached_spectra,
            options,
        )));
    }

    /// Returns the new area light index.
    fn add_area_light(&mut self, light: SceneEntity) -> i32 {
        self.area_lights.push(light);
        (self.area_lights.len() - 1) as i32
    }

    fn add_shapes(&mut self, shapes: &[ShapeSceneEntity]) {
        self.shapes.extend_from_slice(shapes);
    }

    // TODO add_animated_shapes().

    fn add_instance_definition(&mut self, instance: InstanceDefinitionSceneEntity) {
        self.instance_definitions.push(instance);
    }

    fn add_instance_uses(&mut self, instances: &[InstanceSceneEntity]) {
        self.instances.extend_from_slice(instances);
    }

    fn done(&mut self) {
        // TODO Check for unused textures, lights, etc and warn about them.
    }

    fn load_normal_map(&mut self, parameters: &mut ParameterDictionary) {
        let normal_map_filename = parameters.get_one_string("normalmap", "");
        if normal_map_filename.is_empty() {
            return;
        }
        let filename = Path::new(&normal_map_filename);
        if !filename.exists() {
            warn!("Normal map \"{}\" not found.", filename.display());
        }

        let image_and_metadata = Image::read(
            filename,
            Some(ColorEncoding::get("linear", None)),
        );

        let image = image_and_metadata.image;
        let rgb_desc = image.get_channel_desc(&["R", "G", "B"]);
        if rgb_desc.is_none() {
            panic!(
                "Normal map \"{}\" should have RGB channels.",
                filename.display()
            );
        }
        let rgb_desc = rgb_desc.unwrap();
        if rgb_desc.size() != 3 {
            panic!(
                "Normal map \"{}\" should have RGB channels.",
                filename.display()
            );
        }
        let image = Arc::new(image);
        self.normal_maps.insert(normal_map_filename, image);
    }

    pub fn create_textures(
        &mut self,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        string_interner: &StringInterner,
        options: &Options,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) -> NamedTextures {
        // TODO Note that albedo spectrum and float textures were created
        //  earlier; if we switch to asynch, we will want to resolve them here.

        for tex in &self.async_spectrum_textures {
            let render_from_texture = tex.1.render_from_object;

            // These are all image textures, so nullptr is fine for the textures, as in float.
            let mut tex_dict = TextureParameterDictionary::new(tex.1.base.parameters.clone());

            let unbounded_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Unbounded,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            );

            let illum_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Illuminant,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            );

            self.textures
                .unbounded_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(unbounded_tex));
            self.textures
                .illuminant_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(illum_tex));
        }

        for tex in &self.serial_float_textures {
            let render_from_texture = tex.1.render_from_object;

            let mut tex_dict = TextureParameterDictionary::new(tex.1.base.parameters.clone());

            // TODO Will need to pass self.textures to create() functions, so they can resolve textures.
            // Not encessary right now as we only have the FloatConstant texture.
            let float_texture = FloatTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                &tex.1.base.loc,
                &self.textures,
                options,
                texture_cache,
                gamma_encoding_cache,
            );

            self.textures
                .float_textures
                .insert(tex.0.to_owned(), Arc::new(float_texture));
        }

        for tex in &self.serial_spectrum_textures {
            let render_from_texture = tex.1.render_from_object;

            let mut tex_dict = TextureParameterDictionary::new(tex.1.base.parameters.clone());

            // TODO Will need to pass self.textures to create() functions, so they can resolve textures.
            // Not encessary right now as we only have the ConstantSpectrum texture.
            let albedo_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Albedo,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            );

            let unbounded_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Unbounded,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            );

            let illum_tex = SpectrumTexture::create(
                string_interner
                    .resolve(tex.1.base.name)
                    .expect("Unexpected symbol"),
                render_from_texture,
                &mut tex_dict,
                SpectrumType::Illuminant,
                cached_spectra,
                &self.textures,
                &tex.1.base.loc,
                options,
                texture_cache,
                gamma_encoding_cache,
            );

            self.textures
                .albedo_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(albedo_tex));
            self.textures
                .unbounded_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(unbounded_tex));
            self.textures
                .illuminant_spectrum_textures
                .insert(tex.0.to_owned(), Arc::new(illum_tex));
        }

        // TODO It would probably be better to not have to clone the textures here.
        //  Can we return a reference?
        //  Storing self.textures as Arc or Rc doesn't work since we need it mutable.
        //  This is fine for now.
        self.textures.clone()
    }

    // Returns a vector of the lights, and a map from shape index to lights.
    pub fn create_lights(
        &mut self,
        textures: &NamedTextures,
        string_interner: &StringInterner,
        options: &Options,
    ) -> (Arc<Vec<Arc<Light>>>, HashMap<usize, Vec<Arc<Light>>>) {
        let mut shape_index_to_area_lights = HashMap::new();
        // TODO We'll want to handle media and alpha textures, but hold off for now.

        let mut lights = Vec::new();

        for i in 0..self.shapes.len() {
            let shape = &mut self.shapes[i];

            if shape.light_index == -1 {
                continue;
            }

            let material_name = if !shape.material_name.is_empty() {
                let mut material = self
                    .named_materials
                    .iter_mut()
                    .find(|m| m.0 == shape.material_name);
                if material.is_none() {
                    panic!(
                        "{}: Couldn't find named material {}.",
                        shape.base.loc, shape.material_name
                    );
                }
                let material = material.as_mut().unwrap();
                assert!(
                    material
                        .1
                        .parameters
                        .get_one_string("type", "")
                        .len()
                        > 0
                );
                material.1.parameters.get_one_string("type", "")
            } else {
                assert!(
                    shape.material_index >= 0
                        && (shape.material_index as usize) < self.materials.len()
                );
                string_interner
                    .resolve(self.materials[shape.material_index as usize].name)
                    .unwrap()
                    .to_owned()
            };

            if material_name == "interface" || material_name == "none" || material_name == "" {
                warn!(
                    "{}: Ignoring area light specification for shape with interface material",
                    shape.base.loc
                );
                continue;
            }

            let shape_objects = Shape::create(
                string_interner.resolve(shape.base.name).unwrap(),
                &shape.render_from_object,
                &shape.object_from_render,
                shape.reverse_orientation,
                &mut shape.base.parameters,
                &textures.float_textures,
                &shape.base.loc,
                options,
            );

            // TODO Support an alpha texture if parameters.get_texture("alpha") is specified.
            let alpha = shape.base.parameters.get_one_float("alpha", 1.0);
            let alpha = Arc::new(FloatTexture::Constant(FloatConstantTexture::new(alpha)));

            // TODO create medium_interface

            let mut shape_lights = Vec::new();
            let area_light_entity = &mut self.area_lights[shape.light_index as usize];
            for ps in shape_objects.iter() {
                let area = Arc::new(Light::create_area(
                    string_interner.resolve(area_light_entity.name).unwrap(),
                    &mut area_light_entity.parameters,
                    shape.render_from_object,
                    ps.clone(),
                    alpha.clone(),
                    &area_light_entity.loc,
                    options,
                ));

                lights.push(area.clone());
                shape_lights.push(area);
            }

            shape_index_to_area_lights.insert(i, shape_lights);
        }
        trace!("Finished area lights");

        // TODO We could create other lights in parallel here;
        //  for now, we are creating them in add_light() in self.lights.
        self.lights.append(&mut lights);

        // TODO We'd rather move self.lights out rather than an expensive clone.
        //   We can switch to make lights vec in this fn though when we parallelize,
        //   which obviates this issue.
        (Arc::new(self.lights.clone()), shape_index_to_area_lights)
    }

    /// Returns a tuple with a map of named materials, and a vec of unnamed materials.
    pub fn create_materials(
        &mut self,
        textures: &NamedTextures,
        string_interner: &StringInterner,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) -> (HashMap<String, Arc<Material>>, Vec<Arc<Material>>) {
        // TODO Note that we'd create normal_maps here if/when we parallelize.
        //  For now they're already been loaded into self.normal_maps.

        let mut named_materials_out: HashMap<String, Arc<Material>> = HashMap::new();
        for (name, material) in &mut self.named_materials {
            if named_materials_out.iter().find(|nm| nm.0 == name).is_some() {
                panic!("{}: Named material {} redefined.", material.loc, name);
            }

            let ty = material.parameters.get_one_string("type", "");
            if ty.is_empty() {
                panic!("{}: No type specified for material {}", material.loc, name);
            }

            let filename = resolve_filename(
                options,
                &material
                    .parameters
                    .get_one_string("normalmap", ""),
            );
            let normal_map = if filename.is_empty() {
                None
            } else {
                let image = self.normal_maps.get(&filename);
                if image.is_none() {
                    panic!("{}: Normal map \"{}\" not found.", material.loc, filename);
                }
                Some(image.unwrap().clone())
            };

            let mut tex_dict = TextureParameterDictionary::new(material.parameters.clone());
            let m = Arc::new(Material::create(
                &ty,
                &mut tex_dict,
                textures,
                normal_map,
                &mut named_materials_out,
                cached_spectra,
                &material.loc,
            ));
            named_materials_out.insert(name.to_string(), m);
        }

        let mut materials_out = Vec::with_capacity(self.materials.len());
        for mtl in &mut self.materials {
            let filename = resolve_filename(
                options,
                &mtl.parameters.get_one_string("normalmap", ""),
            );
            let normal_map = if filename.is_empty() {
                None
            } else {
                let image = self.normal_maps.get(&filename);
                if image.is_none() {
                    panic!("{}: Normal map \"{}\" not found.", mtl.loc, filename);
                }
                Some(image.unwrap().clone())
            };

            let mut tex_dict = TextureParameterDictionary::new(mtl.parameters.clone());
            let m = Arc::new(Material::create(
                string_interner.resolve(mtl.name).unwrap(),
                &mut tex_dict,
                textures,
                normal_map,
                &mut named_materials_out,
                cached_spectra,
                &mtl.loc,
            ));
            materials_out.push(m);
        }

        (named_materials_out, materials_out)
    }

    pub fn create_aggregate(
        &mut self,
        textures: &NamedTextures,
        shape_index_to_area_lights: &HashMap<usize, Vec<Arc<Light>>>,
        _media: &HashMap<String, Arc<Medium>>, // TODO This will be used in future aggregate creation.
        named_materials: &HashMap<String, Arc<Material>>,
        materials: &[Arc<Material>],
        string_interner: &StringInterner,
        options: &Options,
    ) -> Arc<Primitive> {
        // TODO We'll need lambdas for find_medium and get_alpha_texture.

        let create_primitives_for_shapes =
            |shapes: &mut [ShapeSceneEntity]| -> Vec<Arc<Primitive>> {
                let mut shape_vectors: Vec<Vec<Arc<Shape>>> = vec![Vec::new(); shapes.len()];
                // TODO parallelize
                for i in 0..shapes.len() {
                    let sh = &mut shapes[i];
                    shape_vectors[i] = Shape::create(
                        string_interner.resolve(sh.base.name).unwrap(),
                        &sh.render_from_object,
                        &sh.object_from_render,
                        sh.reverse_orientation,
                        &mut sh.base.parameters,
                        &textures.float_textures,
                        &sh.base.loc,
                        options,
                    );
                }

                let mut primitives = Vec::new();
                for i in 0..shapes.len() {
                    let sh = &mut shapes[i];
                    let shapes = &shape_vectors[i];
                    if shapes.is_empty() {
                        continue;
                    }

                    // TODO get alpha texture here

                    let mtl = if !sh.material_name.is_empty() {
                        named_materials
                            .get(sh.material_name.as_str())
                            .expect("No material name defined")
                    } else {
                        assert!(
                            sh.material_index >= 0
                                && (sh.material_index as usize) < materials.len()
                        );
                        &materials[sh.material_index as usize]
                    };

                    // TODO Create medium interface

                    let area_lights = shape_index_to_area_lights.get(&i);
                    for j in 0..shapes.len() {
                        // Possibly create area light for shape
                        let area = if sh.light_index != -1 && area_lights.is_some() {
                            let area_light = area_lights.unwrap();
                            Some(area_light[j].clone())
                        } else {
                            None
                        };

                        // TODO Also check against !mi.is_medium_transition() and alpha_tex.is_none()
                        if area.is_none() {
                            let prim = Arc::new(Primitive::Simple(SimplePrimitive {
                                shape: shapes[j].clone(),
                                material: mtl.clone(),
                            }));
                            primitives.push(prim);
                        } else {
                            let prim = Arc::new(Primitive::Geometric(GeometricPrimitive::new(
                                shapes[j].clone(),
                                mtl.clone(),
                                area,
                            )));
                            primitives.push(prim);
                        }
                    }
                }
                primitives
            };

        trace!("Starting shapes");
        let mut primitives = create_primitives_for_shapes(&mut self.shapes);

        self.shapes.clear();
        self.shapes.shrink_to_fit();

        // TODO Animated shapes, when added.
        trace!("Finished shapes");

        trace!("Starting instances");
        // TODO Can we use a SymbolU32 here for the key instead of String?
        let mut instance_definitions: HashMap<String, Option<Arc<Primitive>>> = HashMap::new();
        for inst in &mut self.instance_definitions {
            let instance_primitives = create_primitives_for_shapes(&mut inst.shapes);
            // TODO animated instance primitives

            let instance_primitives = if instance_primitives.len() > 1 {
                // TODO Use a better split method
                let bvh = BvhAggregate::new(
                    instance_primitives,
                    1,
                    crate::aggregate::SplitMethod::Middle,
                );
                vec![Arc::new(Primitive::BvhAggregate(bvh))]
            } else {
                instance_primitives
            };

            if instance_primitives.is_empty() {
                instance_definitions
                    .insert(string_interner.resolve(inst.name).unwrap().to_owned(), None);
            } else {
                instance_definitions.insert(
                    string_interner.resolve(inst.name).unwrap().to_owned(),
                    Some(instance_primitives[0].clone()),
                );
            }
            todo!()
        }

        // Don't need these anymore, we've created them as primitives.
        self.instance_definitions.clear();
        self.instance_definitions.shrink_to_fit();

        // Use those instance definitions to create actual instances
        for inst in &self.instances {
            let instance = instance_definitions
                .get(string_interner.resolve(inst.name).unwrap())
                .expect("Unknown instance name");

            if instance.is_none() {
                continue;
            }

            let instance = instance.as_ref().unwrap();

            // TODO Handle animated instances
            let prim = Arc::new(Primitive::Transformed(TransformedPrimitive::new(
                instance.clone(),
                inst.render_from_instance,
            )));
            primitives.push(prim);
        }

        // Likewise don't need the instance scene entities anymore, as we've made them
        // as primitives.
        self.instances.clear();
        self.instances.shrink_to_fit();

        trace!("Finished instances");

        trace!("Starting top-level accelerator");
        let aggregate = Arc::new(create_accelerator(
            string_interner
                .resolve(self.accelerator.as_ref().unwrap().name)
                .unwrap(),
            primitives,
            &mut self.accelerator.as_mut().unwrap().parameters,
        ));
        trace!("Finished top-level accelerator");

        aggregate
    }

    pub fn create_integrator(
        &mut self,
        camera: Camera,
        sampler: Sampler,
        accelerator: Arc<Primitive>,
        lights: Arc<Vec<Arc<Light>>>,
        string_interner: &StringInterner,
    ) -> Box<dyn Integrator> {
        create_integrator(
            string_interner
                .resolve(self.integrator.as_ref().unwrap().name)
                .unwrap(),
            &mut self.integrator.as_mut().unwrap().parameters,
            camera,
            sampler,
            accelerator,
            lights,
            self.film_color_space.as_ref().unwrap().clone(),
        )
    }
}

/// All objects in the scene are described by various *Entity classes.
/// The SceneEntity is the simplest; it records the name of the entity,
/// the file location of the of the associated statement in the scene description,
/// and any user-provided parameters.
/// It is used for the film, sampler, integrator, pixel filter, and accelerator,
/// and is also used as a "base" for some of the other scene entity types.
#[derive(Debug, Clone)]
pub struct SceneEntity {
    name: SymbolU32,
    loc: FileLoc,
    parameters: ParameterDictionary,
}

impl SceneEntity {
    pub fn new(
        name: &str,
        loc: FileLoc,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner,
    ) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            loc,
            parameters,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShapeSceneEntity {
    base: SceneEntity,
    render_from_object: Transform,
    object_from_render: Transform,
    reverse_orientation: bool,
    // TODO It should be one of these two - enum?
    // It just makes instatiation a bit more complex
    material_index: i32,
    material_name: String,
    light_index: i32,
    _inside_medium: String,
    _outside_medium: String,
}

#[derive(Debug, Clone)]
pub struct CameraSceneEntity {
    base: SceneEntity,
    camera_transform: CameraTransform,
    // TODO medium: String,
}

#[derive(Debug, Clone)]
pub struct TransformedSceneEntity {
    base: SceneEntity,
    // TODO this may need to be an AnimatedTransform
    render_from_object: Transform,
}

impl TransformedSceneEntity {
    pub fn new(
        name: &str,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        render_from_object: Transform,
    ) -> Self {
        Self {
            base: SceneEntity::new(name, loc, parameters, string_interner),
            render_from_object,
        }
    }
}

pub type MediumSceneEntity = TransformedSceneEntity;
pub type TextureSceneEntity = TransformedSceneEntity;

#[derive(Debug, Clone)]
pub struct LightSceneEntity {
    base: TransformedSceneEntity,
    _medium: String,
}

impl LightSceneEntity {
    fn new(
        name: &str,
        parameters: ParameterDictionary,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        render_from_light: Transform,
        medium: &str,
    ) -> LightSceneEntity {
        let base =
            TransformedSceneEntity::new(name, parameters, string_interner, loc, render_from_light);
        LightSceneEntity {
            base,
            _medium: medium.to_owned(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstanceSceneEntity {
    name: SymbolU32,
    _loc: FileLoc,
    render_from_instance: Transform,
    // TODO Possibly aniamted transform
}

impl InstanceSceneEntity {
    pub fn new(
        name: &str,
        loc: FileLoc,
        string_interner: &mut StringInterner,
        render_from_instance: Transform,
    ) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            _loc: loc,
            render_from_instance,
        }
    }
}

pub struct InstanceDefinitionSceneEntity {
    // TODO we will need a stringinterner on this, same as in SceneEntity
    name: SymbolU32,
    _loc: FileLoc,
    shapes: Vec<ShapeSceneEntity>,
    // TODO aniamted_shapes: Vec<AnimatedShapeSceneEntity>,
}

impl InstanceDefinitionSceneEntity {
    pub fn new(name: &str, loc: FileLoc, string_interner: &mut StringInterner) -> Self {
        Self {
            name: string_interner.get_or_intern(name),
            _loc: loc,
            shapes: Default::default(),
        }
    }
}

const MAX_TRANSFORMS: usize = 2;

#[derive(Debug, Copy, Clone, PartialEq)]
struct TransformSet {
    t: [Transform; MAX_TRANSFORMS],
}

impl TransformSet {
    fn inverse(ts: &TransformSet) -> TransformSet {
        let mut t_inv = TransformSet::default();
        for i in 0..MAX_TRANSFORMS {
            t_inv.t[i] = Transform::inverse(&ts.t[i]);
        }
        t_inv
    }

    fn is_animated(&self) -> bool {
        for i in 0..(MAX_TRANSFORMS - 1) {
            if self.t[i] != self.t[i + 1] {
                return true;
            }
        }
        false
    }
}

impl Default for TransformSet {
    fn default() -> Self {
        Self {
            t: [Transform::default(); MAX_TRANSFORMS],
        }
    }
}

impl Index<usize> for TransformSet {
    type Output = Transform;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < MAX_TRANSFORMS);
        &self.t[index]
    }
}

impl IndexMut<usize> for TransformSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < MAX_TRANSFORMS);
        &mut self.t[index]
    }
}

#[derive(Debug, Clone)]
struct GraphicsState {
    current_inside_medium: String,
    current_outside_medium: String,

    // TODO It's one or the other - use an enum.
    current_material_index: i32,
    current_named_material: String,

    area_light_name: String,
    area_light_params: ParameterDictionary,
    area_light_loc: FileLoc,

    shape_attributes: ParsedParameterVector,
    light_attributes: ParsedParameterVector,
    material_attributes: ParsedParameterVector,
    medium_attributes: ParsedParameterVector,
    texture_attributes: ParsedParameterVector,
    reverse_orientation: bool,
    color_space: Arc<RgbColorSpace>,
    ctm: TransformSet,
    active_transform_bits: u32,
    transform_start_time: Float,
    transform_end_time: Float,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            current_inside_medium: Default::default(),
            current_outside_medium: Default::default(),
            current_material_index: 0,
            current_named_material: Default::default(),
            area_light_name: Default::default(),
            area_light_params: Default::default(),
            area_light_loc: Default::default(),
            shape_attributes: Default::default(),
            light_attributes: Default::default(),
            material_attributes: Default::default(),
            medium_attributes: Default::default(),
            texture_attributes: Default::default(),
            reverse_orientation: Default::default(),
            color_space: RgbColorSpace::get_named(crate::colorspace::NamedColorSpace::SRGB).clone(),
            ctm: Default::default(),
            active_transform_bits: BasicSceneBuilder::ALL_TRANSFORM_BITS,
            transform_start_time: 0.0,
            transform_end_time: 1.0,
        }
    }
}

impl GraphicsState {
    pub fn for_active_transforms(&mut self, func: impl Fn(&mut Transform)) {
        for i in 0..MAX_TRANSFORMS {
            if self.active_transform_bits & (1 << i) != 0 {
                func(&mut self.ctm[i]);
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum BlockState {
    OptionsBlock,
    WorldBlock,
}

struct ActiveInstanceDefinition {
    pub active_imports: atomic::AtomicI32,
    pub entity: InstanceDefinitionSceneEntity,
    pub parent: Option<Arc<ActiveInstanceDefinition>>,
}

impl ActiveInstanceDefinition {
    pub fn new(name: &str, loc: FileLoc, string_interner: &mut StringInterner) -> Self {
        Self {
            active_imports: atomic::AtomicI32::new(1),
            entity: InstanceDefinitionSceneEntity::new(name, loc, string_interner),
            parent: None,
        }
    }
}

pub struct BasicSceneBuilder {
    scene: Box<BasicScene>,
    current_block: BlockState,
    graphics_state: GraphicsState,
    named_coordinate_systems: HashMap<String, TransformSet>,
    render_from_world: Transform,
    // TODO Transform cache - we will implement this in the future.
    pushed_graphics_states: Vec<GraphicsState>,
    // TODO Should the push_stack be an enum instead? Natural representation, each variant can store a FileLoc too.
    push_stack: Vec<(u8, FileLoc)>, // 'a' attribute, 'o' object
    // TODO instance definition stuff

    // Buffered for consistent ordering across runs
    shapes: Vec<ShapeSceneEntity>,
    instance_uses: Vec<InstanceSceneEntity>,

    named_material_names: HashSet<String>,
    _medium_names: HashSet<String>,
    float_texture_names: HashSet<String>,
    spectrum_texture_names: HashSet<String>,
    instance_names: HashSet<String>,

    current_material_index: i32,
    sampler: SceneEntity,
    film: SceneEntity,
    integrator: SceneEntity,
    filter: SceneEntity,
    accelerator: SceneEntity,
    camera: CameraSceneEntity,

    active_instance_definition: Option<ActiveInstanceDefinition>,
}

impl BasicSceneBuilder {
    const START_TRANSFORM_BITS: u32 = 1 << 0;
    const END_TRANSFORM_BITS: u32 = 1 << 1;
    const ALL_TRANSFORM_BITS: u32 = (1 << MAX_TRANSFORMS) - 1;

    pub fn new(scene: Box<BasicScene>, string_interner: &mut StringInterner) -> BasicSceneBuilder {
        // TODO Rather than Optional SceneEntities, we need to instantiate them here.
        //   PBRT provides some defaults for their names, I guess...

        // TODO Update default to zsobol
        let sampler = SceneEntity {
            name: string_interner.get_or_intern("independent"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::default(),
        };

        let film = SceneEntity {
            name: string_interner.get_or_intern("rgb"),
            loc: FileLoc::default(),
            parameters: ParameterDictionary::new(
                ParsedParameterVector::new(),
                RgbColorSpace::get_named(crate::colorspace::NamedColorSpace::SRGB).clone(),
            ),
        };

        // TODO Change default to volpath when available.
        let integrator = SceneEntity {
            name: string_interner.get_or_intern("path"),
            loc: FileLoc::default(),
            parameters: Default::default(),
        };

        // TODO Update default to gaussian when available
        let filter = SceneEntity {
            name: string_interner.get_or_intern("box"),
            loc: FileLoc::default(),
            parameters: Default::default(),
        };

        let accelerator = SceneEntity {
            name: string_interner.get_or_intern("bvh"),
            loc: FileLoc::default(),
            parameters: Default::default(),
        };

        let camera = CameraSceneEntity {
            base: SceneEntity {
                name: string_interner.get_or_intern("perspective"),
                loc: FileLoc::default(),
                parameters: Default::default(),
            },
            camera_transform: CameraTransform::default(),
        };

        let mut builder = BasicSceneBuilder {
            scene,
            current_block: BlockState::OptionsBlock,
            graphics_state: GraphicsState::default(),
            named_coordinate_systems: HashMap::new(),
            render_from_world: Transform::default(),
            pushed_graphics_states: Vec::new(),
            push_stack: Vec::new(),
            shapes: Vec::new(),
            instance_uses: Vec::new(),
            named_material_names: HashSet::new(),
            _medium_names: HashSet::new(),
            float_texture_names: HashSet::new(),
            spectrum_texture_names: HashSet::new(),
            instance_names: HashSet::new(),
            current_material_index: 0,
            sampler,
            film,
            integrator,
            filter,
            accelerator,
            camera,
            active_instance_definition: None,
        };

        let dict = ParameterDictionary::new(
            ParsedParameterVector::new(),
            RgbColorSpace::get_named(crate::colorspace::NamedColorSpace::SRGB).clone(),
        );
        let diffuse = SceneEntity::new("diffuse", FileLoc::default(), dict, string_interner);
        builder.current_material_index = builder.scene.add_material(diffuse);

        builder
    }

    /// Drops self, returning the scene.
    pub fn done(self) -> Box<BasicScene> {
        self.scene
    }

    fn render_from_object(&self) -> Transform {
        // TODO Want to create a version for AnimatedTransform that uses both of ctm.
        self.render_from_world * self.graphics_state.ctm[0]
    }

    fn ctm_is_animated(&self) -> bool {
        self.graphics_state.ctm.is_animated()
    }
}

impl ParserTarget for BasicSceneBuilder {
    fn shape(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        // TODO Verify world
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.shape_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        let area_light_index = if self.graphics_state.area_light_name.is_empty() {
            -1
        } else {
            if self.active_instance_definition.is_some() {
                warn!("{} Area lights not supported with object instancing", loc)
            }
            self.scene.add_area_light(SceneEntity::new(
                &self.graphics_state.area_light_name,
                self.graphics_state.area_light_loc.clone(),
                self.graphics_state.area_light_params.clone(),
                string_interner,
            ))
        };

        if self.ctm_is_animated() {
            todo!(); // Not yet implemented! We don't have animated transforms yet.
        } else {
            // TODO Use transform cache.
            let render_from_object = self.render_from_object();
            let object_from_render = Transform::inverse(&render_from_object);

            let entity = ShapeSceneEntity {
                base: SceneEntity::new(name, loc, dict, string_interner),
                render_from_object: render_from_object,
                object_from_render: object_from_render,
                reverse_orientation: self.graphics_state.reverse_orientation,
                material_index: self.graphics_state.current_material_index,
                material_name: self.graphics_state.current_named_material.clone(),
                light_index: area_light_index,
                _inside_medium: self.graphics_state.current_inside_medium.clone(),
                _outside_medium: self.graphics_state.current_outside_medium.clone(),
            };
            if let Some(active_instance_definition) = &mut self.active_instance_definition {
                active_instance_definition.entity.shapes.push(entity)
            } else {
                self.shapes.push(entity)
            }
        }
    }

    fn option(&mut self, name: &str, value: &str, options: &mut Options, loc: FileLoc) {
        let name = normalize_arg(name);

        match name.as_str() {
            "disablepixeljitter" => match value {
                "true" => options.disable_pixel_jitter = true,
                "false" => options.disable_pixel_jitter = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "disabletexturefiltering" => match value {
                "true" => options.disable_texture_filtering = true,
                "false" => options.disable_texture_filtering = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "disablewavelengthjitter" => match value {
                "true" => options.disable_wavelength_jitter = true,
                "false" => options.disable_wavelength_jitter = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "displacementedgescale" => {
                options.displacement_edge_scale = value
                    .parse()
                    .unwrap_or_else(|_| panic!("{} Unable to parse option value {}", loc, value));
            }
            "msereferenceimage" => {
                if value.len() < 3 || !value.starts_with("\"") || !value.ends_with('\"') {
                    panic!("{} Expected quotes string for option value {}", loc, value);
                }
                options.mse_reference_image = Some(value[1..value.len() - 1].to_owned());
            }
            "msereferenceout" => {
                if value.len() < 3 || !value.starts_with("\"") || !value.ends_with('\"') {
                    panic!("{} Expected quotes string for option value {}", loc, value);
                }
                options.mse_reference_output = Some(value[1..value.len() - 1].to_owned());
            }
            "rendercoordsys" => {
                if value.len() < 3 || !value.starts_with("\"") || !value.ends_with("\"") {
                    panic!("{} Expected quotes string for option value {}", loc, value);
                }
                let render_coord_sys = value[1..value.len() - 1].to_owned();
                match render_coord_sys.as_str() {
                    "camera" => {
                        options.rendering_coord_system =
                            crate::options::RenderingCoordinateSystem::Camera
                    }
                    "cameraworld" => {
                        options.rendering_coord_system =
                            crate::options::RenderingCoordinateSystem::CameraWorld
                    }
                    "world" => {
                        options.rendering_coord_system =
                            crate::options::RenderingCoordinateSystem::World
                    }
                    _ => panic!("{} Unknown option value {}", loc, value),
                }
            }
            "seed" => {
                options.seed = value
                    .parse()
                    .unwrap_or_else(|_| panic!("{} Unable to parse option value {}", loc, value));
            }
            "forcediffuse" => match value {
                "true" => options.force_diffuse = true,
                "false" => options.force_diffuse = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "pixelstats" => match value {
                "true" => options.record_pixel_statistics = true,
                "false" => options.record_pixel_statistics = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            "wavefront" => match value {
                "true" => options.wavefront = true,
                "false" => options.wavefront = false,
                _ => panic!("{} Unknown option value {}", loc, value),
            },
            _ => panic!("{} Unknown option {}", loc, name),
        }
    }

    fn identity(&mut self, _loc: FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| *t = Transform::default());
    }

    fn translate(&mut self, dx: crate::Float, dy: crate::Float, dz: crate::Float, _loc: FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = *t * Transform::translate(Vector3f::new(dx, dy, dz))
            });
    }

    fn scale(&mut self, sx: crate::Float, sy: crate::Float, sz: crate::Float, _loc: FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| *t = *t * Transform::scale(sx, sy, sz));
    }

    fn rotate(
        &mut self,
        angle: crate::Float,
        ax: crate::Float,
        ay: crate::Float,
        az: crate::Float,
        _loc: FileLoc,
    ) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = *t * Transform::rotate(angle, &Vector3f::new(ax, ay, az))
            });
    }

    fn look_at(
        &mut self,
        ex: crate::Float,
        ey: crate::Float,
        ez: crate::Float,
        lx: crate::Float,
        ly: crate::Float,
        lz: crate::Float,
        ux: crate::Float,
        uy: crate::Float,
        uz: crate::Float,
        _loc: FileLoc,
    ) {
        let transform = Transform::look_at(
            &Point3f::new(ex, ey, ez),
            &Point3f::new(lx, ly, lz),
            &Vector3f::new(ux, uy, uz),
        );
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| *t = *t * transform);
    }

    fn transform(&mut self, transform: [crate::Float; 16], _loc: FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = Transform::transpose(&Transform::new_calc_inverse(SquareMatrix::<4>::from(
                    transform.as_slice(),
                )));
            })
    }

    fn concat_transform(&mut self, transform: [crate::Float; 16], _loc: FileLoc) {
        self.graphics_state
            .for_active_transforms(|t: &mut Transform| {
                *t = *t
                    * Transform::transpose(&Transform::new_calc_inverse(SquareMatrix::<4>::from(
                        transform.as_slice(),
                    )));
            })
    }

    fn coordinate_system(&mut self, name: &str, _loc: FileLoc) {
        // TODO Normalize name to UTF-8.
        self.named_coordinate_systems
            .insert(name.to_owned(), self.graphics_state.ctm.clone());
    }

    fn coordinate_sys_transform(&mut self, name: &str, loc: FileLoc) {
        // TODO Normalize name to UTF-8.
        if let Some(ctm) = self.named_coordinate_systems.get(name) {
            self.graphics_state.ctm = ctm.clone();
        } else {
            warn!("{}: Couldn't find named coordinate system {}.", loc, name);
        }
    }

    fn active_transform_all(&mut self, _loc: FileLoc) {
        self.graphics_state.active_transform_bits = Self::ALL_TRANSFORM_BITS;
    }

    fn active_transform_end_time(&mut self, _loc: FileLoc) {
        self.graphics_state.active_transform_bits = Self::END_TRANSFORM_BITS;
    }

    fn active_transform_start_time(&mut self, _loc: FileLoc) {
        self.graphics_state.active_transform_bits = Self::START_TRANSFORM_BITS;
    }

    fn transform_times(&mut self, start: crate::Float, end: crate::Float, _loc: FileLoc) {
        // TODO verify options
        self.graphics_state.transform_start_time = start;
        self.graphics_state.transform_end_time = end;
    }

    fn color_space(&mut self, n: &str, _loc: FileLoc) {
        let cs = RgbColorSpace::get_named(n.into());
        self.graphics_state.color_space = cs.clone();
    }

    fn pixel_filter(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.filter = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn film(
        &mut self,
        film_type: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.film = SceneEntity::new(film_type, loc, dict, string_interner);
    }

    fn accelerator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.accelerator = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn integrator(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.integrator = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn camera(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());

        let camera_from_world = &self.graphics_state.ctm;
        let world_from_camera = TransformSet::inverse(&camera_from_world);
        // TODO Animated transform
        let camera_transform = CameraTransform::new(&world_from_camera[0], options);
        self.named_coordinate_systems
            .insert("camera".to_owned(), world_from_camera);

        self.render_from_world = camera_transform.render_from_world();

        self.camera = CameraSceneEntity {
            base: SceneEntity::new(name, loc, dict, string_interner),
            camera_transform,
        };
    }

    fn make_named_medium(&mut self, _name: &str, _params: ParsedParameterVector, _loc: FileLoc) {
        todo!("Mediums not yet implemented; can't make named medium.")
    }

    fn medium_interface(&mut self, _inside_name: &str, _outside_name: &str, _loc: FileLoc) {
        todo!("Mediums not yet implemented; can't create medium interface.")
    }

    fn sampler(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        let dict = ParameterDictionary::new(params, self.graphics_state.color_space.clone());
        // TODO Verify options
        self.sampler = SceneEntity::new(name, loc, dict, string_interner);
    }

    fn world_begin(
        &mut self,
        string_interner: &mut StringInterner,
        _loc: FileLoc,
        options: &Options,
    ) {
        self.current_block = BlockState::WorldBlock;
        for i in 0..MAX_TRANSFORMS {
            self.graphics_state.ctm[i] = Transform::default();
        }
        self.graphics_state.active_transform_bits = Self::ALL_TRANSFORM_BITS;
        self.named_coordinate_systems
            .insert("world".to_owned(), self.graphics_state.ctm.clone());

        // Pass pre-world-begin entities to the scene
        self.scene.set_options(
            self.filter.clone(),
            self.film.clone(),
            self.camera.clone(),
            self.sampler.clone(),
            self.integrator.clone(),
            self.accelerator.clone(),
            string_interner,
            options,
        );
    }

    fn attribute_begin(&mut self, loc: FileLoc) {
        // TODO Verify world
        self.pushed_graphics_states
            .push(self.graphics_state.clone());
        self.push_stack.push(('a' as u8, loc));
    }

    fn attribute_end(&mut self, loc: FileLoc) {
        // TODO Verify world
        if self.push_stack.is_empty() {
            panic!("{} Unmatched attribute_end statement.", loc);
        }

        // Note: Must keep the following consistent with code in ObjectEnd.
        self.graphics_state = self.pushed_graphics_states.pop().unwrap();

        if self.push_stack.last().unwrap().0 == 'o' as u8 {
            panic!(
                "{} Masmatched nesting: open ObjectBegin from {} at attribute_end.",
                loc,
                self.push_stack.last().unwrap().1
            );
        } else {
            assert!(self.push_stack.last().unwrap().0 == 'a' as u8);
        }
    }

    fn attribute(&mut self, target: &str, mut attrib: ParsedParameterVector, loc: FileLoc) {
        let current_attributes = match target {
            "shape" => &mut self.graphics_state.shape_attributes,
            "light" => &mut self.graphics_state.light_attributes,
            "material" => &mut self.graphics_state.material_attributes,
            "medium" => &mut self.graphics_state.medium_attributes,
            "texture" => &mut self.graphics_state.texture_attributes,
            _ => panic!("{} Unknown attribute target {}", loc, target),
        };

        // We hold onto the curent color space and associate it with the parameters.
        for p in attrib.iter_mut() {
            p.may_be_unused = true;
            p.color_space = Some(self.graphics_state.color_space.clone());
            current_attributes.push(p.to_owned())
        }
    }

    fn texture(
        &mut self,
        name: &str,
        texture_type: &str,
        tex_name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        options: &Options,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        texture_cache: &Arc<Mutex<HashMap<TexInfo, Arc<MIPMap>>>>,
        gamma_encoding_cache: &mut ColorEncodingCache,
    ) {
        // TODO Normalize name to UTF8
        // TODO Verify world

        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.texture_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if texture_type != "float" && texture_type != "spectrum" {
            panic!("{} Texture type \"{}\" unknown.", loc, texture_type);
        }

        let names = if texture_type == "float" {
            &mut self.float_texture_names
        } else {
            &mut self.spectrum_texture_names
        };

        if names.insert(name.to_owned()) {
            match texture_type {
                "float" => {
                    self.scene.add_float_texture(
                        name,
                        TextureSceneEntity::new(
                            tex_name,
                            dict,
                            string_interner,
                            loc,
                            self.render_from_object(),
                        ),
                        &string_interner,
                        options,
                        texture_cache,
                        gamma_encoding_cache,
                    );
                }
                "spectrum" => {
                    self.scene.add_spectrum_texture(
                        name,
                        TextureSceneEntity::new(
                            tex_name,
                            dict,
                            string_interner,
                            loc,
                            self.render_from_object(),
                        ),
                        &string_interner,
                        cached_spectra,
                        options,
                        texture_cache,
                        gamma_encoding_cache,
                    );
                }
                _ => panic!("{} Unknown texture type {}", loc, texture_type),
            }
        } else {
            // TODO defer error instead
            panic!("{} Texture \"{}\" redefined.", loc, name);
        }
    }

    fn material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        // TODO Verify world
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.material_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        self.graphics_state.current_material_index =
            self.scene
                .add_material(SceneEntity::new(name, loc, dict, string_interner));
        self.graphics_state.current_named_material.clear();
    }

    fn make_named_material(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
    ) {
        // TODO Normalize name to UTF8
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.material_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );

        if self.named_material_names.insert(name.to_owned()) {
            self.scene
                .add_named_material(name, SceneEntity::new(name, loc, dict, string_interner));
        } else {
            // TODO defer error instead
            panic!("{} Named material {} redefined.", loc, name);
        }
    }

    fn named_material(&mut self, name: &str, _loc: FileLoc) {
        // TODO Normalize name to UTF8
        // TODO Verify world
        self.graphics_state.current_named_material = name.to_owned();
        self.current_material_index = -1;
    }

    fn light_source(
        &mut self,
        name: &str,
        params: ParsedParameterVector,
        string_interner: &mut StringInterner,
        loc: FileLoc,
        cached_spectra: &mut HashMap<String, Arc<Spectrum>>,
        options: &Options,
    ) {
        let dict = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.light_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );
        self.scene.add_light(
            &mut LightSceneEntity::new(
                name,
                dict,
                string_interner,
                loc,
                self.render_from_object(),
                &self.graphics_state.current_outside_medium,
            ),
            &string_interner,
            cached_spectra,
            options,
        );
    }

    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector, loc: FileLoc) {
        // TODO Verify world
        self.graphics_state.area_light_name = name.to_owned();
        self.graphics_state.area_light_params = ParameterDictionary::new_with_unowned(
            params,
            self.graphics_state.light_attributes.clone(),
            self.graphics_state.color_space.clone(),
        );
        self.graphics_state.area_light_loc = loc;
    }

    fn reverse_orientation(&mut self, _loc: FileLoc) {
        // TODO verify world
        self.graphics_state.reverse_orientation = !self.graphics_state.reverse_orientation;
    }

    fn object_begin(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner) {
        // TODO Verify world
        // TODO Normalize name to UTF8

        self.pushed_graphics_states
            .push(self.graphics_state.clone());

        if self.active_instance_definition.is_some() {
            panic!("{} ObjectBegin called inside of instance definition.", loc);
        }

        let inserted = self.instance_names.insert(name.to_owned());
        if !inserted {
            panic!(
                "{} ObjectBegin trying to redefine object instance {}.",
                loc, name
            );
        }

        self.active_instance_definition =
            Some(ActiveInstanceDefinition::new(name, loc, string_interner));
    }

    fn object_end(&mut self, loc: FileLoc) {
        // TODO Verify world
        if self.active_instance_definition.is_none() {
            panic!("{} ObjectEnd called outside of instance definition.", loc);
        }
        if self
            .active_instance_definition
            .as_ref()
            .unwrap()
            .parent
            .is_some()
        {
            panic!(
                "{} ObjectEnd called inside Import for instance definition.",
                loc
            );
        }

        // Note: Must keep the following consistent with AttributeEnd.
        if self.pushed_graphics_states.last().is_none() {
            panic!("{} Unmatched ObjectEnd statement.", loc);
        }
        self.graphics_state = self.pushed_graphics_states.pop().unwrap();

        if self.push_stack.last().unwrap().0 == 'a' as u8 {
            panic!(
                "{} Mismatched nesting: open AttributeBegin from {} at ObjectEnd.",
                loc,
                self.push_stack.last().unwrap().1
            );
        } else {
            assert!(self.push_stack.last().unwrap().0 == 'o' as u8);
        }
        self.push_stack.pop();

        let active_instance_definition = self.active_instance_definition.take().unwrap();
        active_instance_definition
            .active_imports
            .fetch_sub(1, atomic::Ordering::SeqCst);
        // TODO Technically, the value could change between fetch_sub() and load().
        // It would be better to do this entirely within one atomic operation.

        // Otherwise will be taken care of in MergeImported().
        if active_instance_definition
            .active_imports
            .load(atomic::Ordering::SeqCst)
            == 0
        {
            self.scene
                .add_instance_definition(active_instance_definition.entity);
        }

        self.active_instance_definition = None;
    }

    fn object_instance(&mut self, name: &str, loc: FileLoc, string_interner: &mut StringInterner) {
        // TODO Normalize name to UTF8
        // TODO Verify world

        if self.active_instance_definition.is_some() {
            panic!(
                "{} ObjectInstance called inside of instance definition.",
                loc
            );
        }

        let worlf_from_render = self.render_from_world.inverse();

        if self.ctm_is_animated() {
            todo!()
        }

        // TODO Use transformCache
        let render_from_instance = self.render_from_object() * worlf_from_render;

        let entity = InstanceSceneEntity::new(name, loc, string_interner, render_from_instance);
        self.instance_uses.push(entity);
    }

    fn end_of_files(&mut self) {
        if self.current_block != BlockState::WorldBlock {
            panic!("End of files before WorldBegin.");
        }

        // Ensure there are no pushed graphics states
        while !self.pushed_graphics_states.is_empty() {
            self.pushed_graphics_states.pop();
            panic!("Missing end to AttributeBegin.");
        }

        // TODO If we start deferring error handling rather than panicing, this would be a good spot
        //   to check for any deferred errors and report them and actually error out.
        //   (Deferred error handling would be good because it lets the user see all the issues
        //    so they can fix them before running the scene again.)

        if !self.shapes.is_empty() {
            self.scene.add_shapes(&self.shapes);
        }
        if !self.instance_uses.is_empty() {
            self.scene.add_instance_uses(&self.instance_uses);
        }

        self.scene.done();
    }
}
