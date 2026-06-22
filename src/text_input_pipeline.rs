use crate::TextInputBuffer;
use crate::TextInputGlyph;
use crate::TextInputLayoutInfo;
use crate::TextInputNode;
use crate::TextInputPrompt;
use crate::TextInputPromptLayoutInfo;
use bevy::asset::AssetEvent;
use bevy::asset::AssetId;
use bevy::asset::Assets;
use bevy::ecs::change_detection::DetectChanges;
use bevy::ecs::message::MessageReader;
use bevy::ecs::resource::Resource;
use bevy::ecs::system::Query;
use bevy::ecs::system::Res;
use bevy::ecs::system::ResMut;
use bevy::ecs::world::Ref;
use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::math::Rect;
use bevy::math::UVec2;
use bevy::math::Vec2;
use bevy::platform::collections::HashMap;
use bevy::render::render_resource::Extent3d;
use bevy::render::render_resource::TextureDimension;
use bevy::render::render_resource::TextureFormat;
use bevy::text::Font;
use bevy::text::FontAtlas;
use bevy::text::FontSmoothing;
use bevy::text::GlyphAtlasInfo;
use bevy::text::GlyphCacheKey;
use bevy::text::LineBreak;
use bevy::text::LineHeight;
use bevy::text::TextBounds;
use bevy::text::TextError;
use bevy::text::TextFont;
use bevy::text::{FontSize, FontSource};
use bevy::ui::ComputedNode;
use cosmic_text;
use cosmic_text::Buffer;
use cosmic_text::CacheKey;
use cosmic_text::Edit;
use cosmic_text::Metrics;
use cosmic_text::SwashContent;
use std::sync::Arc;

/// Key identifying the set of [`FontAtlas`]es for a particular rasterized font + size + smoothing.
///
/// In Bevy 0.18 this crate reused bevy's `FontAtlasKey`/`FontAtlasSet`. Bevy 0.19's
/// `FontAtlasKey` is tied to bevy's own parley-based font ids, which this crate (which keeps
/// its own cosmic-text pipeline) does not have. So we key our atlases by the cosmic-text
/// font id + the physical font-size bits + the font smoothing mode instead.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub(crate) struct CosmicFontAtlasKey {
    font_id: cosmic_text::fontdb::ID,
    font_size_bits: u32,
    font_smoothing: FontSmoothing,
}

#[derive(Resource)]
pub struct TextInputPipeline {
    pub(crate) handle_to_font_id_map: HashMap<AssetId<Font>, (cosmic_text::fontdb::ID, Arc<str>)>,
    pub font_system: cosmic_text::FontSystem,
    pub(crate) swash_cache: cosmic_text::SwashCache,
    /// Rasterized glyph atlases, keyed first by bevy font asset (so they can be dropped when the
    /// font asset is removed) and then by the physical font size / smoothing.
    pub(crate) font_atlas_sets:
        HashMap<AssetId<Font>, HashMap<CosmicFontAtlasKey, Vec<FontAtlas>>>,
}

impl Default for TextInputPipeline {
    fn default() -> Self {
        let locale = sys_locale::get_locale().unwrap_or_else(|| String::from("en-US"));
        let db = cosmic_text::fontdb::Database::new();
        Self {
            handle_to_font_id_map: Default::default(),
            font_system: cosmic_text::FontSystem::new_with_locale_and_db(locale, db),
            swash_cache: cosmic_text::SwashCache::new(),
            font_atlas_sets: Default::default(),
        }
    }
}

#[derive(Clone)]
struct FontFaceInfo {
    stretch: cosmic_text::fontdb::Stretch,
    style: cosmic_text::fontdb::Style,
    weight: cosmic_text::fontdb::Weight,
    family_name: Arc<str>,
}

fn load_font_to_fontdb(
    font_id: AssetId<Font>,
    font_system: &mut cosmic_text::FontSystem,
    map_handle_to_font_id: &mut HashMap<AssetId<Font>, (cosmic_text::fontdb::ID, Arc<str>)>,
    fonts: &Assets<Font>,
) -> FontFaceInfo {
    let (face_id, family_name) = map_handle_to_font_id
        .entry(font_id)
        .or_insert_with(|| {
            let font = fonts.get(font_id).expect(
                "Tried getting a font that was not available, probably due to not being loaded yet",
            );
            // In Bevy 0.19 `Font::data` is a `parley::fontique::Blob<u8>` rather than the
            // `Arc<Vec<u8>>` it was in 0.18. Copy the bytes into a fresh `Arc<Vec<u8>>` so
            // cosmic-text's `fontdb` can own them.
            let data: Arc<Vec<u8>> = Arc::new(font.data.data().to_vec());
            let ids = font_system
                .db_mut()
                .load_font_source(cosmic_text::fontdb::Source::Binary(data));

            // TODO: it is assumed this is the right font face
            let face_id = *ids.last().unwrap();
            let face = font_system.db().face(face_id).unwrap();
            let family_name = Arc::from(face.families[0].0.as_str());

            (face_id, family_name)
        });
    let face = font_system.db().face(*face_id).unwrap();

    FontFaceInfo {
        stretch: face.stretch,
        style: face.style,
        weight: face.weight,
        family_name: family_name.clone(),
    }
}

/// Convert a bevy [`Justify`] into a cosmic-text [`Align`].
///
/// In Bevy 0.18 `cosmic_text::Align` implemented `From<Justify>`. In 0.19 that auto-conversion is
/// gone (and `Justify` gained `Start`/`End` variants), so we convert explicitly.
fn justify_to_align(justify: bevy::text::Justify) -> cosmic_text::Align {
    use bevy::text::Justify;
    match justify {
        Justify::Left => cosmic_text::Align::Left,
        Justify::Right => cosmic_text::Align::Right,
        Justify::Center => cosmic_text::Align::Center,
        Justify::Justified => cosmic_text::Align::Justified,
        Justify::Start => cosmic_text::Align::Left,
        Justify::End => cosmic_text::Align::Right,
    }
}

/// Extract the f32 font size from a [`TextFont`], falling back to 12.0 for non-pixel sizes.
fn text_font_size(text_font: &TextFont) -> f32 {
    match text_font.font_size {
        FontSize::Px(v) => v,
        _ => 12.0,
    }
}

/// Extract the bevy [`Font`] asset id from a [`TextFont`], if it uses a handle-based font source.
///
/// This crate only supports handle-based fonts (it loads the font bytes into its own cosmic-text
/// `fontdb`). Other `FontSource` variants (family / generic families) are unsupported.
fn text_font_handle_id(text_font: &TextFont) -> Option<AssetId<Font>> {
    match &text_font.font {
        FontSource::Handle(handle) => Some(handle.id()),
        _ => None,
    }
}

/// Rasterize a cosmic-text glyph via cosmic-text's own swash cache and add it to a bevy
/// [`FontAtlas`], returning the [`GlyphAtlasInfo`] describing where it landed.
///
/// In Bevy 0.18 this crate used bevy's `add_glyph_to_atlas`, which rasterized cosmic-text glyphs
/// through bevy's `TextureAtlasLayout`-based path. Bevy 0.19's `add_glyph_to_atlas` requires a
/// swash `Scaler` built from bevy's own font ids, which don't exist here. Instead we rasterize the
/// glyph directly with cosmic-text's `SwashCache` and feed the resulting image into a bevy
/// [`FontAtlas`] via [`FontAtlas::add_glyph`], preserving the previous rendering behavior.
fn add_cosmic_glyph_to_atlas(
    font_atlases: &mut Vec<FontAtlas>,
    textures: &mut Assets<Image>,
    font_system: &mut cosmic_text::FontSystem,
    swash_cache: &mut cosmic_text::SwashCache,
    cache_key: CacheKey,
    font_smoothing: FontSmoothing,
) -> Result<GlyphAtlasInfo, TextError> {
    let glyph_key = GlyphCacheKey {
        glyph_id: cache_key.glyph_id,
    };

    // Already rasterized? Return the cached info.
    if let Some(info) = get_atlas_info(font_atlases, glyph_key) {
        return Ok(info);
    }

    let image = swash_cache
        .get_image_uncached(font_system, cache_key)
        .ok_or(TextError::FailedToGetGlyphImage(cache_key.glyph_id))?;

    let width = image.placement.width;
    let height = image.placement.height;

    // Convert the swash image into an RGBA bevy `Image`.
    let (rgba, is_alpha_mask) = match image.content {
        SwashContent::Mask => {
            let px = (width * height) as usize;
            let mut rgba = vec![0u8; px * 4];
            match font_smoothing {
                FontSmoothing::AntiAliased => {
                    for i in 0..px {
                        let a = image.data[i];
                        rgba[i * 4] = 255;
                        rgba[i * 4 + 1] = 255;
                        rgba[i * 4 + 2] = 255;
                        rgba[i * 4 + 3] = a;
                    }
                }
                FontSmoothing::None => {
                    for i in 0..px {
                        let a = image.data[i];
                        rgba[i * 4] = 255;
                        rgba[i * 4 + 1] = 255;
                        rgba[i * 4 + 2] = 255;
                        rgba[i * 4 + 3] = if 127 < a { 255 } else { 0 };
                    }
                }
            }
            (rgba, true)
        }
        SwashContent::Color | SwashContent::SubpixelMask => (image.data, false),
    };

    // Guard against zero-sized glyphs (e.g. spaces) which can't be packed into an atlas.
    if width == 0 || height == 0 {
        return Err(TextError::FailedToGetGlyphImage(cache_key.glyph_id));
    }

    let glyph_image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        rgba,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD,
    );

    let offset = Vec2::new(image.placement.left as f32, -image.placement.top as f32);

    let mut try_add = |atlas: &mut FontAtlas| -> Result<(), TextError> {
        atlas.add_glyph(textures, glyph_key, &glyph_image, offset, is_alpha_mask)
    };

    if !font_atlases.iter_mut().any(|atlas| try_add(atlas).is_ok()) {
        // Create a new atlas large enough for this glyph.
        let glyph_max_size = width.max(height);
        let containing = (1u32 << (32 - glyph_max_size.leading_zeros())).max(512);
        let mut new_atlas = FontAtlas::new(textures, UVec2::splat(containing), font_smoothing);
        new_atlas.add_glyph(textures, glyph_key, &glyph_image, offset, is_alpha_mask)?;
        font_atlases.push(new_atlas);
    }

    get_atlas_info(font_atlases, glyph_key).ok_or(TextError::InconsistentAtlasState)
}

/// Look up the [`GlyphAtlasInfo`] for an already-rasterized glyph across a font's atlases.
fn get_atlas_info(font_atlases: &[FontAtlas], glyph_key: GlyphCacheKey) -> Option<GlyphAtlasInfo> {
    font_atlases.iter().find_map(|atlas| {
        atlas.get_glyph_index(glyph_key).map(|location| GlyphAtlasInfo {
            offset: location.offset,
            rect: atlas.texture_atlas.textures[location.glyph_index].as_rect(),
            texture: atlas.texture.id(),
            is_alpha_mask: location.is_alpha_mask,
        })
    })
}

fn buffer_dimensions(buffer: &cosmic_text::Buffer) -> Vec2 {
    let (width, height) = buffer
        .layout_runs()
        .map(|run| (run.line_w, run.line_height))
        .reduce(|(w1, h1), (w2, h2)| (w1.max(w2), h1 + h2))
        .unwrap_or((0.0, 0.0));

    Vec2::new(width, height).ceil()
}

pub fn text_input_system(
    mut textures: ResMut<Assets<Image>>,
    fonts: Res<Assets<Font>>,
    mut text_input_pipeline: ResMut<TextInputPipeline>,
    mut text_query: Query<(
        Ref<ComputedNode>,
        Ref<TextFont>,
        Ref<LineHeight>,
        &mut TextInputLayoutInfo,
        &mut TextInputBuffer,
        Ref<TextInputNode>,
    )>,
) {
    for (node, text_font, line_height, text_input_layout_info, mut editor, input) in
        text_query.iter_mut()
    {
        let layout_info = text_input_layout_info.into_inner();

        // This crate only supports handle-based fonts. Skip other `FontSource` variants.
        let Some(font_id) = text_font_handle_id(&text_font) else {
            continue;
        };
        let text_font_size = text_font_size(&text_font);

        if editor.needs_update
            || text_font.is_changed()
            || line_height.is_changed()
            || node.is_changed()
            || input.is_changed()
        {
            let bounds = TextBounds {
                width: Some(node.size().x),
                height: Some(node.size().y),
            };

            let line_height = match *line_height {
                LineHeight::Px(h) => h,
                LineHeight::RelativeToFont(r) => r * text_font_size,
            };

            let result = editor.editor.with_buffer_mut(|buffer| {
                let TextInputPipeline {
                    font_system,
                    handle_to_font_id_map: map_handle_to_font_id,
                    ..
                } = &mut *text_input_pipeline;
                if !fonts.contains(font_id) {
                    return Err(TextError::NoSuchFont);
                }

                let face_info =
                    load_font_to_fontdb(font_id, font_system, map_handle_to_font_id, &fonts);

                let mut metrics = Metrics::new(text_font_size, line_height)
                    .scale(node.inverse_scale_factor().recip());

                metrics.font_size = metrics.font_size.max(0.000001);
                metrics.line_height = metrics.line_height.max(0.000001);

                buffer.set_metrics_and_size(font_system, metrics, bounds.width, bounds.height);

                buffer.set_wrap(font_system, input.mode.wrap());

                let attrs = cosmic_text::Attrs::new()
                    .metadata(0)
                    .family(cosmic_text::Family::Name(&face_info.family_name))
                    .stretch(face_info.stretch)
                    .style(face_info.style)
                    .weight(face_info.weight)
                    .metrics(metrics);

                let text = crate::get_text(buffer);
                let align = Some(justify_to_align(input.justification));
                buffer.set_text(
                    font_system,
                    &text,
                    &attrs,
                    cosmic_text::Shaping::Advanced,
                    align,
                );

                Ok(())
            });

            if result.is_ok() {
                editor.needs_update = false;
                editor.editor.set_redraw(true);
            } else {
                editor.needs_update = true;
                continue;
            }
        }

        editor
            .editor
            .shape_as_needed(&mut text_input_pipeline.font_system, false);

        let selection = editor.editor.selection_bounds();
        let TextInputBuffer {
            editor,
            selection_rects,
            ..
        } = &mut *editor;

        if editor.redraw() {
            layout_info.glyphs.clear();
            selection_rects.clear();

            let result = editor.with_buffer_mut(|buffer| {
                let box_size = buffer_dimensions(buffer);
                let result = buffer.layout_runs().try_for_each(|run| {
                    if let Some(selection) = selection
                        && let Some((x0, w)) = run.highlight(selection.0, selection.1)
                    {
                        let y0 = run.line_top;
                        let y1 = y0 + run.line_height;
                        let x1 = x0 + w;
                        let r = Rect::new(x0, y0, x1, y1);
                        selection_rects.push(r);
                    }

                    run.glyphs
                        .iter()
                        .map(move |layout_glyph| (layout_glyph, run.line_y, run.line_i))
                        .try_for_each(|(layout_glyph, line_y, line_i)| {
                            let mut temp_glyph;
                            let span_index = layout_glyph.metadata;
                            let font_smoothing = text_font.font_smoothing;

                            let layout_glyph = if font_smoothing == FontSmoothing::None {
                                // If font smoothing is disabled, round the glyph positions and sizes,
                                // effectively discarding all subpixel layout.
                                temp_glyph = layout_glyph.clone();
                                temp_glyph.x = temp_glyph.x.round();
                                temp_glyph.y = temp_glyph.y.round();
                                temp_glyph.w = temp_glyph.w.round();
                                temp_glyph.x_offset = temp_glyph.x_offset.round();
                                temp_glyph.y_offset = temp_glyph.y_offset.round();
                                temp_glyph.line_height_opt =
                                    temp_glyph.line_height_opt.map(f32::round);

                                &temp_glyph
                            } else {
                                layout_glyph
                            };

                            let TextInputPipeline {
                                font_system,
                                swash_cache,
                                font_atlas_sets,
                                ..
                            } = &mut *text_input_pipeline;

                            let font_atlas_set = font_atlas_sets.entry(font_id).or_default();

                            let physical_glyph = layout_glyph.physical((0., 0.), 1.);

                            let font_atlases = font_atlas_set
                                .entry(CosmicFontAtlasKey {
                                    font_id: physical_glyph.cache_key.font_id,
                                    font_size_bits: physical_glyph.cache_key.font_size_bits,
                                    font_smoothing,
                                })
                                .or_default();

                            let atlas_info = add_cosmic_glyph_to_atlas(
                                font_atlases,
                                &mut textures,
                                font_system,
                                swash_cache,
                                physical_glyph.cache_key,
                                font_smoothing,
                            )?;

                            let glyph_size =
                                UVec2::new(atlas_info.rect.width() as u32, atlas_info.rect.height() as u32);
                            // `atlas_info.offset` is `Vec2(placement.left, -placement.top)`.
                            let left = atlas_info.offset.x;
                            let top = -atlas_info.offset.y;

                            // offset by half the size because the origin is center
                            let x = glyph_size.x as f32 / 2.0 + left + physical_glyph.x as f32;
                            let y = line_y.round() + physical_glyph.y as f32 - top
                                + glyph_size.y as f32 / 2.0;

                            let position = Vec2::new(x, y);

                            let pos_glyph = TextInputGlyph {
                                position,
                                size: glyph_size.as_vec2(),
                                atlas_info,
                                span_index,
                                byte_index: layout_glyph.start,
                                byte_length: layout_glyph.end - layout_glyph.start,
                                line_index: line_i,
                            };
                            layout_info.glyphs.push(pos_glyph);
                            Ok(())
                        })
                });

                // Check result.
                result?;

                layout_info.size = box_size;
                Ok(())
            });

            match result {
                Err(TextError::NoSuchFont) => {
                    // There was an error processing the text layout, try again next frame
                }
                Err(
                    e @ (TextError::FailedToAddGlyph(_)
                    | TextError::FailedToGetGlyphImage(_)
                    | TextError::MissingAtlasLayout
                    | TextError::MissingAtlasTexture
                    | TextError::NoSuchFontFamily(_)
                    | TextError::DegenerateScaleFactor
                    | TextError::InconsistentAtlasState),
                ) => {
                    panic!("Fatal error when processing text: {e}.");
                }
                Ok(()) => {
                    layout_info.size.x *= node.inverse_scale_factor();
                    layout_info.size.y *= node.inverse_scale_factor();
                    editor.set_redraw(false);
                }
            }
        }
    }
}

pub fn text_input_prompt_system(
    mut textures: ResMut<Assets<Image>>,
    fonts: Res<Assets<Font>>,
    mut text_input_pipeline: ResMut<TextInputPipeline>,
    mut text_query: Query<(
        Ref<ComputedNode>,
        Ref<TextFont>,
        Ref<LineHeight>,
        &mut TextInputPromptLayoutInfo,
        &mut TextInputBuffer,
        Ref<TextInputNode>,
        Ref<TextInputPrompt>,
    )>,
) {
    for (node, text_font, line_height, text_input_layout_info, mut editor, input, prompt) in
        text_query.iter_mut()
    {
        let layout_info = text_input_layout_info.into_inner();
        if prompt.is_changed()
            || input.is_changed()
            || editor.prompt_buffer.is_none()
            || layout_info.glyphs.is_empty()
            || (text_font.is_changed() || line_height.is_changed()) && prompt.font.is_none()
            || node.is_changed()
        {
            layout_info.glyphs.clear();

            if prompt.text.is_empty() {
                editor.prompt_buffer = None;
                continue;
            }

            let TextInputPipeline {
                font_system,
                handle_to_font_id_map: map_handle_to_font_id,
                ..
            } = &mut *text_input_pipeline;

            let font = prompt.font.as_ref().unwrap_or(text_font.as_ref());

            // This crate only supports handle-based fonts. Skip other `FontSource` variants.
            let Some(font_id) = text_font_handle_id(font) else {
                editor.prompt_buffer = None;
                continue;
            };
            if !fonts.contains(font_id) {
                editor.prompt_buffer = None;
                continue;
            }

            let font_pixel_size = text_font_size(font);

            let line_height = match *line_height {
                LineHeight::Px(h) => h,
                LineHeight::RelativeToFont(r) => r * font_pixel_size,
            };

            let metrics = Metrics::new(font_pixel_size, line_height)
                .scale(node.inverse_scale_factor().recip());

            if metrics.font_size <= 0. || metrics.line_height <= 0. {
                editor.prompt_buffer = None;
                continue;
            }

            let buffer = editor
                .prompt_buffer
                .get_or_insert(Buffer::new(font_system, metrics));

            let linebreak = LineBreak::WordBoundary;
            let bounds = TextBounds {
                width: Some(node.size().x),
                height: Some(node.size().y),
            };

            let face_info =
                load_font_to_fontdb(font_id, font_system, map_handle_to_font_id, &fonts);

            buffer.set_size(font_system, bounds.width, bounds.height);

            buffer.set_wrap(
                font_system,
                match linebreak {
                    LineBreak::WordBoundary => cosmic_text::Wrap::Word,
                    LineBreak::AnyCharacter => cosmic_text::Wrap::Glyph,
                    LineBreak::WordOrCharacter => cosmic_text::Wrap::WordOrGlyph,
                    LineBreak::NoWrap => cosmic_text::Wrap::None,
                },
            );

            let attrs = cosmic_text::Attrs::new()
                .metadata(0)
                .family(cosmic_text::Family::Name(&face_info.family_name))
                .stretch(face_info.stretch)
                .style(face_info.style)
                .weight(face_info.weight)
                .metrics(metrics);

            let align = Some(justify_to_align(input.justification));
            buffer.set_text(
                font_system,
                &prompt.text,
                &attrs,
                cosmic_text::Shaping::Advanced,
                align,
            );

            buffer.shape_until_scroll(font_system, false);

            let box_size = buffer_dimensions(buffer);
            let result = buffer.layout_runs().try_for_each(|run| {
                run.glyphs
                    .iter()
                    .map(move |layout_glyph| (layout_glyph, run.line_y, run.line_i))
                    .try_for_each(|(layout_glyph, line_y, line_i)| {
                        let mut temp_glyph;
                        let span_index = layout_glyph.metadata;
                        let font_smoothing = text_font.font_smoothing;

                        let layout_glyph = if font_smoothing == FontSmoothing::None {
                            // If font smoothing is disabled, round the glyph positions and sizes,
                            // effectively discarding all subpixel layout.
                            temp_glyph = layout_glyph.clone();
                            temp_glyph.x = temp_glyph.x.round();
                            temp_glyph.y = temp_glyph.y.round();
                            temp_glyph.w = temp_glyph.w.round();
                            temp_glyph.x_offset = temp_glyph.x_offset.round();
                            temp_glyph.y_offset = temp_glyph.y_offset.round();
                            temp_glyph.line_height_opt = temp_glyph.line_height_opt.map(f32::round);

                            &temp_glyph
                        } else {
                            layout_glyph
                        };

                        let TextInputPipeline {
                            font_system,
                            swash_cache,
                            font_atlas_sets,
                            ..
                        } = &mut *text_input_pipeline;

                        let font_atlas_set = font_atlas_sets.entry(font_id).or_default();

                        let physical_glyph = layout_glyph.physical((0., 0.), 1.);

                        let font_atlases = font_atlas_set
                            .entry(CosmicFontAtlasKey {
                                font_id: physical_glyph.cache_key.font_id,
                                font_size_bits: physical_glyph.cache_key.font_size_bits,
                                font_smoothing,
                            })
                            .or_default();

                        let atlas_info = add_cosmic_glyph_to_atlas(
                            font_atlases,
                            &mut textures,
                            font_system,
                            swash_cache,
                            physical_glyph.cache_key,
                            font_smoothing,
                        )?;

                        let glyph_size =
                            UVec2::new(atlas_info.rect.width() as u32, atlas_info.rect.height() as u32);
                        // `atlas_info.offset` is `Vec2(placement.left, -placement.top)`.
                        let left = atlas_info.offset.x;
                        let top = -atlas_info.offset.y;

                        // offset by half the size because the origin is center
                        let x = glyph_size.x as f32 / 2.0 + left + physical_glyph.x as f32;
                        let y = line_y.round() + physical_glyph.y as f32 - top
                            + glyph_size.y as f32 / 2.0;

                        let position = Vec2::new(x, y);

                        let pos_glyph = TextInputGlyph {
                            position,
                            size: glyph_size.as_vec2(),
                            atlas_info,
                            span_index,
                            byte_index: layout_glyph.start,
                            byte_length: layout_glyph.end - layout_glyph.start,
                            line_index: line_i,
                        };
                        layout_info.glyphs.push(pos_glyph);
                        Ok(())
                    })
            });

            layout_info.size = box_size;

            match result {
                Err(TextError::NoSuchFont) => {
                    editor.prompt_buffer = None;
                    // There was an error processing the text layout, try again next frame
                }
                Err(
                    e @ (TextError::FailedToAddGlyph(_)
                    | TextError::FailedToGetGlyphImage(_)
                    | TextError::MissingAtlasLayout
                    | TextError::MissingAtlasTexture
                    | TextError::NoSuchFontFamily(_)
                    | TextError::DegenerateScaleFactor
                    | TextError::InconsistentAtlasState),
                ) => {
                    panic!("Fatal error when processing text: {e}.");
                }
                Ok(()) => {
                    layout_info.size.x *= node.inverse_scale_factor();
                    layout_info.size.y *= node.inverse_scale_factor();
                }
            }
        }
    }
}

pub fn remove_dropped_font_atlas_sets_from_text_input_pipeline(
    mut text_input_pipeline: ResMut<TextInputPipeline>,
    mut font_events: MessageReader<AssetEvent<Font>>,
) {
    for event in font_events.read() {
        if let AssetEvent::Removed { id } = event {
            text_input_pipeline.font_atlas_sets.remove(id);
        }
    }
}
