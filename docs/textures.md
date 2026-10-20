# Textures

Every texture in the level comes from one of three places, and each is listed here:

1. **Stock game textures**, referenced by their path in BeamNG.drive's own content. They are not copied
   into the mod.
2. **CC0 downloads** from [ambientCG](https://ambientcg.com/) and [Poly Haven](https://polyhaven.com/),
   registered in `src/brmap/data/textures.yaml` with the URL and the sha256 of every downloaded file.
3. **Generated textures**, drawn by scripts in this repo.

Materials name a texture as `stock:<path>` or `tex:<tex_id>`
(`src/brmap/data/landmark_materials.yaml`, `tree_materials.yaml`). The `textures` stage
(`python -B -m brmap textures`) writes the CC0 and generated textures as DDS (BC1 colour, BC5 normals,
at most `textures.max_size` pixels on the longer side, full mip chain) and `textures/index.json`;
`landmarks`, `styled` and `trees` copy the ones their materials use into `art/textures/br/`. Only the
`.dds` files ship, and the materials reference them by their `.png` names
(`/levels/<level>/art/textures/br/clock_face_b.color.png` for `clock_face_b.color.dds`), as the stock
materials do: BeamNG.drive 0.40's `NOTE_FOR_MODDERS.md` says to ship the DDS files where the PNGs were
"but keep referencing the `.png` in `materials.json`", and the game loads the `.dds` it finds under that name.

## CC0 downloads

Both sites publish these under the CC0 1.0 Universal public-domain dedication
(<https://creativecommons.org/publicdomain/zero/1.0/>; see <https://docs.ambientcg.com/license/> and
<https://polyhaven.com/license>), so no attribution is required. The level credits them anyway
(`level.CREDITS`: "Textures: ambientCG and Poly Haven (CC0).").

All six are the 1K JPG downloads. The `textures` stage fetches and converts only the sets a material
binds (`textures.bound_textures`: a `tex:<tex_id>` map in `landmark_materials.yaml` or `tree_materials.yaml`;
today only `turf`) and logs the others as unbound; their entries and rows stay, not fetched while unbound.
`fetch` keeps the files in `data/raw/textures/<tex_id>/` (shared by every config), compares each file with
the sha256 recorded in the registry, and refuses a file that differs.
Normal maps are the OpenGL variant (`NormalGL`, `nor_gl`: green = +V up), which the in-game spike S1
(2026-10-19) found the game reads right.

| tex_id | Source and asset | Download | Maps | Licence | Bound to |
|---|---|---|---|---|---|
| `roof_barrel` | ambientCG [RoofingTiles012A](https://ambientcg.com/view?id=RoofingTiles012A) | `https://ambientcg.com/get?file=RoofingTiles012A_1K-JPG.zip` | Color, NormalGL, Roughness | CC0-1.0 | not bound, not fetched while unbound: kept in the registry for later use (the roof tiles went back to the stock Italian ones) |
| `limestone` | ambientCG [Tiles143](https://ambientcg.com/view?id=Tiles143) | `https://ambientcg.com/get?file=Tiles143_1K-JPG.zip` | Color, NormalGL, Roughness | CC0-1.0 | not bound, not fetched while unbound: the State Capitol's limestone is the stock marble slabs, tinted (below) |
| `turf` | ambientCG [Grass005](https://ambientcg.com/view?id=Grass005) | `https://ambientcg.com/get?file=Grass005_1K-JPG.zip` | Color, NormalGL, Roughness | CC0-1.0 | `br_lm_turf` (Tiger Stadium's field, Alex Box Stadium's field and infield grass) |
| `bark_cypress` | Poly Haven [metasequoia_bark](https://polyhaven.com/a/metasequoia_bark) | `https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/metasequoia_bark/metasequoia_bark_<map>_1k.jpg` | diff, nor_gl, rough | CC0-1.0 | not bound, not fetched while unbound: the cypress keeps the stock fir bark |
| `bark_tupelo` | ambientCG [Bark004](https://ambientcg.com/view?id=Bark004) | `https://ambientcg.com/get?file=Bark004_1K-JPG.zip` | Color, NormalGL, Roughness | CC0-1.0 | not bound, not fetched while unbound: the water tupelo binds the stock poplar bark (below) |
| `needles` | ambientCG [LeafSet019](https://ambientcg.com/view?id=LeafSet019) | `https://ambientcg.com/get?file=LeafSet019_1K-JPG.zip` | Color, NormalGL, Roughness, Opacity | CC0-1.0 | not bound, not fetched while unbound: the cypress keeps the stock sprays |

Poly Haven answers 403 to a request without a User-Agent header; `brmap.fetch.download` always sends one.
The ambientCG zips are untrusted downloads: `fetch` reads only the registry's images out of them (never
extracting a member by its own name), refuses a zip with an absolute or `..` member name, and deletes the
zip afterwards.

### Why the roofs, the cypress and the Capitol stayed on the stock textures

`docs/screenshots/v02/textures_cc0_blender.jpg` compares the stock and CC0 sets in the Blender Workbench
previews (`blender/tools/preview.py`).

- **Roof tiles: switched, then back to stock.** RoofingTiles012A reads as the tower's clean terracotta red in
  the preview, where the stock Italian roof tiles read mottled brown. But the texture covers about 2.9 m, so
  at the models' 1 m-per-repeat roof UVs each barrel tile drew about 7.7 cm wide. `br_lm_roof_tile` went back
  to the stock tiles, which looked right in game on the Memorial Tower; `roof_barrel` stays in
  `textures.yaml` for later use (not fetched while unbound).
- **Cypress bark: kept stock.** metasequoia_bark is a flat chocolate brown, and the trunk's UVs are laid out
  for the stock fir bark's tall 512 x 2048 texture, not a square one.
- **Capitol limestone: kept stock.** Tiles143 would tile right on the Capitol (six 0.6 m blocks in the model's
  3.6 m repeat), but it is a saturated peach travertine (mean sRGB 217, 180, 141) that reads brown on the tower
  (`docs/screenshots/v02/capitol_stone_blender.jpg`); a `baseColorFactor` can only darken a channel, so it can't
  be pulled to the building's pale buff. `br_lm_limestone` is the stock marble slabs tinted (0.93, 0.89, 0.79).
- **Tupelo bark: stock poplar, not Bark004.** Bark004 is a pine's bark: scaly plates with pink-orange
  inner bark, about 15 cm each if the square covered 1 m (ambientCG gives no size for it, the roof_barrel
  lesson). Water tupelo bark is grey-brown with long narrow ridges, which the stock `t_poplar_bark`
  (512 x 2048, ridges about 6 cm wide at the trunk's 1 m round, 4 m up per repeat, as the cypress's fir
  bark) shows. `br_tree_tupelo_bark` binds it, tinted a little warmer.
- **Cypress needles: kept stock.** The crown cards' UVs are cut for the stock spray atlas; on LeafSet019
  they show cut-off fronds with straight edges.

## Generated textures

Drawn by Pillow and numpy code in this repo; they carry no third-party rights. Each is a `GENERATED` entry
in `src/brmap/textures.py` and goes into `index.json` with `"source": "generated"`.

| tex_id | Generating function | Maps | Licence | Bound to |
|---|---|---|---|---|
| `clock_face` | `texgen.clock_face()` | colour | original, generated by brmap | `br_lm_clock` (the Memorial Tower's four dials) |
| `bevel` | `texgen.bevel_normal()` | normal (OpenGL) | original, generated by brmap | `br_lm_trim_stone` |
| `lsu_wall` | `texgen.lsu_wall()` | colour, normal (OpenGL), roughness | original, generated by brmap | `br_sty_lsu_wall` (the LSU campus core's walls: one 3 m bay of one 3.6 m floor, tan stucco with an arched window) |
| `field_atlas` | `logos.field_atlas()` | colour with alpha | original, generated by brmap | `br_lm_field_marks` (Tiger Stadium's end zones, numerals, arrows and midfield mark) |
| `board` | `logos.board()` | colour | original, generated by brmap | `br_lm_board` (Tiger Stadium's scoreboard) |
| `facade_ribbon_dark` | `texgen.facade("ribbon_dark")` | colour, roughness | original, generated by brmap | `br_sty_ribbon_dark` (One American Place: dark glass ribbons between bronze spandrels; 4 floors x 10 bays of 1.5 m per repeat; metallic 0.3) |
| `facade_precast_punched` | `texgen.facade("precast_punched")` | colour, roughness | original, generated by brmap | `br_sty_precast_punched` (the towers at 451 Florida and on North Laurel: light precast panels, punched windows; 4 floors x 8 bays of 2 m) |
| `facade_brick_hotel` | `texgen.facade("brick_hotel")` | colour, roughness | original, generated by brmap | `br_sty_brick_hotel` (the Heidelberg Hotel: brick, double-hung windows with stone sills and lintels; 4 floors x 6 bays of 2.4 m) |
| `facade_precast_arena` | `texgen.facade("precast_arena")` | colour, roughness | original, generated by brmap | `br_sty_precast_arena` (the River Center arena: fluted beige precast panels; 4 floors x 5 bays of 3 m) |
| `facade_glass_blue` | `texgen.facade("glass_blue")` | colour, roughness | original, generated by brmap | `br_sty_glass_blue` (the River Center library: a blue-green curtain wall; 4 floors x 10 bays of 1.5 m; metallic 0.5) |
| `gothic_window` | `texgen.gothic_window()` | colour with alpha | original, generated by brmap | `br_lm_gothic_window` (the Old State Capitol's pointed-arch windows, decal quads cut out by the alpha) |
| `moss` | `texgen.moss_card()` | colour, opacity, normal (OpenGL); 512 x 1024 | original, generated by brmap | `br_tree_moss` (Spanish moss: grey-green filaments (#6F7865 .. #A2A894) hanging from the card's top edge, drawn at `baseColorFactor` (0.65, 0.65, 0.5, 1), since the card alone read near white-blue under the 0.40 sky; the strands of `br_cypress_b` and of the `br_moss_*` overlays on the live oaks, each a 1.1 m wide slice of the card over its whole height) |

## Stock textures

Source: BeamNG.drive's own content. Licence: the game's, as BeamNG's shared assets; the level references them
by path from the installed game and never copies them into the mod, as the stock levels do.

Paths are in the game's `content/assets` (inside `tileable.zip`, `tree.zip` and so on; BeamNG.drive 0.39.4,
checked again on 0.40.0.0), unless a section says otherwise.
The `.png` names resolve to the `.dds` files the game ships. `tests/test_game_assets.py` checks that the
landmark, tree, dressing, water and licence plate paths exist in an installed game.

### Landmarks (`landmark_materials.yaml`)

| Material | Texture folder (under `/assets/materials/tileable/`) | Files |
|---|---|---|
| `br_lm_stucco_tan` | `plaster/m_stucco_white/` (the stock garage's `t_stucco_white`) | `t_stucco_white_b.color.png`, `_nm.normal.png`, `_r.data.png` |
| `br_lm_roof_tile` | `roofing/t_italy_bld_roof_tiles/` (the Italian roof tiles; the Memorial Tower's and the habitat's roofs, the LSU campus core's styled roofs) | `t_italy_bld_roof_tiles_b.color.png`, `_nm.normal.png`, `_r.data.png` (no ao map) |
| `br_lm_trim_stone` | `marble/slabs_marble/` (`t_slabs_marble`; also the styled buildings' cornices) | `t_slabs_marble_b.color.png`, `_r.data.png`; its normal map is the generated `bevel` |
| `br_lm_concrete_tan` | `concrete/concrete_plain/` (`t_concrete_plain`, as the buildings use it), tinted tan | `t_concrete_plain_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png` |
| `br_lm_concrete` | `concrete/concrete_plain/` (`t_concrete_plain`), untinted (PMAC, Alex Box Stadium, the habitat's curb) | `t_concrete_plain_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png` |
| `br_lm_rock` | `stone/rockwall/` (`t_rock_wall`; Mike's habitat) | `t_rock_wall_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png` |
| `br_lm_limestone` | `marble/slabs_marble/` (`t_slabs_marble`), tinted buff (the State Capitol) | `t_slabs_marble_b.color.png`, `_nm.normal.png`, `_r.data.png` |
| `br_lm_stucco_white` | `plaster/stucco1_white/` (`t_stucco1_white`; the Old State Capitol) | `t_stucco1_white_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png` |
| `br_lm_truss_steel` | `metal/metal_paint/` (`t_metal_paint`, an off-white paint), tinted grey-green (the Horace Wilkinson Bridge's truss) | `t_metal_paint_b.color.png`, `_nm.normal.png`, `_r.data.png` |

`br_lm_clay` (Alex Box Stadium's infield) is the terrain's crumbly dirt, tinted orange:
`/assets/materials/terrain/soil/dirt_crumbly/t_dirt_crumbly_{b.color,nm.normal,r.data,ao.data}.png` (in `terrain.zip`).

`br_lm_seats`, `br_lm_field_paint`, `br_lm_steel` and `br_lm_lamp` (Tiger Stadium; the steel also the Old State
Capitol's roofs and the downtown towers' copings) have no texture: a colour factor, and on the seats the purple and
gold vertex colours. Nor do `br_lm_paint` (the ballpark's wall, poles and rails, the Old State Capitol's glazed
rotunda roof: vertex colours), `br_lm_water` (the habitat's pool and stream) and `br_lm_glass` (its viewing glass,
translucent).

### Custom trees (`tree_materials.yaml`)

| Material | Texture folder (under `/assets/materials/tree/`) | Files |
|---|---|---|
| `br_tree_cypress_bark` | `fir/t_fir_bark/` (the Douglas fir's `t_fir_bark`) | `t_fir_bark_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png` |
| `br_tree_cypress_leaves` | `cypress/t_cypress_leaves/` | `t_cypress_leaves_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_o.data.png`, `_ao.data.png` |
| `br_tree_tupelo_bark` | `poplar/t_poplar_bark/` (the stock `m_poplar_bark`'s maps) | `t_poplar_bark_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png` |
| `br_tree_tupelo_leaves` | `beech/t_beech_leaves/` (the stock `beech_leaves`' sprays, tinted deep green) | `t_beech_leaves_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_o.data.png`, `_ao.data.png` |

`br_tree_moss` has no stock texture: its colour, opacity and normal maps are the generated `moss`, the
colour drawn at `baseColorFactor` (0.65, 0.65, 0.5, 1) (the in-game A/B on 0.40: 0.75 too pale, 0.4 dark blocks).

### Buildings (`buildings.py`, `TEXTURE_SETS`)

All under `/assets/materials/tileable/<folder>/<stem>_<suffix>.png`:

| Materials | Folder | Stem |
|---|---|---|
| `br_bld_wall_res` | `wood/weatherboards` | `t_weatherboards` |
| `br_bld_wall_brick` | `brick/brick_plain` | `t_bricks_red_01` |
| `br_bld_wall_concrete`, `br_bld_roof_flat` | `concrete/concrete_plain` | `t_concrete_plain` |
| `br_bld_wall_glass` | `glass/windows1` | `t_bld_ind_windows` |
| `br_bld_roof_shingle` | `roofing/asphalt_roof` | `t_asphalt_roof` |

### Deck dressing (`dressing.py`)

The barriers and piers bind the game's own global material `concrete_plain` (defined in
`art/shapes/objects/main.materials.json` in `content/art_shapes.zip`, loaded for every level), so the
level writes no material for them. It uses `/assets/materials/tileable/concrete/concrete_plain/`
(`t_concrete_plain_b.color.png`, `_nm.normal.png`, `_r.data.png`, `_ao.data.png`), as the buildings do.

### Water (`bayous.py`, the stock muddy River of the italy level)

- `/assets/materials/tileable/water/depthcolor_ramp/depthcolor_ramp_italy_muddy_b.png` (depth colour ramp)
- `/assets/materials/tileable/water/water_effects/foam2_b.color.dds` (foam)
- `/assets/materials/tileable/water/water_effects/ripple3_nm.normal.dds` (ripples)
- `/core/art/water/mud_depthcolor_ramp_b.png` (the bayous' mud ramp, `RIVER_ONLY`)

The Mississippi's and the lakes' water blocks (`beamng_format.WATER_STYLE`, v0.1) use
`/core/art/water/depthcolor_ramp.png`, `foam.dds` and `ripple.dds` (in `gameengine.zip`, as `.link` files to
`depthcolor_ramp_b.png`, `foam_b.color.dds` and `ripple_nm.normal.dds`).

### Licence plate (`beamng_format.plate_design`)

The level's plate design (`vehicles/common/licenseplates/<level>/licensePlate-default.sktemplate.json` in
the zip) draws its text on the stock default design's plain white background (a colour, no texture) and
names one stock texture, the plate's embossed rim: `vehicles/common/licenseplates/default/licenseplate-default_n.png`
(in `content/vehicles/common.zip`). The default folder's `licenseplate-default_d.png` is not used: it is a
state's plate (Utah's).

### Roads, terrain and the stock trees (v0.1, unchanged)

- Roads (`beamng_format.ROAD_MATERIAL_DEFS`): the `/assets/materials/decalroad/` folders
  `asphalt/m_asphalt_02/` (arterials), `asphalt/asphalt_decayed/` (streets), `concrete/decalroad_concrete/`
  (interstates), `asphalt/asphalt_mesh_road/` (the deck tops) and `concrete/street_concrete1/` (the deck
  sides and undersides).
- Paint (`markings.py`): `/assets/materials/decalroad/lines/` (`line_white`, `line_white_discontinue`,
  `line_yellow_double`, `line_yellow_double_discontinue`, `m_marking_zebra`).
- Terrain (`beamng_format.terrain_materials`): `/assets/materials/terrain/grass/ind_grass/` and
  `/assets/materials/terrain/soil/macro_clumply/`, and the template level's `/levels/template/art/terrains/`
  (`t_dirt_crumbly`, `t_asphalt_02`, `t_mud`, `t_ind_grass` and the `t_macro_*` maps).
- Stock trees (`stock_trees.json`, copied from the game's own levels): the `/assets/materials/tree/` folders
  `cork_oak`, `eucalyptus`, `fir`, `holm_oak`, `oak`, `palm`, `scots_pine`, `t_pine` and `tropical`.

### The level's own terrain maps (`level.py`)

- `art/terrains/br_basecolor.png`: the NAIP aerial photo mosaic the `imagery` stage makes. Source: USDA NAIP;
  licence: public domain (a US government work).
- `art/terrains/br_base_nm.png`, `br_base_r.png`, `br_base_ao.png`, `br_base_h.png`: flat single-colour maps
  (`beamng_format.BASE_MAPS`). Licence: original, generated by brmap.

## Adding a texture

1. CC0: add an entry to `src/brmap/data/textures.yaml` (source, asset, maps, `license: CC0-1.0`) and, once a
   material references it (step 2; the stage fetches only bound sets), run
   `python -B -m brmap textures --config config.lsu.yaml`. The log prints the sha256 of each new file; record
   them under `sha256`. Generated: add a `GENERATED` entry in `src/brmap/textures.py`.
2. Reference it from a material as `tex:<tex_id>`.
3. If a material cuts it out with an alpha test (`alphaTest: true`), declare that material's `alphaRef` as the
   texture's `alpha_ref` (`textures.ALPHA_REF` for a generated one, an `alpha_ref:` line in `textures.yaml` for a
   CC0 set). See "Alpha-tested textures" below.
4. Add a row to this file.

## Alpha-tested textures

A material with `alphaTest` and an `alphaRef` shows a texel only where the opacity is at or above the
reference. The mip chain is what the game samples from a distance, and plain halvings average thin
strands into grey: the moss card's 1-3 px filaments crossed `alphaRef` 67 over whole texels and drew as a
solid pale rectangle (the full-map check on 0.40, `docs/screenshots/v02/full_moss_street.jpg`).

So a texture that declares an `alpha_ref` (1-255) is written with coverage-preserving mips
(`dds.write_dds(..., alpha_coverage_ref=alpha_ref / 255)`): after each mip level is made, its opacity is
scaled by the factor, found by bisection and clamped to 0..4, that puts the same share of its texels at or
above the reference as the top level has, measured on the blocks as they decode. The opacity is the `o` map
and the alpha of a `b` map that has one. `test_alpha_tested_textures_declare_the_alphaRef_of_their_materials`
keeps the declared values equal to the materials' `alphaRef`.

`docs/screenshots/v02/moss_mips.png` shows levels 0, 2, 4 and 6 of the moss opacity through the alpha test
(white where it passes `alphaRef` 67): plain halvings above (42 % of the texels pass at the top, 53 %, 60 % and
59 % lower down), coverage-preserving mips below (42 % at every level).

| Texture | `alpha_ref` | Material |
|---|---|---|
| `moss` | 67 | `br_tree_moss` |
| `field_atlas` | 127 | `br_lm_field_marks` |
| `gothic_window` | 127 | `br_lm_gothic_window` |

`needles` (the CC0 leaf set, which has an Opacity map) is not bound to any material, so it declares none.
