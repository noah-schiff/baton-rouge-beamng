# Known issues and limits

Version 0.2 was tested in BeamNG.drive 0.40.0 (Direct3D 12) on a laptop with integrated
Radeon 680M graphics and 15 GB of RAM, using the in-game checks in `tools/ingame/`:
camera views of the landmarks, the downtown towers, the Horace Wilkinson Bridge's truss
and the deck dressing, the swamps, the trees on the LSU Quad and the paint on Government
Street, and a traffic check, in which 6 of 6 AI cars drove more than 100 m in a minute.
Version 0.1 was also driven in 0.39.4 by a scripted car over I-10 both ways (the
I-10/I-110 split flyover, the Mississippi River bridge crest and both approaches, 4.2 km
each), up the I-110 from the split, and along two streets by the LSU lakes. On the decks
the car stayed within 0.2 m of its usual height above the deck the whole way. The game
logs no errors from the level, and one renderer warning (see
[Performance](#performance)). Not yet tested on GeForce NOW.

## What the map does not have (yet)

* **No traffic lights, stop signs or other signs, street lights, sidewalks or
  parking-lot paint.** The game's traffic still drives without them. Stop bars and
  crosswalks are painted where OpenStreetMap maps signals, stop signs and marked
  crossings.
* **Nothing beyond the 16 x 16 km box.** The terrain stops at its edge.
* **No Huey P. Long Bridge.** The old US-190 Mississippi bridge lies outside the box, 181 m
  or more north of its north edge (its decks run at y 8,373 to 8,532 m, the edge is at
  8,192 m), so it is left out.

## Looks

* **The aerial photo is 4 m per pixel** (4096 pixels across 16 km, the largest terrain
  base texture the stock maps use), so it is soft up close. The near ground uses the
  stock grass, dirt, asphalt and mud detail textures chosen from NLCD land cover.
* **Aerial tiles were flown on different days,** so a few tile edges show as straight
  color changes, most visibly on the Mississippi south of the I-10 bridge.
* **Roads are flat decals** with three looks (concrete interstates, asphalt arterials,
  worn asphalt streets). Overlapping roads at intersections can flicker where two decals
  meet.
* **Ordinary buildings are plain extruded shapes** with tiled wall textures (house
  siding, brick, concrete or glass): no modeled windows, doors or roof details. Small
  houses have hipped roofs; all other roofs are flat. Only the landmarks, the LSU campus
  core and six downtown standouts are modelled or styled.
* **The lidar and the aerial photo are a few years apart.** The lidar is from 2017 to
  2019 and the aerial photo from 2023, so a few new or demolished buildings disagree
  with the photo under them.
* **Microsoft footprints are traced from aerial imagery by machine learning.** They
  occasionally merge neighboring buildings into one or miss small ones.
* **Thin tops are clipped.** The 2 m lidar misses the thin top of a tower, so an
  ordinary tower with a spire or a slim top comes out short. The State Capitol (137 m)
  and the Memorial Tower (53 m) are models at their published heights.
* **Landmarks keep clear of the lanes beside them.** A landmark's footprint is cut back to
  stay 0.5 m from the lanes of the roads next to it, so a wall can stand a little back
  from the real building: Tiger Stadium's east wall steps in by up to 1.9 m along
  Powerhouse Lane.
* **Most trees are stock stand-ins.** BeamNG's own tree models from its other maps stand
  in for the live oaks, water oaks, pines and palms (stock oaks for the live and water
  oaks, a pine for loblolly). Only the bald cypress, the water tupelo and the Spanish moss
  are modelled for this map.
* **Spanish moss is drawn as cards of hanging strands.** Its texture keeps the strands'
  share of the card at every mip level and the material draws it grey-green, so the moss
  no longer turns into pale cards at a distance. The card is densest near its top: far off,
  where a card is only a few dozen pixels tall, the moss fades into the crown as a faint
  grey-green band rather than strands.
* **Edge lines at skewed junctions** can run a few meters into a side street's mouth.

## Driving

* **Bridge decks are slabs with concrete Jersey barriers and piers,** but no expansion
  joints. Barriers stand on all 252 decks and run along their edges, but for the two 6 m
  ramp stubs (w40007900 and w112804149), which get only a 2.5 m and a 3 m piece at one
  edge. They run on across the joint where one OpenStreetMap deck piece carries on from
  the next at its level, within 30 degrees of straight on: there the two pieces'
  barriers meet end to end, and a wider piece's barriers taper in to the narrower one's
  line, 5 m along for every metre across. Where a
  deck forks in two at such a joint, or two pieces merge into one, each outer edge runs
  on onto the piece whose edge is outermost on that side (at the I-10 fork at the
  Mississippi River bridge's west end the outer barrier runs on onto the ramp's). Gaps
  remain only where a deck meets the ground (the barriers stop 2 m short of its end),
  where another road's pavement meets the deck at its level (a ramp merging, leaving or
  forking off, from where the two pavements meet to where they part), and where two
  decks meet at a sharper angle. Measured between the nearest barrier ends on each edge,
  at the joints between deck pieces on the full map the outer barriers meet, or hand over
  between a narrow ramp and the deck beside it within 1.05 m (where an on-ramp joins
  eastbound I-10 at the bridge's west end); at a fork the two branches' barriers facing
  each other start up to 10 m apart, where the branches part (the off-ramp leaving the
  bridge's east end). Openings where a ramp or road meets a deck's side, from where the
  two pavements meet to where they part, are not counted in these figures. Where
  OpenStreetMap's two carriageways of the Mississippi River bridge touch there is no
  median barrier. Piers stand every 30 m along a deck that is 2.5 m or more
  over the ground or the river bed, off the roads underneath (116 of the 252 decks have
  any); two parallel decks get a pier each, side by side. The Mississippi crossing (the
  Horace Wilkinson Bridge) has its own cantilever truss and river piers, and no other
  pier comes within 2 m of the
  truss; the 190 m of deck between its east end and the east bank stands on ordinary
  piers in the river. Deck heights are estimated from the lidar and
  OpenStreetMap: pieces that meet are joined into one deck (the Mississippi River
  bridge with its I-10 approaches, the I-10/I-110 split), a ramp starts at the height
  of the deck it leaves and stays level with it until the two part, and decks clear
  the roads and lower decks under them by 5.5 m,
  climbing at 5 % where they can and at most 8 % from their ends. The Horace Wilkinson
  (I-10) bridge crests at about 61 m over the river (`bridges.overrides` in
  `config.yaml`).
* **On narrow decks the barriers overhang the slab's edge.** On decks narrower than 7 m
  (the one-lane ramps and the narrow two-lane street bridges) the barriers stand just
  outside the carriageway, their bases reaching only 5 cm onto it, so a one-lane 3.5 m
  ramp keeps 3.4 m between them. The other 0.55 m of each base hangs past the edge of the
  slab, and seen from below that overhang is open: the barriers have no bottom face.
* **A few decks fall short.** The build log lists 18 decks with a need "out of reach",
  mostly short overpasses (30 to 80 m) that cannot reach full clearance within 8 % of
  their ends, so a tall vehicle on the road underneath may touch the deck. Six short
  decks are steeper than 8.5 % (up to 11 %) because the ground at their two ends
  differs that much (two parking-garage ramps, culvert bridges).
* **`problems.txt` lists three spots,** all from the OpenStreetMap geometry: two short
  decks (Mall of Louisiana Boulevard and a parking ramp) sit 0.8 and 1.1 m under the
  ground next to an abutment, where the level cuts a trench, and one I-10 on-ramp near
  the I-10/I-110 split starts on the ground under the edge of the I-10 deck, 3.2 m
  below it.
* **The terrain is cut away under decks** where the ground would poke through (116
  decks, up to 1.8 m). The cut is hidden under the deck but can show as a small dip
  beside a bridge end.
* **Ground roads do not climb onto decks.** Where OpenStreetMap ends a bridge on an
  elevated deck but carries on as an ordinary road, that road stays on the terrain,
  so there is a step at the joint (the I-10 on-ramp above is one).
* **Trees can overhang a road edge.** Trees are kept clear of the roads from the ground
  up to 3 m, but higher up a tree's branches can reach over a road edge, so a tall truck
  at the edge may brush one.
* **Covered drives dead-end at the building wall.** Drives that OpenStreetMap maps under
  buildings (`tunnel=building_passage`, `covered=yes`) are drawn, but the buildings are
  solid. They are left out of the AI road network, so traffic does not use them.
* **Ground spawns face south.** Only the I-10 bridge spawn is turned along the road.

## Water

* **Water is drawn with rectangular water blocks,** one per 512 m strip of each lake or
  river. Low ground inside a block's rectangle but outside the real shoreline can show
  water, mostly in the batture along the Mississippi.
* **Water levels come from the lidar** (the Mississippi at about 4 m, which is a low
  river stage). The river bed is carved 15 m below the surface.
* **Bayou and swamp shorelines are jagged.** The bayous and the swamp pools are not
  carved into the terrain: their water lies in the channels and hollows the lidar shows,
  and the shoreline is wherever it meets the ground. The terrain is a 4 m grid, so the
  shorelines step and zigzag along its cells. The 2 m terrain planned for version 0.4
  will make them finer.
* **Swamp pools can show water over another water body's edge, and can end in the air.**
  Up to 2 cells (4 m grid) per swamp-pool block may still show pool water over the river's
  or a lake's edge (49 cells, about 780 m2, on the full map, mostly on the Mississippi's
  bank), and in places a pool block's side ends above lower ground, so the water plane
  stops in the air for a few meters (about 1.2 km of edges in all, most beside the river
  bank top and the batture lakes). Both come from the 4 m terrain grid and the box-shaped
  water blocks; version 0.3 will refine the pools, and the 2 m terrain planned for version
  0.4 will help.
* **A bayou the lidar shows no channel for has no water.** Where a ditch or bayou is
  flat or filled in the terrain, it is left dry (most of Monte Sano Bayou).

## Performance

* **The first load takes the longest.** The game compiles every mesh the first time a
  level loads and builds the trees' imposters, and a GeForce NOW session starts fresh
  each time, so the zip ships the compiled meshes and the imposters (about 242 MB in
  all).
  With the compiled building meshes a cold first load of version 0.1 took 98 s on
  the test laptop, against 165 s without them; the game used about 5.4 GB of memory while
  loading and about 3.1 GB while driving. The release zip of version 0.2 loaded cold in 65 s on
  BeamNG.drive 0.40.0, with every shape compiled, and the game used 5.0 GB of memory at
  the peak. The frame rate held the game's 30 fps cap at the I-10 bridge, over the
  swamps, at the truss and at the Memorial Tower.
* **BeamNG.drive 0.40 logs `material cbv array exhausted (8192 slots)`** (a `bindless`
  warning, not an error) while the map loads. It is a performance fallback of the 0.40
  renderer, which the map's many building tiles trigger by filling its 8,192 material
  slots. The map draws correctly; what the fallback costs and how to stay under the limit
  are under investigation.
* The map has about 24,000 decal roads (16,000 of them service roads: driveways and
  parking aisles) and 11,389 painted markings, which is many more than stock maps, plus
  72,807 buildings in 3,019 mesh tiles, 168,072 trees, 301 bayous drawn as River objects
  and 228 tiles of deck barriers and piers. If it runs slowly, dropping service roads in
  `brmap/roads.py` (`KEPT`) is the first thing to try, and lowering `trees.max_count` in
  `config.yaml` the second.
