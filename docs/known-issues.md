# Known issues and limits

Tested in BeamNG.drive 0.39.4 (Direct3D 12) on a laptop with integrated Radeon 680M
graphics and 15 GB of RAM, using the in-game checks in `tools/ingame/`: camera views of
the landmarks, the downtown towers, the Capitol tower, the trees on the LSU Quad and the
paint on Government Street; a traffic check, in which 6 of 6 AI cars drove 400 to 780 m
in a minute; and a scripted car driven over I-10 both ways (the I-10/I-110 split
flyover, the Mississippi River bridge crest and both approaches, 4.2 km each), up the
I-110 from the split, and along two streets by the LSU lakes. On the decks the car
stayed within 0.2 m of its usual height above the deck the whole way. The game logs no
errors from the level. Not yet tested on GeForce NOW.

## What the map does not have (yet)

* **No traffic lights, stop signs or other signs, street lights, sidewalks or
  parking-lot paint.** The game's traffic still drives without them. Stop bars and
  crosswalks are painted where OpenStreetMap maps signals, stop signs and marked
  crossings.
* **Nothing beyond the 16 x 16 km box.** The terrain stops at its edge.

## Looks

* **The aerial photo is 4 m per pixel** (4096 pixels across 16 km, the largest terrain
  base texture the stock maps use), so it is soft up close. The near ground uses the
  stock grass, dirt, asphalt and mud detail textures chosen from NLCD land cover.
* **Aerial tiles were flown on different days,** so a few tile edges show as straight
  color changes, most visibly on the Mississippi south of the I-10 bridge.
* **Roads are flat decals** with three looks (concrete interstates, asphalt arterials,
  worn asphalt streets). Overlapping roads at intersections can flicker where two decals
  meet.
* **Buildings are plain extruded shapes** with tiled wall textures (house siding, brick,
  concrete or glass): no modeled windows, doors or roof details. Small houses have
  hipped roofs; all other roofs are flat.
* **The lidar and the aerial photo are a few years apart.** The lidar is from 2017 to
  2019 and the aerial photo from 2023, so a few new or demolished buildings disagree
  with the photo under them.
* **Microsoft footprints are traced from aerial imagery by machine learning.** They
  occasionally merge neighboring buildings into one or miss small ones.
* **Thin tops are clipped.** The 2 m lidar misses the thin top of a tower: the State
  Capitol's tower reaches about 114 m in the map, while the real tower is 137 m.
* **Trees are stock stand-ins.** BeamNG's own tree models from its other maps stand in
  for the local species (stock oaks for the live and water oaks, a pine for loblolly, a
  redwood scaled to the measured height for bald cypress), so there is no Spanish moss
  and no real bald cypress.
* **Edge lines at skewed junctions** can run a few meters into a side street's mouth.

## Driving

* **Bridge decks are plain slabs** without railings, piers or expansion joints. Their
  heights are estimated from the lidar and OpenStreetMap: pieces that meet are joined
  into one deck (the Mississippi River bridge with its I-10 approaches, the I-10/I-110
  split), a ramp starts at the height of the deck it leaves and stays level with it
  until the two part, and decks clear the roads and lower decks under them by 5.5 m,
  climbing at 5 % where they can and at most 8 % from their ends. The Horace Wilkinson
  (I-10) bridge crests at about 61 m over the river (`bridges.overrides` in
  `config.yaml`).
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

## Performance

* **The first load takes the longest.** The game compiles every building mesh the first time a
  level loads, and a GeForce NOW session starts fresh each time, so the zip ships the
  compiled meshes (218 MB in all). With them a cold first load took 98 s on the
  test laptop, against 165 s without them. The game used about 5.4 GB of memory while
  loading and about 3.1 GB while driving.
* The map has about 24,000 decal roads (16,000 of them service roads: driveways and
  parking aisles) and 11,389 painted markings, which is many more than stock maps, plus
  72,816 buildings in 3,007 mesh tiles and 166,138 trees. If it runs slowly, dropping
  service roads in `brmap/roads.py` (`KEPT`) is the first thing to try, and lowering
  `trees.max_count` in `config.yaml` the second.
