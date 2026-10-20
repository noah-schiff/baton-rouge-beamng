# Putting the map on the BeamNG repository (and on GeForce NOW)

GeForce NOW runs BeamNG.drive on NVIDIA's servers, so a zip on your own computer never
reaches it. Mods from the official repository can be subscribed to from inside the game,
which is the way to get this map into a GeForce NOW session. BeamNG's FAQ confirms the
game runs on GeForce NOW but does not document mod support there; players on BeamNG's
forum report that only repository mods work there, and one thinks each session starts
fresh, so you may have to subscribe again in each session. Check in game once the map is
approved.

## Before uploading

Run the full build and the checks:

```
.venv\Scripts\python -m brmap all
.venv\Scripts\python -m pytest tests -q
```

`data/out/problems.txt` should list only the three known spots described in
[known-issues.md](known-issues.md) (two decks just under the ground and one on-ramp
under the edge of I-10); anything else is new and worth a look before uploading.

Then add the compiled meshes and the tree imposters, so a fresh GeForce NOW session does
not build them on every load (the details are in the
[README](../README.md#shipping-the-compiled-meshes-and-imposters)):

1. Copy `dist/baton_rouge.zip` into the mods folder and load the map once in the game.
2. Run `.venv\Scripts\python tools\ingame\collect_meshes.py`. It must run with no warnings
   (it warns for each tree it has no imposters for).
3. Run `.venv\Scripts\python -m brmap level` again. Its log should show
   `imposters: N pairs shipped, 0 withheld` and end the compiled meshes line with
   `0 left for the game to compile`. If it does not, or if step 2 warned, copy the new zip
   into the mods folder, load the level once more and repeat from step 2.

The file to upload is then `dist/baton_rouge.zip` (about 242 MB for version 0.2). Last,
decide where the merged map data or the method used to make it goes (see
[The ODbL and the merged building data](#the-odbl-and-the-merged-building-data)) and put
its link on the last line of the overview text.

BeamNG's [modding guidelines](https://beamng.com/game/support/policies/modding-guidelines/)
and how this build meets them:

| Guideline | This map |
|---|---|
| Works by dropping the zip into the mods folder, no extra steps | yes: the zip holds `levels/baton_rouge/` and the level's licence plate design, `vehicles/common/licenseplates/baton_rouge/`, at its root |
| Is recognised as a map by the mod manager | not on a fresh install: BeamNG's mod manager guesses a zip's type from its file names (`core/modmanager.lua`, `checkType`: `vehicles/` means a vehicle, `main/*.level.json` a map, and a zip that matches both is typed "unknown"), so the plate design's `vehicles/` folder beside `levels/` makes a fresh install show type "unknown" (an existing install keeps its type: the manager types a zip only when it first sees it). Mounting does not depend on the type: the level and its plate design both load. A zip without the plate folder is typed as a map, but then 0.40 logs a warning that the level has no licence plate design. The folder stays, and you check the type shown after a truly fresh install (step 2 of [posting the update](#version-02-posting-the-update) says how to get one) |
| Unique zip and file names, only letters, digits, `-` and `_`, no version in the name | `baton_rouge.zip`; every material the map makes is prefixed `br_` (the stock tree models' materials keep their stock names, which the models need); the plate design's folder is named after the level, as the stock levels' are; no other Baton Rouge map was found on the repository |
| You made it or have permission for the content | built from public data with the attribution below (OpenStreetMap and the Microsoft footprints under the ODbL, see below); the landmarks, styled buildings, deck dressing, cypress, tupelo and moss are this repo's own Blender and numpy models; textures are the game's own (referenced, not copied), CC0 downloads from ambientCG and Poly Haven, or drawn by the repo's scripts ([textures.md](textures.md) lists each); every logo is original, with no real logo, wordmark or trademark |
| Does not overwrite default game data | nothing outside `levels/baton_rouge/` and `vehicles/common/licenseplates/baton_rouge/`, a folder no stock file uses (the game looks there for the level's plate design) |
| No unused files or readmes in the zip | only the level files and the plate design, plus each shape's compiled mesh (`.cdae`) beside its `.dae` and the custom trees' imposters, as the stock maps ship them; about 242 MB in all |
| An overview and at least two images | text below; use the screenshots listed below |
| No errors in the console | the game logs no errors from the level in 0.40.0 (checked with `tools/ingame/brmapTest.lua`) |
| Works with the latest game version | tested in 0.40.0 |

## Uploading

1. Sign in at [beamng.com](https://www.beamng.com/) with your forum account.
2. Open the mod repository, choose to upload a new mod, and pick the maps/terrains
   category.
3. Title: **Baton Rouge 1:1**. Version: **0.1**.
4. Attach `dist/baton_rouge.zip` (the one with the compiled meshes).
5. Paste the overview below, with the real link on its last line.
6. Upload at least two images from `docs/screenshots/`: `lsu_tiger_stadium.jpg`,
   `capitol.jpg`, `i10_mississippi_bridge.jpg`, `downtown_river.jpg`, `i10_elevated.jpg`
   and `university_lake.jpg` were all taken in game. `tools/ingame/brmapTest.lua` saves
   fresh ones to `screenshots\brmap\` in the game's user folder.
7. Submit (the form's **Save** button). Every mod is reviewed by BeamNG's moderators
   before it becomes public.

What the form enforces (found on the first upload, 2026-10-17):

* File and image uploads need two-step verification turned on for the account
  (Your Account, Two-Step Verification); without it every upload fails with "Please
  enable Two-Step verification in order to upload files!".
* The subtitle is at most 100 characters. The one used: "Baton Rouge, Louisiana at real
  scale: real terrain, roads, buildings, trees and AI traffic".
* Only existing tags can be used ("You may not create new tags"). `usa` and `lidar` exist.
* Unique ID: `baton_rouge`. The additional information URL takes the GitHub link.

When it is approved: start BeamNG.drive on GeForce NOW, open **Repository** from the main
menu, search for "Baton Rouge", press **Subscribe**, then open **Freeroam** and pick
**Baton Rouge 1:1**.

For updates, use **Post resource update** on the mod's page, upload a zip with exactly
the same file name, raise the version (0.1 to 0.2) and say what changed.

## Version 0.2: posting the update

Version 0.2 goes up once version 0.1 is approved, or, if the moderators ask for changes,
together with those fixes.

1. Build and check it as in [Before uploading](#before-uploading), compiled meshes and
   imposters included.
2. Check the mod type, on a fresh install. 0.40's mod manager types a zip only when it
   first sees it (`core/modmanager.lua`: `updateZIPEntry` runs `checkType` only for a new
   entry, and keeps an entry it already knows with the type it has), and it drops the
   entry of a deleted zip only when the game starts without that zip. So a new zip copied
   over an old install shows the old type: an existing install keeps its type, and only a
   fresh install shows what the new zip is typed as. To get one, quit the game, move the
   old `baton_rouge.zip` out of the mods folder, start the game and quit it once, then
   copy the new zip in, start the game again and check the type the mod manager shows for
   Baton Rouge. By the mod manager's code (`checkType`) it should read "unknown" instead
   of a map, which has not been seen in game yet: the zip holds the plate design's
   `vehicles/` folder beside `levels/`, the manager types a zip as a vehicle if it finds a
   `vehicles/` folder and as a map if it finds a `main/*.level.json` file, and when both
   match it gives up. That does not stop the mod from mounting: the level and its plate
   design both load, and the map starts from Freeroam. If you would rather have it typed
   as a map, drop the `vehicles/common/licenseplates/baton_rouge/` folder from the zip
   (`zip_mod` in `src/brmap/level.py`), which brings back 0.40's licence plate warning in
   the game's log instead. The plate folder stays unless you decide otherwise.
3. On the mod's page, choose **Post resource update**.
4. Attach `dist/baton_rouge.zip`: exactly the same file name, the same unique ID
   (`baton_rouge`), about 242 MB.
5. Version: **0.2**.
6. Paste the update text below, and replace the overview with the updated
   [overview text](#overview-text-to-paste).
7. Add images from `docs/screenshots/v02/`, all taken in game: `full_capitol.jpg`,
   `full_truss_side.jpg`, `full_downtown_towers.jpg`, `full_old_capitol_se.jpg`,
   `full_cypress_moss_close.jpg` and `full_swamp_far.jpg`
   on the full map, and `tiger_stadium_close.jpg`, `pmac_close.jpg`, `alex_box_close.jpg`
   and `mike_habitat_close.jpg` on the LSU test map.
8. The subtitle may stay, or change to (95 characters): "Baton Rouge, Louisiana at
   real scale: Tiger Stadium, the Capitol, cypress swamps and AI traffic". The tags stay
   `usa` and `lidar`.
9. Save. The update is reviewed like the first upload.

### Update text to paste

> **Version 0.2: landmarks and swamps**
>
> * Modelled landmarks, each with its own levels of detail and collision: Tiger Stadium,
>   the Memorial Tower, the Pete Maravich Assembly Center (PMAC), Alex Box Stadium, Mike
>   the Tiger's habitat, the Louisiana State Capitol (at its full 137 m) and the Old State
>   Capitol
> * The downtown towers with their real massing and window-grid facades, and the LSU
>   campus core in tan stucco with red tile roofs
> * The Horace Wilkinson Bridge (I-10 over the Mississippi) has its steel cantilever truss
> * Concrete barriers on all 252 bridge decks (the two 6 m ramp stubs get only a short
>   piece), and piers under the 116 decks that stand 2.5 m or more over the ground or the
>   river bed
> * Bald cypress, water tupelo and Spanish moss: cypress and tupelo in the wetlands and
>   along the bayous, moss on cypress and on live oaks
> * Bayous, creeks and canals as murky water in their channels, and swamp pools
> * Clean outlines for the tall buildings that were stair-stepped
> * Original logos on the stadium's field and scoreboard; no real logos
> * Tested on BeamNG.drive 0.40
>
> The zip is now about 242 MB.

## The ODbL and the merged building data

The map merges two databases under the Open Database License, OpenStreetMap and
Microsoft's building footprints, into one set of buildings. The ODbL treats that merged
data as a derivative database, and the map made from it as a produced work. Publishing
the map then requires offering, with it, either the merged database or the method used
to make it, in a machine-readable form
([ODbL 1.0, section 4.6](https://opendatacommons.org/licenses/odbl/1-0/)). This
repository's source code is one way to offer the method (`python -m brmap all` rebuilds
the merged data from the public sources); the other is the merged buildings themselves,
`data/out/buildings.json`.

Before uploading, decide which of the two to offer and where it will be, then put its
link in place of the placeholder on the last line of the overview text.

## Overview text to paste

> **Baton Rouge 1:1**
>
> Baton Rouge, Louisiana at real scale: a 16 x 16 km map built entirely from public map
> data. Drive the real street grid on the real terrain, from LSU and Tiger Stadium to
> downtown and the State Capitol, across the I-10 Mississippi River bridge (Horace
> Wilkinson Bridge), along the elevated I-110 and I-10, around the LSU lakes and past the
> cypress swamps.
>
> * Modelled landmarks: Tiger Stadium, the Memorial Tower, the Pete Maravich Assembly
>   Center (PMAC), Alex Box Stadium, Mike the Tiger's habitat, the State Capitol, the Old
>   State Capitol, the downtown towers and the LSU campus core
> * Terrain from USGS 1 m lidar on a 4 m grid, with the 2023 NAIP aerial photo on top
> * About 73,000 buildings from OpenStreetMap and Microsoft footprints, at heights
>   measured by lidar
> * About 168,000 trees where the lidar finds them, with bald cypress, water tupelo and
>   Spanish moss made for this map
> * Bayous, creeks and swamp pools as murky still water
> * Lane markings on the main roads: lane lines, double yellow center lines, edge
>   lines, stop bars and crosswalks
> * About 24,000 roads from OpenStreetMap, from interstates to parking lots
> * AI traffic, with lane counts and speed limits from OpenStreetMap
> * 252 bridges and elevated roads as solid decks, which the traffic drives over too:
>   concrete barriers on all of them (the two 6 m ramp stubs get only a short piece),
>   piers under the 116 decks that stand 2.5 m or more over the ground or the river bed,
>   and the Horace Wilkinson Bridge's steel truss
> * The Mississippi River, the LSU lakes, City Park Lake, Capitol Lake and more
> * Spawn points at the LSU Tiger Stadium lot, the State Capitol and on top of the I-10
>   Mississippi River bridge
>
> Ordinary buildings are simple shapes without modeled windows or doors, and most trees
> are the game's stock models standing in for the local oaks and pines. There are no
> traffic lights, signs or street lights yet. Every logo is original. Tested on
> BeamNG.drive 0.40.
>
> **How to use:** Freeroam, then Baton Rouge 1:1. Pick a spawn point from the map's
> spawn list. To add AI traffic, switch it on in Freeroam's options.
>
> Map data © OpenStreetMap contributors (ODbL). Building footprints © Microsoft (ODbL).
> Elevation and surface: USGS 3DEP. Imagery: USDA NAIP. Land cover: MRLC NLCD.
> Textures: ambientCG and Poly Haven (CC0).
>
> The merged map data, or the method used to make it: https://github.com/noah-schiff/baton-rouge-beamng

Keep the credits wherever the map is shared: the ODbL, which covers OpenStreetMap and
the Microsoft footprints, requires the attribution. The USGS, USDA and MRLC data are US
government works in the public domain. The CC0 textures need no attribution, but the
credits name their sources anyway.
