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

Then add the compiled building meshes, so a fresh GeForce NOW session does not compile
them on every load (the details are in the
[README](../README.md#shipping-the-compiled-building-meshes)):

1. Copy `dist/baton_rouge.zip` into the mods folder and load the map once in the game.
2. Run `.venv\Scripts\python tools\ingame\collect_meshes.py`.
3. Run `.venv\Scripts\python -m brmap level` again. Its log should end the compiled
   meshes line with `0 left for the game to compile`.

The file to upload is then `dist/baton_rouge.zip` (about 218 MB). Last, decide where the
merged map data or the method used to make it goes (see
[The ODbL and the merged building data](#the-odbl-and-the-merged-building-data)) and put
its link on the last line of the overview text.

BeamNG's [modding guidelines](https://beamng.com/game/support/policies/modding-guidelines/)
and how this build meets them:

| Guideline | This map |
|---|---|
| Works by dropping the zip into the mods folder, no extra steps | yes: the zip holds `levels/baton_rouge/` at its root |
| Unique zip and file names, only letters, digits, `-` and `_`, no version in the name | `baton_rouge.zip`; every material the map makes is prefixed `br_` (the stock tree models' materials keep their stock names, which the models need); no other Baton Rouge map was found on the repository |
| You made it or have permission for the content | built from public data with the attribution below (OpenStreetMap and the Microsoft footprints under the ODbL, see below); textures and tree models are the game's own, referenced and not copied |
| Does not overwrite default game data | nothing outside `levels/baton_rouge/` |
| No unused files or readmes in the zip | only the level files, plus each building tile's compiled mesh (`.cdae`) beside its `.dae`, as the stock maps ship them; 218 MB in all |
| An overview and at least two images | text below; use the screenshots listed below |
| No errors in the console | the game logs no errors from the level in 0.39.4 (checked with `tools/ingame/brmapTest.lua`) |
| Works with the latest game version | tested in 0.39.4 |

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
7. Submit. Every mod is reviewed by BeamNG's moderators before it becomes public.

When it is approved: start BeamNG.drive on GeForce NOW, open **Repository** from the main
menu, search for "Baton Rouge", press **Subscribe**, then open **Freeroam** and pick
**Baton Rouge 1:1**.

For updates, use **Post resource update** on the mod's page, upload a zip with exactly
the same file name, raise the version (0.1 to 0.2) and say what changed.

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
> Wilkinson Bridge), along the elevated I-110 and I-10, and around the LSU lakes.
>
> * Terrain from USGS 1 m lidar on a 4 m grid, with the 2023 NAIP aerial photo on top
> * About 73,000 buildings from OpenStreetMap and Microsoft footprints, at heights
>   measured by lidar, downtown towers and the Capitol included
> * About 166,000 trees where the lidar finds them, using the game's own tree models
> * Lane markings on the main roads: lane lines, double yellow center lines, edge
>   lines, stop bars and crosswalks
> * About 24,000 roads from OpenStreetMap, from interstates to parking lots
> * AI traffic, with lane counts and speed limits from OpenStreetMap
> * 252 bridges and elevated roads as solid decks, which the traffic drives over too
> * The Mississippi River, the LSU lakes, City Park Lake, Capitol Lake and more
> * Spawn points at the LSU Tiger Stadium lot, the State Capitol and on top of the I-10
>   Mississippi River bridge
>
> Buildings are simple shapes without modeled windows or doors, and the trees are the
> game's stock models standing in for the local oaks and cypress. There are no traffic
> lights, signs or street lights yet, and bridges are plain decks without railings.
>
> **How to use:** Freeroam, then Baton Rouge 1:1. Pick a spawn point from the map's
> spawn list. To add AI traffic, switch it on in Freeroam's options.
>
> Map data © OpenStreetMap contributors (ODbL). Building footprints © Microsoft (ODbL).
> Elevation and surface: USGS 3DEP. Imagery: USDA NAIP. Land cover: MRLC NLCD.
>
> The merged map data, or the method used to make it: https://github.com/noah-schiff/baton-rouge-beamng

Keep the credits wherever the map is shared: the ODbL, which covers OpenStreetMap and
the Microsoft footprints, requires the attribution. The USGS, USDA and MRLC data are US
government works in the public domain.
