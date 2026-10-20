-- In-game check for a brmap level (load with -onLevelLoad_ext brmapTest).
-- Logs object counts, the vehicle position and terrain heights at fixed points,
-- then moves the free camera through the level's views (VIEW_LISTS, by level name;
-- one every HOLD seconds). SETTLE seconds into each view it takes an engine screenshot,
-- <userfolder>/screenshots/brmap/<level>_<view>.jpg, which other windows on the desktop
-- can't spoil (capture_views.ps1 can still save screen captures meanwhile). A view with
-- fps = <s> then logs the frame rate over that many seconds: always the frames counted,
-- and also the number the game's own FPS display reads when it is available
-- (VariableRegistry.get('$fps::instantaneous', -1), as core/metrics.lua reads it; nil or
-- negative means not available). After the views comes a traffic check:
-- TRAFFIC_COUNT AI cars drive for TRAFFIC_WAIT seconds, and each one's distance from
-- where it spawned is logged. Messages are tagged brmapTest in beamng.log and the
-- latest one is also written to <userfolder>/temp/brmapTest_status.txt (the log is
-- flushed late, the status file at once); the last one is 'done'.
local M = {}

local HOLD = 15
local SETTLE = 8              -- s into a view before its screenshot
local FPS_DELAY = 1           -- s after the screenshot before the frame rate is measured
local SHOT_DIR = 'screenshots/brmap/'
local TRAFFIC_COUNT = 6
local TRAFFIC_WAIT = 60       -- s the traffic drives before it is measured
local TRAFFIC_TIMEOUT = 90    -- s the traffic may take to spawn
local TRAFFIC_MOVED = 100     -- m a car has to get from where it spawned
local POINTS = {{0, 0}, {0, 1400}, {0, -1400}, {1400, 0}, {-1400, 0}, {700, 700}, {-700, -700}}
-- Positions are level metres (x east, y north, z NAVD88), from brmap.geo.Extent with
-- config.debug.yaml, config.yaml or config.lsu.yaml; dir is the way the camera looks
-- (its z is the slope).
local DEBUG_VIEWS = {  -- baton_rouge_debug (3 km around the map center)
  {name = 'road_closeup', pos = {690, -170, 22}, dir = {0, 1, -0.35}},
  {name = 'spawn_street', pos = {2, -35, 32}, dir = {0, 1, -0.3}},
  {name = 'south_overview', pos = {0, -1300, 500}, dir = {0, 1, -0.55}},
  {name = 'i10_deck', pos = {441, -820, 70}, dir = {0, 1, -0.3}},
  {name = 'lakes', pos = {-692, -250, 140}, dir = {0, -1, -0.45}},
  -- buildings and trees on the blocks north-east of the center, 80 m up from the south;
  -- camera at 30.42791 N 91.15900 W
  {name = 'centre_buildings', pos = {100.0, -230.0, 96.5}, dir = {0.000, 1.000, -0.244}},
  -- trees along Edward Avenue (way 12632852), 25 m up; camera at 30.42941 N 91.16014 W
  {name = 'street_trees', pos = {-12.5, -65.5, 41.5}, dir = {0.999, 0.047, -0.211}},
  -- paint: Perkins Road (way 12628845, two-way tertiary: the double yellow) by the
  -- center, 1.8 m above its southbound lane; camera at 30.43001 N 91.16087 W
  {name = 'perkins_paint', pos = {-83.4, -0.4, 18.3}, dir = {0.412, -0.911, -0.028}},
  -- paint: I-10 (way 132986449, 3 lanes one-way: dividers and edge lines), 2.5 m above
  -- its middle lane; camera at 30.42543 N 91.16215 W
  {name = 'i10_paint', pos = {-198.3, -509.9, 13.3}, dir = {-0.994, 0.113, -0.026}},
}
local FULL_VIEWS = {  -- baton_rouge (16 km); the UI is hidden so the captures are clean
  {name = 'bridge_spawn', pos = {-3420, 760, 95}, dir = {0.6, 1, -0.2}, teleport = 'spawn_i10_bridge'},
  {name = 'lsu_tiger_stadium', pos = {-2300, -2700, 260}, dir = {0.25, 1, -0.55}},
  {name = 'capitol', pos = {-2620, 2330, 230}, dir = {0, 1, -0.65}},
  {name = 'i10_mississippi_bridge', pos = {-3150, 520, 110}, dir = {-0.2, 1, -0.12}, fps = 10},
  -- spike S4: over the biggest swamp (OSM w1462812397, 0.75 km2, about -4830, -1986; ground
  -- 9.5 m), 60 m up from 360 m south, with the frame rate
  {name = 'swamp_far', pos = {-4830.0, -2350.0, 60.0}, dir = {0.000, 1.000, -0.140}, fps = 10},
  {name = 'downtown_river', pos = {-2200, 1100, 160}, dir = {-0.55, 1, -0.3}},
  {name = 'i10_elevated', pos = {441, -820, 70}, dir = {0, 1, -0.3}},
  {name = 'university_lake', pos = {-692, -250, 140}, dir = {0, -1, -0.45}},
  -- the downtown skyline from the south-east, 120 m up, aimed at One American Place
  -- (30.4503 N 91.1867 W); camera at 30.44553 N 91.18137 W
  {name = 'downtown_towers', pos = {-2080.7, 1688.7, 133.7}, dir = {-0.707, 0.707, -0.100}},
  -- the State Capitol (30.4570 N 91.1874 W, 137 m tall) from 330 m south, 45 m up;
  -- camera at 30.45402 N 91.18746 W
  {name = 'capitol_tower', pos = {-2679.8, 2620.3, 60.9}, dir = {0.000, 1.000, 0.088}},
  -- across the LSU Quad (30.4133 N 91.1800 W) from the south, 40 m up;
  -- camera at 30.41231 N 91.18002 W
  {name = 'lsu_quad_trees', pos = {-1891.3, -1992.2, 52.2}, dir = {0.000, 1.000, -0.242}},
  -- paint: Government Street (way 1412996238, 4 lanes two-way primary: the double
  -- yellow, dashed dividers, white edges), 1.8 m above the inner eastbound lane;
  -- camera at 30.44422 N 91.15192 W
  {name = 'government_st_paint', pos = {750.4, 1588.8, 15.8}, dir = {0.999, 0.044, -0.031}},
}
-- The Memorial Tower (LSU, 30.414482 N 91.178903 W; blender/landmarks/memorial_tower.py).
-- Its model's +u axis points north (yaw 92.4 deg: +u = (-0.043, 0.999), +v = (-0.999,
-- -0.043)), so its north and south faces carry the PNG clock dial and its east and west
-- faces the DDS one. The dials are centred 40.5 m above the ground (landmarks.json
-- ground_z); the shaft is 8.0 m along +u and 6.6 m along +v. The views are worked out with
-- brmap.geo.Extent from the level's config and the ground from its out/heights.npy.
local LANDMARK_VIEWS = {  -- baton_rouge (config.yaml); ground_z 12.93
  -- 30 m from the south face, 1.7 m up, looking up 35 deg; camera at 30.41418 N 91.17889 W
  {name = 'tower_close', pos = {-1786.6, -1783.4, 14.3}, dir = {-0.043, 0.999, 0.700}},
  -- 150 m south-west, 20 m up, aimed 30 m up the tower; camera at 30.41354 N 91.18002 W
  {name = 'tower_approach', pos = {-1894.1, -1855.5, 32.0}, dir = {0.707, 0.707, 0.073}},
  -- 1.5 km south-west, 60 m up, with the frame rate; camera at 30.40507 N 91.19012 W
  {name = 'tower_far', pos = {-2848.7, -2810.1, 67.1}, dir = {0.707, 0.707, -0.016}, fps = 10},
  -- 25 m out from the north face (the PNG dial), level with it; camera at 30.41474 N 91.17891 W
  {name = 'tower_clock_n', pos = {-1789.2, -1720.5, 53.4}, dir = {0.043, -0.999, 0.000}},
  -- 25 m out from the east face (the DDS dial), level with it; camera at 30.41449 N 91.17861 W
  {name = 'tower_clock_e', pos = {-1759.7, -1748.2, 53.4}, dir = {-0.999, -0.043, 0.000}},
  -- sign-offs 3-5 (full map, from data/out: landmarks.json, roads.json, dressing.json, trees.json)
  {name = 'capitol_south', pos = {-2677.7, 2788.8, 26.9}, dir = {0.000, 0.976, 0.219}},       -- 170 m S, aim 45 m up
  {name = 'old_capitol_se', pos = {-2752.2, 1719.2, 29.6}, dir = {-0.682, 0.731, 0.002}},     -- 100 m SE
  {name = 'truss_side', pos = {-3640.0, 589.9, 25.0}, dir = {-0.141, 0.985, 0.104}, fps = 10}, -- 420 m S of the truss
  {name = 'truss_deck', pos = {-3150.5, 981.2, 44.6}, dir = {-0.999, 0.050, 0.016}},          -- on the deck, looking W
  {name = 'piers_under', pos = {-2860.0, 935.0, 10.3}, dir = {-0.967, 0.245, 0.072}},       -- I-10's east approach, 17-28 m piers
  {name = 'tupelo_close', pos = {-7884.2, -2574.8, 7.1}, dir = {0.000, 0.995, 0.096}},        -- densest tupelo stand
  {name = 'moss_street', pos = {724.8, -2940.8, 9.0}, dir = {0.706, 0.706, 0.044}},           -- densest moss oaks
  {name = 'stadium_lane', pos = {-2110.1, -2077.9, 8.8}, dir = {-0.372, 0.928, 0.020}},       -- Powerhouse Lane by the bowl's east wall
  {name = 'barrier_joint', pos = {-2675.0, 968.0, 19.9}, dir = {-0.997, 0.050, 0.030}},        -- I-10 WB deck, 32 m before a straight joint
  {name = 'cypress_moss_close', pos = {5594.1, 6424.4, 16.9}, dir = {0.000, 0.986, 0.165}},   -- mossy cypress
}
local LSU_VIEWS = {  -- baton_rouge_lsu (config.lsu.yaml: 3 km around LSU); the same views
  -- sign-off 2: each landmark from the air and from close by, aimed at its footprint centroid
  -- (data/out_lsu/landmarks.json, styled.json); compass bearing target->camera in the comment
  {name = 'tiger_stadium_air', pos = {-194.6, -436.8, 166.9}, dir = {0.320, 0.878, -0.356}},     -- 420 m, 200 deg
  {name = 'tiger_stadium_close', pos = {7.2, -201.9, 46.9}, dir = {-0.336, 0.923, -0.185}},      -- 170 m, 160 deg
  {name = 'pmac_air', pos = {-295.8, 37.7, 97.1}, dir = {0.607, 0.724, -0.327}},                 -- 260 m, 220 deg
  {name = 'pmac_close', pos = {-169.7, 124.1, 22.1}, dir = {0.341, 0.938, -0.058}},              -- 120 m, 200 deg
  {name = 'alex_box_air', pos = {-277.0, -429.2, 97.2}, dir = {-0.472, -0.818, -0.327}},         -- 260 m, 30 deg
  {name = 'alex_box_close', pos = {-444.7, -757.7, 27.2}, dir = {0.340, 0.934, -0.108}},         -- 110 m, 200 deg
  {name = 'mike_habitat_air', pos = {-176.2, 34.8, 42.5}, dir = {0.000, 0.932, -0.362}},         -- 90 m, 180 deg
  {name = 'mike_habitat_close', pos = {-153.7, 85.8, 13.5}, dir = {-0.498, 0.863, -0.089}},      -- 45 m, 150 deg
  {name = 'quad_air', pos = {183.0, -141.3, 92.1}, dir = {0.327, 0.898, -0.294}},                -- 260 m, 200 deg
  {name = 'quad_ground', pos = {308.5, 73.0, 14.0}, dir = {0.000, 0.995, 0.100}},                -- on the lawn, 45 m S of the centre
  {name = 'stadium_lane', pos = {89.7, -77.7, 8.8}, dir = {-0.372, 0.928, 0.020}},             -- Powerhouse Lane by the bowl's east wall
  {name = 'moss_lsu', pos = {684.9, -287.0, 9.4}, dir = {0.766, 0.643, 0.080}},               -- three moss oaks, 28 m
  -- spike S3: Bayou Fountain (w662218183, water at 3.9 m, banks at 6.3 m) from its bank and from
  -- 70 m, and an unnamed campus drain (w549173644, water at 4.3 m)
  {name = 'bayou_bank', pos = {1191.9, -1313.1, 9.5}, dir = {-0.601, -0.720, -0.351}},
  {name = 'bayou_air', pos = {1220.6, -1278.4, 40.0}, dir = {-0.547, -0.660, -0.516}},
  {name = 'drain_bank', pos = {551.1, 1132.3, 9.5}, dir = {-0.664, -0.634, -0.397}},
  {name = 'tower_close', pos = {413.2, 216.8, 14.3}, dir = {-0.043, 0.999, 0.700}},
  {name = 'tower_approach', pos = {305.7, 144.7, 32.0}, dir = {0.707, 0.707, 0.073}},
  {name = 'tower_far', pos = {-648.9, -809.9, 67.1}, dir = {0.707, 0.707, -0.016}, fps = 10},
  {name = 'tower_clock_n', pos = {410.5, 279.7, 53.4}, dir = {0.043, -0.999, 0.000}},
  {name = 'tower_clock_e', pos = {440.0, 252.0, 53.4}, dir = {-0.999, -0.043, 0.000}},
  -- spike S2: the densest bald cypress group (wetland trees around -1164, 871; ground 9.6 m),
  -- from 25 m and from 300 m
  {name = 'cypress_close', pos = {-1164.0, 845.0, 13.0}, dir = {0.000, 1.000, 0.150}},
  {name = 'cypress_far', pos = {-1164.0, 570.0, 50.0}, dir = {0.000, 1.000, -0.120}, fps = 10},
}

local function concat(a, b)
  local out = {}
  for _, list in ipairs({a, b}) do
    for _, v in ipairs(list) do out[#out + 1] = v end
  end
  return out
end

local VIEW_LISTS = {baton_rouge_debug = DEBUG_VIEWS, baton_rouge = concat(FULL_VIEWS, LANDMARK_VIEWS),
                    baton_rouge_lsu = LSU_VIEWS}
local HIDE_UI = {baton_rouge = true, baton_rouge_debug = true, baton_rouge_lsu = true}
local VIEWS = DEBUG_VIEWS

local t, step, ready, n = 0, 0, false, 0
local prefix = ''     -- '<level>_', for the screenshot names
local shot = false    -- the current view's screenshot is taken
local measure = nil   -- the current view's frame-rate measurement, once it has begun
local traffic = nil   -- the traffic check's state while it runs

local function say(fmt, ...)
  local msg = string.format(fmt, ...)
  log('I', 'brmapTest', msg)
  n = n + 1
  writeFile('/temp/brmapTest_status.txt', string.format('%d %s', n, msg))
  -- every line also goes to its own log: beamng.log loses its tail when the game is closed hard
  local f = io.open('/temp/brmapTest_log.txt', n == 1 and 'w' or 'a')
  if f then f:write(string.format('%d %s\n', n, msg)); f:close() end
end

local function report()
  local v = be:getPlayerVehicle(0)
  if v then
    local p = v:getPosition()
    say('vehicle at %.2f %.2f %.2f', p.x, p.y, p.z)
  end
  for _, cls in ipairs({'TerrainBlock', 'DecalRoad', 'MeshRoad', 'WaterBlock', 'SpawnSphere', 'TSStatic', 'Forest'}) do
    say('%s count %d', cls, #scenetree.findClassObjects(cls))
  end
  for _, xy in ipairs(POINTS) do
    say('surface at %d %d = %.3f', xy[1], xy[2], be:getSurfaceHeightBelow(vec3(xy[1], xy[2], 1000)))
  end
end

local function regenerateRoads()
  local names = scenetree.findClassObjects('DecalRoad')
  for _, name in ipairs(names) do
    local r = scenetree.findObject(name)
    if r and r.regenerate then r:regenerate() end
  end
  say('regenerated %d decal roads', #names)
end

local pendingCheck

local function teleportTo(spawnName)
  local sp = scenetree.findObject(spawnName)
  local v = be:getPlayerVehicle(0)
  if not (sp and v) then
    say('teleport: %s or the vehicle is missing', spawnName)
    return
  end
  spawn.safeTeleport(v, sp:getPosition(), quat(0, 0, 1, 0) * sp:getRotation(), nil, nil, nil, true, false)
  local p = sp:getPosition()
  say('teleported to %s at %.2f %.2f %.2f', spawnName, p.x, p.y, p.z)
  pendingCheck = {name = spawnName, t = 0}
end

local function setView(view)
  if view.regenerate then regenerateRoads() end
  if view.teleport then teleportTo(view.teleport) end
  commands.setFreeCamera()
  local q = quatFromDir(vec3(view.dir[1], view.dir[2], view.dir[3]):normalized())
  core_camera.setPosRot(0, view.pos[1], view.pos[2], view.pos[3], q.x, q.y, q.z, q.w)
  shot, measure = false, nil
  say('view %s', view.name)
end

-- an engine screenshot of what the camera shows: <userfolder>/<SHOT_DIR><prefix><name>.jpg
local function screenshot(name)
  if type(createScreenshot2) ~= 'function' then
    say('screenshot %s: createScreenshot2 is not available', name)
    return
  end
  if not FS:directoryExists(SHOT_DIR) then FS:directoryCreate(SHOT_DIR, true) end
  local file = SHOT_DIR .. prefix .. name
  local id = createScreenshot2({filename = file, writeJPG = true, superSampling = 1})
  if not id or id == 0 then
    say('screenshot %s failed', file)
  else
    say('screenshot %s.jpg', file)
  end
end

-- ------------------------------------------------------------------ frame rate

-- how long a view lasts: HOLD, or long enough for its frame-rate measurement
local function holdOf(view)
  if view and view.fps then return math.max(HOLD, SETTLE + FPS_DELAY + view.fps + 2) end
  return HOLD
end

-- the game's fps::instantaneous, read the way its FPS display (core/metrics.lua) reads it;
-- nil when the registry or the value is not available (metrics.lua's default is -1)
local function consoleFps()
  if not VariableRegistry or type(VariableRegistry.get) ~= 'function' then return nil end
  local f = tonumber(VariableRegistry.get('$fps::instantaneous', -1))
  if f and f >= 0 then return f end
  return nil
end

-- one frame of the view's measurement; after view.fps seconds it logs the frames counted
-- per second and, over the frames where it could be read, the game's own fps::instantaneous
-- averaged, with its min and max
local function stepMeasure(view, dt)
  if not measure then
    measure = {t = 0, frames = 0, samples = 0, sum = 0, min = math.huge, max = -math.huge}
  end
  measure.t, measure.frames = measure.t + dt, measure.frames + 1
  local f = consoleFps()
  if f then
    measure.samples, measure.sum = measure.samples + 1, measure.sum + f
    measure.min, measure.max = math.min(measure.min, f), math.max(measure.max, f)
  end
  if measure.t >= view.fps then
    local console = 'fps::instantaneous is not available'
    if measure.samples > 0 then
      console = string.format('fps::instantaneous avg %.1f min %.1f max %.1f (%d of the frames)',
                              measure.sum / measure.samples, measure.min, measure.max, measure.samples)
    end
    say('fps at %s over %.1f s: %.1f (%d frames); %s', view.name, measure.t, measure.frames / measure.t,
        measure.frames, console)
    measure.done = true
  end
end

-- ------------------------------------------------------------------ traffic check

local function startTraffic()
  if not gameplay_traffic then extensions.load('gameplay_traffic') end
  if not gameplay_traffic then
    say('traffic: the gameplay_traffic extension is not available')
    return false
  end
  commands.setGameCamera()  -- traffic spawns near the camera: back to the player's car
  local ok = gameplay_traffic.setupTraffic(TRAFFIC_COUNT)
  say('traffic: setupTraffic(%d) returned %s', TRAFFIC_COUNT, tostring(ok))
  if not ok then return false end
  traffic = {phase = 'spawning', t = 0, spawn = {}, ids = {}, reserved = 0,
             spawnValue = (gameplay_traffic.getTrafficVars() or {}).spawnValue}
  return true
end

-- where each traffic car is the first time it is seen; cars the traffic system keeps
-- parked in reserve (for police or taxi roles) don't drive and are left out
local function noteSpawns()
  local data = gameplay_traffic.getTrafficData() or {}
  for _, id in ipairs(gameplay_traffic.getTrafficList()) do
    local o, veh = getObjectByID(id), data[id]
    if o and traffic.spawn[id] == nil then
      if veh and veh.reserved and not veh.claimed then
        traffic.spawn[id] = false
        traffic.reserved = traffic.reserved + 1
      else
        local p = o:getPosition()
        traffic.spawn[id] = {x = p.x, y = p.y, z = p.z}
        traffic.ids[#traffic.ids + 1] = id
      end
    end
  end
end

local function measureTraffic()
  local data = gameplay_traffic.getTrafficData() or {}
  local moved = 0
  for _, id in ipairs(traffic.ids) do
    local o, p0 = getObjectByID(id), traffic.spawn[id]
    if o then
      local p, vel = o:getPosition(), o:getVelocity()
      local d = math.sqrt((p.x - p0.x) ^ 2 + (p.y - p0.y) ^ 2 + (p.z - p0.z) ^ 2)
      local speed = math.sqrt(vel.x ^ 2 + vel.y ^ 2 + vel.z ^ 2)
      local respawns = data[id] and data[id].respawnCount or 0
      if d > TRAFFIC_MOVED then moved = moved + 1 end
      say('traffic vehicle %d (%s): %.0f m from its spawn at %.0f %.0f, now at %.0f %.0f %.1f, %.1f m/s, %d respawns',
          id, tostring(o.jbeam), d, p0.x, p0.y, p.x, p.y, p.z, speed, respawns)
    else
      say('traffic vehicle %d: gone', id)
    end
  end
  say('traffic moved %d/%d over %d m', moved, #traffic.ids, TRAFFIC_MOVED)
end

local function stepTraffic(dt)
  traffic.t = traffic.t + dt
  if traffic.phase == 'spawning' then
    if gameplay_traffic.getState() == 'on' and #gameplay_traffic.getTrafficList() > 0 then
      noteSpawns()
      -- no respawning during the check, so a car's distance from its spawn is how far it drove
      gameplay_traffic.setTrafficVars({spawnValue = 0})
      say('traffic: %d vehicles driving for %d s (%d more in reserve)', #traffic.ids, TRAFFIC_WAIT,
          traffic.reserved)
      traffic.phase, traffic.t = 'driving', 0
    elseif traffic.t > TRAFFIC_TIMEOUT then
      say('traffic: no vehicles after %d s (state %s)', TRAFFIC_TIMEOUT, tostring(gameplay_traffic.getState()))
      traffic.phase = 'finish'
    end
  elseif traffic.phase == 'driving' then
    noteSpawns()  -- a car that joined late
    if traffic.t > TRAFFIC_WAIT then
      measureTraffic()
      traffic.phase = 'finish'
    end
  end
  if traffic.phase == 'finish' then
    gameplay_traffic.deleteVehicles()
    if traffic.spawnValue then gameplay_traffic.setTrafficVars({spawnValue = traffic.spawnValue}) end
    traffic = nil
    say('done')
  end
end

-- ------------------------------------------------------------------ hooks

local function onWorldReadyState(state)
  if state == 2 and not ready then
    ready, t = true, 0
    local level = getCurrentLevelIdentifier and getCurrentLevelIdentifier() or '?'
    VIEWS = VIEW_LISTS[level] or DEBUG_VIEWS
    prefix = tostring(level) .. '_'
    if HIDE_UI[level] and extensions.ui_visibility then extensions.ui_visibility.setCef(false) end
    say('world ready on %s', tostring(level))
  end
end

local function onUpdate(dtReal)
  if not ready then return end
  t = t + dtReal
  if pendingCheck then
    pendingCheck.t = pendingCheck.t + dtReal
    if pendingCheck.t > 4 then
      local v = be:getPlayerVehicle(0)
      local p, d = v:getPosition(), v:getDirectionVector()
      say('vehicle after %s: at %.2f %.2f %.2f facing %.3f %.3f %.3f, ground below %.2f', pendingCheck.name,
          p.x, p.y, p.z, d.x, d.y, d.z, be:getSurfaceHeightBelow(vec3(p.x, p.y, p.z + 2)))
      pendingCheck = nil
    end
  end
  if traffic then
    stepTraffic(dtReal)
  elseif step == 0 then
    if t > 8 then
      report()
      step, t = 1, 0
      setView(VIEWS[1])
    end
  elseif VIEWS[step] and not shot and t > SETTLE then
    shot = true
    screenshot(VIEWS[step].name)
  elseif VIEWS[step] and VIEWS[step].fps and t > SETTLE + FPS_DELAY and not (measure and measure.done) then
    stepMeasure(VIEWS[step], dtReal)
  elseif t > holdOf(VIEWS[step]) then
    step, t = step + 1, 0
    if VIEWS[step] then
      setView(VIEWS[step])
    elseif step == #VIEWS + 1 then
      if not startTraffic() then say('done') end
    end
  end
end

M.onUpdate = onUpdate
M.onWorldReadyState = onWorldReadyState
return M
