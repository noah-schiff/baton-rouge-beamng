-- In-game check for a brmap level (load with -onLevelLoad_ext brmapTest).
-- Logs object counts, the vehicle position and terrain heights at fixed points,
-- then moves the free camera through VIEWS (one every HOLD seconds). SETTLE seconds
-- into each view it takes an engine screenshot, <userfolder>/screenshots/brmap/
-- <level>_<view>.jpg, which other windows on the desktop can't spoil (capture_views.ps1
-- can still save screen captures meanwhile). After the views comes a traffic check:
-- TRAFFIC_COUNT AI cars drive for TRAFFIC_WAIT seconds, and each one's distance from
-- where it spawned is logged. Messages are tagged brmapTest in beamng.log and the
-- latest one is also written to <userfolder>/temp/brmapTest_status.txt (the log is
-- flushed late, the status file at once); the last one is 'done'.
local M = {}

local HOLD = 15
local SETTLE = 8              -- s into a view before its screenshot
local SHOT_DIR = 'screenshots/brmap/'
local TRAFFIC_COUNT = 6
local TRAFFIC_WAIT = 60       -- s the traffic drives before it is measured
local TRAFFIC_TIMEOUT = 90    -- s the traffic may take to spawn
local TRAFFIC_MOVED = 100     -- m a car has to get from where it spawned
local POINTS = {{0, 0}, {0, 1400}, {0, -1400}, {1400, 0}, {-1400, 0}, {700, 700}, {-700, -700}}
-- Positions are level metres (x east, y north, z NAVD88), from brmap.geo.Extent with
-- config.debug.yaml or config.yaml; dir is the way the camera looks (its z is the slope).
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
  {name = 'i10_mississippi_bridge', pos = {-3150, 520, 110}, dir = {-0.2, 1, -0.12}},
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
local HIDE_UI = {baton_rouge = true, baton_rouge_debug = true}
local VIEWS = DEBUG_VIEWS

local t, step, ready, n = 0, 0, false, 0
local prefix = ''     -- '<level>_', for the screenshot names
local shot = false    -- the current view's screenshot is taken
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
  shot = false
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
    if level == 'baton_rouge' then VIEWS = FULL_VIEWS end
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
  elseif t > HOLD then
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
