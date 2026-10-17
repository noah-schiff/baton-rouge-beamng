-- In-game drive check for a brmap level (load with -onLevelLoad_ext brmapDrive from the
-- user folder's lua/ge/extensions/; an unpacked mod is mounted too late for that).
-- Reads /brmap_routes.json (tools/ingame/drive_routes.py), then for each route puts
-- the player car at its start and drives it along the route with a simple steering
-- and speed controller, logging every half second where the car is against the deck
-- or street under it. Then it parks the car at the map corners and logs how much it
-- moves standing still. Lines go to /temp/brmapDrive_log.txt, the latest status to
-- /temp/brmapDrive_status.txt (both under the user folder).
local M = {}

local LOG_EVERY = 0.5   -- s between log lines while driving
local SETTLE = 3        -- s the car stands after a teleport before it drives
local PARK = 8          -- s the car stands at a corner
local GAIN = 2.0        -- steering per radian of heading error
local OFF_ROUTE = 8     -- m from the route that ends a drive as failed
local TIMEOUT = 420     -- s a single route may take

local ready, n, lines = false, 0, {}
local data, ri, pi = nil, 0, 0
local st = nil          -- state of the current route or park
local sign = nil        -- steering sign, measured on the first drive

local function flush()
  writeFile('/temp/brmapDrive_log.txt', table.concat(lines, '\n') .. '\n')
end

local function say(fmt, ...)
  local msg = string.format(fmt, ...)
  log('I', 'brmapDrive', msg)
  n = n + 1
  lines[#lines + 1] = msg
  writeFile('/temp/brmapDrive_status.txt', string.format('%d %s', n, msg))
end

local function car() return be:getPlayerVehicle(0) end

local function inputs(v, steer, throttle, brake, park)
  v:queueLuaCommand(string.format(
    "input.event('steering', %.4f, 2) input.event('throttle', %.4f, 2) input.event('brake', %.4f, 2) input.event('parkingbrake', %d, 2)",
    steer, throttle, brake, park or 0))
end

local function place(v, p, toward, flip)
  local d = vec3(toward[1] - p[1], toward[2] - p[2], 0):normalized()
  local q = quatFromDir(d)
  if flip then q = quat(0, 0, 1, 0) * q end
  spawn.safeTeleport(v, vec3(p[1], p[2], p[3] + 0.6), q, nil, nil, nil, true, true)  -- repaired
  return d
end

-- nearest point of the route to (x, y) searching segments k0 .. k0 + 40
local function project(pts, k0, x, y)
  local best, bk, bu = math.huge, k0, 0
  for k = math.max(1, k0 - 2), math.min(#pts - 1, k0 + 40) do
    local a, b = pts[k], pts[k + 1]
    local dx, dy = b[1] - a[1], b[2] - a[2]
    local len2 = dx * dx + dy * dy
    local u = len2 > 0 and math.max(0, math.min(1, ((x - a[1]) * dx + (y - a[2]) * dy) / len2)) or 0
    local px, py = a[1] + u * dx, a[2] + u * dy
    local d = (x - px) ^ 2 + (y - py) ^ 2
    if d < best then best, bk, bu = d, k, u end
  end
  local a, b = pts[bk], pts[bk + 1]
  return bk, bu, math.sqrt(best), a[3] + bu * (b[3] - a[3])
end

-- the point `ahead` m further along the route from segment k at fraction u
local function lookahead(pts, k, u, ahead)
  local a, b = pts[k], pts[k + 1]
  local seg = math.sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
  local left = (1 - u) * seg
  while ahead > left and k < #pts - 1 do
    ahead = ahead - left
    k = k + 1
    a, b = pts[k], pts[k + 1]
    seg = math.sqrt((b[1] - a[1]) ^ 2 + (b[2] - a[2]) ^ 2)
    left, u = seg, 0
  end
  local f = seg > 0 and math.min(1, u + ahead / seg) or 1
  return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])
end

local function wrap(a)
  while a > math.pi do a = a - 2 * math.pi end
  while a < -math.pi do a = a + 2 * math.pi end
  return a
end

local function heading(pts, k, u, from, to)
  local x0, y0 = lookahead(pts, k, u, from)
  local x1, y1 = lookahead(pts, k, u, to)
  return math.atan2(y1 - y0, x1 - x0)
end

local function finish(ok, why)
  local s = st
  local v = car()
  if v then inputs(v, 0, 0, 1, 1) end
  table.sort(s.dz)
  local med = #s.dz > 0 and s.dz[math.ceil(#s.dz / 2)] or 0
  local bump, air = 0, 0
  for _, d in ipairs(s.dz) do
    bump = math.max(bump, math.abs(d - med))
    if d > med + 1.0 then air = air + 1 end
  end
  say('route %s %s%s: %.0f m in %.0f s (%.1f m/s), max %.1f m off the line, car z - %s z: median %.2f, min %.2f, max %.2f, largest swing %.2f m, %d of %d samples over 1 m up',
      s.route.name, ok and 'DONE' or 'FAILED', why and (' (' .. why .. ')') or '', s.dist, s.t, s.dist / math.max(s.t, 1e-3),
      s.maxlat, s.route.deck and 'deck' or 'street', med, s.dz[1] or 0, s.dz[#s.dz] or 0, bump, air, #s.dz)
  flush()
  st = nil
end

local function startRoute(route)
  st = {kind = 'route', route = route, pts = route.points, phase = 'settle', t = 0, k = 1, logt = 0,
        dist = 0, maxlat = 0, dz = {}, flipped = false}
  local v = car()
  st.dir = place(v, st.pts[1], st.pts[3] or st.pts[2], false)
  say('route %s (%s): %d points, z %.1f at the start, %.0f m/s', route.name, route.label, #st.pts,
      st.pts[1][3], route.speed)
end

local function startPark(park)
  st = {kind = 'park', park = park, phase = 'settle', t = 0, zmin = math.huge, zmax = -math.huge, vmax = 0}
  place(car(), park.point, park.toward, false)
end

local function stepRoute(v, dt)
  local p, fwd, vel = v:getPosition(), v:getDirectionVector(), v:getVelocity()
  local speed = math.sqrt(vel.x ^ 2 + vel.y ^ 2 + vel.z ^ 2)
  st.t = st.t + dt
  if st.phase == 'settle' then
    inputs(v, 0, 0, 1, 1)
    if st.t > SETTLE then
      local off = math.sqrt((p.x - st.pts[1][1]) ^ 2 + (p.y - st.pts[1][2]) ^ 2)
      if off > 3 and (st.tries or 0) < 2 then  -- the teleport landed somewhere else: again
        st.tries, st.t = (st.tries or 0) + 1, 0
        say('route %s: car %.1f m from the start after the teleport, placing it again', st.route.name, off)
        place(v, st.pts[1], st.pts[3] or st.pts[2], st.flipped)
        return
      end
      if fwd.x * st.dir.x + fwd.y * st.dir.y < 0 and not st.flipped then
        st.flipped, st.t = true, 0
        place(v, st.pts[1], st.pts[3] or st.pts[2], true)
        return
      end
      st.phase, st.t, st.last = 'drive', 0, vec3(p.x, p.y, p.z)
      say('route %s: driving from %.1f %.1f %.1f facing %.2f %.2f', st.route.name, p.x, p.y, p.z, fwd.x, fwd.y)
    end
    return
  end
  local k, u, lat, zr = project(st.pts, st.k, p.x, p.y)
  st.k = k
  st.dist = st.dist + math.sqrt((p.x - st.last.x) ^ 2 + (p.y - st.last.y) ^ 2)
  st.last = vec3(p.x, p.y, p.z)
  st.maxlat = math.max(st.maxlat, lat)
  local ahead = math.max(8, math.min(40, speed * 1.2))
  local tx, ty = lookahead(st.pts, k, u, ahead)
  local yaw = math.atan2(fwd.y, fwd.x)
  local err = wrap(math.atan2(ty - p.y, tx - p.x) - yaw)
  local bend = math.abs(wrap(heading(st.pts, k, u, 15, 60) - heading(st.pts, k, u, 0, 15)))
  local vt = st.route.speed
  if bend > 0.01 then vt = math.min(vt, math.sqrt(3.0 * 45 / bend)) end
  local steer = 0
  if sign == nil then
    if speed > 4 then
      st.cal = st.cal or {t = 0, yaw = yaw}
      st.cal.t = st.cal.t + dt
      steer = 0.15
      if st.cal.t > 0.8 then
        local dyaw = wrap(yaw - st.cal.yaw)
        sign = dyaw > 0 and 1 or -1
        say('steering +0.15 turned the car %.3f rad in 0.8 s, so the sign is %d', dyaw, sign)
      end
    end
  else
    steer = math.max(-1, math.min(1, GAIN * err * sign))
  end
  local throttle = math.max(0, math.min(1, 0.3 * (vt - speed)))
  local brake = math.max(0, math.min(1, 0.3 * (speed - vt - 1)))
  inputs(v, steer, throttle, brake, 0)
  st.logt = st.logt + dt
  if st.logt >= LOG_EVERY then
    st.logt = 0
    st.dz[#st.dz + 1] = p.z - zr
    lines[#lines + 1] = string.format('  %s t %.1f seg %d at %.1f %.1f %.2f %s %.2f speed %.1f off %.1f',
      st.route.name, st.t, k, p.x, p.y, p.z, st.route.deck and 'deck' or 'street', zr, speed, lat)
  end
  if k >= #st.pts - 1 and u > 0.9 then return finish(true) end
  if lat > OFF_ROUTE then return finish(false, string.format('%.1f m off the route at %.0f %.0f', lat, p.x, p.y)) end
  if st.route.deck and p.z < zr - 4 then return finish(false, string.format('fell to %.1f under the deck at %.1f', p.z, zr)) end
  if st.t > TIMEOUT then return finish(false, 'timeout') end
end

local function stepPark(v, dt)
  st.t = st.t + dt
  local p, vel = v:getPosition(), v:getVelocity()
  inputs(v, 0, 0, 1, 1)
  if st.phase == 'settle' then
    if st.t > SETTLE then st.phase, st.t, st.p0 = 'watch', 0, vec3(p.x, p.y, p.z) end
    return
  end
  st.zmin, st.zmax = math.min(st.zmin, p.z), math.max(st.zmax, p.z)
  st.vmax = math.max(st.vmax, math.sqrt(vel.x ^ 2 + vel.y ^ 2 + vel.z ^ 2))
  if st.t > PARK then
    say('park %s at %.1f %.1f (%.1f m from where it was put): z %.3f to %.3f, moved %.3f m and at most %.3f m/s in %d s standing',
        st.park.name, p.x, p.y, math.sqrt((p.x - st.park.point[1]) ^ 2 + (p.y - st.park.point[2]) ^ 2),
        st.zmin, st.zmax, math.sqrt((p.x - st.p0.x) ^ 2 + (p.y - st.p0.y) ^ 2 + (p.z - st.p0.z) ^ 2),
        st.vmax, PARK)
    flush()
    st = nil
  end
end

local function onWorldReadyState(state)
  if state == 2 and not ready then
    ready = true
    data = jsonReadFile('/brmap_routes.json')
    say('world ready on %s; %d routes, %d parks', tostring(getCurrentLevelIdentifier and getCurrentLevelIdentifier()),
        data and #data.routes or -1, data and #data.parks or -1)
    if extensions.ui_visibility then extensions.ui_visibility.setCef(false) end
  end
end

local wait, calm, held = 0, 0, 0
local function onUpdate(dtReal, dtSim)
  if not ready or not data then return end
  local v = car()
  if not v then return end
  local dt = dtSim or dtReal
  if wait < 6 then wait = wait + dtReal return end
  if st == nil then
    inputs(v, 0, 0, 1, 1)  -- stand still for a second (at most 15 s of trying) before the next teleport
    held = held + dtReal
    local vel = v:getVelocity()
    if math.sqrt(vel.x ^ 2 + vel.y ^ 2 + vel.z ^ 2) > 0.3 and held < 15 then calm = 0 return end
    calm = calm + dtReal
    if calm < 1 and held < 15 then return end
    calm, held = 0, 0
    if ri < #data.routes then
      ri = ri + 1
      startRoute(data.routes[ri])
    elseif pi < #data.parks then
      pi = pi + 1
      startPark(data.parks[pi])
    elseif pi == #data.parks then
      pi = pi + 1
      say('done')
      flush()
    end
    return
  end
  if st.kind == 'route' then stepRoute(v, dt) else stepPark(v, dt) end
end

M.onUpdate = onUpdate
M.onWorldReadyState = onWorldReadyState
return M
