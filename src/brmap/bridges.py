"""Stage 6: give every bridge deck a height profile, with decks that meet joined up.

OSM splits one structure into many bridge ways: a new name, ref or layer, a ramp
leaving the mainline. Pieces with the same name (or kind when unnamed), layer and
direction are first merged end to end (merge_bridges). Decks that then meet end to
end are chained through the joint that continues most nearly straight ahead,
whatever their names or layers, and a deck whose end meets another deck anywhere
else (a ramp joining the mainline, the third arm of a fork) starts at that deck's
height there. Only a chain's free ends come down to the terrain.

A chain's profile is the lowest one that never dips below the straight line
between its two ends (bridges span valleys) and

* stays MARGIN above the terrain away from its free ends (the outer END_FRACTION of
  the end piece sits on the abutment fill of a bare-earth DEM),
* clears a road or a lower deck crossing underneath by `clearance`, and
* reaches a named override's crest (the Mississippi River bridge),

climbing toward each need at no more than DESIGN_GRADE, or as steeply as MAX_GRADE
where an end is too close, and giving up whatever part of a need even MAX_GRADE
cannot reach. Dips between needs shorter than FILL_LENGTH are filled, so a viaduct
over a row of streets stays level, and grade changes are rounded into vertical
curves about SMOOTH_LENGTH long. Joint heights and crossings depend on each other,
so the chains are profiled in rounds until no deck moves by more than SETTLE.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import shapely
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .geo import Extent
from .roads import Road
from .terrain import sample_height

log = logging.getLogger(__name__)

MARGIN = 0.2          # m the deck stays above the terrain (keeps it from z-fighting the ground)
END_FRACTION = 0.1    # share of the end piece at a chain's free end treated as abutment
MAX_GRADE = 0.08      # steepest grade a deck may climb at to reach a need
DESIGN_GRADE = 0.05   # grade a deck climbs at when that still reaches the need
JOIN_TOL = 0.5        # m between ends (or an end and a deck) that count as the same joint
APPROACH_TOL = 2.0    # m: crossings this close to an end are just the approach road
JOINT_REACH = 15.0    # m: decks crossing this close to where one ends on the other are merging
MAX_SHARED = 300.0    # m: longest stretch a ramp follows the deck it leaves before parting
TERRAIN_GAP = 0.3     # m the terrain is cut below a deck's top where it would poke through
SIDE_SLOPE = 1.0      # past a deck's edge the cut terrain may rise 1 m per m (a 45 degree bank)
STEP = 2.0            # m between the points a profile is worked out on
FILL_LENGTH = 400.0   # m: a dip between two needs closer than this is filled level
SMOOTH_LENGTH = 120.0  # m: grade changes are rounded over about this length
MAX_TURN = 60.0       # degrees: the sharpest bend at a joint that still continues a chain
ROUNDS = 6            # most rounds of profiling the chains against each other
SETTLE = 0.05         # m: a round in which no deck moves more than this ends the rounds


def _t(xy: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy[:, :2], axis=0).T))]
    return d / d[-1], float(d[-1])


def _ramp(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _cones(peaks: np.ndarray, drop: float) -> np.ndarray:
    """max over j of peaks[j] - drop * |i - j| (peaks is -inf where nothing is needed)."""
    x = np.arange(len(peaks)) * drop
    ahead = np.maximum.accumulate(peaks + x) - x
    behind = np.maximum.accumulate((peaks - x)[::-1])[::-1] + x
    return np.maximum(ahead, behind)


def _morph(z: np.ndarray, radius: float, ds: float, z0: float, z1: float, close: bool) -> np.ndarray:
    """Running max over +-radius m (and then running min, a closing), the ends padded flat."""
    w = int(round(radius / ds))
    if w < 1:
        return z
    out = maximum_filter1d(np.r_[np.full(w, z0), z, np.full(w, z1)], 2 * w + 1, mode="nearest")
    if close:
        out = minimum_filter1d(out, 2 * w + 1, mode="nearest")
    return out[w:-w]


def _smooth(z: np.ndarray, radius: float, ds: float) -> np.ndarray:
    """Running mean over +-radius m; the window shrinks toward the ends so they stay put."""
    w, n = int(round(radius / ds)), len(z)
    if w < 1 or n < 3:
        return z
    i = np.arange(n)
    k = np.minimum(np.minimum(i, n - 1 - i), w)
    c = np.r_[0.0, np.cumsum(z)]
    return (c[i + k + 1] - c[i - k]) / (2 * k + 1)


def _profile(length: float, z0: float, z1: float, need_s: np.ndarray, need_z: np.ndarray):
    """(s, z) of the lowest deck from z0 to z1 meeting the needs (see the module docstring),
    plus how many needs it gives up in part and the largest shortfall."""
    n = max(2, int(math.ceil(length / STEP)) + 1)
    s = np.linspace(0.0, length, n)
    ds = s[1] - s[0]
    lin = z0 + (z1 - z0) * s / length
    if len(need_s) == 0:
        return s, lin, 0, 0.0
    cap = np.minimum(z0 + MAX_GRADE * s, z1 + MAX_GRADE * (length - s))
    reach = np.minimum(need_z, np.interp(need_s, s, cap))
    peaks = np.full(n, -np.inf)
    np.maximum.at(peaks, np.clip(np.rint(need_s / ds).astype(int), 0, n - 1), reach)
    z = np.maximum(lin, np.minimum(_cones(peaks, DESIGN_GRADE * ds), cap))
    z = np.maximum(lin, np.minimum(_morph(z, FILL_LENGTH / 2, ds, z0, z1, close=True), cap))
    z = np.maximum(lin, np.minimum(_morph(z, SMOOTH_LENGTH / 2, ds, z0, z1, close=False), cap))
    z = _smooth(z, SMOOTH_LENGTH / 2, ds)
    short = need_z - reach
    return s, z, int((short > 0.05).sum()), float(short.max())


def deck_profile(xy, heights: np.ndarray, extent: Extent, clearance: float,
                 peak_override: float | None = None, obstacles=(), label: str = "",
                 end_z=(None, None), abutments=None, needs=()) -> np.ndarray:
    """Deck z at each node.

    obstacles: (t along the span, z of a road or deck passing under), cleared by `clearance`.
    needs: (t, z) heights the deck has to reach, such as an override's crest.
    end_z: the height at each end that joins another deck; None comes down to the terrain.
    abutments: m at each free end where the terrain is no need (END_FRACTION of the span).
    peak_override: a need at mid-span that far above the line between the terrain at the ends.
    label: names the deck in the log line when a need is out of reach ("" logs nothing).
    """
    xy = np.asarray(xy, dtype=float)[:, :2]
    t, length = _t(xy)
    s = t * length
    ground = np.asarray(sample_height(heights, extent, xy[:, 0], xy[:, 1]), dtype=float)
    z0 = ground[0] if end_z[0] is None else float(end_z[0])
    z1 = ground[-1] if end_z[1] is None else float(end_z[1])
    a0, a1 = abutments if abutments is not None else (END_FRACTION * length,) * 2
    a0, a1 = (a0 if end_z[0] is None else 0.0), (a1 if end_z[1] is None else 0.0)
    span = (s >= a0) & (s <= length - a1)
    extra = [(tc * length, zc + clearance) for tc, zc in obstacles if 0.0 < tc < 1.0]
    extra += [(tc * length, zc) for tc, zc in needs if 0.0 <= tc <= 1.0]
    if peak_override is not None:
        extra.append((0.5 * length, 0.5 * (ground[0] + ground[-1]) + float(peak_override)))
    es, ez = np.asarray(extra, dtype=float).reshape(-1, 2).T
    grid_s, grid_z, missed, worst = _profile(length, z0, z1, np.r_[s[span], es],
                                             np.r_[ground[span] + MARGIN, ez])
    if missed and label:
        log.info("deck %s: %d of %d needs out of reach, worst %.1f m short (%.0f m span)",
                 label, missed, int(span.sum()) + len(es), worst, length)
    return np.interp(s, grid_s, grid_z)


def _key(r: Road):
    return ("name", r.name) if r.name else ("kind", r.kind)


def merge_bridges(roads: list[Road]) -> list[Road]:
    """Chain bridge pieces that share a joint (< JOIN_TOL) and a name (or kind when unnamed)."""
    pool = list(roads)
    merged: list[Road] = []
    while pool:
        cur = pool.pop(0)
        nodes, ids = cur.nodes.copy(), list(cur.osm_ids)
        width, maxspeed = cur.width, cur.maxspeed
        grew = True
        while grew:
            grew = False
            for i, other in enumerate(pool):
                if (_key(other) != _key(cur) or other.layer != cur.layer
                        or other.oneway != cur.oneway):
                    continue
                s, e = nodes[0, :2], nodes[-1, :2]
                os_, oe = other.nodes[0, :2], other.nodes[-1, :2]
                two_way = not cur.oneway
                if math.dist(e, os_) < JOIN_TOL:
                    nodes, ids = np.vstack([nodes, other.nodes[1:]]), ids + other.osm_ids
                elif math.dist(oe, s) < JOIN_TOL:
                    nodes, ids = np.vstack([other.nodes, nodes[1:]]), other.osm_ids + ids
                elif two_way and math.dist(e, oe) < JOIN_TOL:
                    nodes, ids = np.vstack([nodes, other.nodes[::-1][1:]]), ids + other.osm_ids
                elif two_way and math.dist(s, os_) < JOIN_TOL:
                    nodes, ids = np.vstack([other.nodes[::-1], nodes[1:]]), other.osm_ids + ids
                else:
                    continue
                width = max(width, other.width)
                maxspeed = other.maxspeed if maxspeed is None else maxspeed
                pool.pop(i)
                grew = True
                break
        merged.append(Road(ids, cur.kind, nodes, width, cur.lanes, cur.oneway, True,
                           cur.layer, cur.name, maxspeed))
    return merged


def _points(geom) -> list[tuple[float, float]]:
    return [tuple(c) for c in shapely.get_coordinates(geom)] if not geom.is_empty else []


def _crossings(line: LineString, other: LineString) -> list[float]:
    """Arc lengths (m) along `line` where `other` passes over or under it: away from the ends
    of `line` (its approaches) and from the ends of `other` (a road ending there joins it)."""
    out = []
    o0, o1 = Point(other.coords[0]), Point(other.coords[-1])
    for x, y in _points(line.intersection(other)):
        p = Point(x, y)
        d = line.project(p)
        if (APPROACH_TOL < d < line.length - APPROACH_TOL
                and p.distance(o0) > APPROACH_TOL and p.distance(o1) > APPROACH_TOL):
            out.append(d)
    return out


def deck_crossings(a: LineString, b: LineString) -> list[tuple[float, float]]:
    """(arc along a, arc along b) where decks a and b pass over or under each other. A deck
    ending on the other is a joint, and so is a crossing within JOINT_REACH of that end
    (a ramp merging at a shallow angle overshoots the mainline's centerline a little)."""
    joints = [p for own, other in ((a, b), (b, a)) for p in (Point(own.coords[0]), Point(own.coords[-1]))
              if other.distance(p) < JOIN_TOL]
    out = []
    for x, y in _points(a.intersection(b)):
        p = Point(x, y)
        sa, sb = float(a.project(p)), float(b.project(p))
        if (min(sa, a.length - sa, sb, b.length - sb) >= APPROACH_TOL
                and all(p.distance(q) >= JOINT_REACH for q in joints)):
            out.append((sa, sb))
    return out


def _away(r: Road, end: int) -> np.ndarray:
    """Unit vector from the joint at `end` (0 or -1) into the deck."""
    a, b = (r.nodes[0, :2], r.nodes[1, :2]) if end == 0 else (r.nodes[-1, :2], r.nodes[-2, :2])
    v = b - a
    return v / max(float(np.hypot(*v)), 1e-9)


def _turn(ra: Road, ea: int, rb: Road, eb: int) -> float:
    """Degrees a driver turns going from one deck through the joint into the other."""
    c = -float(np.dot(_away(ra, ea), _away(rb, eb)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _can_chain(ra: Road, ea: int, rb: Road, eb: int) -> bool:
    """Traffic can flow through: one-way decks need one arriving and one leaving."""
    return not (ra.oneway and rb.oneway) or (ea == -1) != (eb == -1)


def _network(decks: list[Road]):
    """Chains of decks, each a list of (deck, reversed), and for deck ends that meet another
    deck without continuing it: {(deck, end): (other deck, arc length along it)}."""
    lines = [LineString(r.nodes[:, :2]) for r in decks]
    tree = STRtree(lines)
    ends = [(i, e) for i in range(len(decks)) for e in (0, -1)]
    pts = [Point(decks[i].nodes[e, :2]) for i, e in ends]
    parent = list(range(len(ends)))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    end_tree = STRtree(pts)
    for a, p in enumerate(pts):
        for b in end_tree.query(p.buffer(JOIN_TOL)):
            if int(b) != a and p.distance(pts[int(b)]) < JOIN_TOL:
                parent[root(a)] = root(int(b))
    groups: dict[int, list[int]] = {}
    for a in range(len(ends)):
        groups.setdefault(root(a), []).append(a)

    link: dict[tuple[int, int], tuple[int, int]] = {}
    attach: dict[tuple[int, int], tuple[int, float]] = {}
    for members in groups.values():
        p = pts[members[0]]
        here = {ends[a][0] for a in members}
        through = [int(j) for j in tree.query(p.buffer(JOIN_TOL))
                   if int(j) not in here and lines[int(j)].distance(p) < JOIN_TOL]
        if through:  # the joint is on another deck's span: everything here starts at its height
            m = max(through, key=lambda j: lines[j].length)
            for a in members:
                attach[ends[a]] = (m, float(lines[m].project(p)))
            continue
        pairs = sorted((_turn(decks[ends[a][0]], ends[a][1], decks[ends[b][0]], ends[b][1]), a, b)
                       for k, a in enumerate(members) for b in members[k + 1:]
                       if ends[a][0] != ends[b][0]
                       and _can_chain(decks[ends[a][0]], ends[a][1], decks[ends[b][0]], ends[b][1]))
        used: list[int] = []
        for turn, a, b in pairs:
            if turn <= MAX_TURN and a not in used and b not in used:
                link[ends[a]], link[ends[b]] = ends[b], ends[a]
                used += [a, b]
        if used:  # the rest of the ends here start at the height of the first chain through it
            d, e = ends[used[0]]
            for a in members:
                if a not in used:
                    attach[ends[a]] = (d, 0.0 if e == 0 else lines[d].length)

    chains: list[list[tuple[int, bool]]] = []
    seen: set[int] = set()

    def walk(i: int, entry: int) -> list[tuple[int, bool]]:
        seq = []
        while i not in seen:
            seen.add(i)
            seq.append((i, entry == -1))
            nxt = link.get((i, 0 if entry == -1 else -1))
            if nxt is None:
                break
            i, entry = nxt
        return seq

    by_length = sorted(range(len(decks)), key=lambda i: -lines[i].length)
    for i in by_length:
        if i not in seen and ((i, 0) not in link or (i, -1) not in link):
            chains.append(walk(i, 0 if (i, 0) not in link else -1))
    for i in by_length:  # closed loops
        if i not in seen:
            chains.append(walk(i, 0))
    return chains, attach


class _Chain:
    """Decks laid end to end along one polyline, sharing the joint nodes."""

    def __init__(self, decks: list[Road], members: list[tuple[int, bool]], attach: dict):
        self.members = members
        parts, self.spans = [], []
        start = 0
        for k, (i, rev) in enumerate(members):
            xy = decks[i].nodes[::-1, :2] if rev else decks[i].nodes[:, :2]
            parts.append(xy if k == 0 else xy[1:])
            self.spans.append((start, start + len(xy)))
            start += len(xy) - 1
        self.xy = np.vstack(parts)
        t, self.length = _t(self.xy)
        self.s = t * self.length
        self.line = LineString(self.xy)
        (i0, rev0), (i1, rev1) = members[0], members[-1]
        ends = ((i0, -1 if rev0 else 0), (i1, 0 if rev1 else -1))
        self.anchors = [attach.get(e) for e in ends]
        self.abutments = tuple(END_FRACTION * _t(decks[i].nodes)[1] for i, _ in ends)
        self.end_widths = (decks[i0].width, decks[i1].width)
        self.layers = np.array([max(decks[i].layer, 1) for i, _ in members])
        self.span_ends = np.array([self.s[b - 1] for _, b in self.spans])
        self.label = " + ".join(dict.fromkeys(decks[i].name or f"way {decks[i].osm_ids[0]}"
                                              for i, _ in members))

    def layer_at(self, s: float) -> int:
        return int(self.layers[min(int(np.searchsorted(self.span_ends, s)), len(self.layers) - 1)])

    def write(self, decks: list[Road], z: np.ndarray) -> None:
        for (i, rev), (a, b) in zip(self.members, self.spans):
            decks[i].nodes[:, 2] = z[a:b][::-1] if rev else z[a:b]


def _deck_z(r: Road, s: float) -> float | None:
    if np.isnan(r.nodes[:, 2]).any():
        return None
    d = np.r_[0.0, np.cumsum(np.hypot(*np.diff(r.nodes[:, :2], axis=0).T))]
    return float(np.interp(s, d, r.nodes[:, 2]))


def apply_bridges(roads: list[Road], heights: np.ndarray, extent: Extent, cfg) -> list[Road]:
    """Ground roads unchanged, plus the merged bridge decks with z filled in."""
    ground = [r for r in roads if not r.is_bridge]
    decks = merge_bridges([r for r in roads if r.is_bridge])
    if not decks:
        return ground
    members, attach = _network(decks)
    chains = [_Chain(decks, m, attach) for m in members]
    owner = {i: c for c, ch in enumerate(chains) for i, _ in ch.members}
    ground_lines = [LineString(r.nodes[:, :2]) for r in ground]
    gtree = STRtree(ground_lines) if ground_lines else None
    ctree = STRtree([ch.line for ch in chains])
    over_roads, over_decks, crests = [], [], []
    for c, ch in enumerate(chains):
        found = []
        for g in (gtree.query(ch.line) if gtree is not None else []):
            for d in _crossings(ch.line, ground_lines[int(g)]):
                if ground[int(g)].layer < ch.layer_at(d):
                    p = ch.line.interpolate(d)
                    found.append((d / ch.length, float(sample_height(heights, extent, p.x, p.y))))
        over_roads.append(found)
        found = []
        for o in (int(o) for o in ctree.query(ch.line)):
            if o == c:
                continue
            for d, so in deck_crossings(ch.line, chains[o].line):
                if chains[o].layer_at(so) < ch.layer_at(d):
                    found.append((d / ch.length, o, so))
        over_decks.append(found)
        found = []
        for (i, _), (a, b) in zip(ch.members, ch.spans):
            rise = cfg.bridge_overrides.get(decks[i].name) if decks[i].name else None
            if rise is not None:
                g = sample_height(heights, extent, decks[i].nodes[[0, -1], 0], decks[i].nodes[[0, -1], 1])
                found.append((0.5 * (ch.s[a] + ch.s[b - 1]) / ch.length, float(np.mean(g)) + rise))
        crests.append(found)

    z: list[np.ndarray | None] = [None] * len(chains)

    def shared(c: int, end: int) -> tuple[list[int], list[float]]:
        """Nodes from chain c's joined `end` (0 or 1) whose centerline still runs on the
        pavement of the deck it joins (or that deck's on this one's), with that deck's z under
        each: a ramp diverging at a shallow angle stays level with the mainline until the two
        part, as long as the rest of it can still reach its other end within MAX_GRADE."""
        ch, a = chains[c], chains[c].anchors[end]
        if a is None or owner[a[0]] == c or z[owner[a[0]]] is None:
            return [], []
        other, zo = chains[owner[a[0]]], z[owner[a[0]]]
        reach = max(ch.end_widths[end], decks[a[0]].width) / 2.0
        far = len(ch.xy) - 1 if end == 0 else 0
        b = ch.anchors[1 - end]
        z_far = (_deck_z(decks[b[0]], b[1]) if b is not None and owner[b[0]] != c else None)
        if z_far is None:
            z_far = float(sample_height(heights, extent, *ch.xy[far]))
        idx, zs = [], []
        for i in (range(len(ch.xy)) if end == 0 else range(len(ch.xy) - 1, -1, -1)):
            p = Point(ch.xy[i])
            if idx and (other.line.distance(p) >= reach or abs(ch.s[i] - ch.s[-end]) > MAX_SHARED):
                break
            zi = float(np.interp(other.line.project(p), other.s, zo))
            if idx and abs(zi - z_far) > MAX_GRADE * abs(ch.s[far] - ch.s[i]):
                break
            idx.append(i)
            zs.append(zi)
        return idx, zs

    def profile(c: int, label: str) -> np.ndarray:
        ch = chains[c]
        out = np.full(len(ch.xy), np.nan)
        ends = [shared(c, 0), shared(c, 1)]
        i0 = ends[0][0][-1] if ends[0][0] else 0
        i1 = ends[1][0][-1] if ends[1][0] else len(ch.xy) - 1
        for idx, zs in ends:
            out[idx] = zs
        if i1 - i0 < 1:  # all of it runs on the decks it joins
            return out
        end_z = [zs[-1] if zs else (None if a is None or owner[a[0]] == c else _deck_z(decks[a[0]], a[1]))
                 for (_, zs), a in zip(ends, ch.anchors)]
        s0, s1 = ch.s[i0], ch.s[i1]

        def part(items):  # (t along the chain, ...) -> (t along nodes i0..i1, ...)
            return [((t * ch.length - s0) / (s1 - s0), *rest) for t, *rest in items
                    if s0 < t * ch.length < s1]

        under = [(t, float(np.interp(so, chains[o].s, z[o]))) for t, o, so in over_decks[c]
                 if z[o] is not None]
        out[i0:i1 + 1] = deck_profile(ch.xy[i0:i1 + 1], heights, extent, cfg.clearance_m,
                                      obstacles=part(over_roads[c] + under), label=label,
                                      end_z=tuple(end_z), abutments=ch.abutments,
                                      needs=part(crests[c]))
        return out

    order = sorted(range(len(chains)), key=lambda c: (not crests[c], -chains[c].length))
    rounds, moved = 0, math.inf
    while rounds < ROUNDS and moved >= SETTLE:
        rounds, moved = rounds + 1, 0.0
        for c in order:
            new = profile(c, "")
            moved = max(moved, math.inf if z[c] is None else float(np.abs(new - z[c]).max()))
            z[c] = new
            chains[c].write(decks, new)
    if moved >= SETTLE:
        log.warning("bridge decks still moved %.2f m in round %d", moved, rounds)
    for c in order:  # a last round that also logs the needs out of reach
        z[c] = profile(c, chains[c].label)
        chains[c].write(decks, z[c])
    joined = sum(len(ch.members) - 1 for ch in chains)
    log.info("%d decks in %d chains (%d joints run straight through, %d deck ends start on "
             "another deck); settled in %d rounds", len(decks), len(chains), joined,
             sum(a is not None for ch in chains for a in ch.anchors), rounds)
    for c in order:
        if crests[c]:
            log.info("%s: override crest %.1f m, deck top %.1f m", chains[c].label,
                     max(zc for _, zc in crests[c]), float(z[c].max()))
    return ground + decks


def clear_under_decks(heights: np.ndarray, extent: Extent, roads) -> tuple[np.ndarray, dict]:
    """Copy of heights cut to TERRAIN_GAP below every deck the terrain would poke through.

    Cells whose centers lie within a deck's half width plus 1.5 cells (flat ends) go
    down to deck - TERRAIN_GAP * taper, the taper easing from 0 at the deck's ends to 1
    at END_FRACTION in, so a deck still meets the ground where it starts. Past the
    deck's edge the limit rises SIDE_SLOPE per metre, so banks beside it stay. Returns
    (heights, {deck label: deepest cut in m}) for the decks that needed a cut.
    """
    out = np.array(heights, dtype=np.float32, copy=True)
    n = out.shape[0]
    px, half = extent.pixel_size(n), extent.half_size
    cuts: dict[str, float] = {}
    for r in roads:
        if not r.is_bridge or len(r.nodes) < 2 or np.isnan(r.nodes[:, 2]).any():
            continue
        line = LineString(r.nodes[:, :2])
        if line.length <= 0:
            continue
        foot = line.buffer(r.width / 2.0 + 1.5 * px, cap_style="flat")
        minx, miny, maxx, maxy = foot.bounds
        c0, c1 = max(0, int((minx + half) / px)), min(n, math.ceil((maxx + half) / px) + 1)
        r0, r1 = max(0, int((half - maxy) / px)), min(n, math.ceil((half - miny) / px) + 1)
        if c1 <= c0 or r1 <= r0:
            continue
        X, Y = np.meshgrid(-half + (np.arange(c0, c1) + 0.5) * px, half - (np.arange(r0, r1) + 0.5) * px)
        inside = shapely.contains_xy(foot, X, Y)
        if not inside.any():
            continue
        pts = shapely.points(X[inside], Y[inside])
        t = shapely.line_locate_point(line, pts, normalized=True)
        deck = np.interp(t, _t(r.nodes)[0], r.nodes[:, 2])
        beyond_edge = np.maximum(shapely.distance(line, pts) - r.width / 2.0, 0.0)
        target = (deck - TERRAIN_GAP * _ramp(np.minimum(t, 1.0 - t) / END_FRACTION)
                  + SIDE_SLOPE * beyond_edge)
        win = out[r0:r1, c0:c1]
        cur = win[inside]
        low = cur > target
        if low.any():
            label = r.name or f"way {r.osm_ids[0]}"
            cuts[label] = max(cuts.get(label, 0.0), float((cur - target)[low].max()))
            cur[low] = target[low]
            win[inside] = cur
    return out, cuts
