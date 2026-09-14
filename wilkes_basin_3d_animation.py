"""
Wilkes Subglacial Basin, East Antarctica -- animated version of panel (a)
of wilkes_basin_3d_figure.py (the ice-surface scene with MEaSUREs-derived
glacier flow lines draped on top).

Three things are animated:

  1. Ice flow. Instead of static flow-line tubes, short "parcels" are
     released from every flow-line seed (interior/upstream end) and travel
     down-glacier along the traced line, with a gap between successive
     parcels so the eye reads direction and speed rather than a solid line.
     Each parcel is comet-shaped (thick head, thin tail) and coloured by the
     local ice speed on the same log-scaled 20-500 m/yr plasma scale as the
     static figure. Its advance rate is a log mapping of that same speed
     (see parcel_rate()): fast outlet trunks visibly outrun the slow interior
     network, but the interior still moves perceptibly rather than being
     frozen by the ~3 orders of magnitude of real speed contrast.

  2. Camera. Starts on the static figure's overview (compute_camera() from
     the figure script), zooms in to the George V Coast at Mertz Glacier,
     pans east along the coast past Ninnis Glacier to Cook Glacier, then
     zooms back out to the overview. Key frames are eased with a smoothstep
     so every move starts and ends at rest.

  3. Past warm period. From the overview, the ice-sheet geometry morphs to
     the end state of the Pliocene-like PISM run used for panel (b) of
     wilkes_basin_paleo_compare.py (PLIO_end.nc, 5 km grid, resampled
     bilinearly onto the BedMachine grid) and back again. Every pin column's
     top and colour is linearly interpolated between the two states; where
     the simulation has no ice the pin collapses to the sea-level base and
     vanishes, so the Wilkes Subglacial Basin visibly drains of ice. The
     present-day-only overlays (grounding line, coastline, flight lines, flow
     parcels) fade out during the morph and the simulation's own
     grounded/floating extent line fades in, exactly as in the static
     comparison figure. Mid-morph geometry is a visual interpolation, not a
     modelled intermediate state -- the on-screen state label says so.

Everything scientific -- data loading, AOI masking, flow-line tracing, the
hillshaded elevation-coloured pin/hedgehog ice terrain, the overview camera,
the PLIO subset and extent line -- is imported from the two figure scripts,
so the animation always matches them. What differs is purely
presentational and is driven by the zoom range (a ~6x change in camera
distance), which the static figures never have to cope with:

  - Draped vector lines (grounding line, coastline, islands, BEDMAP3 flight
    lines, the faint static flow-line tracks) are rendered as pixel-width
    lines (render_lines_as_tubes) rather than the figure's fixed world-unit
    tubes, which would swell to tens of pixels wide when zoomed in.
  - The pin columns' on-screen width is rescaled every frame from the
    camera distance so the 500 m columns stay just touching at every zoom
    level (PIN_FILL_FACTOR) instead of separating into a visible bristle
    field close up.
  - Parcel tube radii are likewise set per frame in pixel-equivalent units.
  - The flight lines are drawn as a single dark line, without the static
    figure's pale halo: two coincident pixel-width tube lines z-fight into a
    dashed pattern, and the halo is not needed at video resolution.

Run:
    python wilkes_basin_3d_animation.py            # full 1080p render
    python wilkes_basin_3d_animation.py --draft    # quick low-res preview
    python wilkes_basin_3d_animation.py --stills   # key-frame PNGs only
    python wilkes_basin_3d_animation.py --chunk i/N --frames-dir D   # N parallel
        workers writing PNG frames, then:
    python wilkes_basin_3d_animation.py --assemble --frames-dir D

Output: OUT_DIR/wilkes_basin_3d_animation.mp4 (direct H.264 render), or from
--assemble: ..._web.mp4 (H.264, yuv420p) and ..._broadcast.mov (ProRes 422 HQ)
and ..._small.gif (480 px, 10 fps) -- see ENCODE_VARIANTS / GIF_*.
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LightSource
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter, maximum_filter1d, uniform_filter1d

import wilkes_basin_3d_figure as base
import wilkes_basin_paleo_compare as paleo

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUT_DIR = base.OUT_DIR
OUT_BASENAME = "wilkes_basin_3d_animation"

FRAME_W, FRAME_H = 1920, 1080
# Playback speed-up: the TIMELINE / parcel / morph clocks below are defined
# in "design seconds"; the rendered movie runs them PLAYBACK_SPEEDUP times
# faster (1/0.7 -> 30% shorter playtime) at the same standard FPS.
PLAYBACK_SPEEDUP = 1.0 / 0.7
FPS = 30
VIEW_ANGLE_DEG = 32.0  # same as render_panel() in the figure script

# Include the BEDMAP3 flight-line drape (as in the static panel a). They add
# a lot of fine linework that competes with the moving parcels close up;
# switch off for a cleaner flow-only animation.
FLIGHTLINES_ENABLE = True
# Draw the full traced flow-line network as faint static tracks beneath the
# moving parcels, so the routes are visible even in the gaps between parcels.
STATIC_TRACKS_ENABLE = True
STATIC_TRACK_COLOR = "#d0d4da"
STATIC_TRACK_OPACITY = 0.30
STATIC_TRACK_WIDTH_PX = 1.2

# --- parcel animation -------------------------------------------------------
TRACK_RESAMPLE_KM = 2.0     # uniform arc-length spacing tracks are resampled to
MIN_TRACK_KM = 40.0         # drop stubs too short to carry a parcel
PARCEL_LENGTH_KM = 30.0     # arc length of one parcel
LAUNCH_INTERVAL_S = 4.0     # seconds between successive parcels leaving a seed
# Parcel advance rate (km of arc length per second of animation) mapped from
# local ice speed by the same log scale used for colour
# (base.FLOWLINE_SPEED_CLIM_MYR): slowest -> RATE_MIN, fastest -> RATE_MAX.
SPEED_BAR_LABELS_MYR = (20, 100, 250, 500)   # ice-speed colour bar tick labels
RATE_MIN_KM_S = 20.0
RATE_MAX_KM_S = 100.0
# Comet profile, in on-screen pixel radius at the head and tail.
PARCEL_HEAD_PX = 4.0
PARCEL_TAIL_PX = 0.7
PARCEL_TUBE_SIDES = 6
PARCEL_SEED = 12345  # RNG seed for per-track launch phase

# --- terrain / drape line styling (pixels) ----------------------------------
# Pin on-screen width = PIN_FILL_FACTOR * (grid spacing in px); ~1.3 keeps
# adjacent 500 m columns overlapping slightly at every zoom level.
PIN_FILL_FACTOR = 1.3
PIN_WIDTH_MIN_PX = 2.0
GL_WIDTH_PX = 3.0
COAST_WIDTH_PX = 2.5
ISLAND_WIDTH_PX = 2.0
FLIGHT_WIDTH_PX = 1.2
FLIGHT_COLOR = "#2a2a2a"
FLIGHT_OPACITY = 0.8
PLIO_EXTENT_WIDTH_PX = 2.5
# Along-line smoothing window (km) for draped z. The figure's drapes look up
# elevation per vertex; at 500 m vertex spacing and 100x exaggeration the
# sub-km surface roughness becomes a vertical zigzag of tens of km that
# renders as a thick jagged ribbon rather than a line. A running maximum
# (keeps the line clear of the terrain) followed by a running mean (removes
# the steps) over this window gives a smooth upper envelope instead.
DRAPE_SMOOTH_KM = 10.0

# --- camera path ------------------------------------------------------------
# Coastal targets (label, lon, lat) in pan order. Mertz's anchor sits on the
# grounded trunk just upstream of the grounding zone; Ninnis and Cook reuse
# the static figure's label anchors.
COAST_TARGETS = [
    ("Mertz Glacier", 145.0, -67.9),
    ("Ninnis Glacier / Ice Shelf", 147.5, -68.3),
    ("Cook Glacier / Ice Shelf", 152.5, -68.6),
]
CLOSEUP_DIST_KM = 650.0        # camera-to-focal distance when zoomed in
CLOSEUP_ELEVATION_DEG = 32.0   # camera elevation above the focal point
# Pull the close-up focal point inland (toward the pole) by this much so the
# glacier trunk, not just the calving front, sits mid-frame.
CLOSEUP_INLAND_KM = 60.0
# Focal-point height above the draped ice surface (km, exaggerated units).
CLOSEUP_FOCAL_LIFT_KM = 25.0
# Overview camera: compute_camera() from the figure script, but the 16:9
# frame has more horizontal room than the figure's 1.42:1 panel and there is
# no post-render crop-to-content, so the camera sits closer.
OVERVIEW_DIST_FACTOR = 0.72
# Raise the overview focal point (km, exaggerated units) above compute_camera()'s
# bed-median height so the ice sheet sits mid-frame rather than in the top half.
OVERVIEW_FOCAL_LIFT_KM = 70.0

# Timeline as (duration_s, camera_target, morph_end) legs. camera_target is
# "hold" (keep the previous camera), "overview", or a COAST_TARGETS label;
# morph_end is the ice-geometry state at the end of the leg (0 = present
# day, 1 = Pliocene-like simulation), smoothstep-eased across the leg.
TIMELINE = [
    (3.0, "hold", 0.0),                      # overview
    (5.0, "Mertz Glacier", 0.0),
    (3.0, "hold", 0.0),
    (4.0, "Ninnis Glacier / Ice Shelf", 0.0),
    (3.0, "hold", 0.0),
    (4.0, "Cook Glacier / Ice Shelf", 0.0),
    (3.0, "hold", 0.0),
    (5.0, "overview", 0.0),
    (2.0, "hold", 0.0),
    (4.0, "hold", 1.0),                      # morph to Pliocene-like state
    (3.0, "hold", 1.0),
    (4.0, "hold", 0.0),                      # morph back to present day
    (2.5, "hold", 0.0),
]
# Glacier labels are shown only while the camera is closer than this fraction
# of the overview distance (at the overview they would overlap each other).
LABEL_SHOW_FRAC = 0.55
# Sized so glacier labels stay legible in the 640-px GIF (~13 px there).
LABEL_FONT_PX = 40
# Labels sit to the screen-right of their anchor dot; targets whose name starts
# with one of these are placed to the screen-left instead (as seen from that
# target's close-up camera), e.g. so the Cook label does not cover the glacier.
LABEL_LEFT_OF_POINT = ("Cook",)
LABEL_POINT_GAP_PX = 10

TITLE_TEXT = "Wilkes Subglacial Basin"
# Upper-right data/state caption (Present day / Morphing / Pliocene-like);
# off for the clean title-only top row.
SHOW_STATE_TEXT = False
STATE_TEXT_PRESENT = "Present day (BedMachine v3)"
STATE_TEXT_MORPH = "Morphing (visual interpolation only)"
STATE_TEXT_PLIO = "Pliocene-like ice sheet (PISM simulation)"

# ---------------------------------------------------------------------------


def log_speed_frac(speed):
    lo, hi = base.FLOWLINE_SPEED_CLIM_MYR
    s = np.clip(np.asarray(speed, dtype=float), 1e-3, None)
    return np.clip((np.log10(s) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0.0, 1.0)


def parcel_rate(speed):
    return RATE_MIN_KM_S + (RATE_MAX_KM_S - RATE_MIN_KM_S) * log_speed_frac(speed)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def smooth_envelope(z, spacing_km, smooth_km=DRAPE_SMOOTH_KM):
    """Running max then running mean along a line (see DRAPE_SMOOTH_KM)."""
    size = int(round(smooth_km / max(spacing_km, 1e-6)))
    if size < 2 or z.size < 3:
        return z
    return uniform_filter1d(maximum_filter1d(z, size, mode="nearest"), size, mode="nearest")


def drape_lines_polydata(lines, region, elev_m, lift_km=1e-3, local_max_cells=0):
    """Same drape as base.drape_lines_mesh() (nearest-cell elevation, optional
    local-max window, VERT_EXAG, lift) but returned as plain polylines for
    pixel-width line rendering rather than a world-unit tube, with the
    draped z smoothed along the line (smooth_envelope)."""
    if not lines:
        return None
    x = region["x"][0, :]
    y = region["y"][:, 0]
    if local_max_cells > 0:
        elev_lookup = maximum_filter(np.nan_to_num(elev_m, nan=-1e9), size=2 * local_max_cells + 1)
    else:
        elev_lookup = elev_m
    pts, conn, offset = [], [], 0
    for coords in lines:
        if coords.shape[0] < 2:
            continue
        ix = np.clip(np.searchsorted(x, coords[:, 0]), 0, len(x) - 1)
        iy = np.clip(np.searchsorted(-y, -coords[:, 1]), 0, len(y) - 1)
        z_km = elev_lookup[iy, ix] * base.VERT_EXAG / 1000.0 + lift_km
        seg = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1])) / 1000.0
        z_km = smooth_envelope(z_km, float(np.median(seg)) if seg.size else 1.0)
        xyz = np.column_stack([coords[:, 0] / 1000.0, coords[:, 1] / 1000.0, z_km])
        pts.append(xyz)
        n = xyz.shape[0]
        conn.append(np.hstack([[n], np.arange(offset, offset + n)]))
        offset += n
    if not pts:
        return None
    # lines= in the constructor: PolyData(points) alone would also create a
    # vertex cell per point, drawn as 5 px point sprites along every line.
    return pv.PolyData(np.vstack(pts), lines=np.concatenate(conn))


def prepare_tracks(flowlines, region, rng):
    """Turn compute_flowlines() output ((N, 3) x/y/speed polylines) into
    uniformly resampled, draped 3-D tracks with a travel-time
    parameterisation t(s) for the parcel animation."""
    x = region["x"][0, :]
    y = region["y"][:, 0]
    elev_lookup = maximum_filter(
        np.nan_to_num(region["surface"], nan=-1e9), size=2 * base.FLOWLINE_LOCAL_MAX_CELLS + 1
    )
    tracks = []
    for coords in flowlines:
        ix = np.clip(np.searchsorted(x, coords[:, 0]), 0, len(x) - 1)
        iy = np.clip(np.searchsorted(-y, -coords[:, 1]), 0, len(y) - 1)
        z_km = elev_lookup[iy, ix] * base.VERT_EXAG / 1000.0 + base.FLOWLINE_LIFT_KM
        x_km = coords[:, 0] / 1000.0
        y_km = coords[:, 1] / 1000.0
        seg = np.hypot(np.diff(x_km), np.diff(y_km))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < MIN_TRACK_KM:
            continue
        s_u = np.arange(0.0, s[-1], TRACK_RESAMPLE_KM)
        if s_u[-1] < s[-1]:
            s_u = np.append(s_u, s[-1])
        xu = np.interp(s_u, s, x_km)
        yu = np.interp(s_u, s, y_km)
        zu = smooth_envelope(np.interp(s_u, s, z_km), TRACK_RESAMPLE_KM)
        vu = np.interp(s_u, s, coords[:, 2])
        rate = parcel_rate(vu)
        dt = np.diff(s_u) / (0.5 * (rate[1:] + rate[:-1]))
        t = np.concatenate([[0.0], np.cumsum(dt)])
        tracks.append({
            "s": s_u, "t": t, "x": xu, "y": yu, "z": zu, "speed": vu,
            "rate_end": float(rate[-1]),
            "phase": float(rng.uniform(0.0, LAUNCH_INTERVAL_S)),
        })
    return tracks


def build_parcels(tracks, T, km_per_px):
    """PolyData of every parcel visible at animation time T, as comet-shaped
    tubes with point scalars "speed" (colour) and "radius" (km, absolute)."""
    r_head = PARCEL_HEAD_PX * km_per_px
    r_tail = PARCEL_TAIL_PX * km_per_px
    pts, conn, speeds, radii, offset = [], [], [], [], 0
    for tr in tracks:
        s, t = tr["s"], tr["t"]
        s_end, t_end = s[-1], t[-1]
        age_max = t_end + PARCEL_LENGTH_KM / tr["rate_end"]
        k_min = int(np.ceil((T - tr["phase"] - age_max) / LAUNCH_INTERVAL_S))
        k_max = int(np.floor((T - tr["phase"]) / LAUNCH_INTERVAL_S))
        if k_max < k_min:
            continue
        ages = T - tr["phase"] - np.arange(k_min, k_max + 1) * LAUNCH_INTERVAL_S
        for age in ages:
            if age <= t_end:
                head = float(np.interp(age, t, s))
            else:
                head = s_end + tr["rate_end"] * (age - t_end)
            tail = head - PARCEL_LENGTH_KM
            if tail >= s_end or head <= 0.0:
                continue
            lo, hi = max(tail, 0.0), min(head, s_end)
            if hi - lo < 1e-3:
                continue
            inner = s[(s > lo) & (s < hi)]
            sp = np.concatenate([[lo], inner, [hi]])
            xyz = np.column_stack([
                np.interp(sp, s, tr["x"]), np.interp(sp, s, tr["y"]), np.interp(sp, s, tr["z"])
            ])
            w = np.clip((sp - tail) / PARCEL_LENGTH_KM, 0.0, 1.0)
            pts.append(xyz)
            speeds.append(np.interp(sp, s, tr["speed"]))
            radii.append(r_tail + (r_head - r_tail) * w)
            n = sp.size
            conn.append(np.hstack([[n], np.arange(offset, offset + n)]))
            offset += n
    if not pts:
        return None
    poly = pv.PolyData(np.vstack(pts), lines=np.concatenate(conn))
    poly["speed"] = np.concatenate(speeds)
    poly["radius"] = np.concatenate(radii)
    return poly.tube(scalars="radius", absolute=True, n_sides=PARCEL_TUBE_SIDES, capping=True)


def ice_rgb_from_surface(surface_m, dx, dy):
    """Panel-a ice colouring from build_meshes(): 'Blues' by elevation over
    ICE_SURF_ELEV_CLIM_M, hillshaded at VERT_EXAG with HILLSHADE_STRENGTH."""
    lo, hi = base.ICE_SURF_ELEV_CLIM_M
    filled = np.where(np.isnan(surface_m), lo, surface_m)
    rgb = plt.get_cmap("Blues")(np.clip((filled - lo) / (hi - lo), 0.0, 1.0))[..., :3]
    hs = LightSource(azdeg=315, altdeg=45).hillshade(
        np.where(np.isnan(surface_m), 0.0, surface_m), vert_exag=base.VERT_EXAG, dx=dx, dy=dy
    )
    shade = (1.0 - base.HILLSHADE_STRENGTH) + 2.0 * base.HILLSHADE_STRENGTH * hs
    return (np.clip(rgb * shade[..., None], 0.0, 1.0) * 255).astype(np.uint8)


def prepare_morph(region, meshes, substride):
    """Per-pin present-day and Pliocene-like targets (top z in km, RGB) in
    the exact pin order pin_columns_mesh() emits, plus the simulation's
    grounded/floating extent line. PLIO usurf is resampled bilinearly, its
    mask by nearest neighbour, from the 5 km PISM grid onto the BedMachine
    grid; pins where the simulation has no ice collapse to the z=0 base."""
    plio = paleo.load_plio_region()
    py = plio["y"][:, 0]
    px = plio["x"][0, :]
    yo = np.argsort(py)
    usurf_i = RegularGridInterpolator((py[yo], px), plio["surface"][yo, :],
                                      method="linear", bounds_error=False, fill_value=0.0)
    ice_i = RegularGridInterpolator((py[yo], px), np.isin(plio["mask"], [2, 3])[yo, :].astype(np.float64),
                                    method="nearest", bounds_error=False, fill_value=0.0)
    q = np.column_stack([region["y"].ravel(), region["x"].ravel()])
    usurf = usurf_i(q).reshape(region["x"].shape)
    has_ice = ice_i(q).reshape(region["x"].shape) > 0.5
    usurf = np.where(has_ice, usurf, 0.0)

    dx = float(abs(region["x"][0, 1] - region["x"][0, 0]))
    dy = float(abs(region["y"][1, 0] - region["y"][0, 0]))
    rgb_plio = ice_rgb_from_surface(usurf, dx, dy)

    ss = substride
    valid = meshes["ice_bool"][::ss, ::ss]
    top_plio = (usurf * base.VERT_EXAG / 1000.0)[::ss, ::ss][valid]
    pins = meshes["ice_pins"]
    pts = np.asarray(pins.points)
    top_now = pts[1::2, 2].copy()
    rgb_now = np.asarray(pins["pin_rgb"])[1::2].astype(np.float32)
    rgb_plio = rgb_plio[::ss, ::ss][valid].astype(np.float32)
    # Pins that vanish in the simulation keep their present-day colour while
    # they sink, rather than picking up the hillshade of the ice-free field.
    keep = ~has_ice[::ss, ::ss][valid]
    rgb_plio[keep] = rgb_now[keep]
    print(f"Morph targets: {top_now.size} pins; Pliocene-like ice under {has_ice[meshes['ice_bool']].mean() * 100:.0f}% "
          f"of present-day ice pins")

    paleo.VOXEL_SUBSTRIDE = 4  # its pins are discarded; only the extent line is used
    extent = paleo.build_plio_meshes(plio)["plio_extent_line"]
    return {"top_now": top_now, "top_plio": top_plio, "rgb_now": rgb_now, "rgb_plio": rgb_plio,
            "extent_line": extent, "u_applied": 0.0}


def apply_morph(pins_mesh, morph, u):
    if abs(u - morph["u_applied"]) < 1e-4:
        return
    pts = np.array(pins_mesh.points)
    pts[1::2, 2] = morph["top_now"] + (morph["top_plio"] - morph["top_now"]) * u
    pins_mesh.points = pts
    rgb = morph["rgb_now"] + (morph["rgb_plio"] - morph["rgb_now"]) * u
    pins_mesh["pin_rgb"] = np.repeat(np.clip(rgb, 0, 255).astype(np.uint8), 2, axis=0)
    morph["u_applied"] = u


def rotated_outward(x_km, y_km):
    """Horizontal unit vector pointing away from the pole at (x, y), rotated by
    base.CAMERA_AZIMUTH_DEG -- the same convention as base.compute_camera()."""
    v = np.array([x_km, y_km]) / np.hypot(x_km, y_km)
    th = np.deg2rad(base.CAMERA_AZIMUTH_DEG)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return rot @ v


def closeup_camera(lon, lat, region):
    """Camera looking landward at one coastal target: focal point on the ice
    surface a little inland of the anchor, camera seaward at CLOSEUP_DIST_KM."""
    ax, ay, az = base.label_world_xyz(lon, lat, region)
    out = rotated_outward(ax, ay)
    fx, fy = np.array([ax, ay]) - out * CLOSEUP_INLAND_KM
    fz = az + CLOSEUP_FOCAL_LIFT_KM
    el = np.deg2rad(CLOSEUP_ELEVATION_DEG)
    cx, cy = np.array([fx, fy]) + out * CLOSEUP_DIST_KM * np.cos(el)
    cz = fz + CLOSEUP_DIST_KM * np.sin(el)
    return np.array([cx, cy, cz]), np.array([fx, fy, fz])


def build_keyframes(meshes, region):
    """Expand TIMELINE into camera key frames (time_s, cam_pos, focal) and
    morph key frames (time_s, u)."""
    base.CAMERA_DIST_FACTOR = OVERVIEW_DIST_FACTOR
    ov_pos, ov_focal, _ = base.compute_camera(meshes, region)
    ov_pos, ov_focal = np.asarray(ov_pos), np.asarray(ov_focal)
    ov_pos[2] += OVERVIEW_FOCAL_LIFT_KM
    ov_focal[2] += OVERVIEW_FOCAL_LIFT_KM
    cams = {"overview": (ov_pos, ov_focal)}
    for name, lon, lat in COAST_TARGETS:
        cams[name] = closeup_camera(lon, lat, region)

    cam_keys = [(0.0, *cams["overview"])]
    morph_keys = [(0.0, 0.0)]
    t = 0.0
    for dur, target, u_end in TIMELINE:
        t += dur
        if target == "hold":
            cam_keys.append((t, cam_keys[-1][1], cam_keys[-1][2]))
        else:
            cam_keys.append((t, *cams[target]))
        morph_keys.append((t, float(u_end)))
    return cam_keys, morph_keys


def camera_at(keys, T):
    times = [k[0] for k in keys]
    if T <= times[0]:
        return keys[0][1], keys[0][2]
    if T >= times[-1]:
        return keys[-1][1], keys[-1][2]
    i = int(np.searchsorted(times, T, side="right")) - 1
    t0, p0, f0 = keys[i]
    t1, p1, f1 = keys[i + 1]
    u = smoothstep((T - t0) / (t1 - t0))
    return p0 + (p1 - p0) * u, f0 + (f1 - f0) * u


def morph_at(keys, T):
    times = [k[0] for k in keys]
    if T <= times[0]:
        return keys[0][1]
    if T >= times[-1]:
        return keys[-1][1]
    i = int(np.searchsorted(times, T, side="right")) - 1
    t0, u0 = keys[i]
    t1, u1 = keys[i + 1]
    return float(u0 + (u1 - u0) * smoothstep((T - t0) / (t1 - t0)))


def km_per_pixel(pos, focal, frame_w):
    d = float(np.linalg.norm(np.asarray(pos) - np.asarray(focal)))
    return 2.0 * d * np.tan(np.deg2rad(VIEW_ANGLE_DEG / 2.0)) / frame_w


def load_scene(pin_substride):
    region = base.load_region()
    vel = base.load_velocity(region)
    if vel is None:
        sys.exit("Velocity data required for the flow animation.")
    base.VOXEL_SUBSTRIDE = pin_substride
    meshes = base.build_meshes(region)
    meshes["bed_pins"] = None  # not rendered here; free the memory

    gl_lines = base.lines_from_geodataframe(base.load_grounding_line())
    coast_gdf = base.load_moa_coastline()
    coast_lines = base.lines_from_geodataframe(coast_gdf) if not coast_gdf.empty else []
    islands_gdf = base.load_moa_islands()
    island_lines = base.lines_from_geodataframe(islands_gdf) if not islands_gdf.empty else []
    flightlines = base.load_bedmap_flightlines(region) if FLIGHTLINES_ENABLE else []
    flowlines = base.compute_flowlines(region, vel["vx"], vel["vy"])
    print(f"Flow lines traced: {len(flowlines)}; flight-line segments: {len(flightlines)}")

    surf = region["surface"]
    drapes = {
        "gl": drape_lines_polydata(gl_lines, region, surf),
        "coast": drape_lines_polydata(coast_lines, region, surf),
        "island": drape_lines_polydata(island_lines, region, surf),
        "flight": drape_lines_polydata(flightlines, region, surf, lift_km=base.FLIGHTLINE_LIFT_KM,
                                       local_max_cells=base.FLIGHTLINE_LOCAL_MAX_CELLS),
        "tracks": drape_lines_polydata([fl[:, :2] for fl in flowlines], region, surf,
                                       lift_km=base.FLOWLINE_LIFT_KM,
                                       local_max_cells=base.FLOWLINE_LOCAL_MAX_CELLS),
    }
    rng = np.random.default_rng(PARCEL_SEED)
    tracks = prepare_tracks(flowlines, region, rng)
    print(f"Animated tracks: {len(tracks)}")
    morph = prepare_morph(region, meshes, pin_substride)
    return region, meshes, drapes, tracks, morph


def _text_box(actor, opacity=0.55):
    tp = actor.GetTextProperty()
    tp.SetBackgroundColor(0.0, 0.0, 0.0)
    tp.SetBackgroundOpacity(opacity)


def _custom_labels(scalar_bar, values):
    """Tick labels at the given values (placed on the bar's log scale)."""
    import vtk
    arr = vtk.vtkDoubleArray()
    for v in values:
        arr.InsertNextValue(float(v))
    scalar_bar.SetCustomLabels(arr)
    scalar_bar.SetUseCustomLabels(True)


def _bar_box(scalar_bar, opacity=0.55):
    """Dark translucent backing so scalar-bar text stays legible over pale bed."""
    scalar_bar.SetBarRatio(0.3)   # more room for title + tick labels
    for tp in (scalar_bar.GetTitleTextProperty(), scalar_bar.GetLabelTextProperty()):
        tp.SetBackgroundColor(0.0, 0.0, 0.0)
        tp.SetBackgroundOpacity(opacity)


class Scene:
    """Plotter plus the handles update() needs each frame."""

    def __init__(self, meshes, drapes, tracks, morph, region, frame_size, cam_keys):
        self.frame_w = frame_size[0]
        self.tracks = tracks
        self.morph = morph
        self.overview_dist = float(np.linalg.norm(cam_keys[0][1] - cam_keys[0][2]))
        self.state_shown = None

        pl = pv.Plotter(off_screen=True, window_size=list(frame_size))
        pl.set_background("black")
        self.pl = pl

        self.pins_mesh = meshes["ice_pins"]
        self.pins = pl.add_mesh(
            self.pins_mesh, scalars="pin_rgb", rgb=True, show_scalar_bar=False,
            render_lines_as_tubes=True, line_width=PIN_WIDTH_MIN_PX, lighting=False,
        )

        # (actor, full opacity) for every present-day-only overlay that fades
        # out during the Pliocene morph.
        self.fading = []
        for key, color, width in (("gl", "black", GL_WIDTH_PX), ("coast", "#6b4226", COAST_WIDTH_PX),
                                  ("island", "#6b4226", ISLAND_WIDTH_PX)):
            if drapes.get(key) is not None:
                a = pl.add_mesh(drapes[key], color=color, line_width=width,
                                render_lines_as_tubes=True, lighting=False)
                self.fading.append((a, 1.0))
        if drapes.get("flight") is not None:
            a = pl.add_mesh(drapes["flight"], color=FLIGHT_COLOR, opacity=FLIGHT_OPACITY,
                            line_width=FLIGHT_WIDTH_PX, render_lines_as_tubes=True, lighting=False)
            self.fading.append((a, FLIGHT_OPACITY))
        if STATIC_TRACKS_ENABLE and drapes.get("tracks") is not None:
            a = pl.add_mesh(drapes["tracks"], color=STATIC_TRACK_COLOR, opacity=STATIC_TRACK_OPACITY,
                            line_width=STATIC_TRACK_WIDTH_PX, render_lines_as_tubes=True, lighting=False)
            self.fading.append((a, STATIC_TRACK_OPACITY))

        # Parcel actor: one PolyData object updated in place every frame
        # (copy_from), so the actor, its mapper and the scalar bar persist.
        pos0, foc0 = cam_keys[0][1], cam_keys[0][2]
        self.parcel_mesh = build_parcels(tracks, 0.0, km_per_pixel(pos0, foc0, self.frame_w))
        self.parcels = pl.add_mesh(
            self.parcel_mesh, scalars="speed", cmap=base.FLOWLINE_SPEED_CMAP,
            clim=base.FLOWLINE_SPEED_CLIM_MYR, log_scale=True, specular=0.0, name="parcels",
            scalar_bar_args=dict(
                title="Ice speed (m/yr)", vertical=False, position_x=0.03, position_y=0.04,
                width=0.28, height=0.17, title_font_size=36, label_font_size=30, color="white",
                fmt="%.0f", n_labels=4, shadow=False,
            ),
        )
        self.scalar_bar = pl.scalar_bars["Ice speed (m/yr)"]
        _bar_box(self.scalar_bar)
        _custom_labels(self.scalar_bar, SPEED_BAR_LABELS_MYR)
        self.fading.append((self.parcels, 1.0))

        self.plio_extent = None
        if morph["extent_line"].n_points > 0:
            self.plio_extent = pl.add_mesh(morph["extent_line"], color="black", opacity=0.0,
                                           line_width=PLIO_EXTENT_WIDTH_PX, render_lines_as_tubes=True,
                                           lighting=False)
            self.plio_extent.SetVisibility(False)

        self.labels, self.label_points = {}, {}
        for name, lon, lat in COAST_TARGETS:
            xyz = np.asarray(base.label_world_xyz(lon, lat, region))
            anchor = xyz
            if name.startswith(LABEL_LEFT_OF_POINT):
                # Shift the (left-anchored) label a full label width to the
                # screen-left of the dot as seen from this target's close-up.
                cpos, cfoc = closeup_camera(lon, lat, region)
                view = cfoc - cpos
                right = np.cross(view, np.array([0.0, 0.0, 1.0]))
                right /= np.linalg.norm(right)
                km_per_px = (2.0 * np.linalg.norm(xyz - cpos) * np.tan(np.deg2rad(VIEW_ANGLE_DEG / 2))
                             / frame_size[1])
                width_px = 0.52 * LABEL_FONT_PX * max(len(l) for l in name.split("\n")) + 12
                anchor = xyz - right * (width_px + LABEL_POINT_GAP_PX) * km_per_px
            self.labels[name] = pl.add_point_labels(
                anchor[None, :], [name], font_size=LABEL_FONT_PX, text_color="white",
                font_family="arial", italic=True, shape="rounded_rect", shape_color="black",
                shape_opacity=0.55, margin=6, always_visible=True, show_points=False,
                name=f"label_{name}",
            )
            self.label_points[name] = pl.add_points(
                xyz[None, :], color="white", point_size=12, render_points_as_spheres=True,
                name=f"label_{name}-points")

        _text_box(pl.add_text(TITLE_TEXT, position="upper_left", font_size=18, color="white"))
        self.state_text = pl.add_text(STATE_TEXT_PRESENT, position="upper_right", font_size=15,
                                      color="white")
        _text_box(self.state_text)
        self.state_text.SetVisibility(SHOW_STATE_TEXT)
        _text_box(pl.add_text(f"Vertical exaggeration: {base.VERT_EXAG}×", position="lower_right",
                              font_size=13, color="white"))
        pl.camera_position = (tuple(pos0), tuple(foc0), (0.0, 0.0, 1.0))
        pl.camera.view_angle = VIEW_ANGLE_DEG

    def update(self, cam_keys, morph_keys, T, grid_km):
        pl = self.pl
        pos, foc = camera_at(cam_keys, T)
        pl.camera_position = (tuple(pos), tuple(foc), (0.0, 0.0, 1.0))
        pl.camera.view_angle = VIEW_ANGLE_DEG
        kpp = km_per_pixel(pos, foc, self.frame_w)
        self.pins.prop.line_width = max(PIN_WIDTH_MIN_PX, PIN_FILL_FACTOR * grid_km / kpp)

        u = morph_at(morph_keys, T)
        apply_morph(self.pins_mesh, self.morph, u)

        if u < 0.999:
            new = build_parcels(self.tracks, T, kpp)
            if new is not None:
                self.parcel_mesh.copy_from(new)
        for actor, full in self.fading:
            actor.prop.opacity = full * (1.0 - u)
            actor.SetVisibility(u < 0.999)
        self.scalar_bar.SetVisibility(u < 0.5)
        if self.plio_extent is not None:
            self.plio_extent.prop.opacity = u
            self.plio_extent.SetVisibility(u > 0.001)

        if u < 0.02:
            state = STATE_TEXT_PRESENT
        elif u > 0.98:
            state = STATE_TEXT_PLIO
        else:
            state = STATE_TEXT_MORPH
        if SHOW_STATE_TEXT and state != self.state_shown:
            self.state_text.SetText(3, state)  # 3 = upper-right corner slot
            self.state_shown = state

        show = (np.linalg.norm(pos - foc) < LABEL_SHOW_FRAC * self.overview_dist) and u < 0.02
        for name, actor in self.labels.items():
            actor.SetVisibility(bool(show))
            self.label_points[name].SetVisibility(bool(show))
        pl.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="960x540, 12 fps, thinned pins")
    ap.add_argument("--stills", action="store_true", help="render key-frame PNGs only")
    ap.add_argument("--stills-dir", default=None)
    ap.add_argument("--times", default=None, help="comma-separated seconds for --stills")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--substride", type=int, default=None,
                    help="pin thinning factor (default 1, or 2 with --draft)")
    ap.add_argument("--chunk", default=None, metavar="I/N",
                    help="render only frames i::N as PNGs into --frames-dir (parallel workers)")
    ap.add_argument("--frames-dir", default=None,
                    help="PNG frame directory for --chunk / --assemble")
    ap.add_argument("--assemble", action="store_true",
                    help="encode the PNGs in --frames-dir to the MP4 and exit")
    args = ap.parse_args()

    frame_size = (960, 540) if args.draft else (FRAME_W, FRAME_H)
    fps = args.fps or (12 if args.draft else FPS)
    substride = args.substride or (2 if args.draft else 1)
    mp4 = os.path.join(OUT_DIR, OUT_BASENAME + ("_draft" if args.draft else "") + ".mp4")
    frames_dir = args.frames_dir or os.path.join(OUT_DIR, OUT_BASENAME + "_frames")

    if args.assemble:
        assemble_frames(frames_dir, mp4, fps)
        return

    t0 = time.time()
    region, meshes, drapes, tracks, morph = load_scene(substride)
    cam_keys, morph_keys = build_keyframes(meshes, region)
    total_s = cam_keys[-1][0]
    grid_km = float(abs(region["x"][0, 1] - region["x"][0, 0])) / 1000.0 * substride
    print(f"Scene ready in {time.time() - t0:.0f} s; duration {total_s:.0f} s at {fps} fps")

    scene = Scene(meshes, drapes, tracks, morph, region, frame_size, cam_keys)

    if args.stills:
        out = args.stills_dir or os.path.join(OUT_DIR, OUT_BASENAME + "_stills")
        os.makedirs(out, exist_ok=True)
        # End of every leg (overview, each close-up hold, morph end points),
        # plus a mid-pan and a mid-morph frame to check the transitions.
        times = [k[0] for k in cam_keys[1:]]
        times.append(0.5 * (cam_keys[3][0] + cam_keys[4][0]))
        times.append(0.5 * (cam_keys[9][0] + cam_keys[10][0]))
        if args.times:
            times = [float(v) for v in args.times.split(",")]
        for T in sorted(times):
            tf = time.time()
            scene.update(cam_keys, morph_keys, T, grid_km)
            path = os.path.join(out, f"still_t{T:05.1f}s.png")
            scene.pl.screenshot(path)
            print(f"  {path}  ({time.time() - tf:.1f} s)", flush=True)
        scene.pl.close()
        return

    n_frames = int(round(total_s / PLAYBACK_SPEEDUP * fps))
    print(f"Playback speed-up {PLAYBACK_SPEEDUP:.2f}x -> {n_frames / fps:.1f} s, {n_frames} frames")
    if args.chunk:
        # Worker mode: frames i, i+N, i+2N, ... as PNGs (software GL is
        # single-process, so several workers give a near-linear speed-up).
        i, n = (int(v) for v in args.chunk.split("/"))
        os.makedirs(frames_dir, exist_ok=True)
        frames = list(range(i, n_frames, n))
    else:
        frames = list(range(n_frames))
        scene.pl.open_movie(mp4, framerate=fps, quality=8)
    tf = time.time()
    if args.chunk:
        # Resumable: skip frames already rendered by an earlier (interrupted) run.
        frames = [f for f in frames
                  if not os.path.exists(os.path.join(frames_dir, f"frame_{f:05d}.png"))]
    for k, f in enumerate(frames):
        T = f / fps * PLAYBACK_SPEEDUP
        scene.update(cam_keys, morph_keys, T, grid_km)
        if args.chunk:
            scene.pl.screenshot(os.path.join(frames_dir, f"frame_{f:05d}.png"))
        else:
            scene.pl.write_frame()
        if k % 25 == 0 or k == len(frames) - 1:
            el = time.time() - tf
            print(f"  frame {k + 1}/{len(frames)}  t={T:5.1f}s  {el / (k + 1):.2f} s/frame  "
                  f"ETA {el / (k + 1) * (len(frames) - k - 1) / 60:.1f} min", flush=True)
    scene.pl.close()
    if not args.chunk:
        print(f"Wrote {mp4}")


# Output variants written by --assemble, all from the same PNG frames:
#   web:       H.264 High, CRF 20, yuv420p, faststart -- plays in any browser/
#              PowerPoint, ~10-20 MB/min at 1080p.
#   broadcast: Apple ProRes 422 HQ in .mov, 10-bit 4:2:2, ~1 GB/min -- the
#              usual mezzanine/broadcast delivery format; edits without
#              generation loss.
GIF_WIDTH_PX = 640
GIF_FPS = 10
# GIF covers only the present-day fly-through: stop at the start of the
# Pliocene morph (design-timeline seconds; None = full animation), play once.
GIF_END_DESIGN_S = 32.0
GIF_LOOP = -1          # ffmpeg gif muxer: 0 = loop forever, -1 = play once
ENCODE_VARIANTS = [
    ("_web.mp4", ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                  "-crf", "20", "-preset", "slow", "-movflags", "+faststart"]),
    ("_broadcast.mov", ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0",
                        "-pix_fmt", "yuv422p10le", "-qscale:v", "9"]),
]


def assemble_frames(frames_dir, mp4, fps):
    """Encode frame_NNNNN.png (from --chunk workers) into the ENCODE_VARIANTS
    with the ffmpeg binary bundled with imageio-ffmpeg. `mp4` is the base
    output path; its extension is replaced by each variant's suffix."""
    import glob
    import subprocess

    import imageio_ffmpeg

    pngs = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not pngs:
        sys.exit(f"No frames in {frames_dir}")
    idx = [int(os.path.basename(p)[6:11]) for p in pngs]
    missing = sorted(set(range(idx[-1] + 1)) - set(idx))
    if missing:
        sys.exit(f"{len(missing)} frames missing, e.g. {missing[:10]}")
    stem = os.path.splitext(mp4)[0]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for suffix, vargs in ENCODE_VARIANTS:
        out = stem + suffix
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
               "-i", os.path.join(frames_dir, "frame_%05d.png"), *vargs, out]
        subprocess.run(cmd, check=True)
        print(f"Wrote {out} ({len(pngs)} frames at {fps} fps, {os.path.getsize(out) / 1e6:.0f} MB)")

    # Small animated GIF: downscaled, frame-decimated, two-pass palette
    # (palettegen/paletteuse with dithering) so the 256-colour limit does not
    # band the ice/bed colour ramps.
    out = stem + "_small.gif"
    vf = (f"fps={GIF_FPS},scale={GIF_WIDTH_PX}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3")
    trim = [] if GIF_END_DESIGN_S is None else ["-t", f"{GIF_END_DESIGN_S / PLAYBACK_SPEEDUP:.3f}"]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), *trim,
           "-i", os.path.join(frames_dir, "frame_%05d.png"), "-filter_complex", vf, "-loop", str(GIF_LOOP), out]
    subprocess.run(cmd, check=True)
    print(f"Wrote {out} ({GIF_WIDTH_PX} px wide, {GIF_FPS} fps, {os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
