"""
Wilkes Subglacial Basin -- "lift the ice off the bed" animation (MP4).

Second, simpler companion to wilkes_basin_3d_animation.py. The ice sheet of
figure panel (a) sits on the bedrock of panel (b) -- ice pins run from the
ice base (BedMachine surface - thickness, i.e. the bed where grounded) up to
the surface, bed pins from a flat floor up to the bed -- and the camera:

  1. holds the overview, zooms half way in towards the Ninnis ice front at
     low elevation, holds, zooms back out;
  2. lifts the whole ice sheet vertically (rigid translation of the ice
     actors) well clear of the bed to reveal the underlying bedrock, holds,
     then fades the ice out;
  3. zooms into the bed (basin centre, data-driven: centroid of grounded bed
     below WSB_BASIN_BED_M inside the AOI) and labels
     "Wilkes Subglacial Basin". No title/state text or glacier labels.

Everything that can be reused from the two figure scripts and the first
animation is imported (data loading, pin meshes, colour scales, drapes,
camera helpers, chunked parallel rendering).

Usage:
  python3 wilkes_basin_bed_reveal_animation.py            # 1080p30 MP4
  python3 wilkes_basin_bed_reveal_animation.py --draft    # 960x540, 12 fps, thinned pins
  python3 wilkes_basin_bed_reveal_animation.py --stills   # key-frame PNGs only
  python3 ... --chunk i/N --frames-dir D   (N workers)  then  --assemble --frames-dir D
  --assemble writes <name>_web.mp4 (H.264) and <name>_broadcast.mov (ProRes 422 HQ)
"""

import argparse
import os
import sys
import time

import numpy as np
import pyvista as pv

import wilkes_basin_3d_animation as anim
import wilkes_basin_3d_figure as base

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASENAME = "wilkes_basin_bed_reveal_animation"

FRAME_W, FRAME_H = anim.FRAME_W, anim.FRAME_H
FPS = anim.FPS
PLAYBACK_SPEEDUP = anim.PLAYBACK_SPEEDUP  # same 30% shorter playtime
anim.GIF_END_DESIGN_S = None  # this GIF keeps the whole timeline (bed reveal is the point)

# --- ice front close-up: Ninnis, a half zoom from the overview at low elevation
FRONT_TARGET_LONLAT = (147.5, -68.3)   # Ninnis Glacier / Ice Shelf
FRONT_DIST_KM = 750.0          # full close-up distance; actual = half way from overview
FRONT_ZOOM_FRAC = 0.5          # 0 = stay at overview distance, 1 = FRONT_DIST_KM
FRONT_ELEVATION_DEG = 20.0
FRONT_INLAND_KM = 80.0
FRONT_FOCAL_LIFT_KM = 25.0

# --- ice lift ---------------------------------------------------------------
ICE_LIFT_CLEARANCE_KM = 350.0  # gap between lifted ice base and the highest bed under ice
ICE_LIFT_BED_PCTL = 99.5       # ...measured at this percentile of sub-ice bed (ignore isolated peaks)
LIFT_CAMERA_PULLBACK = 1.6     # overview camera distance multiplier while the ice is lifted

# --- bed close-up -----------------------------------------------------------
WSB_BASIN_BED_M = -400.0       # grounded bed below this defines the basin floor for the centroid
WSB_LABEL_TEXT = "Wilkes Subglacial\nBasin"   # two lines, centred on the basin centroid
WSB_LABEL_SHIFT_PX = 380.0     # then shifted screen-left (px at 1080p) onto the below-sea-level floor
BED_DIST_KM = 2300.0
BED_ELEVATION_DEG = 50.0
BED_FOCAL_LIFT_KM = 30.0
SEA_LEVEL_WIDTH_PX = 2.0
IBCSO_POINT_PX = 1.5
IBCSO_OPACITY = 0.5
# Ice flow before the lift is animated as in the merged (intro) animation:
# velocity parcels over faint static tracks; the parcels freeze at lift-off
# and ride up with the ice. These parcel settings are applied by
# build_scene_data() only (the intro script sets its own).
PARCEL_STYLE = dict(seed_spacing_div=np.sqrt(2.0), length_km=90.0, head_px=2.25, tail_px=1.125)

# --- timeline: (duration_s, camera target, ice lift fraction, ice opacity) --
TIMELINE = [
    (3.0, "hold", 0.0, 1.0),
    (5.0, "front", 0.0, 1.0),
    (3.0, "hold", 0.0, 1.0),
    (5.0, "overview", 0.0, 1.0),
    (2.0, "hold", 0.0, 1.0),
    (6.0, "lifted", 1.0, 1.0),
    (3.0, "hold", 1.0, 1.0),
    (1.0, "hold", 1.0, 0.0),     # ice sheet cut away (ICE_CUT_S) before the bed zoom
    (6.0, "bed", 1.0, 0.0),
    (5.0, "hold", 1.0, 0.0),
]

# Ice opacity changes are a hard cut at the start of the leg, not a fade:
# semi-transparent pin tubes render as a featureless fog on llvmpipe.
ICE_CUT_S = 1e-3

# Title (anim.TITLE_TEXT) top-left, scalar bars, vertical-exaggeration note
# and the basin label are drawn; no state caption.


def ice_base_z_km(region, bed_floor_km):
    """Ice-base elevation (km, exaggerated): surface - thickness where known,
    else the bed; never below the bed-pin floor."""
    thk = region["thickness"]
    surf = region["surface"]
    base_m = np.where(np.isfinite(thk) & np.isfinite(surf), surf - thk, region["bed"])
    base_m = np.where(np.isfinite(base_m), base_m, region["bed"])
    z = base_m * base.VERT_EXAG / 1000.0
    return np.maximum(np.nan_to_num(z, nan=bed_floor_km), bed_floor_km)


def basin_centroid(region):
    """Centroid (km) of grounded bed below WSB_BASIN_BED_M inside the AOI,
    weighted by depth below that threshold -- the basin floor."""
    sel = region["inbox"] & (region["mask"] == 2) & (region["bed"] < WSB_BASIN_BED_M)
    w = (WSB_BASIN_BED_M - region["bed"])[sel]
    x = region["x"][sel] / 1000.0
    y = region["y"][sel] / 1000.0
    cx, cy = np.average(x, weights=w), np.average(y, weights=w)
    ix = int(np.argmin(np.abs(region["x"][0, :] / 1000.0 - cx)))
    iy = int(np.argmin(np.abs(region["y"][:, 0] / 1000.0 - cy)))
    cz = float(region["bed"][iy, ix]) * base.VERT_EXAG / 1000.0
    return np.array([cx, cy, cz])


def wsb_label_anchor(centre, cam_pos, cam_foc, font_px, frame_h_px):
    """World anchor for the (left/bottom-anchored) basin label so that the
    text block is centred on `centre` as seen from the given camera: shift
    half a label width screen-left and half a height screen-down."""
    view = np.asarray(cam_foc) - np.asarray(cam_pos)
    right = np.cross(view, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)
    km_per_px = (2.0 * np.linalg.norm(view) * np.tan(np.deg2rad(anim.VIEW_ANGLE_DEG / 2))
                 / frame_h_px)
    lines = WSB_LABEL_TEXT.split("\n")
    half_w_px = 0.5 * (0.52 * font_px * max(len(l) for l in lines) + 12)
    half_h_px = 0.5 * (1.15 * font_px * len(lines) + 12)
    shift_px = WSB_LABEL_SHIFT_PX * frame_h_px / 1080.0
    return (np.asarray(centre) + np.array([0.0, 0.0, 5.0])
            - right * (half_w_px + shift_px) * km_per_px - up * half_h_px * km_per_px)


def looking_landward(anchor_xyz, inland_km, dist_km, elev_deg, focal_lift_km):
    out = anim.rotated_outward(anchor_xyz[0], anchor_xyz[1])
    fx, fy = np.array(anchor_xyz[:2]) - out * inland_km
    fz = anchor_xyz[2] + focal_lift_km
    el = np.deg2rad(elev_deg)
    cx, cy = np.array([fx, fy]) + out * dist_km * np.cos(el)
    cz = fz + dist_km * np.sin(el)
    return np.array([cx, cy, cz]), np.array([fx, fy, fz])


def configure_parcels():
    base.FLOWLINE_SEED_SPACING_KM /= PARCEL_STYLE["seed_spacing_div"]
    anim.PARCEL_LENGTH_KM = PARCEL_STYLE["length_km"]
    anim.PARCEL_HEAD_PX = PARCEL_STYLE["head_px"]
    anim.PARCEL_TAIL_PX = PARCEL_STYLE["tail_px"]


def build_scene_data(substride):
    configure_parcels()
    region = base.load_region()
    vel = base.load_velocity(region)
    base.VOXEL_SUBSTRIDE = substride
    meshes = base.build_meshes(region)
    inbox = region["inbox"]
    bed_z = meshes["bed_z"]
    bed_floor_km = float(np.nanmin(bed_z[inbox])) - 0.05 * (
        float(np.nanmax(bed_z[inbox])) - float(np.nanmin(bed_z[inbox]))
    )
    # Re-base the ice pins from z=0 (figure panel a) onto the ice base.
    ice_base = ice_base_z_km(region, bed_floor_km)
    ice_pins = meshes["ice_pins"]
    ib = ice_base[::substride, ::substride][meshes["ice_bool"][::substride, ::substride]]
    pts = ice_pins.points.copy()
    pts[0::2, 2] = ib
    ice_pins.points = pts

    # Lift so the ice base clears (almost) all bed beneath it.
    sub_ice = meshes["ice_bool"] & inbox
    bed_under = bed_z[sub_ice]
    base_under = ice_base[sub_ice]
    lift_km = float(np.nanpercentile(bed_under, ICE_LIFT_BED_PCTL) - np.nanmin(base_under)
                    + ICE_LIFT_CLEARANCE_KM)
    print(f"Ice lift: {lift_km:.0f} km (exaggerated units)")

    gl_lines = base.lines_from_geodataframe(base.load_grounding_line())
    coast_gdf = base.load_moa_coastline()
    coast_lines = base.lines_from_geodataframe(coast_gdf) if not coast_gdf.empty else []
    islands_gdf = base.load_moa_islands()
    island_lines = base.lines_from_geodataframe(islands_gdf) if not islands_gdf.empty else []
    flightlines = base.load_bedmap_flightlines(region) if anim.FLIGHTLINES_ENABLE else []
    flowlines = base.compute_flowlines(region, vel["vx"], vel["vy"]) if vel is not None else []
    surf, bed = region["surface"], region["bed"]
    dl = anim.drape_lines_polydata
    drapes = {
        "gl_ice": dl(gl_lines, region, surf),
        "coast_ice": dl(coast_lines, region, surf),
        "island_ice": dl(island_lines, region, surf),
        "flight": dl(flightlines, region, surf, lift_km=base.FLIGHTLINE_LIFT_KM,
                     local_max_cells=base.FLIGHTLINE_LOCAL_MAX_CELLS),
        "gl_bed": dl(gl_lines, region, bed, lift_km=base.BED_LINE_LIFT_KM,
                     local_max_cells=base.BED_LINE_LOCAL_MAX_CELLS),
        "coast_bed": dl(coast_lines, region, bed, lift_km=base.BED_LINE_LIFT_KM,
                        local_max_cells=base.BED_LINE_LOCAL_MAX_CELLS),
        "island_bed": dl(island_lines, region, bed, lift_km=base.BED_LINE_LIFT_KM,
                         local_max_cells=base.BED_LINE_LOCAL_MAX_CELLS),
    }
    # Animated velocity parcels over faint static tracks (as in animation 1 /
    # the intro).
    drapes["tracks"] = anim.drape_lines_polydata(
        [fl[:, :2] for fl in flowlines], region, surf, lift_km=base.FLOWLINE_LIFT_KM,
        local_max_cells=base.FLOWLINE_LOCAL_MAX_CELLS) if flowlines else None
    tracks = anim.prepare_tracks(flowlines, region, np.random.default_rng(anim.PARCEL_SEED))
    print(f"Animated tracks: {len(tracks)}")

    if base.IBCSO_ENABLE:
        ix, iy, iz = base.load_ibcso_direct_points(region)
        drapes["ibcso"] = base.ibcso_points_mesh(ix, iy, iz) if ix.size else None
    else:
        drapes["ibcso"] = None
    print(f"Flow lines: {len(flowlines)}; flight-line segments: {len(flightlines)}")
    return region, meshes, drapes, lift_km, tracks


def build_keyframes(meshes, region, lift_km):
    base.CAMERA_DIST_FACTOR = anim.OVERVIEW_DIST_FACTOR
    ov_pos, ov_focal, _ = base.compute_camera(meshes, region)
    ov_pos, ov_focal = np.asarray(ov_pos), np.asarray(ov_focal)
    ov_pos[2] += anim.OVERVIEW_FOCAL_LIFT_KM
    ov_focal[2] += anim.OVERVIEW_FOCAL_LIFT_KM

    # While lifted: focal point halfway up the lift, camera pulled back.
    lf_focal = ov_focal + np.array([0.0, 0.0, 0.5 * lift_km])
    lf_pos = lf_focal + (ov_pos - ov_focal) * LIFT_CAMERA_PULLBACK

    fx, fy, fz = base.label_world_xyz(*FRONT_TARGET_LONLAT, region)
    ov_dist = float(np.linalg.norm(ov_pos - ov_focal))
    front_dist = ov_dist + FRONT_ZOOM_FRAC * (FRONT_DIST_KM - ov_dist)
    front = looking_landward((fx, fy, fz), FRONT_INLAND_KM, front_dist,
                             FRONT_ELEVATION_DEG, FRONT_FOCAL_LIFT_KM)
    centre = basin_centroid(region)
    print(f"Basin centroid (km): {centre[0]:.0f}, {centre[1]:.0f}, bed z {centre[2]:.0f}")
    bed_cam = looking_landward(centre, 0.0, BED_DIST_KM, BED_ELEVATION_DEG, BED_FOCAL_LIFT_KM)

    cams = {"overview": (ov_pos, ov_focal), "front": front, "lifted": (lf_pos, lf_focal),
            "bed": bed_cam}
    cam_keys = [(0.0, *cams["overview"])]
    lift_keys = [(0.0, 0.0)]
    opac_keys = [(0.0, 1.0)]
    t = 0.0
    for dur, target, lift, opac in TIMELINE:
        t += dur
        if target == "hold":
            cam_keys.append((t, cam_keys[-1][1], cam_keys[-1][2]))
        else:
            cam_keys.append((t, *cams[target]))
        lift_keys.append((t, float(lift)))
        if float(opac) != opac_keys[-1][1]:
            opac_keys.append((t - dur + ICE_CUT_S, float(opac)))
        opac_keys.append((t, float(opac)))
    return cam_keys, lift_keys, opac_keys, centre


class Scene:
    def __init__(self, meshes, drapes, tracks, region, frame_size, cam_keys, centre):
        self.frame_w = frame_size[0]
        pl = pv.Plotter(off_screen=True, window_size=frame_size)
        pl.set_background("black")
        self.pl = pl

        # --- bed (figure panel b styling) ---
        self.bed_pins = pl.add_mesh(
            meshes["bed_pins"], scalars="pin_scalar", cmap=base.BATHY_CMAP,
            clim=(-base.BED_ELEV_SATURATION_M, base.BED_ELEV_SATURATION_M),
            render_lines_as_tubes=True, line_width=anim.PIN_WIDTH_MIN_PX, lighting=False,
            scalar_bar_args=dict(
                title="Bed elevation (m)", vertical=False, position_x=0.69, position_y=0.04,
                width=0.28, height=0.17, title_font_size=36, label_font_size=30, color="white",
                fmt="%.0f", n_labels=3, shadow=False,
            ),
        )
        anim._bar_box(pl.scalar_bars["Bed elevation (m)"])
        if meshes["sea_level_line"].n_points > 0:
            pl.add_mesh(meshes["sea_level_line"], color="#00e5ff", line_width=SEA_LEVEL_WIDTH_PX,
                        render_lines_as_tubes=True, lighting=False)
        for key, color, width in (("gl_bed", "black", anim.GL_WIDTH_PX),
                                  ("coast_bed", "#6b4226", anim.COAST_WIDTH_PX),
                                  ("island_bed", "#6b4226", anim.ISLAND_WIDTH_PX)):
            if drapes.get(key) is not None:
                pl.add_mesh(drapes[key], color=color, line_width=width,
                            render_lines_as_tubes=True, lighting=False)
        if drapes.get("ibcso") is not None:
            pl.add_mesh(drapes["ibcso"], color="#ff6f00", point_size=IBCSO_POINT_PX,
                        render_points_as_spheres=True, opacity=IBCSO_OPACITY)

        # --- ice (figure panel a styling); all actors kept for the lift ---
        self.ice_actors = []
        self.ice_pins = pl.add_mesh(
            meshes["ice_pins"], scalars="pin_rgb", rgb=True, render_lines_as_tubes=True,
            line_width=anim.PIN_WIDTH_MIN_PX, lighting=False,
        )
        self.ice_actors.append((self.ice_pins, 1.0))
        for key, color, width in (("gl_ice", "black", anim.GL_WIDTH_PX),
                                  ("coast_ice", "#6b4226", anim.COAST_WIDTH_PX),
                                  ("island_ice", "#6b4226", anim.ISLAND_WIDTH_PX)):
            if drapes.get(key) is not None:
                a = pl.add_mesh(drapes[key], color=color, line_width=width,
                                render_lines_as_tubes=True, lighting=False)
                self.ice_actors.append((a, 1.0))
        if drapes.get("flight") is not None:
            a = pl.add_mesh(drapes["flight"], color=anim.FLIGHT_COLOR, opacity=anim.FLIGHT_OPACITY,
                            line_width=anim.FLIGHT_WIDTH_PX, render_lines_as_tubes=True, lighting=False)
            self.ice_actors.append((a, anim.FLIGHT_OPACITY))
        if anim.STATIC_TRACKS_ENABLE and drapes.get("tracks") is not None:
            a = pl.add_mesh(drapes["tracks"], color=anim.STATIC_TRACK_COLOR,
                            opacity=anim.STATIC_TRACK_OPACITY, line_width=anim.STATIC_TRACK_WIDTH_PX,
                            render_lines_as_tubes=True, lighting=False)
            self.ice_actors.append((a, anim.STATIC_TRACK_OPACITY))
        # Parcel actor: one PolyData updated in place every frame (copy_from).
        self.tracks = tracks
        pos0, foc0 = cam_keys[0][1], cam_keys[0][2]
        self.parcel_mesh = anim.build_parcels(tracks, 0.0, anim.km_per_pixel(pos0, foc0, self.frame_w))
        self.parcels = pl.add_mesh(
            self.parcel_mesh, scalars="speed", cmap=base.FLOWLINE_SPEED_CMAP,
            clim=base.FLOWLINE_SPEED_CLIM_MYR, log_scale=True, specular=0.0, name="parcels",
            scalar_bar_args=dict(
                title="Ice speed (m/yr)", vertical=False, position_x=0.03, position_y=0.04,
                width=0.28, height=0.17, title_font_size=36, label_font_size=30, color="white",
                fmt="%.0f", n_labels=4, shadow=False,
            ),
        )
        self.ice_actors.append((self.parcels, 1.0))
        self.ice_bar = pl.scalar_bars["Ice speed (m/yr)"]
        anim._bar_box(self.ice_bar)
        anim._custom_labels(self.ice_bar, anim.SPEED_BAR_LABELS_MYR)

        # Point labels are left/bottom-anchored (centre justification misplaces
        # the backing shape), so offset the anchor so the text block is centred
        # on the basin as seen from the final bed camera.
        wsb_font = anim.LABEL_FONT_PX + 8
        anchor = wsb_label_anchor(centre, cam_keys[-1][1], cam_keys[-1][2], wsb_font, frame_size[1])
        # The anchor is re-derived from the current camera every frame (update()).
        self.wsb_centre, self.wsb_font, self.frame_h = centre, wsb_font, frame_size[1]
        self.wsb_pts = pv.PolyData(anchor[None, :])
        self.wsb_pts["labels"] = [WSB_LABEL_TEXT]
        self.wsb_label = pl.add_point_labels(
            self.wsb_pts, "labels",
            font_size=wsb_font, text_color="white", font_family="arial",
            italic=True, shape="rounded_rect", shape_color="black", shape_opacity=0.55,
            margin=6, show_points=False, always_visible=True, name="label_wsb",
        )
        self.wsb_label.SetVisibility(False)

        anim._text_box(pl.add_text(anim.TITLE_TEXT, position="upper_left", font_size=18, color="white"))
        anim._text_box(pl.add_text(f"Vertical exaggeration: {base.VERT_EXAG}×",
                                   position="lower_edge", font_size=13, color="white"))

        pos0, foc0 = cam_keys[0][1], cam_keys[0][2]
        self.overview_dist = float(np.linalg.norm(pos0 - foc0))
        pl.camera_position = (tuple(pos0), tuple(foc0), (0.0, 0.0, 1.0))
        pl.camera.view_angle = anim.VIEW_ANGLE_DEG

    def update(self, cam_keys, lift_keys, opac_keys, T, grid_km, lift_km):
        pl = self.pl
        pos, foc = anim.camera_at(cam_keys, T)
        pl.camera_position = (tuple(pos), tuple(foc), (0.0, 0.0, 1.0))
        pl.camera.view_angle = anim.VIEW_ANGLE_DEG
        kpp = anim.km_per_pixel(pos, foc, self.frame_w)
        w = max(anim.PIN_WIDTH_MIN_PX, anim.PIN_FILL_FACTOR * grid_km / kpp)
        self.bed_pins.prop.line_width = w
        self.ice_pins.prop.line_width = w

        lift = anim.morph_at(lift_keys, T)
        opac = anim.morph_at(opac_keys, T)
        dz = lift * lift_km
        # Flow animation pauses at lift-off: parcels freeze where they are
        # and ride up with the ice.
        t_lift = next(t for t, v in lift_keys if v > 0.0)
        t_lift = max(t for t, v in lift_keys if t < t_lift)
        if opac > 0.001:
            new = anim.build_parcels(self.tracks, min(T, t_lift), kpp)
            if new is not None:
                self.parcel_mesh.copy_from(new)
        for actor, full in self.ice_actors:
            actor.SetPosition(0.0, 0.0, dz)
            actor.prop.opacity = full * opac
            actor.SetVisibility(opac > 0.001)
        if self.ice_bar is not None:
            self.ice_bar.SetVisibility(opac > 0.5)

        self.wsb_label.SetVisibility(bool(opac < 0.5))
        if opac < 0.5:
            self.wsb_pts.points = wsb_label_anchor(self.wsb_centre, pos, foc, self.wsb_font,
                                                   self.frame_h)[None, :]
        pl.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true", help="960x540, 12 fps, thinned pins")
    ap.add_argument("--stills", action="store_true", help="render key-frame PNGs only")
    ap.add_argument("--stills-dir", default=None)
    ap.add_argument("--times", default=None, help="comma-separated seconds for --stills")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--substride", type=int, default=None)
    ap.add_argument("--chunk", default=None, metavar="I/N")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()

    frame_size = (960, 540) if args.draft else (FRAME_W, FRAME_H)
    fps = args.fps or (12 if args.draft else FPS)
    substride = args.substride or (2 if args.draft else 1)
    mp4 = os.path.join(OUT_DIR, OUT_BASENAME + ("_draft" if args.draft else "") + ".mp4")
    frames_dir = args.frames_dir or os.path.join(OUT_DIR, OUT_BASENAME + "_frames")
    if args.assemble:
        anim.assemble_frames(frames_dir, mp4, fps)
        return

    t0 = time.time()
    region, meshes, drapes, lift_km, tracks = build_scene_data(substride)
    cam_keys, lift_keys, opac_keys, centre = build_keyframes(meshes, region, lift_km)
    total_s = cam_keys[-1][0]
    grid_km = float(abs(region["x"][0, 1] - region["x"][0, 0])) / 1000.0 * substride
    print(f"Scene ready in {time.time() - t0:.0f} s; duration {total_s:.0f} s at {fps} fps")
    scene = Scene(meshes, drapes, tracks, region, frame_size, cam_keys, centre)

    if args.stills:
        out = args.stills_dir or os.path.join(OUT_DIR, OUT_BASENAME + "_stills")
        os.makedirs(out, exist_ok=True)
        times = [k[0] for k in cam_keys[1:]]
        times.append(0.5 * (cam_keys[5][0] + cam_keys[6][0]))  # mid-lift
        times.append(0.5 * (cam_keys[8][0] + cam_keys[9][0]))  # mid bed zoom
        if args.times:
            times = [float(v) for v in args.times.split(",")]
        for T in sorted(times):
            tf = time.time()
            scene.update(cam_keys, lift_keys, opac_keys, T, grid_km, lift_km)
            path = os.path.join(out, f"still_t{T:05.1f}s.png")
            scene.pl.screenshot(path)
            print(f"  {path}  ({time.time() - tf:.1f} s)", flush=True)
        scene.pl.close()
        return

    n_frames = int(round(total_s / PLAYBACK_SPEEDUP * fps))
    print(f"Playback speed-up {PLAYBACK_SPEEDUP:.2f}x -> {n_frames / fps:.1f} s, {n_frames} frames")
    if args.chunk:
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
        scene.update(cam_keys, lift_keys, opac_keys, T, grid_km, lift_km)
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


if __name__ == "__main__":
    main()
