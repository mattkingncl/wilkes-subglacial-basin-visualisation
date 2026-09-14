"""Combined variant of animations 1 and 2 with a whole-of-Antarctica intro:

  1. camera high above the South Pole looking down, full continent in view
     (ice surface only, coarse pins), hold;
  2. swing down to animation 1's overview camera, then animation 1's
     fly-by with half-depth close-ups (Mertz -> Ninnis -> Cook) and the
     zoom back to the overview (no Pliocene morph);
  3. animation 2's bed reveal: the AOI ice block lifts off the bed, fades out
     (the surrounding continent fades with it), camera drops to the Wilkes
     Subglacial Basin and labels it.

Same CLI as wilkes_basin_3d_animation.py (--draft, --stills [--times],
--chunk i/N --frames-dir D, --assemble). Outputs use the basename
wilkes_basin_3d_animation_intro. The GIF is the full sequence.
"""

import os
import sys

import numpy as np
import pyvista as pv
import vtk
import xarray as xr
from pyproj import Transformer

import wilkes_basin_3d_animation as anim
import wilkes_basin_3d_figure as base
import wilkes_basin_bed_reveal_animation as reveal

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

anim.OUT_BASENAME = "wilkes_basin_3d_animation_intro"

# Intro legs (design seconds, before PLAYBACK_SPEEDUP): hold the polar view,
# then one smoothstep swing to the existing overview camera.
INTRO_HOLD_S = 4.5
INTRO_PAN_S = 6.0
INTRO_S = INTRO_HOLD_S + INTRO_PAN_S

# Close-ups stop half way between the overview distance and animation 1's
# CLOSEUP_DIST_KM (0 = stay at overview distance, 1 = full close-up).
CLOSEUP_ZOOM_FRAC = 0.5
LABEL_HANDOVER_FRAC = 0.3   # glacier label stays this far into the pan to the next target
_CLOSEUP_DIST_FULL = anim.CLOSEUP_DIST_KM

# Velocity parcels: twice the seed density, longer streaks with a less
# pronounced head so they read as flow lines rather than tadpoles.
base.FLOWLINE_SEED_SPACING_KM /= np.sqrt(2.0)
anim.PARCEL_LENGTH_KM = 90.0
anim.PARCEL_HEAD_PX = 2.25   # 25% thinner than the first draft (3.0 / 1.5)
anim.PARCEL_TAIL_PX = 1.125

# Two-line glacier labels (half the width, so they no longer overplot at the
# shallower close-ups), smaller than animation 1's. The Pliocene morph legs
# are dropped: the fly-by ends with the zoom back to the overview and goes
# straight into the lift-off.
anim.COAST_TARGETS = [(name.replace(" / ", " /\n"), lon, lat) for name, lon, lat in anim.COAST_TARGETS]
_n_present = next(i for i, leg in enumerate(anim.TIMELINE) if leg[2] > 0.0)
anim.TIMELINE = [(dur, target.replace(" / ", " /\n"), 0.0) for dur, target, _ in anim.TIMELINE[:_n_present]]
anim.LABEL_FONT_PX = 26
LABEL_POINT_PX = 8
WSB_LABEL_FONT_PX = 48
anim.GIF_END_DESIGN_S = None
anim.FLIGHTLINES_ENABLE = False   # no BEDMAP3 radar flight lines

# Bed-reveal tail appended after animation 1's last leg: (duration_s,
# camera_target, ice_lift, ice_opacity) as in the bed-reveal script's
# TIMELINE from its "lifted" leg on. The bed fades in over BED_FADE_S from
# the start of the lift.
TAIL_TIMELINE = [leg for leg in reveal.TIMELINE if leg[1] == "lifted"] + \
    reveal.TIMELINE[[leg[1] for leg in reveal.TIMELINE].index("lifted") + 1:]
BED_FADE_S = 2.0
# Final camera: the bed-reveal script's landward view, with the focus shifted
# this far east (screen-left) of the basin centroid, in front of the basin.
BED_FOCUS_EAST_KM = 300.0
# Basin label: 2-D text at the projected (integer-pixel) anchor, shown from
# this fraction of the way through the final zoom, fading in over
# WSB_LABEL_FADE_S (design s).
WSB_LABEL_ZOOM_FRAC = 0.5
WSB_LABEL_FADE_S = 0.5
# Bed hidden until the lift (fades in over BED_FADE_S from the start of the
# lift, as in the bed-reveal script). Until then a flat, opaque sea-level
# sheet over the ocean/ice-shelf cells, colour-shaded by water depth with a
# light hillshade of the seafloor; it is stripped (fades out) with the lift.
BED_FROM_START = False
# Exposed bedrock (BedMachine mask 1, ice-free land, plus any ocean-mask cell
# with bed above sea level) is drawn from frame 1 in the bed colours; it stays
# on the ground through the lift and is replaced by the full bed once that
# has faded in.
ROCK_FROM_START = True
OCEAN_SURFACE = True
OCEAN_OPACITY = 1.0
OCEAN_DEPTH_COLORS = ["#a6d3ea", "#4f9bd2", "#1f5f9f", "#0d3268", "#061c44"]  # 0 m -> deepest
OCEAN_DEPTH_CLIM_M = (-3000.0, 0.0)
OCEAN_HILLSHADE_STRENGTH = 0.3
OCEAN_MASK_VALUES = (0, 3)   # BedMachine mask: open ocean, floating ice
# Ice-pin base: True = flat at sea level under grounded ice (only the floating
# ice keeps its real draft), so the lifted ice is a flat-bottomed slab rather
# than a bed-shaped keel; False = the bed-reveal script's bed-shaped base.
ICE_BASE_FLAT = False

# Continent-wide ice pins: every Nth 500 m BedMachine point (8 -> 4 km),
# outside the AOI box only (the AOI keeps its fine 1.5 km pins on top).
CONTINENT_STRIDE = 8
if "--draft" in sys.argv:
    CONTINENT_STRIDE *= 2

# Polar camera: straight down from height POLE_HEIGHT_KM over the pole, tilted
# POLE_TILT_DEG off vertical toward the AOI so view-up (0,0,1) stays defined
# and the AOI sits at the bottom of the frame -- the same screen position it
# occupies in the overview, so the swing does not roll the horizon.
POLE_HEIGHT_KM = 9200.0
POLE_TILT_DEG = 6.0

# Whole-continent view: at 100x the outlet-glacier troughs are 100 km canyons
# whose walls read as grey smears from the pole, and the AOI's flow lines /
# flight lines / parcels are sub-pixel grey clutter. The vertical scale ramps
# from POLE_VERT_SCALE to 1 over the swing (the exaggeration note follows),
# the static lines cut in OVERLAY_SHOW_S before the swing ends (camera still
# moving) and the parcels grow from zero radius over PARCEL_GROW (design s).
POLE_VERT_SCALE = 0.1
OVERLAY_SHOW_S = 1.0
PARCEL_GROW = (INTRO_HOLD_S + 0.5 * INTRO_PAN_S, INTRO_S)
# Filled in by load_scene()/build_keyframes() for Scene.update().
STATE = {}


# ---------------------------------------------------------------------------
# CONTINENT
# ---------------------------------------------------------------------------

def load_continent(stride):
    """Coarse whole-continent ice-surface pin mesh, excluding the AOI box."""
    ds = xr.open_dataset(base.BEDMACHINE_NC)
    sub = ds.isel(x=slice(0, None, stride), y=slice(0, None, stride))
    x = sub["x"].values.astype(np.float64)
    y = sub["y"].values.astype(np.float64)
    Xg, Yg = np.meshgrid(x, y)
    surface = sub["surface"].values.astype(np.float64)
    mask = sub["mask"].values
    to_ll = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(Xg, Yg)
    in_aoi = ((lon >= base.LON_MIN) & (lon <= base.LON_MAX)
              & (lat >= base.LAT_MIN) & (lat <= base.LAT_MAX))
    valid = np.isin(mask, [2, 3]) & ~in_aoi
    dx = float(abs(x[1] - x[0]))
    # Shade the whole surface (not just `valid`) so the hillshade has no edge
    # along the AOI box where the coarse and fine pins meet.
    rgb = anim.ice_rgb_from_surface(surface, dx, dx)
    mesh = base.pin_columns_mesh(Xg / 1000.0, Yg / 1000.0,
                                 surface * base.VERT_EXAG / 1000.0, 0.0, valid, rgb=rgb)
    print(f"Continent pins: {int(valid.sum())} at {dx / 1000:.0f} km", flush=True)
    return mesh, dx / 1000.0


# ---------------------------------------------------------------------------
# SCENE DATA: animation 1's scene plus the bed layers of animation 2
# ---------------------------------------------------------------------------

_load_scene_orig = anim.load_scene
_build_keyframes_orig = anim.build_keyframes
_Scene_orig = anim.Scene


def load_scene(pin_substride):
    # anim.load_scene() discards the bed pins; keep a reference.
    keep = {}
    build_meshes_orig = base.build_meshes

    def build_meshes(region):
        m = build_meshes_orig(region)
        keep["bed_pins"] = m["bed_pins"]
        return m

    base.build_meshes = build_meshes
    try:
        region, meshes, drapes, tracks, morph = _load_scene_orig(pin_substride)
    finally:
        base.build_meshes = build_meshes_orig
    meshes["bed_pins"] = keep["bed_pins"]
    meshes["rock_pins"] = None
    if ROCK_FROM_START:
        mask, bed_m = region["mask"], region["bed"]
        rock = region["inbox"] & ~meshes["ice_bool"] & ((mask == 1) | ((mask == 0) & (bed_m > 0.0)))
        # Pins rise from sea level, not the bed-pin floor: a full-depth pin
        # shows below the ocean sheet when seen from beyond the AOI edge.
        meshes["rock_pins"] = base.pin_columns_mesh(meshes["x_km"], meshes["y_km"], meshes["bed_z"], 0.0,
                                                    rock, substride=pin_substride, scalars=bed_m)
        print(f"Exposed rock: {int(rock.sum())} cells drawn from the start")

    # Re-base the ice pins from z=0 onto the ice base and work out the lift,
    # exactly as the bed-reveal script does (the morph only moves pin tops).
    inbox = region["inbox"]
    bed_z = meshes["bed_z"]
    bed_floor_km = float(np.nanmin(bed_z[inbox])) - 0.05 * (
        float(np.nanmax(bed_z[inbox])) - float(np.nanmin(bed_z[inbox])))
    ice_base = reveal.ice_base_z_km(region, bed_floor_km)
    if ICE_BASE_FLAT:
        floating = region["mask"] == 3
        ice_base = np.where(floating, np.minimum(ice_base, 0.0), 0.0)
    ss = pin_substride
    ib = ice_base[::ss, ::ss][meshes["ice_bool"][::ss, ::ss]]
    pts = meshes["ice_pins"].points.copy()
    pts[0::2, 2] = ib
    meshes["ice_pins"].points = pts
    sub_ice = meshes["ice_bool"] & inbox
    STATE["lift_km"] = float(np.nanpercentile(bed_z[sub_ice], reveal.ICE_LIFT_BED_PCTL)
                             - np.nanmin(ice_base[sub_ice]) + reveal.ICE_LIFT_CLEARANCE_KM)
    STATE["centre"] = reveal.basin_centroid(region)
    print(f"Ice lift: {STATE['lift_km']:.0f} km; basin centroid (km): "
          f"{STATE['centre'][0]:.0f}, {STATE['centre'][1]:.0f}")

    # Bed-draped outlines and the IBCSO direct-measurement points.
    gl_lines = base.lines_from_geodataframe(base.load_grounding_line())
    coast_gdf = base.load_moa_coastline()
    coast_lines = base.lines_from_geodataframe(coast_gdf) if not coast_gdf.empty else []
    islands_gdf = base.load_moa_islands()
    island_lines = base.lines_from_geodataframe(islands_gdf) if not islands_gdf.empty else []
    bed = region["bed"]
    for key, lines in (("gl_bed", gl_lines), ("coast_bed", coast_lines), ("island_bed", island_lines)):
        drapes[key] = anim.drape_lines_polydata(lines, region, bed, lift_km=base.BED_LINE_LIFT_KM,
                                                local_max_cells=base.BED_LINE_LOCAL_MAX_CELLS)
    drapes["ibcso"] = None
    if base.IBCSO_ENABLE:
        ix, iy, iz = base.load_ibcso_direct_points(region)
        drapes["ibcso"] = base.ibcso_points_mesh(ix, iy, iz) if ix.size else None
    return region, meshes, drapes, tracks, morph


def ocean_surface_mesh(meshes, region):
    """Flat sheet at sea level (z = 0) over the ocean cells inside the AOI box,
    coloured by water depth (bed elevation) with a light hillshade of the
    seafloor so the shelf/slope/trough pattern reads through the surface."""
    from matplotlib.colors import LinearSegmentedColormap, LightSource
    ocean = region["inbox"] & np.isin(region["mask"], OCEAN_MASK_VALUES)
    bed_m = np.where(np.isfinite(region["bed"]), region["bed"], 0.0)
    lo, hi = OCEAN_DEPTH_CLIM_M
    frac = np.clip((bed_m - lo) / (hi - lo), 0.0, 1.0)          # 0 deep ... 1 sea level
    cmap = LinearSegmentedColormap.from_list("ocean_depth", OCEAN_DEPTH_COLORS[::-1])
    rgb = cmap(frac)[..., :3]
    dx = float(abs(region["x"][0, 1] - region["x"][0, 0]))
    dy = float(abs(region["y"][1, 0] - region["y"][0, 0]))
    hs = LightSource(azdeg=315, altdeg=45).hillshade(bed_m, vert_exag=base.VERT_EXAG, dx=dx, dy=dy)
    shade = (1.0 - OCEAN_HILLSHADE_STRENGTH) + 2.0 * OCEAN_HILLSHADE_STRENGTH * hs
    rgb = (np.clip(rgb * shade[..., None], 0.0, 1.0) * 255).astype(np.uint8)
    grid = pv.StructuredGrid(meshes["x_km"], meshes["y_km"], np.zeros_like(meshes["bed_z"]))
    grid["ocean"] = ocean.ravel(order="F").astype(np.uint8)
    grid["ocean_rgb"] = rgb.reshape(-1, 3, order="F")
    return grid.threshold(0.5, scalars="ocean").extract_surface()


def overview_camera(meshes, region):
    base.CAMERA_DIST_FACTOR = anim.OVERVIEW_DIST_FACTOR
    ov_pos, ov_focal, _ = base.compute_camera(meshes, region)
    ov_pos, ov_focal = np.asarray(ov_pos), np.asarray(ov_focal)
    ov_pos[2] += anim.OVERVIEW_FOCAL_LIFT_KM
    ov_focal[2] += anim.OVERVIEW_FOCAL_LIFT_KM
    return ov_pos, ov_focal


def build_keyframes(meshes, region):
    ov_pos, ov_focal = overview_camera(meshes, region)
    ov_dist = float(np.linalg.norm(ov_pos - ov_focal))
    anim.CLOSEUP_DIST_KM = ov_dist + CLOSEUP_ZOOM_FRAC * (_CLOSEUP_DIST_FULL - ov_dist)
    cam_keys, morph_keys = _build_keyframes_orig(meshes, region)

    # At the shallower close-ups all three glaciers stay in frame, so show
    # only the label of the glacier being flown to: from half way through the
    # pan towards it until LABEL_HANDOVER_FRAC of the pan away from it.
    windows, t, current = [], INTRO_S, None
    for dur, target, _ in anim.TIMELINE:
        if target != "hold" and target != "overview":
            if current is not None:
                windows.append((current[0], t + LABEL_HANDOVER_FRAC * dur, current[1]))
            current = (t + 0.5 * dur, target)
        elif target == "overview" and current is not None:
            windows.append((current[0], t + LABEL_HANDOVER_FRAC * dur, current[1]))
            current = None
        t += dur
    STATE["label_windows"] = windows

    # --- intro: polar view, then swing to the overview ---
    toward = ov_pos[:2] / np.linalg.norm(ov_pos[:2])
    tilt = np.deg2rad(POLE_TILT_DEG)
    pole_pos = np.array([*(toward * POLE_HEIGHT_KM * np.sin(tilt)), POLE_HEIGHT_KM * np.cos(tilt)])
    pole_focal = np.zeros(3)
    cam_keys = ([(0.0, pole_pos, pole_focal), (INTRO_HOLD_S, pole_pos, pole_focal)]
                + [(t + INTRO_S, p, f) for t, p, f in cam_keys])
    morph_keys = [(0.0, 0.0)] + [(t + INTRO_S, u) for t, u in morph_keys]

    # --- tail: lift the ice, fade it out, drop to the basin ---
    lift_km, centre = STATE["lift_km"], STATE["centre"]
    lf_focal = ov_focal + np.array([0.0, 0.0, 0.5 * lift_km])
    lf_pos = lf_focal + (ov_pos - ov_focal) * reveal.LIFT_CAMERA_PULLBACK
    lon = np.arctan2(centre[0], centre[1])                 # EPSG:3031: x = r sin(lon), y = r cos(lon)
    east = np.array([np.cos(lon), -np.sin(lon), 0.0])
    bed_cam = reveal.looking_landward(centre + east * BED_FOCUS_EAST_KM, 0.0, reveal.BED_DIST_KM,
                                      reveal.BED_ELEVATION_DEG, reveal.BED_FOCAL_LIFT_KM)
    cams = {"lifted": (lf_pos, lf_focal), "bed": bed_cam}
    t = cam_keys[-1][0]
    lift_keys = [(0.0, 0.0), (t, 0.0)]
    ice_opac_keys = [(0.0, 1.0), (t, 1.0)]
    bed_opac_keys = [(0.0, 1.0)] if BED_FROM_START else [(0.0, 0.0), (t, 0.0), (t + BED_FADE_S, 1.0)]
    for dur, target, lift, opac in TAIL_TIMELINE:
        if target == "bed":
            STATE["wsb_label_t"] = t + WSB_LABEL_ZOOM_FRAC * dur
        t += dur
        if target == "hold":
            cam_keys.append((t, cam_keys[-1][1], cam_keys[-1][2]))
        else:
            cam_keys.append((t, *cams[target]))
        morph_keys.append((t, 0.0))
        lift_keys.append((t, float(lift)))
        if float(opac) != ice_opac_keys[-1][1]:
            ice_opac_keys.append((t - dur + reveal.ICE_CUT_S, float(opac)))   # hard cut
        ice_opac_keys.append((t, float(opac)))
    STATE.update(lift_keys=lift_keys, ice_opac_keys=ice_opac_keys, bed_opac_keys=bed_opac_keys)
    return cam_keys, morph_keys




# ---------------------------------------------------------------------------
# SCENE
# ---------------------------------------------------------------------------

class Scene(_Scene_orig):
    def __init__(self, meshes, drapes, tracks, morph, region, frame_size, cam_keys):
        # The original measures overview_dist from cam_keys[0]; hand it the
        # overview key (index 2) so label show/hide is unchanged.
        super().__init__(meshes, drapes, tracks, morph, region, frame_size, cam_keys[2:])
        for pts in self.label_points.values():
            pts.GetProperty().SetPointSize(LABEL_POINT_PX)
        pl = self.pl
        self.continent_mesh, self.continent_km = load_continent(CONTINENT_STRIDE)
        self.continent = pl.add_mesh(
            self.continent_mesh, scalars="pin_rgb", rgb=True, show_scalar_bar=False,
            render_lines_as_tubes=True, line_width=anim.PIN_WIDTH_MIN_PX, lighting=False,
        )

        # Vertical-exaggeration note moves to the bottom edge to make room for
        # the bed scalar bar at lower right (corner-annotation slots: 1 =
        # lower right, 4 = lower edge).
        for actor in pl.renderer.actors.values():
            if isinstance(actor, vtk.vtkCornerAnnotation) and \
                    (actor.GetText(1) or "").startswith("Vertical exaggeration"):
                actor.SetText(4, actor.GetText(1))
                actor.SetText(1, "")
                self.exag_text = actor

        # --- bed (bed-reveal script styling), hidden until the lift ---
        self.bed_actors = []
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
        self.bed_bar = pl.scalar_bars["Bed elevation (m)"]
        anim._bar_box(self.bed_bar)
        self.bed_actors.append((self.bed_pins, 1.0))
        if meshes["sea_level_line"].n_points > 0:
            a = pl.add_mesh(meshes["sea_level_line"], color="#00e5ff", line_width=reveal.SEA_LEVEL_WIDTH_PX,
                            render_lines_as_tubes=True, lighting=False)
            self.bed_actors.append((a, 1.0))
        for key, color, width in (("gl_bed", "black", anim.GL_WIDTH_PX),
                                  ("coast_bed", "#6b4226", anim.COAST_WIDTH_PX),
                                  ("island_bed", "#6b4226", anim.ISLAND_WIDTH_PX)):
            if drapes.get(key) is not None:
                a = pl.add_mesh(drapes[key], color=color, line_width=width,
                                render_lines_as_tubes=True, lighting=False)
                self.bed_actors.append((a, 1.0))
        if drapes.get("ibcso") is not None:
            a = pl.add_mesh(drapes["ibcso"], color="#ff6f00", point_size=reveal.IBCSO_POINT_PX,
                            render_points_as_spheres=True, opacity=reveal.IBCSO_OPACITY)
            self.bed_actors.append((a, reveal.IBCSO_OPACITY))
        # Bed-draped lines / IBCSO points only once the ice starts to lift: at
        # the polar distance the coastline and outcrop outlines smear into a
        # solid patch, and the ice hides them until then anyway.
        self.bed_lines = [actor for actor, _ in self.bed_actors if actor is not self.bed_pins]
        for actor, _ in self.bed_actors:
            actor.SetVisibility(BED_FROM_START and actor is self.bed_pins)
        self.bed_bar.SetVisibility(BED_FROM_START)
        self.rock_pins = None
        if meshes.get("rock_pins") is not None:
            self.rock_pins = pl.add_mesh(
                meshes["rock_pins"], scalars="pin_scalar", cmap=base.BATHY_CMAP,
                clim=(-base.BED_ELEV_SATURATION_M, base.BED_ELEV_SATURATION_M),
                render_lines_as_tubes=True, line_width=anim.PIN_WIDTH_MIN_PX, lighting=False,
                show_scalar_bar=False,
            )
        self.ocean = None
        if OCEAN_SURFACE:
            self.ocean = pl.add_mesh(ocean_surface_mesh(meshes, region), scalars="ocean_rgb", rgb=True,
                                     opacity=OCEAN_OPACITY, lighting=False, show_scalar_bar=False)
        # The bed fades in by colour (black -> full colour) at full opacity:
        # translucent pin tubes render as a featureless fog on llvmpipe.
        self.bed_fade_shown = None
        self.bed_lut = self.bed_pins.mapper.lookup_table
        self.bed_lut_rgba = np.array(self.bed_lut.values, dtype=float)
        self.bed_colors = [(actor, np.array(actor.prop.color.float_rgb, dtype=float))
                           for actor, _ in self.bed_actors if actor is not self.bed_pins]

        # Basin label: a 2-D text actor placed each frame at the projected,
        # integer-rounded screen position of the basin centre (shifted
        # screen-left as in the bed-reveal script). A world-anchored point
        # label re-placed by VTK every frame jittered during the zoom.
        self.wsb_centre = np.asarray(STATE["centre"]) + np.array([0.0, 0.0, 5.0])
        px_scale = frame_size[1] / 1080.0
        self.wsb_shift_px = reveal.WSB_LABEL_SHIFT_PX * px_scale
        self.wsb_label = vtk.vtkTextActor()
        self.wsb_label.SetInput(reveal.WSB_LABEL_TEXT)
        self.wsb_label.SetTextScaleModeToNone()
        tp = self.wsb_label.GetTextProperty()
        tp.SetFontFamilyToArial()
        tp.SetFontSize(int(round(WSB_LABEL_FONT_PX * px_scale)))
        tp.SetColor(1.0, 1.0, 1.0)
        tp.ItalicOn()
        tp.BoldOff()
        tp.SetJustificationToLeft()
        tp.SetVerticalJustificationToBottom()
        tp.SetBackgroundColor(0.0, 0.0, 0.0)
        tp.SetBackgroundOpacity(0.55)
        tp.SetLineSpacing(1.0)
        self.wsb_label.GetPositionCoordinate().SetCoordinateSystemToDisplay()
        self.wsb_label.SetVisibility(False)
        pl.renderer.AddActor2D(self.wsb_label)
        self.wsb_shown = None

        # Polar-view handling (see POLE_VERT_SCALE): every 3-D actor is
        # z-scaled about sea level; the static flow/flight lines stay hidden.
        self.scaled = [a for a in pl.renderer.actors.values() if isinstance(a, vtk.vtkActor)]
        self.zscale_shown = None
        hide_rgb = {tuple(np.round(pv.Color(c).float_rgb, 3))
                    for c in (anim.FLIGHT_COLOR, anim.STATIC_TRACK_COLOR)}
        self.pole_hidden = [a for a, _ in self.fading
                            if tuple(np.round(a.prop.color.float_rgb, 3)) in hide_rgb]

    def update_wsb_label(self, T):
        """Place the basin label at the projected basin centre, centred on it
        and shifted WSB_LABEL_SHIFT_PX screen-left, snapped to whole pixels."""
        fade = anim.smoothstep((T - STATE["wsb_label_t"]) / WSB_LABEL_FADE_S)
        if fade <= 0.001:
            self.wsb_label.SetVisibility(False)
            return
        ren = self.pl.renderer
        ren.SetWorldPoint(*self.wsb_centre, 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        size = [0, 0]
        self.wsb_label.GetSize(ren, size)
        x = int(round(dx - 0.5 * size[0] - self.wsb_shift_px))
        y = int(round(dy - 0.5 * size[1]))
        self.wsb_label.SetPosition(x, y)
        tp = self.wsb_label.GetTextProperty()
        tp.SetOpacity(fade)
        tp.SetBackgroundOpacity(0.55 * fade)
        self.wsb_label.SetVisibility(True)

    def update(self, cam_keys, morph_keys, T, grid_km):
        pl = self.pl
        pos, foc = anim.camera_at(cam_keys, T)
        pl.camera_position = (tuple(pos), tuple(foc), (0.0, 0.0, 1.0))
        pl.camera.view_angle = anim.VIEW_ANGLE_DEG
        kpp = anim.km_per_pixel(pos, foc, self.frame_w)
        w = max(anim.PIN_WIDTH_MIN_PX, anim.PIN_FILL_FACTOR * grid_km / kpp)
        self.pins.prop.line_width = w
        self.bed_pins.prop.line_width = w
        self.continent.prop.line_width = max(anim.PIN_WIDTH_MIN_PX,
                                             anim.PIN_FILL_FACTOR * self.continent_km / kpp)

        u = anim.morph_at(morph_keys, T)
        anim.apply_morph(self.pins_mesh, self.morph, u)
        lift = anim.morph_at(STATE["lift_keys"], T)
        ice_opac = anim.morph_at(STATE["ice_opac_keys"], T)
        bed_opac = anim.morph_at(STATE["bed_opac_keys"], T)
        dz = lift * STATE["lift_km"]

        zs = POLE_VERT_SCALE + (1.0 - POLE_VERT_SCALE) * anim.smoothstep((T - INTRO_HOLD_S) / INTRO_PAN_S)
        if zs != self.zscale_shown:
            for a in self.scaled:
                a.SetScale(1.0, 1.0, zs)
            self.exag_text.SetText(4, f"Vertical exaggeration: {base.VERT_EXAG * zs:.0f}×")
            self.zscale_shown = zs
        grow = anim.smoothstep((T - PARCEL_GROW[0]) / (PARCEL_GROW[1] - PARCEL_GROW[0]))
        if u < 0.999 and ice_opac > 0.001 and grow > 0.01:
            new = anim.build_parcels(self.tracks, T, kpp * grow)
            if new is not None:
                self.parcel_mesh.copy_from(new)
        # Everything riding on the ice lifts with it and fades with it.
        self.pins.SetPosition(0.0, 0.0, dz)
        self.pins.prop.opacity = ice_opac
        self.pins.SetVisibility(ice_opac > 0.001)
        for actor, full in self.fading:
            actor.SetPosition(0.0, 0.0, dz)
            actor.prop.opacity = full * (1.0 - u) * ice_opac
            actor.SetVisibility(u < 0.999 and ice_opac > 0.001)
        if grow <= 0.01:
            self.parcels.SetVisibility(False)
        if T < INTRO_S - OVERLAY_SHOW_S:
            for a in self.pole_hidden:
                a.SetVisibility(False)
        # The surrounding continent lifts with the AOI ice (left on the ground
        # it reads as a second ice sheet beside the lifted one) and is cut
        # away with it.
        self.continent.SetPosition(0.0, 0.0, dz)
        self.continent.prop.opacity = ice_opac
        self.continent.SetVisibility(ice_opac > 0.001)
        self.scalar_bar.SetVisibility(u < 0.5 and ice_opac > 0.5 and grow > 0.5)
        if self.plio_extent is not None:
            self.plio_extent.prop.opacity = u
            self.plio_extent.SetVisibility(u > 0.001)

        if bed_opac > 0.001 and bed_opac != self.bed_fade_shown:
            rgba = self.bed_lut_rgba.copy()
            rgba[:, :3] *= bed_opac
            self.bed_lut.values = rgba.astype(np.uint8)
            for actor, rgb in self.bed_colors:
                actor.prop.color = tuple(rgb * bed_opac)
            self.bed_fade_shown = bed_opac
        for actor, full in self.bed_actors:
            actor.prop.opacity = full
            actor.SetVisibility(bed_opac > 0.001)
        for actor in self.bed_lines:
            actor.SetVisibility(bed_opac > 0.001 and lift > 0.001)
        self.bed_bar.SetVisibility(bed_opac > 0.999)
        if self.ocean is not None:
            self.ocean.prop.opacity = OCEAN_OPACITY * (1.0 - bed_opac)   # stripped as the bed fades in
            self.ocean.SetVisibility(bed_opac < 0.999)
        if self.rock_pins is not None:
            self.rock_pins.prop.line_width = w
            self.rock_pins.SetVisibility(bed_opac < 0.999)
        self.update_wsb_label(T)

        if u < 0.02:
            state = anim.STATE_TEXT_PRESENT
        elif u > 0.98:
            state = anim.STATE_TEXT_PLIO
        else:
            state = anim.STATE_TEXT_MORPH
        if anim.SHOW_STATE_TEXT and state != self.state_shown:
            self.state_text.SetText(3, state)
            self.state_shown = state

        for name, actor in self.labels.items():
            show = any(t0 <= T < t1 and n == name for t0, t1, n in STATE["label_windows"])
            actor.SetVisibility(bool(show))
            self.label_points[name].SetVisibility(bool(show))
        pl.render()



anim.load_scene = load_scene
anim.build_keyframes = build_keyframes
anim.Scene = Scene

if __name__ == "__main__":
    anim.main()
