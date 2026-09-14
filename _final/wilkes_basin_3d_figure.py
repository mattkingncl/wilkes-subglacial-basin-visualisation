"""
Wilkes Subglacial Basin, East Antarctica -- 3D perspective figure for a policy brief.

Two panels, each a separate off-screen PyVista scene rendered with an
identical oblique camera (looking landward from the Southern Ocean, across
the continental shelf, through the grounding line, into the subglacial basin
interior -- see compute_camera(), computed once and reused unchanged for
both), composited together into one PDF/PNG:
  (a) the ice surface, its HEIGHT and COLOUR both encoding true (vertically
      exaggerated) surface elevation -- a continuous white (low, coastal
      margin) to dark blue (high, interior plateau) scale -- with BEDMAP3
      flight lines and velocity-derived glacier flow lines draped on top.
  (b) the same domain with the ice removed -- bed topography and offshore
      bathymetry, coloured by elevation, with IBCSO direct seafloor
      soundings marked.
Rendering both panels with the same camera_position/focal_point/view_angle/
window_size is what makes them directly comparable: panel (b) shows exactly
what is hidden beneath the opaque ice surface in panel (a).

Terrain rendering technique -- vertical "pin"/hedgehog columns, not a
smoothed continuous surface: each surviving grid point becomes one thin
vertical line from a common flat base up to that point's (exaggerated)
elevation, rendered cheaply with VTK's render_lines_as_tubes (a GPU-shaded
line, not real 3-D tube geometry -- essential at this point count). This is
a deliberate choice, not a rendering shortcut: real, densely surveyed
small-scale relief makes neighbouring pins land at very different heights (a
rough, bristly silhouette -- see the BEDMAP3 flight-line corridors), while
coarsely interpolated/no-data regions vary little from pin to pin (a
smooth-looking surface). Data sparsity is thus visible directly in the
terrain itself, without needing a separate coverage overlay. See
pin_columns_mesh().

Vector text/lines (labels, colourbars, scale bars, panel tags) are added in
matplotlib on top of each panel's raster scene by projecting 3D anchor
points to 2D screen space with VTK's coordinate transform. Output is a
composite PDF (vector text/lines + embedded raster scenes) and a PNG at
print DPI.

Data (all read-only, all local -- no data are modified):
  - BedMachine Antarctica v3 (Morlighem et al., 2020, Nature Geoscience;
    doi:10.5067/E1QL9HFQ7A8M): bed, ice surface, thickness, mask.
  - MEaSUREs InSAR-based Antarctic grounding line v02.1 (Mouginot, Scheuchl,
    Rignot; NSIDC), same EPSG:3031 projection as BedMachine, no reprojection
    needed.
  - MOA (MODIS Mosaic of Antarctica) grounding line/coastline/island vector
    products (Scambos, Haran, Bohlander, Fahnestock; NSIDC).
  - MEaSUREs Antarctic ice velocity, 450 m (Mouginot, Rignot, Scheuchl; NASA
    MEaSUREs / NSIDC), EPSG:3031, resampled (nearest) onto the same grid as
    the rendered surfaces. VX/VY are draped on the ice panel as a threshold
    speed colour map AND integrated into glacier flow lines (matplotlib
    streamplot, restricted to grounded ice) draped on top of it.
  - BEDMAP3 standardised point data (Pritchard, Fretwell, Fremand et al.,
    2024; NERC EDS UK Polar Data Centre, doi:10.5285/2d0e4791-8e20-46a3-80e4-
    f5f6716025d2), "bm3_streamlines_pt" product: individual airborne/ground
    radar survey points grouped by flight/traverse line (field `streamln`),
    draped on the ice surface as thin tracks -- the actual flight lines used
    to help constrain BedMachine's bed, not a synthetic proxy.
  - IBCSO v2 Type-ID grid (International Bathymetric Chart of the Southern
    Ocean; CRS EPSG:9354, reprojected here to EPSG:3031), used to mark
    individual direct seafloor soundings (single/multibeam, seismic, lidar,
    isolated -- excluding gravity-predicted/interpolated cells) as small
    points on the offshore seafloor.

Run:
    python wilkes_basin_3d_figure.py

Requires: xarray, numpy, pyproj, geopandas, pyvista, vtk, matplotlib, scipy,
rasterio, shapely.
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import rasterio
import vtk
import xarray as xr
from rasterio.windows import from_bounds
from matplotlib import font_manager
from matplotlib import patheffects as pe
from matplotlib.colors import LightSource, ListedColormap, TwoSlopeNorm
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates, maximum_filter

# ---------------------------------------------------------------------------
# CONFIG -- edit these to repoint at your own files / change the figure
# ---------------------------------------------------------------------------

BEDMACHINE_NC = os.path.expanduser(
    "~/data/Antarctic_topography/BEDMACHINE/BedMachineAntarctica-v3.nc"
)
GROUNDINGLINE_SHP = os.path.expanduser(
    "~/data/Antarctic_groundinglines/InSAR_GL_Antarctica_v02.1.shp"
)
# MOA (MODIS Mosaic of Antarctica) coastline/grounding-line/island vector
# products (Scambos, Haran, Bohlander, Fahnestock; NSIDC), same Polar
# Stereographic (true-scale 71S) definition as EPSG:3031, no reprojection
# needed.
MOA_GROUNDINGLINE_SHP = os.path.expanduser("~/data/MOA/coastlines/moa_groundingline.shp")
MOA_COASTLINE_SHP = os.path.expanduser("~/data/MOA/coastlines/moa_coastline.shp")
MOA_ISLANDS_SHP = os.path.expanduser("~/data/MOA/coastlines/moa_islands.shp")
MEASURES_VELOCITY_NC = os.path.expanduser(
    "~/data/Antarctic_velocities/Measures_v2/antarctica_ice_velocity_450m_v2.nc"
)
BEDMAP_FLIGHTLINES_SHP = os.path.expanduser(
    "~/data/Antarctic_topography/BEDMAP3/bm3_streamlines_pt/bm3_streamlines_pt.shp"
)
FLIGHTLINES_ENABLE = True
FLIGHTLINE_MIN_POINTS = 5  # drop degenerate/near-empty in-AOI line fragments

IBCSO_TID_TIF = os.path.expanduser("~/data/Antarctic_topography/IBSCO/IBCSO_TID.tif")
IBCSO_ENABLE = True
IBCSO_STRIDE = 4  # native grid is 500 m -> stride 4 = 2 km sample spacing
# IBCSO Type-ID codes for direct soundings (single/multibeam, seismic, lidar,
# isolated soundings, combined direct methods) -- everything else (40s = predicted
# from satellite gravity/interpolation, 70s = pre-generated interpolated grid) is
# NOT a real measurement and is deliberately excluded here.
IBCSO_DIRECT_CODES = {10, 11, 12, 13, 14, 15, 16, 17, 18}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASENAME = "wilkes_basin_3d"

# Region of interest: the full Wilkes Subglacial Basin, Wilkes Land, East
# Antarctica -- coast to deep interior. The render masks to the TRUE lon/lat
# box (not just its bounding rectangle in polar stereographic space), so
# changing these numbers changes exactly what you see, with no leakage from
# neighbouring basins. The WSB is centred near 75S, 145E, running south of
# the George V Coast (Ninnis/Cook glaciers, ~142-153E) and bounded to the
# west by the Aurora Subglacial Basin (centred ~76S, 117-118E) and to the
# east by the Prince Albert Mountains (~161E). The previous box (100-150E)
# was centred too far west and mostly showed the Aurora Subglacial Basin
# instead -- shifted east here to actually capture the WSB.
LON_MIN, LON_MAX = 110.0, 170.0
LAT_MIN, LAT_MAX = -85.0, -64.0

VERT_EXAG = 100  # vertical exaggeration factor. MUST be stated in the caption.
DECIMATE_STRIDE = 1  # take every Nth native 500 m BedMachine grid point (-> 1.5 km);
# this is the resolution used for masks, contouring, and nearest-neighbour
# lookups elsewhere in the script (draping lines, IBCSO points).

# Extra thinning applied ONLY to the pin/hedgehog terrain columns, on top of
# the already-decimated (DECIMATE_STRIDE) grid. 1 = full 1.5 km pin density
# (the most texture, the slowest render); raise this if a render is too slow.
VOXEL_SUBSTRIDE = 1
PIN_LINE_WIDTH_ICE = 2.0  # on-screen pixel width of each pin (render_lines_as_tubes)
PIN_LINE_WIDTH_BED = 2.0

VELOCITY_ENABLE = True

# Panel (a) ice-surface hillshade (relief cue independent of pin height and
# of elevation colour): max fractional brightness change either side of
# neutral, e.g. 0.45 -> shading multiplies colour by [0.55, 1.45].
HILLSHADE_STRENGTH = 0.75

FLOWLINE_ENABLE = True
# Flow lines are traced (not just decorated) from the MEaSUREs velocity field
# -- see compute_flowlines(). Seeds are placed on a regular grid over grounded
# ice at this spacing; smaller = more tributaries traced, denser convergence
# pattern into trunk outlet glaciers, but more render/compute cost and more
# visual clutter if pushed too far.
FLOWLINE_SEED_SPACING_KM = 30.0
# Fixed arc-length step per integration step (km) and hard cap on steps --
# max_steps * step_km is the longest any one traced line could be (a line
# normally ends much sooner, on leaving grounded ice/the AOI or on stalling
# below FLOWLINE_MIN_SPEED_MYR).
FLOWLINE_STEP_KM = 4.0
FLOWLINE_MAX_STEPS = 500
# Ice slower than this (m/yr) is treated as stalled -- tracing stops rather
# than following numerically-noisy near-zero velocity directions.
FLOWLINE_MIN_SPEED_MYR = 10.0
# Colour scale for the flow-line ice-speed tubes (m/yr), used with
# log_scale=True in render_panel() because grounded-ice speed in the AOI
# spans several orders of magnitude (interior plateau ~few m/yr, fast outlet
# trunks into the hundreds-to-low-thousands m/yr; measured 90th/99th
# percentiles here are ~42/~210 m/yr). Values below the low end (slow,
# near-stagnant ice) still get traced/drawn but clip to the scale's floor
# colour; log scaling keeps both the broad slow interior network and the
# narrow fast trunks visually legible in the same tube.
FLOWLINE_SPEED_CLIM_MYR = (20.0, 500.0)
FLOWLINE_SPEED_CMAP = "plasma"  # colourblind-safe, perceptually uniform sequential scale
# drape_lines_mesh()'s local_max_cells window radius (in grid cells) used for
# the flow-line drape: fast flow lines cross the roughest terrain (icefalls/
# crevasse fields right at outlet-glacier convergence), where true-elevation
# roughness of a few hundred m becomes many exaggerated km at VERT_EXAG=50 --
# easily burying the line if only its own grid cell's height is used. Sitting
# it at the local peak within this window instead is what actually clears
# that roughness (see drape_lines_mesh()'s docstring).
FLOWLINE_LOCAL_MAX_CELLS = 4
# Small fixed lift (km, exaggerated render-space) on top of that local peak,
# just for a clean visual gap above the terrain -- not doing the clearance
# work itself (FLOWLINE_LOCAL_MAX_CELLS does that).
FLOWLINE_LIFT_KM = 1.0
# Same idea for panel (b): the bed/bathymetry surface has its own local
# roughness (subglacial troughs, seamounts) that can bury the grounding
# line/coastline drapes the same way.
BED_LINE_LOCAL_MAX_CELLS = 3
BED_LINE_LIFT_KM = 1.0
# BEDMAP3 flight lines criss-cross the whole ice-sheet surface, including the
# same rough outlet-glacier terrain the flow lines cross -- draping them with
# no lift/local-max window (the old behaviour) buried them inside the pin
# forest almost everywhere except the calmest terrain, which is why they were
# effectively invisible. Same treatment as the flow lines fixes it.
FLIGHTLINE_LOCAL_MAX_CELLS = 4
FLIGHTLINE_LIFT_KM = 1.0
# Clearing the terrain wasn't enough on its own: at a thin radius + partial
# opacity, the flight lines were still washed out next to the far bolder,
# fully-opaque, brightly-coloured flow lines and easy to lose against the
# ice-surface colour ramp (dark against the dark-blue interior, but low
# contrast against the pale near-white margin). A pale halo tube drawn just
# underneath a slightly thicker, fully-opaque dark line -- the same
# dark-text/white-stroke trick used for the map labels -- keeps it legible
# against both light and dark terrain.
FLIGHTLINE_COLOR = "#1a1a1a"
FLIGHTLINE_RADIUS = 0.7
FLIGHTLINE_HALO_COLOR = "white"
FLIGHTLINE_HALO_RADIUS = 1.4

# Ice-shelf text labels: (label text, true anchor lon, true anchor lat,
# text_fx, text_fy). text_fx/text_fy place the label TEXT at a fixed,
# clear spot (fraction of panel width/height, fy=0 at panel top) -- see the
# label-drawing loop in compose_figure() for why the text is offset from its
# true anchor via a leader line rather than drawn at the anchor directly.
ICE_SHELF_LABELS = [
    ("Ninnis Glacier / Ice Shelf", 147.5, -68.3, 0.62, 0.50),
    ("Cook Glacier / Ice Shelf", 152.5, -68.6, 0.37, 0.50),
]

CAMERA_DIST_FACTOR = 1.50  # camera distance as a multiple of domain diagonal
CAMERA_INLAND_SHIFT = -0.1  # focal point shift inland, as a fraction of domain radius
VIEW_ELEVATION_DEG = 25.0
# Rotates the camera's horizontal "outward" direction (see compute_camera())
# around the vertical axis, in degrees, before applying CAMERA_DIST_FACTOR/
# CAMERA_INLAND_SHIFT. 0 = the default: camera sits seaward (away from the
# South Pole) of the domain centroid, which is what puts north at the bottom
# of the frame. Positive = counterclockwise as viewed from above (standard
# maths convention); use this to pan the oblique view around the domain
# without having to re-derive the rest of the camera geometry.
CAMERA_AZIMUTH_DEG = -10.0

# Panel (b) bed/bathymetry colour scale is clipped (saturated) at +-this
# elevation (m) rather than auto-scaled to the data's 98th percentile, so the
# colour scale is a fixed, citable value rather than one that silently shifts
# if the AOI or data source changes.
BED_ELEV_SATURATION_M = 2000.0

# Panel (a) ice-surface colour scale range (m): a fixed, citable range (0 m at
# the coast/margin to the approximate summit elevation of the AOI's interior
# plateau) rather than one that auto-scales to the data's actual min/max.
ICE_SURF_ELEV_CLIM_M = (0.0, 3600.0)

INSET_ENABLE = True
INSET_LAT_LIMIT = -55.0  # southern-hemisphere latitude the inset map extends to

COLUMN_WIDTH_MM = 183.0  # Nature: 89 mm single column, 183 mm double column
PANEL_ASPECT = 1.42  # width / height of each individual panel
# Vertical gap between panel (a) and panel (b). Zero: both renders have solid
# black backgrounds, so the two panels sit flush with no visible (non-black)
# seam between them -- panel identity comes from the "a"/"b" labels, not from
# a dividing strip.
PANEL_GAP_MM = 0.0
# Shared legend strip below both panels (white matplotlib background, not
# overlaid on the terrain render) -- keeps the line-type legend entries out of
# the rendered scenes entirely, so panel a's and panel b's actual 3-D imagery
# can sit directly adjacent to one another across PANEL_GAP_MM alone.
FOOTER_HEIGHT_MM = 24.0
PANEL_LABELS = ("a", "b")

FONT_FAMILY = "Arial"
LABEL_FONTSIZE = 7
CALLOUT_FONTSIZE = 7
EMPHASIS_FONTSIZE = 8
TITLE_FONTSIZE = 9

PNG_DPI = 600
# Panels are rendered at RENDER_SUPERSAMPLE x the final print pixel size,
# then explicitly downsampled (PIL LANCZOS -- a real low-pass filter) back
# down before compositing. This matters much more here than in a normal
# figure: the pin/hedgehog terrain is a dense field of thousands of
# near-parallel thin lines, and in-renderer antialiasing alone leaves a
# visible moire/aliasing pattern (regular pale vertical streaks) where
# column spacing beats against the pixel grid under perspective.
# Capped just under 8192 px per side: this headless GL context (software
# rasterizer, no DRI3) was found by testing to fail silently (blank/black
# screenshot, "Frame buffer object was not initialized correctly") above
# that texture size, with print-resolution panels around 4300 px wide.
RENDER_SUPERSAMPLE = 1.8

# ---------------------------------------------------------------------------


OLERON_CMAP_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colormaps", "oleron.txt")


def load_oleron_cmap():
    """Oleron: a unified hypsometric/bathymetric scale (dark blue -> pale ->
    green/brown) from Fabio Crameri's Scientific Colour Maps v8
    (https://zenodo.org/records/8409685), used here as a single continuous
    scale across land and ocean bed elevation together -- not separately
    normalised -- so panel (b) reads as one physical surface. Used with
    TwoSlopeNorm(vcenter=0), so the scale's midpoint always lands exactly on
    bed = 0 m regardless of the data's actual min/max."""
    rgb = np.loadtxt(OLERON_CMAP_TXT)
    return ListedColormap(rgb, name="oleron")


BATHY_CMAP = load_oleron_cmap()


def _set_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (FONT_FAMILY, "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            return candidate
    plt.rcParams["font.family"] = "sans-serif"
    return "sans-serif"


def load_region():
    """Subset BedMachine to a rectangular EPSG:3031 crop around the lon/lat
    box, then build a boolean mask of points that are truly inside the
    lon/lat box (a lon/lat rectangle is a curved sector in polar
    stereographic space, so its bounding box is *not* the same shape)."""
    ds = xr.open_dataset(BEDMACHINE_NC)

    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    lons = np.linspace(LON_MIN, LON_MAX, 400)
    lats = np.linspace(LAT_MIN, LAT_MAX, 400)
    LON, LAT = np.meshgrid(lons, lats)
    X, Y = to_xy.transform(LON, LAT)
    xmin, xmax, ymin, ymax = X.min(), X.max(), Y.min(), Y.max()
    pad = 20_000  # m
    sub = ds.sel(x=slice(xmin - pad, xmax + pad), y=slice(ymax + pad, ymin - pad))

    sub = sub.isel(x=slice(0, None, DECIMATE_STRIDE), y=slice(0, None, DECIMATE_STRIDE))

    x = sub["x"].values.astype(np.float64)
    y = sub["y"].values.astype(np.float64)
    Xg, Yg = np.meshgrid(x, y)

    to_ll = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon_g, lat_g = to_ll.transform(Xg, Yg)
    inbox = (
        (lon_g >= LON_MIN) & (lon_g <= LON_MAX) & (lat_g >= LAT_MIN) & (lat_g <= LAT_MAX)
    )

    return {
        "x": Xg,
        "y": Yg,
        "lon": lon_g,
        "lat": lat_g,
        "inbox": inbox,
        "bed": sub["bed"].values.astype(np.float64),
        "surface": sub["surface"].values.astype(np.float64),
        "thickness": sub["thickness"].values.astype(np.float64),
        "mask": sub["mask"].values,
    }


def load_velocity(region):
    """MEaSUREs Antarctic ice velocity (Mouginot et al., NASA MEaSUREs /
    NSIDC), native 450 m EPSG:3031 grid. Cropped to the AOI's bounding box
    *before* loading values (the full file is ~7 GB), then resampled
    (nearest neighbour -- adequate since the rendered grid is coarser, 1.5 km,
    than the native 450 m velocity grid) onto the exact x/y coordinates of
    the rendered surfaces so it can be draped on the ice mesh directly and
    (via VX/VY) integrated into glacier flow lines. Returns None (rather than
    raising) if the file is missing, so the figure can still be produced
    without the velocity layer."""
    if not os.path.exists(MEASURES_VELOCITY_NC):
        print(f"Note: velocity file not found at {MEASURES_VELOCITY_NC} -- skipping velocity layer.")
        return None

    ds = xr.open_dataset(MEASURES_VELOCITY_NC)
    x1d = region["x"][0, :]
    y1d = region["y"][:, 0]
    pad = 5_000.0
    sub = ds.sel(
        x=slice(x1d.min() - pad, x1d.max() + pad),
        y=slice(y1d.max() + pad, y1d.min() - pad),
    )
    sub = sub.interp(x=x1d, y=y1d, method="nearest")
    vx = sub["VX"].values.astype(np.float64)
    vy = sub["VY"].values.astype(np.float64)
    return {"vx": vx, "vy": vy, "speed": np.hypot(vx, vy)}


def compute_flowlines(region, vx, vy):
    """Glacier flow lines traced from the MEaSUREs velocity field by direct
    downstream integration (fixed-arc-length midpoint/RK2 stepping), seeded on
    a regular grid over grounded ice at FLOWLINE_SEED_SPACING_KM. Each seed is
    followed in the direction the ice is actually moving (the velocity vector
    itself, not its negative) until it leaves grounded ice/the AOI or the
    local speed drops below FLOWLINE_MIN_SPEED_MYR.

    This replaces an earlier matplotlib-streamplot-based version: streamplot's
    output (sp.lines.get_segments()) is a flat bag of tiny, unordered 2-point
    fragments with no record of which fragments belong to the same line or
    which end is upstream -- fine for just drawing lines, but unusable for
    showing flow DIRECTION or controlling seed placement to reveal tributary
    convergence. Tracing our own lines keeps each one as a single ordered
    polyline from its seed (upstream/interior) to where it stops (typically
    downstream, near the coast) -- exactly what's needed both to place a
    direction arrowhead (see FLOWLINE_ARROW_* and compose_figure) and to seed
    densely enough that nearby tributaries visibly merge into shared trunk
    paths, the way real outlet glaciers do.

    Returns a list of (N, 2) x/y polyline arrays in metres -- the same "list
    of polyline arrays" shape load_bedmap_flightlines() returns, so both feed
    the same generic draping function (drape_lines_mesh)."""
    x1d = region["x"][0, :]
    y1d = region["y"][:, 0]
    dx = float(x1d[1] - x1d[0])
    dy = float(y1d[1] - y1d[0])  # BedMachine y descends north-to-south -> dy < 0

    grounded = region["inbox"] & (region["mask"] == 2)
    vx_f = np.where(grounded, np.nan_to_num(vx), 0.0).astype(np.float64)
    vy_f = np.where(grounded, np.nan_to_num(vy), 0.0).astype(np.float64)
    grounded_f = grounded.astype(np.float64)

    def sample(arr, xq, yq):
        col = (xq - x1d[0]) / dx
        row = (yq - y1d[0]) / dy
        return map_coordinates(arr, [row, col], order=1, mode="constant", cval=0.0)

    xs = np.arange(x1d.min(), x1d.max(), FLOWLINE_SEED_SPACING_KM * 1000.0)
    ys = np.arange(y1d.min(), y1d.max(), FLOWLINE_SEED_SPACING_KM * 1000.0)
    Xs, Ys = np.meshgrid(xs, ys)
    seed_x = Xs.ravel()
    seed_y = Ys.ravel()
    keep = sample(grounded_f, seed_x, seed_y) > 0.5
    seed_x, seed_y = seed_x[keep], seed_y[keep]

    step_m = FLOWLINE_STEP_KM * 1000.0
    n = len(seed_x)
    x = seed_x.copy()
    y = seed_y.copy()
    alive = np.ones(n, dtype=bool)
    paths = [[(seed_x[i], seed_y[i])] for i in range(n)]

    for _ in range(FLOWLINE_MAX_STEPS):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        xa, ya = x[idx], y[idx]

        vx0, vy0 = sample(vx_f, xa, ya), sample(vy_f, xa, ya)
        speed0 = np.hypot(vx0, vy0)
        ok0 = speed0 >= FLOWLINE_MIN_SPEED_MYR
        s0 = np.where(ok0, speed0, 1.0)
        xm = xa + (vx0 / s0) * step_m * 0.5
        ym = ya + (vy0 / s0) * step_m * 0.5

        vxm, vym = sample(vx_f, xm, ym), sample(vy_f, xm, ym)
        speedm = np.hypot(vxm, vym)
        okm = speedm >= FLOWLINE_MIN_SPEED_MYR
        sm = np.where(okm, speedm, 1.0)
        x_new = xa + (vxm / sm) * step_m
        y_new = ya + (vym / sm) * step_m

        g_new = sample(grounded_f, x_new, y_new) > 0.5
        step_ok = ok0 & okm

        for local_i, global_i in enumerate(idx):
            if not step_ok[local_i]:
                alive[global_i] = False
                continue
            paths[global_i].append((x_new[local_i], y_new[local_i]))
            x[global_i] = x_new[local_i]
            y[global_i] = y_new[local_i]
            if not g_new[local_i]:
                alive[global_i] = False

    lines = [np.asarray(p) for p in paths if len(p) >= 4]

    # Attach per-vertex ice speed (m/yr) as a 3rd column, re-sampling vx_f/vy_f
    # at each finalized vertex with the same bilinear sample() used while
    # tracing -- this is what lets the flow lines be coloured by speed rather
    # than drawn in a single flat colour (see FLOWLINE_SPEED_CLIM_MYR/
    # FLOWLINE_SPEED_CMAP and render_panel()).
    lines_with_speed = []
    for line in lines:
        spd = np.hypot(sample(vx_f, line[:, 0], line[:, 1]), sample(vy_f, line[:, 0], line[:, 1]))
        lines_with_speed.append(np.column_stack([line, spd]))
    return lines_with_speed


def load_ibcso_direct_points(region):
    """IBCSO v2 Type-ID (TID) grid, cropped to the AOI, subsampled to a coarse
    (2 km) stride, and filtered to cells flagged as direct soundings (single/
    multibeam, seismic, lidar, isolated -- codes in IBCSO_DIRECT_CODES) rather
    than gravity-predicted or interpolated fill. Deliberately sparse and
    coarse: the point is to show *how little* of the seafloor here has actual
    soundings, not to produce a filled coverage map. Returns (x_km, y_km, z_km)
    EPSG:3031 arrays restricted to open ocean within the AOI -- z_km is the
    BedMachine bed elevation *bilinearly interpolated* at each point's exact
    location, plus a small lift, so points sit on or above the rendered
    seafloor -- or empty arrays if the file is missing."""
    if not os.path.exists(IBCSO_TID_TIF):
        print(f"Note: IBCSO TID file not found at {IBCSO_TID_TIF} -- skipping seafloor-data-coverage layer.")
        return np.array([]), np.array([]), np.array([])

    to_ibcso = Transformer.from_crs("EPSG:4326", "EPSG:9354", always_xy=True)
    lons = np.linspace(LON_MIN, LON_MAX, 200)
    lats = np.linspace(LAT_MIN, LAT_MAX, 200)
    LONg, LATg = np.meshgrid(lons, lats)
    Xi, Yi = to_ibcso.transform(LONg, LATg)
    pad = 20_000.0

    with rasterio.open(IBCSO_TID_TIF) as src:
        win = from_bounds(
            Xi.min() - pad, Yi.min() - pad, Xi.max() + pad, Yi.max() + pad,
            transform=src.transform,
        ).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        win_transform = src.window_transform(win)

    sub = arr[::IBCSO_STRIDE, ::IBCSO_STRIDE]
    rows, cols = np.meshgrid(
        np.arange(0, arr.shape[0], IBCSO_STRIDE),
        np.arange(0, arr.shape[1], IBCSO_STRIDE),
        indexing="ij",
    )
    direct = np.isin(sub, list(IBCSO_DIRECT_CODES))
    if not direct.any():
        return np.array([]), np.array([]), np.array([])

    xs, ys = rasterio.transform.xy(win_transform, rows[direct], cols[direct])
    xs, ys = np.asarray(xs), np.asarray(ys)

    to_ll = Transformer.from_crs("EPSG:9354", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(xs, ys)
    inbox_ll = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
    if not inbox_ll.any():
        return np.array([]), np.array([]), np.array([])

    to_3031 = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x3031, y3031 = to_3031.transform(lon[inbox_ll], lat[inbox_ll])

    x1d = region["x"][0, :]
    y1d = region["y"][:, 0]
    ix = np.clip(np.searchsorted(x1d, x3031), 0, len(x1d) - 1)
    iy = np.clip(np.searchsorted(-y1d, -y3031), 0, len(y1d) - 1)
    is_ocean = region["mask"][iy, ix] == 0

    # A nearest-grid-cell bed lookup here leaves a majority of points sitting
    # *below* the rendered surface at their true continuous position, since
    # PyVista/VTK render a piecewise-bilinear surface between grid nodes --
    # bilinearly interpolating (below) matches that instead.
    order = np.argsort(y1d)
    bed_interp = RegularGridInterpolator(
        (y1d[order], x1d), region["bed"][order, :], bounds_error=False, fill_value=None
    )
    bed_m = bed_interp(np.column_stack([y3031[is_ocean], x3031[is_ocean]]))
    lift_km = 0.15  # generous lift (well above the ~50 m-scale interpolation residual)
    z_km = bed_m * VERT_EXAG / 1000.0 + lift_km
    return x3031[is_ocean] / 1000.0, y3031[is_ocean] / 1000.0, z_km


def load_bedmap_flightlines(region):
    """BEDMAP3 standardised survey points (bm3_streamlines_pt), clipped to the
    AOI bounding box and grouped into per-flight-line point sequences by the
    `streamln` field. Points within a line are ordered by `streampt`, which
    (checked against actual point spacing -- median ~500 m, matching
    BedMachine's native grid) increases monotonically along a flight, i.e. it
    is a genuine acquisition-order index, not just a global point ID.

    Each line is further split into contiguous runs over grounded ice
    (BedMachine mask == 2) only: a survey line's ground track over open ocean
    or floating ice shelf is not a subglacial-bed measurement here, and
    without this split, a single straight tube would jump across those gaps.
    Returns a list of (N, 2) x/y arrays in metres, one per grounded-ice run,
    or an empty list if the file is missing."""
    if not os.path.exists(BEDMAP_FLIGHTLINES_SHP):
        print(f"Note: BEDMAP3 flight-line file not found at {BEDMAP_FLIGHTLINES_SHP} -- skipping flight-line layer.")
        return []

    gdf = gpd.read_file(BEDMAP_FLIGHTLINES_SHP)
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values

    to_ll = Transformer.from_crs("EPSG:3031", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(x, y)
    inbox = (lon >= LON_MIN) & (lon <= LON_MAX) & (lat >= LAT_MIN) & (lat <= LAT_MAX)
    if not inbox.any():
        return []

    sub = gdf.loc[inbox, ["streamln", "streampt"]].copy()
    sub["x"] = x[inbox]
    sub["y"] = y[inbox]
    sub = sub.sort_values(["streamln", "streampt"])

    x1d = region["x"][0, :]
    y1d = region["y"][:, 0]

    lines = []
    for _, grp in sub.groupby("streamln", sort=False):
        if len(grp) < FLIGHTLINE_MIN_POINTS:
            continue
        xy = grp[["x", "y"]].to_numpy()
        ix = np.clip(np.searchsorted(x1d, xy[:, 0]), 0, len(x1d) - 1)
        iy = np.clip(np.searchsorted(-y1d, -xy[:, 1]), 0, len(y1d) - 1)
        grounded = region["mask"][iy, ix] == 2
        # Contiguous True-runs of `grounded`, as [start, stop) index pairs.
        edges = np.flatnonzero(np.diff(np.concatenate([[False], grounded, [False]])))
        for start, stop in edges.reshape(-1, 2):
            if stop - start < FLIGHTLINE_MIN_POINTS:
                continue
            lines.append(xy[start:stop])
    return lines


def _aoi_sector_polygon():
    """The AOI's true lon/lat box, as a polygon in EPSG:3031 x/y space.

    A lon/lat box maps to a curved "pie slice" sector in polar-stereographic
    x/y space, not a rectangle. Clipping against the axis-aligned rectangle
    that bounds that sector instead (as an earlier version of this code did)
    still admits geometry from well outside [LON_MIN, LON_MAX] that happens
    to fall in the rectangle's corners -- this produced stray, disconnected
    squiggles at the left/right edges of the rendered scene. Clipping against
    the actual sector polygon (built from the true lon/lat boundary, densely
    sampled so its arcs are faithful in x/y space) fixes this."""
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    n = 400
    lon_bottom = np.linspace(LON_MIN, LON_MAX, n)
    lat_right = np.linspace(LAT_MIN, LAT_MAX, n)
    lon_top = np.linspace(LON_MAX, LON_MIN, n)
    lat_left = np.linspace(LAT_MAX, LAT_MIN, n)

    boundary_lon = np.concatenate(
        [lon_bottom, np.full(n, LON_MAX), lon_top, np.full(n, LON_MIN)]
    )
    boundary_lat = np.concatenate(
        [np.full(n, LAT_MIN), lat_right, np.full(n, LAT_MAX), lat_left]
    )
    bx, by = to_xy.transform(boundary_lon, boundary_lat)

    from shapely.geometry import Polygon

    return Polygon(np.column_stack([bx, by])).buffer(20_000)


def _clip_to_aoi_sector(gdf, sector):
    prefilter = gdf.cx[sector.bounds[0] : sector.bounds[2], sector.bounds[1] : sector.bounds[3]]
    if prefilter.empty:
        return prefilter
    return gpd.clip(prefilter, sector)


def _read_moa_shp(path):
    """MOA vector products ship a Polar Stereographic (true-scale 71S) .prj
    without an EPSG tag; the parameters are numerically identical to
    EPSG:3031, so the CRS is reassigned (not reprojected) to make that
    explicit and let gpd.clip compare directly against the AOI sector, which
    is built in EPSG:3031."""
    gdf = gpd.read_file(path)
    return gdf.set_crs("EPSG:3031", allow_override=True)


def load_grounding_line():
    """Load the MOA grounding line, clipped to the true lon/lat AOI (see
    _aoi_sector_polygon)."""
    gdf = _read_moa_shp(MOA_GROUNDINGLINE_SHP)
    return _clip_to_aoi_sector(gdf, _aoi_sector_polygon())


def load_moa_coastline():
    """Load the MOA coastline (rock/ice edge against open water, including
    ice-shelf fronts -- distinct from the grounding line, which marks the
    grounded/floating ice boundary further inland), clipped to the AOI."""
    gdf = _read_moa_shp(MOA_COASTLINE_SHP)
    return _clip_to_aoi_sector(gdf, _aoi_sector_polygon())


def load_moa_islands():
    """Load MOA island outlines intersecting the AOI, as line geometry (the
    polygon boundary) ready for the same draping function used for the
    grounding line and coastline."""
    gdf = _read_moa_shp(MOA_ISLANDS_SHP)
    clipped = _clip_to_aoi_sector(gdf, _aoi_sector_polygon())
    if clipped.empty:
        return clipped
    clipped = clipped.copy()
    clipped["geometry"] = clipped.geometry.boundary
    return clipped


def load_full_continent_outline():
    """Full-continent grounding line, for the inset locator map -- deliberately
    NOT clipped to the AOI (unlike load_grounding_line()) and heavily
    simplified since it is only ever rendered a couple of cm across."""
    gdf = gpd.read_file(GROUNDINGLINE_SHP)
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.simplify(5000.0, preserve_topology=False)
    return gdf


def add_inset_map(fig, rect_xy=(0.015, 0.64, 0.28, 0.28)):
    """Small circumpolar locator map: whole-Antarctica coastline (from the
    same InSAR grounding-line dataset, unclipped), a box marking the WSB AOI,
    and bearing arrows toward the southern tips of the three nearest
    continents (direction only, not to scale -- real distances at this
    latitude range from ~950 km to South America to ~3500 km to Africa,
    far more than the ~30 mm inset could show to scale)."""
    continent = load_full_continent_outline()

    ax = fig.add_axes(rect_xy)
    ax.set_aspect("equal")
    ax.axis("off")

    for geom in continent.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            coords = np.asarray(part.coords) / 1000.0  # km
            ax.plot(coords[:, 0], coords[:, 1], color="#555555", lw=0.35, zorder=2)

    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    n = 200
    lon_bottom = np.linspace(LON_MIN, LON_MAX, n)
    lat_right = np.linspace(LAT_MIN, LAT_MAX, n)
    lon_top = np.linspace(LON_MAX, LON_MIN, n)
    lat_left = np.linspace(LAT_MAX, LAT_MIN, n)
    boundary_lon = np.concatenate([lon_bottom, np.full(n, LON_MAX), lon_top, np.full(n, LON_MIN)])
    boundary_lat = np.concatenate([np.full(n, LAT_MIN), lat_right, np.full(n, LAT_MAX), lat_left])
    bx, by = to_xy.transform(boundary_lon, boundary_lat)
    ax.plot(bx / 1000.0, by / 1000.0, color="#c9184a", lw=1.1, zorder=4)

    r_lim = np.hypot(*to_xy.transform(0.0, INSET_LAT_LIMIT)) / 1000.0

    targets = {"South\nAmerica": -68.0, "Africa": 20.0, "Australia": 147.0}
    for label, lon in targets.items():
        tx, ty = to_xy.transform(lon, INSET_LAT_LIMIT)
        tx, ty = tx / 1000.0, ty / 1000.0
        u = np.hypot(tx, ty)
        ux, uy = tx / u, ty / u
        r0, r1 = r_lim * 0.88, r_lim * 1.0
        ax.annotate(
            "", xy=(ux * r1, uy * r1), xytext=(ux * r0, uy * r0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5, mutation_scale=5),
            zorder=5,
        )
        ax.text(ux * r1 * 1.12, uy * r1 * 1.12, label, fontsize=LABEL_FONTSIZE - 2,
                ha="center", va="center")

    circle = plt.Circle((0, 0), r_lim, fill=False, edgecolor="#999999", lw=0.4)
    ax.add_patch(circle)
    ax.set_xlim(-r_lim * 1.3, r_lim * 1.3)
    ax.set_ylim(-r_lim * 1.3, r_lim * 1.3)


def pin_columns_mesh(x_km, y_km, top_z_km, base_z_km, valid, substride=1, scalars=None, rgb=None):
    """Vertical "pin"/hedgehog terrain: one thin line per surviving grid
    point, from a common flat base up to that point's (exaggerated)
    elevation. Intended to be rendered with render_lines_as_tubes=True (a
    cheap GPU-shaded line, not real tube geometry -- essential at this point
    count: hundreds of thousands to millions of columns).

    This -- not a smoothed continuous surface -- is the terrain rendering
    technique used throughout this figure: real, densely-sampled small-scale
    relief makes neighbouring pins land at very different heights (a rough,
    bristly silhouette); coarsely interpolated/no-data regions vary little
    from pin to pin (a smooth-looking surface). Data sparsity is thus visible
    directly in the render, without needing a separate coverage overlay.

    `base_z_km` may be a scalar (one common flat floor) or an array matching
    x_km's shape (a per-column base, e.g. sea level only over ice). Returns
    None if no points survive `valid`."""
    xs = x_km[::substride, ::substride]
    ys = y_km[::substride, ::substride]
    tops = top_z_km[::substride, ::substride]
    v = valid[::substride, ::substride]
    if not v.any():
        return None

    xs = xs[v]
    ys = ys[v]
    tops = tops[v]
    if np.isscalar(base_z_km):
        bases = np.full_like(tops, float(base_z_km))
    else:
        bases = base_z_km[::substride, ::substride][v]

    n = xs.size
    pts = np.empty((2 * n, 3), dtype=np.float64)
    pts[0::2, 0] = xs
    pts[0::2, 1] = ys
    pts[0::2, 2] = bases
    pts[1::2, 0] = xs
    pts[1::2, 1] = ys
    pts[1::2, 2] = tops

    lines = np.empty((n, 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(0, 2 * n, 2)
    lines[:, 2] = np.arange(1, 2 * n, 2)

    poly = pv.PolyData(pts, lines=lines.ravel())  # no vertex cells (point sprites)

    if rgb is not None:
        rgb_v = rgb[::substride, ::substride, :][v]
        poly["pin_rgb"] = np.repeat(rgb_v, 2, axis=0)
    elif scalars is not None:
        s_v = scalars[::substride, ::substride][v]
        poly["pin_scalar"] = np.repeat(s_v, 2)

    return poly


def lines_from_geodataframe(gdf, simplify_tol_m=800.0):
    """Convert a line/multiline GeoDataFrame (grounding line, coastline, or
    island outlines) to the plain "list of (N, 2) x/y arrays, metres" format
    shared with load_bedmap_flightlines() / compute_flowlines(), simplifying
    each geometry first: native shapefile vertex spacing is much finer than
    the 1.5 km decimated surface grid, so draping every vertex at its
    nearest-neighbour elevation makes the tube zigzag visibly once
    exaggerated -- simplifying removes that jitter without changing the
    outline's shape at print scale."""
    lines = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        geom = geom.simplify(simplify_tol_m, preserve_topology=False)
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            coords = np.asarray(part.coords)
            if coords.shape[0] >= 2:
                lines.append(coords)
    return lines


def drape_lines_mesh(lines, region, elev_m, radius=0.6, lift_km=1e-3, local_max_cells=0):
    """Drape a list of (N, 2) x/y polylines (metres, EPSG:3031) onto a given
    TRUE-elevation field, returning a PyVista tube, or None if there are no
    lines. Generic over both the source (MOA grounding line/coastline/
    islands, BEDMAP3 flight lines, MEaSUREs-derived flow lines) and the
    target surface -- pass region["surface"] to drape on the ice panel,
    region["bed"] to drape on the bed/bathymetry panel.

    local_max_cells: if > 0, the height used is the MAXIMUM of elev_m over a
    (2*local_max_cells+1)-wide window centred on each point's nearest grid
    cell, instead of that single cell's own value. This matters once
    VERT_EXAG is applied: local terrain roughness (e.g. crevasse fields/
    icefalls right where fast outlet-glacier flow lines cross) of a few
    hundred true metres becomes many exaggerated km -- easily dwarfing any
    reasonably-sized fixed lift_km and burying the line inside the
    surrounding pin/hedgehog terrain. Sitting the line at the local peak
    instead (plus a modest lift_km on top, just for a clean visual gap) is
    what actually guarantees clearance, rather than a fixed lift chosen to
    cover the worst case everywhere (which would look like it's floating far
    above the surface everywhere else).

    Lines may optionally carry a 3rd column (e.g. compute_flowlines()'s
    per-vertex ice speed in m/yr); when present it is attached to the
    returned tube as point-data array "speed", so render_panel() can colour
    the tube by that scalar (VTK interpolates point data across the tube's
    circumferential rings automatically)."""
    if not lines:
        return None

    x = region["x"][0, :]
    y = region["y"][:, 0]
    has_scalar = lines[0].shape[1] >= 3

    if local_max_cells > 0:
        elev_lookup = maximum_filter(np.nan_to_num(elev_m, nan=-1e9), size=2 * local_max_cells + 1)
    else:
        elev_lookup = elev_m

    pts = []
    line_conn = []
    scalars = []
    offset = 0
    for coords in lines:
        if coords.shape[0] < 2:
            continue
        ix = np.clip(np.searchsorted(x, coords[:, 0]), 0, len(x) - 1)
        iy = np.clip(np.searchsorted(-y, -coords[:, 1]), 0, len(y) - 1)
        z_m = elev_lookup[iy, ix]
        xyz = np.column_stack([
            coords[:, 0] / 1000.0,
            coords[:, 1] / 1000.0,
            z_m * VERT_EXAG / 1000.0 + lift_km,
        ])
        pts.append(xyz)
        if has_scalar:
            scalars.append(coords[:, 2])
        n = xyz.shape[0]
        line = np.hstack([[n], np.arange(offset, offset + n)])
        line_conn.append(line)
        offset += n

    if not pts:
        return None

    all_pts = np.vstack(pts)
    all_lines = np.concatenate(line_conn)
    poly = pv.PolyData(all_pts, lines=all_lines)
    if has_scalar:
        poly["speed"] = np.concatenate(scalars)
    return poly.tube(radius=radius)


def build_meshes(region):
    x_km = region["x"] / 1000.0
    y_km = region["y"] / 1000.0
    inbox = region["inbox"]
    mask = region["mask"]

    def z_km(elev_m):
        return elev_m * VERT_EXAG / 1000.0

    bed_z = z_km(region["bed"])
    ice_visible = inbox & np.isin(mask, [2, 3])
    ice_top_z = z_km(region["surface"])

    # Contour-only helper mesh (never added to the render itself): traces the
    # bed = 0 m isoline for the sea-level reference line.
    bed_contour_mesh = pv.StructuredGrid(x_km, y_km, bed_z)
    bed_contour_mesh["bed_m"] = region["bed"].ravel(order="F")
    bed_contour_mesh.hide_points(~inbox.ravel(order="F"))
    sea_level_line = bed_contour_mesh.contour(isosurfaces=[0.0], scalars="bed_m")

    # Ice-surface colour: continuous white (low elevation, near the
    # coast/grounding line) to dark blue (high elevation, interior plateau),
    # using matplotlib's "Blues" ColorBrewer scale as-is. "Blues" is a
    # single-hue sequential scale -- monotonic in brightness, so it still
    # reads correctly in greyscale/print and is colourblind-safe -- unlike a
    # velocity-style scheme, this reads the AOI's actual topography (summit
    # of the interior plateau vs. low-lying coastal/outlet-glacier ice).
    surf_lo, surf_hi = ICE_SURF_ELEV_CLIM_M
    surf_filled_m = np.where(np.isnan(region["surface"]), surf_lo, region["surface"])
    surf_frac = np.clip((surf_filled_m - surf_lo) / (surf_hi - surf_lo), 0.0, 1.0)
    rgb = plt.get_cmap("Blues")(surf_frac)[..., :3]

    # Hillshade panel (a)'s colour from the TRUE ice-surface elevation
    # (matplotlib's standard relief-shading algorithm, not VTK lighting --
    # see the `lighting=False` note in render_panel for why VTK's own
    # Phong/specular lighting is disabled on the pin meshes instead of relied
    # on here). Pin HEIGHT alone does not read as relief at this oblique, low
    # camera angle, especially over the near-flat East Antarctic interior
    # plateau; this gives panel (a) a genuine, independent elevation cue
    # (light from the northwest) on top of the velocity colouring, using the
    # same VERT_EXAG so the shading matches the exaggerated pin heights.
    # A plain multiplicative brightness band (not LightSource.shade_rgb's
    # HSV/soft blend modes) is used deliberately: those blend modes can push
    # already-saturated highlight colour toward near-black on shadowed
    # slopes, which over-reads as "high velocity everywhere" rather than
    # relief. Scaling brightness within a fixed +/-45% band preserves hue.
    dx = float(abs(region["x"][0, 1] - region["x"][0, 0]))
    dy = float(abs(region["y"][1, 0] - region["y"][0, 0]))
    ls = LightSource(azdeg=315, altdeg=45)
    surface_filled = np.where(np.isnan(region["surface"]), 0.0, region["surface"])
    hs = ls.hillshade(surface_filled, vert_exag=VERT_EXAG, dx=dx, dy=dy)
    shade = (1.0 - HILLSHADE_STRENGTH) + 2.0 * HILLSHADE_STRENGTH * hs
    rgb = rgb * shade[..., None]
    ice_rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    # Pin/hedgehog terrain columns -- see pin_columns_mesh() docstring for
    # why this replaces a smooth continuous surface.
    bed_floor_km = float(np.nanmin(bed_z[inbox])) - 0.05 * (
        float(np.nanmax(bed_z[inbox])) - float(np.nanmin(bed_z[inbox]))
    )
    bed_pins = pin_columns_mesh(
        x_km, y_km, bed_z, bed_floor_km, inbox,
        substride=VOXEL_SUBSTRIDE, scalars=region["bed"],
    )
    ice_pins = pin_columns_mesh(
        x_km, y_km, ice_top_z, 0.0, ice_visible,
        substride=VOXEL_SUBSTRIDE, rgb=ice_rgb,
    )

    return {
        "bed_pins": bed_pins,
        "ice_pins": ice_pins,
        "bed_m": region["bed"],
        "sea_level_line": sea_level_line,
        "ice_bool": ice_visible,
        "x_km": x_km,
        "y_km": y_km,
        "bed_z": bed_z,
        "ice_z": ice_top_z,
    }


def ibcso_points_mesh(x_km, y_km, z_km):
    """Build a PyVista point cloud from IBCSO direct-sounding locations,
    already in the render grid's km/VERT_EXAG convention. Returns None if
    there are no points."""
    if x_km.size == 0:
        return None
    pts = np.column_stack([x_km, y_km, z_km + 0.05])  # tiny lift off the bed pins
    return pv.PolyData(pts)


def compute_camera(meshes, region, azimuth_deg=CAMERA_AZIMUTH_DEG):
    valid = region["inbox"]
    cx = meshes["x_km"][valid].mean()
    cy = meshes["y_km"][valid].mean()

    r = np.hypot(cx, cy)
    outward = np.array([cx, cy]) / r  # away from South Pole (0,0) = seaward
    if azimuth_deg:
        theta = np.deg2rad(azimuth_deg)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        outward = rot @ outward

    xspan = meshes["x_km"][valid].max() - meshes["x_km"][valid].min()
    yspan = meshes["y_km"][valid].max() - meshes["y_km"][valid].min()
    diag = np.hypot(xspan, yspan)

    dist = diag * CAMERA_DIST_FACTOR
    elev = np.deg2rad(VIEW_ELEVATION_DEG)
    cam_horiz = np.array([cx, cy]) + outward * dist * np.cos(elev)
    cam_z = dist * np.sin(elev)

    focal_xy = np.array([cx, cy]) - outward * diag * CAMERA_INLAND_SHIFT
    focal_z = np.nanmedian(meshes["bed_z"][valid])

    camera_pos = (cam_horiz[0], cam_horiz[1], cam_z)
    focal_point = (focal_xy[0], focal_xy[1], focal_z)
    view_up = (0.0, 0.0, 1.0)
    return camera_pos, focal_point, view_up


def world_to_display(plotter, xyz):
    """Project a world-space point to 2D pixel coordinates in the *downsampled*
    (target print resolution) image frame that compose_figure()'s axes are
    built on -- NOT the plotter's own window_size, which is rendered at
    RENDER_SUPERSAMPLE x that resolution (see render_panel()/main()). Dividing
    out RENDER_SUPERSAMPLE here keeps every caller (labels, scale bar)
    consistent with img_ice.shape / img_bed.shape without each needing to
    know about the supersample factor itself."""
    coord = vtk.vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    coord.SetValue(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    disp = coord.GetComputedDisplayValue(plotter.renderer)
    w, h = plotter.window_size
    return disp[0] / RENDER_SUPERSAMPLE, (h - disp[1]) / RENDER_SUPERSAMPLE


def content_extent(img, thresh=10, pad_frac=0.015):
    """(top, bottom, left, right) bounding box of `img` containing any
    non-background pixel (the render background is solid black), with a
    small padding fraction of each dimension kept on each side so the crop
    doesn't clip anti-aliased edges. Used to trim the blank canvas around the
    actual rendered terrain in each panel, so panel a's and panel b's real
    data can sit directly adjacent (vertically) and fill the full column
    width (horizontally) instead of floating in a mostly-black frame."""
    mask = np.any(img[..., :3] > thresh, axis=-1)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, img.shape[0], 0, img.shape[1]
    pad_r = int(round(img.shape[0] * pad_frac))
    pad_c = int(round(img.shape[1] * pad_frac))
    top = max(0, rows.min() - pad_r)
    bot = min(img.shape[0], rows.max() + 1 + pad_r)
    left = max(0, cols.min() - pad_c)
    right = min(img.shape[1], cols.max() + 1 + pad_c)
    return top, bot, left, right


def label_world_xyz(lon, lat, region):
    """World-space (km) position of a lon/lat point draped on the ice's true
    (exaggerated) upper surface, using the same nearest-neighbour surface
    lookup as the other ice-panel drapes, so a label lands on the rendered
    ice sheet rather than floating above or sinking below it."""
    to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x_m, y_m = to_xy.transform(lon, lat)

    x = region["x"][0, :]
    y = region["y"][:, 0]
    ix = int(np.clip(np.searchsorted(x, x_m), 0, len(x) - 1))
    iy = int(np.clip(np.searchsorted(-y, -y_m), 0, len(y) - 1))
    z_m = region["surface"][iy, ix]

    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    z_km = z_m * VERT_EXAG / 1000.0
    return x_km, y_km, z_km


def render_panel(kind, meshes, drapes, camera, window_size):
    """Render one panel (kind == 'ice' or 'bed') off-screen with PyVista. Both
    panels are always given the exact same camera tuple and window_size, so
    'same perspective as panel a' holds pixel-for-pixel between the two."""
    px_w, px_h = window_size
    pl = pv.Plotter(off_screen=True, window_size=(px_w, px_h))
    # Black background in both panels: area outside the rendered domain
    # (empty space beyond the pin/hedgehog terrain) reads unambiguously as
    # "outside the AOI", distinct from any in-domain colour in either panel's
    # scale (panel (a)'s "Blues" ice scale runs white-to-dark-blue; panel
    # (b)'s Oleron scale runs dark-navy-to-cream).
    pl.set_background("black")
    # NOT pl.enable_anti_aliasing("ssaa"): VTK's SSAA renders internally at a
    # further multiple of window_size, and on this headless software-GL
    # context that silently blows past the framebuffer's texture-size limit
    # once window_size is already large (confirmed by testing: identical
    # setup at 4322x3044 renders fine, at 7781x5480 comes back solid black
    # with "Frame buffer object was not initialized correctly" on stderr but
    # no raised exception). RENDER_SUPERSAMPLE + the LANCZOS downsample in
    # main() is this script's antialiasing; stacking VTK's own on top is
    # both redundant and what breaks the render.

    vmax = BED_ELEV_SATURATION_M

    if kind == "bed":
        pl.add_mesh(
            meshes["bed_pins"],
            scalars="pin_scalar",
            cmap=BATHY_CMAP,
            clim=(-vmax, vmax),
            show_scalar_bar=False,
            render_lines_as_tubes=True,
            line_width=PIN_LINE_WIDTH_BED,
            lighting=False,
        )
        if meshes["sea_level_line"].n_points > 0:
            pl.add_mesh(meshes["sea_level_line"], color="#00e5ff", line_width=2.5)

        for key, color in (("gl_bed", "black"), ("coast_bed", "#6b4226"), ("island_bed", "#6b4226")):
            tube = drapes.get(key)
            if tube is not None:
                pl.add_mesh(tube, color=color, specular=0.0)

        if drapes.get("ibcso_cloud") is not None:
            pl.add_mesh(
                drapes["ibcso_cloud"],
                color="#ff6f00",
                point_size=4.0,
                render_points_as_spheres=True,
                opacity=0.75,
            )

    else:  # kind == "ice"
        # lighting=False on both pin meshes (bed and ice): VTK's default
        # Phong/specular lighting on these dense near-parallel line fields
        # otherwise washes a bright sheen across the terrain -- worst over
        # near-flat regions (the East Antarctic interior plateau) and at
        # certain viewing angles -- which is what was reported as a flat
        # pale "mask" and as the velocity colouring/flow lines being hard to
        # see. Every other mesh in this scene (the drape tubes below) already
        # sets specular=0.0 for the same reason; only the two main terrain
        # meshes had been left on default lighting.
        pl.add_mesh(
            meshes["ice_pins"],
            scalars="pin_rgb",
            rgb=True,
            show_scalar_bar=False,
            render_lines_as_tubes=True,
            line_width=PIN_LINE_WIDTH_ICE,
            lighting=False,
        )

        for key, color in (("gl_ice", "black"), ("coast_ice", "#6b4226"), ("island_ice", "#6b4226")):
            tube = drapes.get(key)
            if tube is not None:
                pl.add_mesh(tube, color=color, specular=0.0)

        if drapes.get("flight_tube") is not None:
            # Halo drawn first (wider, pale, beneath) so the thinner, fully
            # opaque dark line on top of it reads clearly against both the
            # dark-blue interior and the pale near-margin ice -- see
            # FLIGHTLINE_HALO_RADIUS/FLIGHTLINE_COLOR above for why.
            pl.add_mesh(drapes["flight_tube_halo"], color=FLIGHTLINE_HALO_COLOR, opacity=0.85, specular=0.0)
            pl.add_mesh(drapes["flight_tube"], color=FLIGHTLINE_COLOR, opacity=1.0, specular=0.0)

        if drapes.get("flow_tube") is not None:
            # Coloured by ice speed (m/yr, log scale -- see
            # FLOWLINE_SPEED_CLIM_MYR/FLOWLINE_SPEED_CMAP) rather than a flat
            # colour, and drawn at a local-peak elevation + small lift
            # (FLOWLINE_LOCAL_MAX_CELLS/FLOWLINE_LIFT_KM, applied at the
            # drape_lines_mesh() call in main()) so the lines clear the
            # roughest terrain -- crevasse fields/icefalls right at the fast
            # outlet-glacier convergences this colouring is meant to show --
            # rather than being buried in the pin forest exactly where the
            # fastest (brightest) ice is.
            pl.add_mesh(
                drapes["flow_tube"],
                scalars="speed",
                cmap=FLOWLINE_SPEED_CMAP,
                clim=FLOWLINE_SPEED_CLIM_MYR,
                log_scale=True,
                show_scalar_bar=False,
                specular=0.0,
            )

    pl.camera_position = camera
    pl.camera.view_angle = 32.0
    pl.render()

    img = pl.screenshot(return_img=True)
    return pl, img, vmax


def find_anchor_points(region, meshes):
    """Summary stats reported on the console (and usable in the caption):
    the deepest subglacial bed elevation in the box."""
    inbox = region["inbox"]
    mask = region["mask"]

    grounded_inbox = inbox & (mask == 2)
    bed = np.where(grounded_inbox, region["bed"], np.nan)
    deepest_bed_m = np.nanmin(bed) if grounded_inbox.any() else np.nan

    return {"deepest_bed_m": deepest_bed_m}


def scale_bar_pixels(pl, meshes, region, img_width, target_frac=0.15):
    """Length in pixels of a round-number-km bar laid along the sea-surface
    plane in the foreground (an oblique 3D view has no single scale
    everywhere; the bar is only exact at that plane -- state this in the
    caption). The reference distance is auto-picked (from a fixed set of
    round numbers) so the bar occupies roughly ``target_frac`` of the image
    width regardless of how zoomed-in the camera is."""
    valid = region["inbox"] & (region["mask"] == 0)
    if not valid.any():
        valid = region["inbox"]
    y0 = meshes["y_km"][valid].max()
    x0 = meshes["x_km"][valid][np.argmax(meshes["y_km"][valid])]
    z0 = 0.0
    p0 = world_to_display(pl, (x0, y0, z0))
    p1 = world_to_display(pl, (x0 + 200.0, y0, z0))
    px_per_km = np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 200.0

    target_km = (img_width * target_frac) / px_per_km
    nice_options = np.array([1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000])
    nice_km = nice_options[np.argmin(np.abs(nice_options - target_km))]

    return px_per_km * nice_km, nice_km, p0


def compose_figure(img_ice, img_bed, pl, region, meshes, anchors, vmax_bed):
    font = _set_font()

    # Crop each panel's rendered image tightly to the terrain it actually
    # contains (the render background is solid black). Both panels share one
    # camera and render resolution, so:
    #  - vertically, each panel is cropped to its OWN row range and given its
    #    own axes height in inches (below) -- no stretching, so the
    #    km-per-pixel scale is untouched, it's just a direct slice.
    #  - horizontally, the SAME column range (the widest range needed to
    #    avoid clipping either panel) is cropped from BOTH images, then that
    #    shared width is stretched to fill the fixed COLUMN_WIDTH_MM figure
    #    width -- an identical zoom factor for both panels, so they still
    #    share one scale, while using the full printed width for content
    #    instead of leaving it mostly black either side.
    top_a, bot_a, left_a, right_a = content_extent(img_ice)
    top_b, bot_b, left_b, right_b = content_extent(img_bed)
    left, right = min(left_a, left_b), max(right_a, right_b)
    img_ice = img_ice[top_a:bot_a, left:right]
    img_bed = img_bed[top_b:bot_b, left:right]

    mm_to_in = 1.0 / 25.4
    fig_w_in = COLUMN_WIDTH_MM * mm_to_in
    # Each panel's own height in inches, derived from its cropped pixel
    # height at the figure's fixed PNG_DPI density -- NOT a shared
    # panel_h_in -- so panel a and panel b can have different physical
    # heights (matching however much vertical content each actually has)
    # while still sharing the exact same physical scale (mm per km).
    panel_h_in = {"a": img_ice.shape[0] / PNG_DPI, "b": img_bed.shape[0] / PNG_DPI}
    gap_in = PANEL_GAP_MM * mm_to_in
    footer_h_in = FOOTER_HEIGHT_MM * mm_to_in
    fig_h_in = panel_h_in["a"] + panel_h_in["b"] + 2 * gap_in + footer_h_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=PNG_DPI)

    panel_h_frac = {k: v / fig_h_in for k, v in panel_h_in.items()}
    gap_frac = gap_in / fig_h_in
    footer_h_frac = footer_h_in / fig_h_in
    # panel a on top, panel b directly beneath it (only PANEL_GAP_MM between
    # the two rendered scenes), shared legend footer at the very bottom.
    row_y0 = {
        "a": footer_h_frac + gap_frac + panel_h_frac["b"] + gap_frac,
        "b": footer_h_frac + gap_frac,
    }

    def panel_rect(row, lx, ly, lw, lh):
        return [lx, row_y0[row] + ly * panel_h_frac[row], lw, lh * panel_h_frac[row]]

    # Terrain images are displayed at their NATIVE pixel scale (1 pixel =
    # 1/PNG_DPI inch) in both x and y -- no horizontal stretch-to-fill-width.
    # Vertically there was never any stretch (panel_h_in above is already
    # native). Horizontally, stretching the shared crop width out to fill
    # the full figure width would silently make 1 km cover a different
    # number of mm left-right than it does up-down (and the render's
    # perspective/vertical-exaggeration geometry is only correct if x and y
    # share one physical scale) -- so the ax width here is sized to match
    # img_ice.shape[1] (== img_bed.shape[1], since both panels were cropped
    # to the identical shared column range) at that same native density,
    # capped so it can never grow into the reserved colourbar sidebar.
    PANEL_CONTENT_FRAC = min((img_ice.shape[1] / PNG_DPI) / fig_w_in, 0.80)
    # Colourbars sit a small fixed gap to the right of the terrain's actual
    # (native-scale) right edge, rather than at a hardcoded figure fraction --
    # otherwise, whenever PANEL_CONTENT_FRAC shrinks (as it now does, since it
    # tracks the real cropped content width), the colourbar stays put and a
    # widening dead strip of blank white opens up between terrain and
    # colourbar. Tying its x-origin to PANEL_CONTENT_FRAC keeps it flush
    # regardless of how wide the cropped terrain happens to be.
    CB_GAP_FRAC = 0.025
    CB_WIDTH_FRAC = 0.02
    CB_X0 = PANEL_CONTENT_FRAC + CB_GAP_FRAC

    # origin="lower" (not the more common "upper"): the axes' y-limits below
    # are inverted (h, 0) so that data-space y matches the pixel-space
    # convention used everywhere else in this function (world_to_display(),
    # the scale bar, label placement) -- y=0 at the top of the panel. With
    # that inverted ylim, origin="lower" is what places img row 0 (the top of
    # the actual rendered image) at the top of the panel; origin="upper"
    # here would silently flip the panel vertically.
    ax_a = fig.add_axes([0.0, row_y0["a"], PANEL_CONTENT_FRAC, panel_h_frac["a"]])
    ax_a.imshow(img_ice, extent=[0, img_ice.shape[1], 0, img_ice.shape[0]], origin="lower", aspect="auto")
    ax_a.set_xlim(0, img_ice.shape[1])
    ax_a.set_ylim(img_ice.shape[0], 0)
    ax_a.axis("off")

    ax_b = fig.add_axes([0.0, row_y0["b"], PANEL_CONTENT_FRAC, panel_h_frac["b"]])
    ax_b.imshow(img_bed, extent=[0, img_bed.shape[1], 0, img_bed.shape[0]], origin="lower", aspect="auto")
    ax_b.set_xlim(0, img_bed.shape[1])
    ax_b.set_ylim(img_bed.shape[0], 0)
    ax_b.axis("off")

    # --- panel a: ice-surface elevation colourbar (vertical, right edge --
    # frees the full panel width at the bottom for panel b to sit directly
    # underneath with only PANEL_GAP_MM between the two rendered scenes) ---
    cax_e = fig.add_axes(panel_rect("a", CB_X0, 0.22, CB_WIDTH_FRAC, 0.62))
    elev_lo, elev_hi = ICE_SURF_ELEV_CLIM_M
    sm_e = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=elev_lo, vmax=elev_hi), cmap="Blues")
    cb_e = fig.colorbar(sm_e, cax=cax_e, orientation="vertical")
    ticks = np.linspace(elev_lo, elev_hi, 6)
    cb_e.set_ticks(ticks)
    tick_labels = [f"{t:.0f}" for t in ticks]
    tick_labels[-1] = f"≥{elev_hi:.0f}"
    cb_e.set_ticklabels(tick_labels)
    cb_e.set_label("Ice surface elevation (m)", fontsize=LABEL_FONTSIZE, labelpad=4)
    cb_e.ax.tick_params(labelsize=LABEL_FONTSIZE - 1, length=2, pad=1)
    cb_e.outline.set_linewidth(0.5)

    # --- shared legend footer (below both panels, on the plain white figure
    # background -- not overlaid on any rendered terrain, so it never needs a
    # translucent backdrop and never competes for space with panel a/b's
    # imagery). A single column, one entry per row: the two-column layout
    # previously used here gave each entry only ~half the figure width for
    # its text, which was too narrow for the longer descriptions (e.g. the
    # ice-flow-speed entry, ~90 characters) -- that text had no wrap/clip
    # applied, so it simply ran off the right edge of the figure. A single
    # column gives every entry the full row width, comfortably fitting even
    # the longest description on one line at this font size.
    def footer_swatch(x0, y, color, text, lw=1.5, alpha=1.0, marker=False, gradient=None):
        y_fig = y * footer_h_frac
        bar_h = 0.05 * footer_h_frac
        bar_w = 0.03
        sax = fig.add_axes([x0, y_fig, bar_w, bar_h])
        if gradient is not None:
            cmap, vmin, vmax = gradient
            grad = np.log10(np.linspace(vmin, vmax, 256)).reshape(1, -1)
            sax.imshow(grad, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1),
                       vmin=np.log10(vmin), vmax=np.log10(vmax))
        elif marker:
            sax.scatter([0.5], [0.5], s=4, color=color, alpha=alpha)
        else:
            sax.plot([0, 1], [0.5, 0.5], color=color, lw=lw, alpha=alpha, solid_capstyle="butt")
        sax.set_xlim(0, 1)
        sax.set_ylim(0, 1)
        sax.axis("off")
        fig.text(x0 + bar_w + 0.012, y_fig + bar_h / 2, text,
                  fontsize=LABEL_FONTSIZE - 1, va="center", ha="left")

    vmin_fl, vmax_fl = FLOWLINE_SPEED_CLIM_MYR
    col_x = 0.045
    row_ys = np.linspace(0.90, 0.10, 6)
    footer_swatch(col_x, row_ys[0], "black", "Grounding line (MOA)")
    footer_swatch(col_x, row_ys[1], "#6b4226", "Coastline and island outlines (MOA)")
    footer_swatch(col_x, row_ys[2], FLIGHTLINE_COLOR, "BEDMAP3 airborne/ground radar survey lines (panel a)")
    footer_swatch(col_x, row_ys[3], None,
                  f"Ice-flow lines (MEaSUREs velocity), coloured by ice speed "
                  f"({vmin_fl:.0f}-{vmax_fl:.0f} m/yr, log scale) (panel a)",
                  gradient=(FLOWLINE_SPEED_CMAP, vmin_fl, vmax_fl))
    footer_swatch(col_x, row_ys[4], "#00e5ff", "Sea level (bed = 0 m) -- ice below this line is marine-based (panel b)")
    footer_swatch(col_x, row_ys[5], "#ff6f00", "IBCSO direct seafloor soundings, rest gravity-predicted (panel b)", marker=True)

    # --- panel b: bed elevation colourbar (vertical, right edge) ---
    cax_b = fig.add_axes(panel_rect("b", CB_X0, 0.22, CB_WIDTH_FRAC, 0.62))
    norm_b = TwoSlopeNorm(vcenter=0.0, vmin=-vmax_bed, vmax=vmax_bed)
    sm_b = plt.cm.ScalarMappable(norm=norm_b, cmap=BATHY_CMAP)
    cb_b = fig.colorbar(sm_b, cax=cax_b, orientation="vertical")
    cb_b.set_label("Bed elevation (m)", fontsize=LABEL_FONTSIZE, labelpad=4)
    cb_b.ax.tick_params(labelsize=LABEL_FONTSIZE - 1, length=2, pad=1)
    cb_b.outline.set_linewidth(0.5)

    # --- locator inset (panel a only) -- drawn BEFORE the ice-shelf labels
    # below so the labels' leader lines/markers/text always render on top of
    # it, never hidden underneath (see the label loop's comment for why this
    # ordering matters).
    if INSET_ENABLE:
        add_inset_map(fig, rect_xy=panel_rect("a", 0.015, 0.64, 0.28, 0.28))

    # --- panel a: ice-shelf text labels ---
    # Each anchor (lon, lat) is that feature's true approximate location.
    # Ninnis Glacier/Ice Shelf and Cook Glacier/Ice Shelf sit only ~150 km apart along
    # this stretch of coast, and both anchors happen to project into the same
    # screen region as the locator inset above -- drawing the label text
    # directly at the anchor collided with both the inset and each other.
    # Each label's text is instead placed at a fixed, clear (text_fx, text_fy)
    # position (fractions of panel width/height, fy=0 at panel top) chosen to
    # sit below the inset and clear of the terrain edges, with a thin leader
    # line + marker dot back to the true anchor -- so the readable text never
    # collides, even though the anchor itself may render under the inset.
    # This loop runs AFTER add_inset_map() above so the leader lines/markers/
    # text are guaranteed to paint on top of it.
    for label_text, lon, lat, text_fx, text_fy in ICE_SHELF_LABELS:
        xyz_km = label_world_xyz(lon, lat, region)
        px, py = world_to_display(pl, xyz_km)
        # world_to_display() returns pixel coords in the pre-crop render
        # frame; shift by the panel-a crop origin (top_a, left) to match
        # img_ice's now-cropped row/column indexing.
        px -= left
        py -= top_a
        tx, ty = text_fx * img_ice.shape[1], text_fy * img_ice.shape[0]
        if 0 <= px <= img_ice.shape[1] and 0 <= py <= img_ice.shape[0]:
            ax_a.plot([px, tx], [py, ty], color="black", lw=0.6, zorder=6)
            ax_a.plot(px, py, marker="o", markersize=2.2, color="black", zorder=6)
            ax_a.text(tx, ty, label_text, fontsize=LABEL_FONTSIZE - 1, style="italic",
                       color="black", ha="center", va="center", zorder=6,
                       path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # --- panel b: basin name label -- a broad-region label rather than a
    # point feature, so (unlike the ice-shelf labels above) no anchor/leader
    # line: placed directly over the pale, below-sea-level basin floor
    # visible in the bed render, in axes-fraction coordinates.
    ax_b.text(0.33, 0.615, "Wilkes Subglacial Basin", transform=ax_b.transAxes,
              fontsize=LABEL_FONTSIZE - 1, style="italic", ha="center", va="center",
              color="black", zorder=6,
              path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    # --- scale bars: identical scale in both panels (same camera) ---
    bar_px, bar_km, _ = scale_bar_pixels(pl, meshes, region, img_ice.shape[1])
    margin = img_ice.shape[1] * 0.03
    for ax, img in ((ax_a, img_ice), (ax_b, img_bed)):
        bx, by = img.shape[1] - margin - bar_px, img.shape[0] * 0.93
        ax.plot([bx, bx + bar_px], [by, by], color="black", lw=1.2, solid_capstyle="butt")
        ax.text(bx + bar_px / 2, by - 12, f"{bar_km:.0f} km*", ha="center", va="bottom",
                 fontsize=LABEL_FONTSIZE)

    # --- panel labels ---
    # White text with a thin black stroke: the tightly-cropped panel corner
    # here is the render's black background (nothing plotted that far out),
    # so plain black text would be invisible; the stroke keeps the label
    # legible if the crop ever leaves lighter terrain in that corner instead.
    for ax, lbl in ((ax_a, PANEL_LABELS[0]), (ax_b, PANEL_LABELS[1])):
        ax.text(0.02, 0.97, lbl, transform=ax.transAxes, fontsize=TITLE_FONTSIZE,
                fontweight="bold", ha="left", va="top", color="white",
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    # --- vertical exaggeration note (non-negotiable, both panels) ---
    # White with a black stroke, same reasoning as the panel labels above:
    # this corner of the tightly-cropped render is black background.
    for ax in (ax_a, ax_b):
        ax.text(
            0.98, 0.02, f"Vertical exaggeration: {VERT_EXAG}×",
            transform=ax.transAxes, fontsize=LABEL_FONTSIZE - 1, ha="right", va="bottom",
            style="italic", color="white",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
        )

    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    region = load_region()
    vel = load_velocity(region) if VELOCITY_ENABLE else None
    meshes = build_meshes(region)

    gl_gdf = load_grounding_line()
    gl_lines = lines_from_geodataframe(gl_gdf)

    coast_gdf = load_moa_coastline()
    coast_lines = lines_from_geodataframe(coast_gdf) if not coast_gdf.empty else []

    islands_gdf = load_moa_islands()
    island_lines = lines_from_geodataframe(islands_gdf) if not islands_gdf.empty else []
    print(f"MOA coastline parts in box: {len(coast_gdf)}, island outlines in box: {len(islands_gdf)}")

    flightlines = load_bedmap_flightlines(region) if FLIGHTLINES_ENABLE else []
    print(f"BEDMAP3 flight-line segments drawn in box: {len(flightlines)}")

    flowlines = compute_flowlines(region, vel["vx"], vel["vy"]) if (FLOWLINE_ENABLE and vel is not None) else []
    print(f"Ice-flow lines traced in box: {len(flowlines)}")

    if IBCSO_ENABLE:
        ibcso_x, ibcso_y, ibcso_z = load_ibcso_direct_points(region)
    else:
        ibcso_x = np.array([])
    ibcso_cloud = ibcso_points_mesh(ibcso_x, ibcso_y, ibcso_z) if ibcso_x.size else None
    print(f"IBCSO direct-sounding points drawn in box: {ibcso_x.size}")

    drapes = {
        "gl_ice": drape_lines_mesh(gl_lines, region, region["surface"], radius=1.2),
        "coast_ice": drape_lines_mesh(coast_lines, region, region["surface"], radius=0.9),
        "island_ice": drape_lines_mesh(island_lines, region, region["surface"], radius=0.7),
        "flight_tube": drape_lines_mesh(flightlines, region, region["surface"], radius=FLIGHTLINE_RADIUS,
                                         lift_km=FLIGHTLINE_LIFT_KM,
                                         local_max_cells=FLIGHTLINE_LOCAL_MAX_CELLS),
        "flight_tube_halo": drape_lines_mesh(flightlines, region, region["surface"], radius=FLIGHTLINE_HALO_RADIUS,
                                              lift_km=FLIGHTLINE_LIFT_KM,
                                              local_max_cells=FLIGHTLINE_LOCAL_MAX_CELLS),
        "flow_tube": drape_lines_mesh(flowlines, region, region["surface"], radius=0.5,
                                       lift_km=FLOWLINE_LIFT_KM,
                                       local_max_cells=FLOWLINE_LOCAL_MAX_CELLS),
        "gl_bed": drape_lines_mesh(gl_lines, region, region["bed"], radius=1.2,
                                    lift_km=BED_LINE_LIFT_KM,
                                    local_max_cells=BED_LINE_LOCAL_MAX_CELLS),
        "coast_bed": drape_lines_mesh(coast_lines, region, region["bed"], radius=0.9,
                                       lift_km=BED_LINE_LIFT_KM,
                                       local_max_cells=BED_LINE_LOCAL_MAX_CELLS),
        "island_bed": drape_lines_mesh(island_lines, region, region["bed"], radius=0.7,
                                        lift_km=BED_LINE_LIFT_KM,
                                        local_max_cells=BED_LINE_LOCAL_MAX_CELLS),
        "ibcso_cloud": ibcso_cloud,
    }

    # Computed once, reused for BOTH panels unchanged: this is what makes
    # panel (b) directly comparable to panel (a) ("same perspective").
    camera = compute_camera(meshes, region)

    mm_to_in = 1.0 / 25.4
    fig_w_in = COLUMN_WIDTH_MM * mm_to_in
    panel_h_in = fig_w_in / PANEL_ASPECT
    target_w = int(fig_w_in * PNG_DPI)
    target_h = int(panel_h_in * PNG_DPI)
    px_w = int(target_w * RENDER_SUPERSAMPLE)
    px_h = int(target_h * RENDER_SUPERSAMPLE)

    pl_ice, img_ice, _ = render_panel("ice", meshes, drapes, camera, (px_w, px_h))
    pl_bed, img_bed, vmax_bed = render_panel("bed", meshes, drapes, camera, (px_w, px_h))

    if RENDER_SUPERSAMPLE != 1.0:
        from PIL import Image
        img_ice = np.asarray(Image.fromarray(img_ice).resize((target_w, target_h), Image.LANCZOS))
        img_bed = np.asarray(Image.fromarray(img_bed).resize((target_w, target_h), Image.LANCZOS))

    anchors = find_anchor_points(region, meshes)

    fig = compose_figure(img_ice, img_bed, pl_ice, region, meshes, anchors, vmax_bed)

    pdf_path = os.path.join(OUT_DIR, OUT_BASENAME + ".pdf")
    png_path = os.path.join(OUT_DIR, OUT_BASENAME + ".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=PNG_DPI)
    pl_ice.close()
    pl_bed.close()

    print(f"Deepest subglacial bed in box: {anchors['deepest_bed_m']:.0f} m")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
