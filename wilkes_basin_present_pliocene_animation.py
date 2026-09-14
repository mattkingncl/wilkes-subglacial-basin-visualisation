"""Animation 3: present day <-> Pliocene-like ice sheet, fixed camera.

Camera held at the Wilkes Subglacial Basin overview (animation 1/2's first
view). Present-day ice with flowing MEaSUREs velocity tracks and the caption
"Present day"; the tracks switch off and pause; image dissolve (no morph)
to the PISM Pliocene-like ice surface with the caption "Past warm climate
due to higher CO2", held; dissolve back to present day; tracks back on; end.
No radar flight lines; parcel style as in the intro animation.

Same CLI as wilkes_basin_3d_animation.py (--draft, --stills [--times],
--chunk i/N --frames-dir D, --assemble). Outputs use the basename
wilkes_basin_present_pliocene_animation.
"""

import copy

import numpy as np
import pyvista as pv
import vtk

import wilkes_basin_3d_animation as anim
import wilkes_basin_3d_figure as base

anim.OUT_BASENAME = "wilkes_basin_present_pliocene_animation"
anim.GIF_END_DESIGN_S = None          # GIF is the whole (short) sequence
anim.FLIGHTLINES_ENABLE = False       # no BEDMAP3 radar flight lines in this animation

# Velocity parcel style as in the intro animation: denser seeds, long comet
# tails, thinner heads.
base.FLOWLINE_SEED_SPACING_KM /= np.sqrt(2.0)
anim.PARCEL_LENGTH_KM = 90.0
anim.PARCEL_HEAD_PX = 2.25
anim.PARCEL_TAIL_PX = 1.125

# Segments in PLAYED seconds: (hold_s, state). Each transition (a FADE_S
# dissolve) is counted inside the segment it leads into, so the segments
# sum to the total playtime.
SEGMENTS = [
    (5.0, "present"),
    (8.5, "pliocene"),
    (5.5, "present"),
]
FADE_S = 2.5        # smoothstep-eased image dissolve; most of the change in the middle ~1.5 s
# The velocity parcels/tracks (and their bar) are switched off LINES_PAUSE_S
# before the dissolve to the Pliocene state starts, and come back on
# LINES_PAUSE_S after the dissolve back to present day has ended.
LINES_PAUSE_S = 1.0
LINES_OFF = []      # (t0, t1) design-second windows, filled by build_keyframes

CAPTION_PRESENT = "Present day"
CAPTION_PLIO = "Past warm climate\ndue to higher CO₂"
CAPTION_FONT = 18          # add_text points (title is 18); upper-right corner slot
CAPTION_FONT_FILE = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"   # has the subscript 2

_speed = anim.PLAYBACK_SPEEDUP   # keep the parcel clock identical to the other animations


def build_keyframes(meshes, region):
    """Fixed overview camera; 'morph' keys carry the present->Pliocene
    crossfade weight w (0 = present, 1 = Pliocene-like), eased by morph_at."""
    base.CAMERA_DIST_FACTOR = anim.OVERVIEW_DIST_FACTOR
    ov_pos, ov_focal, _ = base.compute_camera(meshes, region)
    ov_pos, ov_focal = np.asarray(ov_pos, float), np.asarray(ov_focal, float)
    ov_pos[2] += anim.OVERVIEW_FOCAL_LIFT_KM
    ov_focal[2] += anim.OVERVIEW_FOCAL_LIFT_KM

    keys, t, prev = [(0.0, 0.0)], 0.0, "present"
    LINES_OFF.clear()
    for hold_s, state in SEGMENTS:
        w = 1.0 if state == "pliocene" else 0.0
        if state != prev:
            keys.append((t * _speed, keys[-1][1]))
            keys.append(((t + FADE_S) * _speed, w))
            if state == "pliocene":
                LINES_OFF.append([(t - LINES_PAUSE_S) * _speed, np.inf])
            else:
                LINES_OFF[-1][1] = (t + FADE_S + LINES_PAUSE_S) * _speed
        t += hold_s
        prev = state
    keys.append((t * _speed, keys[-1][1]))
    # Fixed camera; one key per design second (anim.main's default still
    # times index cam_keys[3], [4], [9], [10]).
    n = int(np.ceil(t * _speed)) + 1
    cam_keys = [(min(float(k), t * _speed), ov_pos, ov_focal) for k in range(n)]
    if cam_keys[-1][0] < t * _speed:
        cam_keys.append((t * _speed, ov_pos, ov_focal))
    return cam_keys, keys


class Scene(anim.Scene):
    def __init__(self, meshes, drapes, tracks, morph, region, frame_size, cam_keys):
        # The static tracks are added here (not by the base class) so they can
        # be switched off with the parcels, separately from the other overlays.
        static_tracks = anim.STATIC_TRACKS_ENABLE
        anim.STATIC_TRACKS_ENABLE = False
        super().__init__(meshes, drapes, tracks, morph, region, frame_size, cam_keys)
        anim.STATIC_TRACKS_ENABLE = static_tracks
        pl = self.pl
        self.velocity = [(self.parcels, 1.0)]
        self.fading = [(a, f) for a, f in self.fading if a is not self.parcels]
        if static_tracks and drapes.get("tracks") is not None:
            a = pl.add_mesh(drapes["tracks"], color=anim.STATIC_TRACK_COLOR,
                            opacity=anim.STATIC_TRACK_OPACITY, line_width=anim.STATIC_TRACK_WIDTH_PX,
                            render_lines_as_tubes=True, lighting=False)
            self.velocity.append((a, anim.STATIC_TRACK_OPACITY))
        for name in self.labels:                       # no glacier labels here
            self.labels[name].SetVisibility(False)
            self.label_points[name].SetVisibility(False)

        # Second copy of the ice pins fixed at the Pliocene-like state; the
        # present-day copy is never morphed. The two are crossfaded.
        self.pins_plio_mesh = self.pins_mesh.copy(deep=True)
        m = dict(morph)
        m["u_applied"] = 0.0
        anim.apply_morph(self.pins_plio_mesh, m, 1.0)
        self.pins_plio = pl.add_mesh(
            self.pins_plio_mesh, scalars="pin_rgb", rgb=True, show_scalar_bar=False,
            render_lines_as_tubes=True, line_width=anim.PIN_WIDTH_MIN_PX, lighting=False,
            opacity=0.0,
        )
        self.pins_plio.SetVisibility(False)

        # Captions in the upper-right corner (where animation 1 had its state
        # text), swapped with the ice state. The corner annotation is shared,
        # so use two separate actors.
        self.captions = []
        for text in (CAPTION_PRESENT, CAPTION_PLIO):
            a = pl.add_text(text, position="upper_right", font_size=CAPTION_FONT, color="white",
                            name=f"caption_{len(self.captions)}")
            tp = a.GetTextProperty()
            tp.SetFontFamily(vtk.VTK_FONT_FILE)
            tp.SetFontFile(CAPTION_FONT_FILE)
            tp.SetBackgroundColor(0.0, 0.0, 0.0)
            tp.SetBackgroundOpacity(0.55)
            tp.SetFrame(False)
            self.captions.append(a)
        self._w = 0.0
        self._pl_screenshot = pl.screenshot
        pl.screenshot = self._screenshot
        pl.write_frame = self._write_frame

    def _apply(self, w, T, kpp):
        """Put every actor into the pure state w in {0, 1} (present / Pliocene);
        the caption opacities follow the crossfade weight self._w so the old
        caption is gone before the new one appears."""
        self.pins.prop.opacity = 1.0 - w
        self.pins.SetVisibility(w < 0.5)
        self.pins_plio.prop.opacity = w
        self.pins_plio.SetVisibility(w > 0.5)
        lines_on = w < 0.5 and not any(t0 <= T < t1 for t0, t1 in LINES_OFF)
        if lines_on:
            new = anim.build_parcels(self.tracks, T, kpp)
            if new is not None:
                self.parcel_mesh.copy_from(new)
        for actor, full in self.fading:
            actor.prop.opacity = full * (1.0 - w)
            actor.SetVisibility(w < 0.5)
        for actor, full in self.velocity:
            actor.prop.opacity = full
            actor.SetVisibility(lines_on)
        self.scalar_bar.SetVisibility(lines_on)
        if self.plio_extent is not None:
            self.plio_extent.prop.opacity = w
            self.plio_extent.SetVisibility(w > 0.5)
        wc = self._w
        ops = (np.clip(1.0 - 2.0 * wc, 0, 1) if w < 0.5 else 0.0,
               np.clip(2.0 * wc - 1.0, 0, 1) if w > 0.5 else 0.0)
        for a, op in zip(self.captions, ops):
            tp = a.GetTextProperty()
            tp.SetOpacity(op)
            tp.SetBackgroundOpacity(0.55 * op)
            a.SetVisibility(op > 0.001)

    def update(self, cam_keys, morph_keys, T, grid_km):
        pl = self.pl
        pos, foc = anim.camera_at(cam_keys, T)
        pl.camera_position = (tuple(pos), tuple(foc), (0.0, 0.0, 1.0))
        pl.camera.view_angle = anim.VIEW_ANGLE_DEG
        kpp = anim.km_per_pixel(pos, foc, self.frame_w)
        width = max(anim.PIN_WIDTH_MIN_PX, anim.PIN_FILL_FACTOR * grid_km / kpp)
        self.pins.prop.line_width = width
        self.pins_plio.prop.line_width = width

        # 0 = present, 1 = Pliocene-like. The transition is an image-space
        # dissolve (see _screenshot): the scene itself is only ever rendered
        # in one of the two pure states, so no translucent pins overlap.
        self._w = anim.morph_at(morph_keys, T)
        self._T, self._kpp = T, kpp
        self._apply(0.0 if self._w < 0.5 else 1.0, T, kpp)
        pl.render()

    def _screenshot(self, filename=None, **kw):
        w = self._w
        if w <= 0.001 or w >= 0.999:
            return self._pl_screenshot(filename, **kw)
        self._apply(0.0, self._T, self._kpp)
        self.pl.render()
        a = self._pl_screenshot(None, return_img=True).astype(np.float32)
        self._apply(1.0, self._T, self._kpp)
        self.pl.render()
        b = self._pl_screenshot(None, return_img=True).astype(np.float32)
        img = np.clip((1.0 - w) * a + w * b + 0.5, 0, 255).astype(np.uint8)
        if filename:
            from PIL import Image
            Image.fromarray(img).save(filename)
        return img

    def _write_frame(self):
        self.pl._get_mwriter_not_none().append_data(self._screenshot(None))


anim.build_keyframes = build_keyframes
anim.Scene = Scene

if __name__ == "__main__":
    anim.main()
