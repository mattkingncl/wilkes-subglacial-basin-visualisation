"""YouTube-ready, University of Tasmania-branded cut of any of the Wilkes
Subglacial Basin animations.

    python youtube_version.py intro  --frames-dir _frames/intro          # PNG frames (best)
    python youtube_version.py flow   --video wilkes_basin_3d_animation_web.mp4
    python youtube_version.py bed    --frames-dir _frames/anim2 --draft   # 960x540 source

Output (next to this script):
    <basename>_youtube.mp4   1920x1080, 30 fps, H.264 High CRF 18, yuv420p,
                             BT.709 tags, silent AAC track, faststart
    <basename>_youtube.txt   title / description / tags to paste into YouTube

Structure:  title card (logo, title, author)  -xfade->
            animation with a branding bar (logo + credit) along the bottom
            -xfade->  end card (author, affiliation, data credits, logo).

The animation frame is scaled to 1920 x (1080 - BAR_H) above the bar, so the
whole 1080p canvas stays a standard 16:9 upload (BRAND_MODE = "bar").
BRAND_MODE = "corner" instead keeps the frame full size and overlays a small
logo + credit at CORNER (top-right by default) -- check it does not collide
with the animation's own corner text before using it.

Cards and the bar are drawn with Pillow (the bundled ffmpeg has no drawtext),
so fonts come from FONT_REGULAR / FONT_BOLD below.
"""

import argparse
import glob
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# BRANDING / CREDITS
# ---------------------------------------------------------------------------

AUTHOR = "Matt King"
AFFILIATION = "University of Tasmania"
CREDIT_LINE = f"{AUTHOR}  |  {AFFILIATION}"
LOGO = os.path.join(HERE, "branding", "utas_h_rev_col.png")   # reverse (white) logo for dark backgrounds
LOGO_MONO = os.path.join(HERE, "branding", "utas_h_rev_mono.png")

# Data / model credits: short lines for the end card, full citations for the
# YouTube description (.txt). Lines containing "Pliocene" are dropped for
# animations without the PISM state. PISM run: PLIO_end.nc
# (RCP-PLIO-sgm-AIS-N1, PISM 0.7.3, 5 km, year 5000) from the Golledge et al.
# (2019) model-output archive, https://osf.io/3edxf/.
DATA_CREDITS = [
    "Bed and ice geometry: BedMachine Antarctica v3 (Morlighem et al., 2020)",
    "Ice velocity: MEaSUREs InSAR-based Antarctic ice velocity v2 (Rignot, Mouginot & Scheuchl, 2017)",
    "Coastline and grounding line: MODIS Mosaic of Antarctica 2004 (Haran et al.)",
    "Radar survey lines: Bedmap3 (Pritchard et al., 2025); bathymetry: IBCSO v2 (Dorschel et al., 2022)",
    "Pliocene-like ice sheet: PISM simulation (Golledge et al., 2019)",
]
DATA_CREDITS_FULL = [
    "Bed topography and ice geometry: MEaSUREs BedMachine Antarctica, Version 3 (NSIDC, "
    "https://doi.org/10.5067/FPSU0V1MWUB6); Morlighem, M. et al. (2020), Deep glacial troughs and stabilizing "
    "ridges unveiled beneath the margins of the Antarctic ice sheet, Nature Geoscience 13, 132-137, "
    "https://doi.org/10.1038/s41561-019-0510-8",
    "Ice velocity: MEaSUREs InSAR-Based Antarctica Ice Velocity Map, Version 2, 450 m (NSIDC, "
    "https://doi.org/10.5067/D7GK8F5J8M8R); Rignot, E., J. Mouginot & B. Scheuchl (2017); Mouginot, J., "
    "E. Rignot & B. Scheuchl (2019), Continent-wide, interferometric SAR phase mapping of Antarctic ice "
    "velocity, Geophysical Research Letters 46, 9710-9718, https://doi.org/10.1029/2019GL083826",
    "Coastline, grounding line and islands: MODIS Mosaic of Antarctica 2003-2004 (MOA2004) Image Map, "
    "Version 1 (NSIDC, https://doi.org/10.7265/N5ZK5DM5); Haran, T., J. Bohlander, T. Scambos, T. Painter & "
    "M. Fahnestock (2005, updated 2013); Scambos, T. A. et al. (2007), Remote Sensing of Environment 111, "
    "242-257, https://doi.org/10.1016/j.rse.2006.12.020",
    "Radar survey lines: Bedmap3 standardised point data (NERC EDS UK Polar Data Centre, "
    "https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2); Pritchard, H. D. et al. (2025), Bedmap3 "
    "updated ice bed, surface and thickness gridded datasets for Antarctica, Scientific Data 12, 414, "
    "https://doi.org/10.1038/s41597-025-04672-y",
    "Bathymetry data-coverage mask: International Bathymetric Chart of the Southern Ocean Version 2 (IBCSO v2); "
    "Dorschel, B. et al. (2022), Scientific Data 9, 275, https://doi.org/10.1038/s41597-022-01366-7",
    "Pliocene-like ice sheet: Parallel Ice Sheet Model (PISM) whole-Antarctica simulation under Pliocene-like "
    "(EPI) climate forcing, final state at model year 5000, 5 km grid; Golledge, N. R. et al. (2019), Global "
    "environmental consequences of twenty-first-century ice-sheet melt, Nature 566, 65-72, "
    "https://doi.org/10.1038/s41586-019-0889-9; model outputs: https://osf.io/3edxf/",
    "PISM: Bueler, E. & J. Brown (2009), Journal of Geophysical Research 114, F03008, "
    "https://doi.org/10.1029/2008JF001179; https://www.pism.io",
]
# Archive acknowledgement for the NSIDC-distributed datasets (BedMachine,
# MEaSUREs velocity, MOA2004), written after the credit list in the .txt.
NSIDC_ACK = ("BedMachine Antarctica, the MEaSUREs ice velocity map and MOA2004 were obtained from the "
             "National Snow and Ice Data Center (NSIDC), Boulder, Colorado, USA, including the NASA NSIDC "
             "Distributed Active Archive Center (DAAC) for the MEaSUREs products: https://nsidc.org")
VERTICAL_EXAG_NOTE = "Vertical exaggeration 100x. Topography drawn as columns; column roughness shows data density."

# Per-animation text. Key = first CLI argument.
ANIMATIONS = {
    "flow": dict(
        basename="wilkes_basin_3d_animation",
        title="Wilkes Subglacial Basin, East Antarctica",
        subtitle="Ice surface, ice flow and a Pliocene-like ice sheet",
        description=("Fly-through of the Wilkes Subglacial Basin ice sheet, East Antarctica, showing present-day "
                     "ice flow (MEaSUREs velocity) at the Mertz, Ninnis and Cook glaciers, then a visual morph to a "
                     "Pliocene-like ice sheet configuration and back."),
        tags=["Antarctica", "ice sheet", "Wilkes Subglacial Basin", "sea level", "glaciology", "Pliocene",
              "University of Tasmania"],
    ),
    "bed": dict(
        basename="wilkes_basin_bed_reveal_animation",
        title="Beneath the Wilkes Subglacial Basin",
        subtitle="Lifting the East Antarctic Ice Sheet off its bed",
        description=("The East Antarctic Ice Sheet over the Wilkes Subglacial Basin is lifted off its bed to reveal "
                     "the deep marine basin beneath it (BedMachine Antarctica v3)."),
        tags=["Antarctica", "ice sheet", "Wilkes Subglacial Basin", "bedrock", "sea level", "glaciology",
              "University of Tasmania"],
        credits=[c for c in DATA_CREDITS if "Pliocene" not in c],
        credits_full=[c for c in DATA_CREDITS_FULL if "Pliocene" not in c and "PISM:" not in c],
    ),
    "intro": dict(
        basename="wilkes_basin_3d_animation_intro",
        title="Wilkes Subglacial Basin, East Antarctica",
        subtitle="From the South Pole to the ice sheet's bed",
        description=("From above the South Pole down to the Wilkes Subglacial Basin: present-day ice flow at the "
                     "Mertz, Ninnis and Cook glaciers, then the ice sheet lifted off its bed to reveal the deep "
                     "marine basin beneath."),
        tags=["Antarctica", "ice sheet", "Wilkes Subglacial Basin", "sea level", "glaciology",
              "bedrock", "University of Tasmania"],
        credits=[c for c in DATA_CREDITS if "Pliocene" not in c],
        credits_full=[c for c in DATA_CREDITS_FULL if "Pliocene" not in c and "PISM:" not in c],
    ),
    "plio": dict(
        basename="wilkes_basin_present_pliocene_animation",
        title="Wilkes Subglacial Basin, East Antarctica",
        subtitle="Present day and a past warm climate",
        description=("The ice sheet over the Wilkes Subglacial Basin today (with MEaSUREs ice flow), dissolving "
                     "to a Pliocene-like ice-sheet configuration simulated with the Parallel Ice Sheet Model under "
                     "a past warm climate with higher atmospheric CO2, and back."),
        tags=["Antarctica", "ice sheet", "Wilkes Subglacial Basin", "sea level", "glaciology", "Pliocene",
              "climate change", "University of Tasmania"],
        title_s=2.5, end_s=4.0,     # short clip: shorter cards
    ),
}

# ---------------------------------------------------------------------------
# LAYOUT / ENCODE
# ---------------------------------------------------------------------------

OUT_W, OUT_H = 1920, 1080
OUT_FPS = 30
BRAND_MODE = "bar"          # "bar" (default) or "corner"
BAR_H = 72                  # branding bar height (px) in "bar" mode
BAR_LOGO_H = 44
BAR_BG = (0, 0, 0)          # matches the animation background; use (14, 14, 14) for a visible strip
CORNER = "top-right"        # "corner" mode: top-left / top-right / bottom-left / bottom-right
CORNER_LOGO_H = 56
CORNER_MARGIN = 24

TITLE_CARD_S = 4.0
END_CARD_S = 6.0
XFADE_S = 0.75
BG = (0, 0, 0)
FG = (255, 255, 255)
FG_DIM = (190, 196, 204)
FONT_REGULAR = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

X264 = ["-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart"]


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def logo_scaled(height, path=LOGO):
    im = Image.open(path).convert("RGBA")
    w = int(round(im.width * height / im.height))
    return im.resize((w, height), Image.LANCZOS)


def title_card(spec, path):
    im = Image.new("RGB", (OUT_W, OUT_H), BG)
    d = ImageDraw.Draw(im)
    m = 120
    lg = logo_scaled(96)
    im.paste(lg, (m, m), lg)
    y = OUT_H // 2 - 40                 # title only (no subtitle line)
    d.text((m, y), spec["title"], font=font(76, bold=True), fill=FG)
    d.text((m, OUT_H - m - 40), CREDIT_LINE, font=font(34), fill=FG)
    im.save(path)


def end_card(spec, path):
    im = Image.new("RGB", (OUT_W, OUT_H), BG)
    d = ImageDraw.Draw(im)
    m = 120
    d.text((m, m), "Produced by", font=font(34), fill=FG_DIM)
    d.text((m, m + 46), AUTHOR, font=font(60, bold=True), fill=FG)
    d.text((m, m + 122), AFFILIATION, font=font(40), fill=FG)
    y = m + 230
    d.text((m, y), "Data and models", font=font(30, bold=True), fill=FG_DIM)
    y += 48
    for line in spec.get("credits", DATA_CREDITS):
        d.text((m, y), line, font=font(28), fill=FG)
        y += 40
    y += 16
    d.text((m, y), VERTICAL_EXAG_NOTE, font=font(26), fill=FG_DIM)
    lg = logo_scaled(96)
    im.paste(lg, (OUT_W - m - lg.width, OUT_H - m - lg.height), lg)
    im.save(path)


def brand_bar(path):
    """RGBA strip: logo at left, credit at right."""
    im = Image.new("RGBA", (OUT_W, BAR_H), (*BAR_BG, 255))
    d = ImageDraw.Draw(im)
    lg = logo_scaled(BAR_LOGO_H)
    im.paste(lg, (32, (BAR_H - BAR_LOGO_H) // 2), lg)
    f = font(28)
    tw = d.textlength(CREDIT_LINE, font=f)
    d.text((OUT_W - 32 - tw, (BAR_H - 32) // 2), CREDIT_LINE, font=f, fill=FG)
    im.save(path)


def corner_badge(path):
    """RGBA badge (logo over credit) for "corner" mode."""
    lg = logo_scaled(CORNER_LOGO_H)
    f = font(24)
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw = int(tmp.textlength(CREDIT_LINE, font=f))
    w, h = max(lg.width, tw), CORNER_LOGO_H + 8 + 30
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    im.paste(lg, (w - lg.width, 0), lg)
    ImageDraw.Draw(im).text((w - tw, CORNER_LOGO_H + 8), CREDIT_LINE, font=f, fill=FG)
    im.save(path)
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("animation", choices=sorted(ANIMATIONS))
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", help="frame_NNNNN.png sequence from the --chunk workers")
    src.add_argument("--video", help="an already-encoded MP4/MOV")
    ap.add_argument("--fps", type=float, default=None, help="frame rate of --frames-dir (default 30, 12 with --draft)")
    ap.add_argument("--draft", action="store_true", help="source is a draft render (960x540, 12 fps)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = ANIMATIONS[args.animation]
    out = args.out or os.path.join(HERE, spec["basename"] + ("_draft" if args.draft else "") + "_youtube.mp4")
    work = os.path.join(HERE, "_frames", "youtube_cards")
    os.makedirs(work, exist_ok=True)
    tcard = os.path.join(work, f"{args.animation}_title.png")
    ecard = os.path.join(work, f"{args.animation}_end.png")
    title_card(spec, tcard)
    end_card(spec, ecard)

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    if args.frames_dir:
        fps = args.fps or (12 if args.draft else 30)
        n = len(glob.glob(os.path.join(args.frames_dir, "frame_*.png")))
        if n == 0:
            sys.exit(f"No frames in {args.frames_dir}")
        main_in = ["-framerate", str(fps), "-i", os.path.join(args.frames_dir, "frame_%05d.png")]
        main_s = n / fps
    else:
        probe = subprocess.run([ffmpeg, "-i", args.video], capture_output=True, text=True).stderr
        import re
        mm = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe)
        main_s = int(mm[1]) * 3600 + int(mm[2]) * 60 + float(mm[3])
        main_in = ["-i", args.video]

    # Branding overlay
    if BRAND_MODE == "bar":
        bar = os.path.join(work, "brand_bar.png")
        brand_bar(bar)
        vh = OUT_H - BAR_H
        main_chain = (f"[1:v]scale={OUT_W}:{vh}:flags=lanczos,pad={OUT_W}:{OUT_H}:0:0:black,"
                      f"fps={OUT_FPS},format=yuv420p[m0];[m0][2:v]overlay=0:{vh}[main]")
    else:
        badge = os.path.join(work, "corner_badge.png")
        bw, bh = corner_badge(badge)
        bar = badge
        x = CORNER_MARGIN if "left" in CORNER else OUT_W - CORNER_MARGIN - bw
        y = CORNER_MARGIN if "top" in CORNER else OUT_H - CORNER_MARGIN - bh
        main_chain = (f"[1:v]scale={OUT_W}:{OUT_H}:flags=lanczos,fps={OUT_FPS},format=yuv420p[m0];"
                      f"[m0][2:v]overlay={x}:{y}[main]")

    # Title card -> main -> end card, cross-faded; silent stereo track.
    title_s = spec.get("title_s", TITLE_CARD_S)
    end_s = spec.get("end_s", END_CARD_S)
    t_off1 = title_s - XFADE_S
    t_off2 = t_off1 + main_s - XFADE_S
    total = t_off2 + end_s
    fc = (f"[0:v]fps={OUT_FPS},format=yuv420p,trim=duration={title_s},fade=t=in:st=0:d=0.5[t];"
          f"{main_chain};"
          f"[3:v]fps={OUT_FPS},format=yuv420p,trim=duration={end_s},fade=t=out:st={end_s - 0.75}:d=0.75[e];"
          f"[t][main]xfade=transition=fade:duration={XFADE_S}:offset={t_off1:.3f}[tm];"
          f"[tm][e]xfade=transition=fade:duration={XFADE_S}:offset={t_off2:.3f}[v]")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-stats",
           "-loop", "1", "-framerate", str(OUT_FPS), "-t", str(title_s), "-i", tcard,
           *main_in,
           "-i", bar,
           "-loop", "1", "-framerate", str(OUT_FPS), "-t", str(end_s), "-i", ecard,
           "-f", "lavfi", "-t", f"{total:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
           "-filter_complex", fc, "-map", "[v]", "-map", "4:a",
           *X264, "-c:a", "aac", "-b:a", "96k", "-shortest", out]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB, {total:.1f} s)")

    meta = os.path.splitext(out)[0] + ".txt"
    with open(meta, "w") as f:
        f.write(f"TITLE\n{spec['title']}\n\n")
        f.write("DESCRIPTION\n" + spec["description"] + "\n\n")
        f.write(f"Animation: {AUTHOR}, {AFFILIATION}.\n{VERTICAL_EXAG_NOTE}\n\nSource data and models:\n")
        for line in spec.get("credits_full", DATA_CREDITS_FULL):
            f.write(f"- {line}\n")
        f.write(f"\nData archive: {NSIDC_ACK}\n")
        f.write("\nTAGS\n" + ", ".join(spec["tags"]) + "\n")
        f.write("\nUPLOAD SETTINGS\n1920x1080 30 fps H.264 (High), yuv420p BT.709, AAC silent track; "
                "category: Science & Technology; add a custom thumbnail from _frames/youtube_cards or a still.\n")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
