# Wilkes Subglacial Basin visualisation

Scripts that generate the policy-brief animations and figures for the Wilkes
Subglacial Basin, East Antarctica — from above the South Pole down into the
basin, the ice sheet lifted off its bed, and a present-day / Pliocene
comparison.

Animations: Matt King, University of Tasmania.

## Animations

| Script | What it makes | Length |
| --- | --- | --- |
| `wilkes_basin_3d_animation_intro.py` | Continent-wide intro, fly-by of the Mertz, Ninnis and Cook glaciers, then the bed reveal | 44 s |
| `wilkes_basin_bed_reveal_animation.py` | Ice block lifts off the bed to expose the basin | 27 s |
| `wilkes_basin_present_pliocene_animation.py` | Present-day ice surface morphing to a Pliocene reconstruction | 19 s |
| `wilkes_basin_3d_animation.py` | Fly-by only (the intro's middle act, standalone) | — |

Supporting code:

* `wilkes_basin_3d_figure.py` — shared scene builder: grids, columns, colour
  maps, labels, lighting. The animations import it.
* `youtube_version.py` — adds title and credit cards and re-encodes for upload.
* `wilkes_basin_paleo_compare.py` — static present/Pliocene comparison figure.

## Rendering

```sh
python wilkes_basin_3d_animation_intro.py --draft          # fast, low-res check
python wilkes_basin_3d_animation_intro.py --stills         # key frames only
python wilkes_basin_3d_animation_intro.py                  # full frame set
python wilkes_basin_3d_animation_intro.py --chunk 1/8 --frames-dir _frames/run
python wilkes_basin_3d_animation_intro.py --assemble       # frames -> mp4/mov/gif
```

`--chunk i/N` splits a render across processes or machines; `--assemble` then
stitches the frame directory into the deliverables.

Requires Python 3 with `pyvista` (off-screen VTK), `numpy`, `xarray`,
`pyproj` and `ffmpeg` on the path.

Rendered video is not in the repository (see `.gitignore`); `_final/` holds
the locked script snapshots and checksums for the published versions.

## Other contents

* `bluesky/` — alt text and `.vtt` captions for each animation.
* `*_youtube.txt` — titles, descriptions and full data credits as uploaded.
* `data/PLIO_end.nc` — Pliocene ice-surface reconstruction used by the
  present–Pliocene animation.
* `colormaps/oleron.txt` — Scientific colour map (Crameri), topography.
* `branding/` — University of Tasmania marks used on the title cards.

## Source data

BedMachine Antarctica v3, MEaSUREs InSAR ice velocity v2 and MOA2004 (NSIDC);
Bedmap3 standardised point data (UK PDC); IBCSO v2 bathymetry coverage.
Full citations and DOIs are in the `*_youtube.txt` files.

## Vertical exaggeration

All 3D scenes use 100x vertical exaggeration. Topography is drawn as columns;
column roughness indicates data density rather than real relief.
