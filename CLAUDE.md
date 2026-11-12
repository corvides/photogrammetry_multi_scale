# CLAUDE.md

Project: metric photogrammetry pipeline (shape + scale + placement, with error
reports) for medical phantoms. Full background and decisions are in
`README.md`. Read it before any design work.

## Session continuity

- Dated worklogs live in `worklog/YYYY-MM-DD.md` (done / open items / NEXT). At the
  start of a session, read the latest one; at the end, write or update the day's file.

## Hard constraints

- **Scale-agnostic** (mm to dm) and **device-agnostic**: cameras, lenses
  (+ focus setting), targets, marker layouts and compute presets come from
  config files (YAML). Never hard-code sensor sizes, target dimensions or
  scale-dependent thresholds; express thresholds in pixels or relative to
  GSD.
- **Python ≥ 3.10** (laptop 3.10.2; desktop and server 3.12.3). No 3.11+
  features (`tomllib`, `ExceptionGroup`, `typing.Self`, ...). Managed with
  **uv**; Linux only.
- **CPU and GPU**: every stage must run on CPU (`cpu-fast`, `cpu-full`
  presets); GPU (`gpu-full`) is optional acceleration.
  - COLMAP PatchMatch is CUDA-only, so dense reconstruction uses
    **OpenMVS**.
  - GPU server is Blackwell (sm_120): build with CUDA ≥ 12.8 and
    `CMAKE_CUDA_ARCHITECTURES=120`.
- **Stack**: pycolmap (SfM), OpenMVS (dense/mesh/texture, external
  binaries), OpenCV (calibration and markers), rawpy (RAW, lens corrections
  OFF), Open3D/trimesh (evaluation), numpy/scipy.
- **Every stage writes a machine-readable report** (JSON/YAML) with its
  errors. Accuracy claims need numbers.

## User working style

- Terminal + vim. CLIs over GUIs/notebooks; plain-text outputs (YAML/JSON/
  CSV, PNG plots); no interactive windows required (add a `--show` flag
  instead).
- Units: **metres internally** (Kalibr/COLMAP convention); reports may show
  mm. Label units explicitly in field names (e.g. `rms_px`, `err_mm`).

## Domain facts to remember

- Current camera: Canon R6 (5472×3648, 6.56 µm pixels) + RF 35mm f/1.8 Macro
  IS STM.
  - Strong barrel distortion, normally corrected in software: decode RAW
    without correction.
  - Focus breathing: calibrations are **per focus setting**.
  - IS and IBIS off.
- Comparison camera: Sony Xperia 1 III (12 MP).
- Available target: Kalibr AprilGrid on acrylic.
  - 6×6 tags, family 36h11, tagSize 0.04313 m, tagSpacing 0.391 (ratio).
    Values from user measurements; the original YAML is unavailable.
  - **Verified** (2026-11-12, video frame 368): tag 0 at the board's
    bottom-left (printed caption at bottom), ids row-wise left→right, rows
    going up. Board frame: x right, y up, origin = outer bottom-left corner of
    tag 0.
  - OpenCV corner order for these tags (tags are printed 180° from OpenCV's
    canonical orientation): corners[0..3] = board BR, BL, TL, TR of each tag.
    Homography check: 1.4 px RMS (uncorrected distortion) vs 65–94 px for
    other orders. Detection is strong (up to 35/36 tags at 1080p).
  - Detect with OpenCV `DICT_APRILTAG_36h11`.
- Planned targets: ChArUco (A4, certified) for the camera; small matte
  ChArUco for the stereo endoscope.

## Repo conventions

- Package `metrogram` in `src/metrogram/`; CLI `metrogram <subcommand>`
  (argparse, `cli.py`). Existing: `check-config`, `calibrate`,
  `extract-frames`, `sfm`, `markers`, `view`, `dense`. Planned: `evaluate`,
  `register`.
- `config.py`: strict frozen dataclasses loaded from YAML (`kind:
  device|target|preset`; targets also need `type: aprilgrid|charuco`).
  Unknown keys are errors. Add new config fields there and in the shipped YAML.
- Calibration modules: `targets.py` (detectors -> board/image points),
  `sources.py` (video/images/RAW), `calibration.py` (scan, quality filter,
  farthest-point pose selection, holdout, outlier rounds), `calibration_io.py`
  (calibration.yaml, Kalibr camchain, report.json), `plots.py`. COLMAP export
  adds +0.5 px to the principal point.
- SfM: `sfm.py` on pycolmap 4.2 (API changes between versions: introspect
  before using new calls). Intrinsics from calibration.yaml are fixed by
  default (`--refine-intrinsics` to refine); the principal point is never
  refined. Images are downscaled by the preset and intrinsics scaled
  accordingly; `extraction.max_image_size` is set so COLMAP never downscales
  silently. `geometry.umeyama` = similarity alignment (reuse it for markers).
- Markers: `markers.py` detects the target in the SfM images, drops tags
  inconsistent with one board homography per image (occlusion/false decodes),
  triangulates (midpoint + LM on pixels), holds out every 4th tag/cell as
  check points, aligns with robust Umeyama. Outputs `sparse_metric/` (COLMAP,
  board frame, metres) and PLYs written by `ply.py` (binary LE, x y z + rgb;
  overlays are triangle meshes because edge elements are poorly supported).
  `metrogram view` (`viewer.py`, Open3D, Z-up, metres) overlays PLY layers,
  frames on board/markers (box extended upward for objects), renders a
  turntable or `--show`; design adapted from the user's Isaac Sim viewer.
- OpenMVS is built by `scripts/build_openmvs.sh`, default **v2.3.0** (upstream's
  Ubuntu 22.04 release: CGAL 5.4, system OpenCV 4.5). v2.4.0 (`--version`)
  needs OpenCV >= 4.11 and nanoflann >= 1.5, which 22.04 lacks; it also has a
  wrong `REQUIRED` on libjxl that the script patches. Everything is pinned to
  commit SHAs and verified; Python bindings off; installs into `~/opt/openmvs`
  (NOT ~/.local: CMake searches PATH-derived prefixes, so ~/.local would leak
  CGAL into the user's ROS/colcon builds), CGAL into its own sub-prefix passed
  via CGAL_DIR. The script refuses to be sourced, never uses sudo, and only
  strips `/opt/ros/*` from its own CMAKE_PREFIX_PATH (the user's custom
  prefixes matter to them). The dense stage looks in
  `~/opt/openmvs/bin/OpenMVS`, then `$OPENMVS_BIN`, then `$PATH`.
- Dense: `dense.py` runs from `markers/sparse_metric` (board frame) ->
  COLMAP undistort -> OpenMVS tools with cwd=`work/` (OpenMVS resolves image
  paths stored in the .mvs relative to the working dir). TextureMesh exports
  PLY (v2.3.0's OBJ export segfaults); an OBJ copy is made with trimesh for
  Open3D. `ply.read_vertices` handles OpenMVS's per-point list properties.
  Report includes a board-plane check (dense points on the board vs z = 0).
- Tests: `tests/synthetic_scene.py` renders textured quads (cube on a floor)
  from known cameras. Slow tests are marked `@pytest.mark.slow`
  (`-m "not slow"` to skip).
- Synthetic rendering: supersample (plain perspective warps alias edges),
  map texture pixel *edges* to quad edges, and use target dimensions that are
  whole texture pixels (else tag positions are quantized). Tag-corner
  detection floor is ~0.2 px even on perfect renders.
- Config files: `configs/devices/`, `configs/targets/`, `configs/presets/`.
- Base deps stay light; stage-specific heavy deps go in extras (`sfm`, `eval`).
- No `.python-version` pin (3.10 and 3.12 machines share `uv.lock`).
- **Laptop has ROS Humble sourced**: always run with `PYTHONPATH= uv run ...`
  (ROS pytest plugins break collection otherwise).
- Git is managed by the user only (never commit/stage). Checks before handing over: `uv run ruff format . && uv run ruff check . &&
  PYTHONPATH= uv run pytest -q`.
- `for_reference/` holds user-provided files from other repos: read-only, never
  modify or format (excluded from ruff). It may disappear; don't depend on it.
- Data and outputs are outside git (`data/`, `runs/` in `.gitignore`).
- Tests with pytest on synthetic data (render or project known geometry), so
  they run without a camera.
- in the recap, be sure to clearly state the study is in relation to MIS and RAS.
