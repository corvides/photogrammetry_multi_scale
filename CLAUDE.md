# CLAUDE.md

Project: metric photogrammetry pipeline (shape + scale + placement, with error
reports) for medical phantoms. Full background and decisions are in
`README.md`. Read it before any design work.

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
  - Tag ID ordering / origin corner **not yet verified** against an image.
    Kalibr numbers tags row-wise; confirm the starting corner before
    trusting the 3D points.
  - Detect with OpenCV `DICT_APRILTAG_36h11`.
- Planned targets: ChArUco (A4, certified) for the camera; small matte
  ChArUco for the stereo endoscope.

## Repo conventions (to be refined as code lands)

- `src/` layout package, CLI subcommands per stage: `calibrate`, `sfm`,
  `markers`, `dense`, `evaluate`, `register`.
- `configs/devices/`, `configs/targets/`, `configs/presets/`.
- Data and outputs are outside git (`data/`, `runs/` in `.gitignore`).
- Tests with pytest on synthetic data (render or project known geometry), so
  they run without a camera.
