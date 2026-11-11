# Metric photogrammetry for medical phantoms

Goal: reconstruct real objects (simple shapes first, surgical phantoms later) as
**metric 3D meshes placed in a defined reference frame**, and **quantify the
error** in shape, scale and placement. The output serves as a ground truth
(with stated uncertainty) for related research, e.g. stereo-endoscope
reconstruction, and may later be registered to CT.

Design principles:

- **Scale-agnostic**: from decimetres (TAPP: abdominal wall, inguinal region)
  down to millimetres (nerve repair, nodules). Not remote sensing.
- **Device-agnostic**: any calibrated camera (mirrorless, smartphone,
  endoscope). Cameras, lenses and targets are described in config files, never
  hard-coded.
- **Runs on CPU and GPU**: the same pipeline with presets. The CPU presets give
  lower density and resolution but run on a laptop or desktop.
- **Every result comes with an error report.** A model without a measured error
  does not count as a ground truth.

---

## 1. Why photogrammetry

Structure-from-Motion (SfM) followed by Multi-View Stereo (MVS) gives sub-mm
accuracy at almost no cost, if:

- the surface has **texture** to match,
- **scale and frame come from accurate references** (coded markers on a
  measured target, not a ruler),
- the **camera is calibrated and stays stable** (fixed focus, no
  stabilization).

| Method | Typical accuracy | Notes |
|---|---|---|
| Photogrammetry + coded targets | 0.05–0.5 mm | needs texture; our approach |
| Consumer structured light | 0.05–0.2 mm | better on low texture, less control over the frame |
| RGB-D / phone LiDAR | 1–5 mm | too coarse |
| CT | ~voxel/2 (≈0.2–0.3 mm) | sees inside; threshold-dependent surface |
| Gaussian splatting / NeRF | worse surface accuracy | not used for metric ground truth |

Known limitations:

- Only visible surfaces are reconstructed (scan parts before assembly; the
  underside is never seen).
- Glossy, translucent or uniform surfaces (silicone phantoms) need treatment:
  matte scanning spray (sublimating, AESUB-type), a speckle pattern, or
  cross-polarized lighting.
- Translucent silicone causes **subsurface scattering**, which biases depth.
  Coat it.
- **Soft phantoms deform.** Keep the phantom on one rigid base for every
  modality.

## 2. Pipeline

```
capture (RAW/TIFF or video frames)
  → [calibrate]   intrinsics per camera + lens + focus setting (target-based)
  → [sfm]         features + matching + incremental/global SfM + bundle adjustment
  → [markers]     detect coded markers, triangulate, similarity transform → reference frame
  → [dense]       MVS depth maps → fused cloud → mesh → (texture)
  → [evaluate]    shape / scale / placement errors vs reference geometry
  → [register]    (later) sphere fiducials → CT frame, FRE/TRE
```

Theory recap (step: what it does):

1. **Features and matching**: SIFT (classic) or SuperPoint + LightGlue (more
   robust on low texture).
2. **Two-view geometry**: essential/fundamental matrix + RANSAC,
   `x2ᵀ E x1 = 0`.
3. **Camera model**: pinhole + Brown–Conrady distortion (`OPENCV` /
   `FULL_OPENCV`); fisheye models for endoscopes. `x = K [R|t] X`.
4. **Bundle adjustment**: Levenberg–Marquardt on reprojection error (Ceres).
   Optionally with marker constraints.
5. **MVS**: PatchMatch on photometric consistency → depth maps → fusion →
   meshing (Delaunay / Poisson).
6. **Frame alignment**: 7-parameter similarity (Umeyama) to marker
   coordinates, with check points held out.

## 3. Tooling decisions

| Role | Choice | Why |
|---|---|---|
| Orchestration | **Python** (≥3.10), managed with **uv** | laptop 3.10, desktop and server 3.12 |
| SfM | **COLMAP via `pycolmap`** (+ optional `hloc`/LightGlue, GLOMAP) | state of the art, scriptable, SfM on CPU is fine |
| Dense / mesh / texture | **OpenMVS** | CPU *and* CUDA (COLMAP's PatchMatch is CUDA-only) |
| Calibration, markers | **OpenCV** (aruco/ChArUco, AprilTag 36h11 dictionary) | one library for both target types |
| RAW decoding | **rawpy** (LibRaw), lens corrections **off** | the calibration models the distortion |
| Mesh processing, evaluation | **Open3D / trimesh** | ICP, distances, sphere fitting |
| GUI sanity checks | COLMAP GUI, Meshroom, MeshLab, CloudCompare | inspection only |

Considered but not chosen as the core: OpenMVG (less momentum), OpenSfM (less
maintained), PyOPF and the Rust crates (too immature for a full pipeline).
Agisoft Metashape Pro remains a useful commercial cross-check, if a licence
becomes available.

### Compute presets

| Preset | Images | Dense | Mesh |
|---|---|---|---|
| `cpu-fast` | ≈2000 px long side | OpenMVS resolution-level 2, fewer views | decimated (~200k faces), no refine |
| `cpu-full` | ≈3000 px | resolution-level 1 | ~1M faces |
| `gpu-full` | full resolution | level 0, all views, RefineMesh | full |

**GPU server (RTX PRO 6000, Blackwell, sm_120, 98 GB):** prebuilt binaries
may lack sm_120 kernels. Build COLMAP/OpenMVS from source with CUDA ≥ 12.8
and `CMAKE_CUDA_ARCHITECTURES=120`, ideally in a Dockerfile.

## 4. Scene and reference frame

- **Base plate** (separate from the calibration target): rigid, flat, matte,
  carrying coded markers (ArUco/ChArUco/AprilTag) **at measured positions**.
  - Frame: origin at a chosen marker corner, X/Y in the plate plane, Z normal
    to it.
  - Add a few markers **off-plane** (a block of known height) to constrain Z
    and tilt.
  - Use 4–6+ markers spread around the object; hold 1–2 out as **check
    points**.
- **Scale**: comes from the measured marker layout, plus an independent
  **scale bar** (two targets at a measured distance) as a check.
  - **A 0.2 % scale error on 100 mm is already 0.2 mm.** Printed targets are
    often 0.1–0.5 % off. Measure or certify them.
- **Placement ground truth**: a fixture plate with holes at known positions
  and dowel pins, or reference spheres glued to the object at measured
  positions.
- **CT readiness**: 4–6 non-coplanar **ceramic or aluminium spheres** (not
  steel: streak artifacts), matte-coated, fixed to the base.
  - Fit sphere centers in both modalities.
  - Register with Arun/Horn's point-based method.
  - Report FRE (fiducial registration error) and TRE (target registration
    error).
  - CT is *not* a 0.1 mm ground truth: voxel size, partial volume and
    thresholding all add error.
- **Lighting**: diffuse and constant; cross-polarization for glossy surfaces.

## 5. Capture protocol (cm–dm scale)

- Manual everything: focus (then tape the ring), aperture, shutter, ISO, white
  balance.
- **Stabilization off**: lens IS and in-body IS (IBIS). On phones: EIS off,
  one fixed lens (no automatic lens switching), no digital zoom.
- **Stills > video.**
  - If you use video: 4K, highest bitrate, shutter ≥ 1/500 s, slow motion.
    Extract frames by spacing, and reject blurry ones (variance of the
    Laplacian).
- Object fills ~60–80 % of the frame. Shoot rings at ~15°, 40° and 65°
  elevation, plus top views, with 8–10° azimuth steps (≈100–150 images).
  Overlap ≥ 80 %.
- Alternative: a **turntable with a fixed camera**. The markers must then be
  on the turntable, and the background plain or masked.

### Current camera: Canon R6 + RF 35mm f/1.8 Macro IS STM

- 20 MP full frame, 6.56 µm pixels, maximum magnification 0.5×.
- Work at **25–40 cm, f/8** → GSD ≈ 50–70 µm → target surface accuracy
  ≈ 0.05–0.1 mm.
- **Distortion**: the lens has strong barrel distortion that is normally
  corrected in software. Shoot RAW and decode with corrections off.
- **Focus breathing**: the focal length changes with focus. Use one
  calibration per focus setting, at the same setting as the scan.
- The smartphone (Xperia 1 III, 12 MP) is the comparison experiment: expect
  0.15–0.3 mm in photo mode and 0.3–0.5 mm in video.

## 6. Multi-scale roadmap

The math does not depend on scale; the optics do. Depth of field shrinks with
the **square of magnification**.

| Scale | Example objects | Optics | GSD | Specifics |
|---|---|---|---|---|
| 2–20 cm | TAPP organs, wall | R6 + 35 mm @ 25–40 cm | 50–70 µm | baseline |
| 0.5–3 cm | nerve segments, nodules | 0.3–1× macro | 7–20 µm | turntable, **focus stacking**, ring light + polarizers |
| < 5 mm | fine nerve structure | 1–5× macro / microscope objective | 1–5 µm | motorized stages; spray particles become visible; consider micro-CT or profilometry |

Reference for macro photogrammetry: **scAnt** (open-source insect scanner with
focus stacking, 2021).

## 7. Calibration targets

- **Available now**: Kalibr-style **AprilGrid**, printed on acrylic in Taiwan
  (flat, not certified).
  - Layout: 6×6 tags, tag family 36h11, `tagSize` = 43.13 mm (black square),
    `tagSpacing` = 0.391 (gap / tagSize → gap ≈ 16.86 mm), total ≈ 343 mm.
    These values come from earlier measurements, not from the original file
    (no longer downloadable).
  - Caveats: tag *corners* are biased by blur and exposure (ChArUco
    **saddle points** are not), and acrylic expands ~70 µm/m/K (5 K ×
    30 cm ≈ 0.1 mm).
  - Usable for first trials; **re-measure** the spacing across the full
    board.
- **Requested** (if budget allows):

| Target | For | Spec |
|---|---|---|
| A | R6 at the dm scale | ChArUco, ~A4, 12–15 mm squares, matte, glass or aluminium composite, **certified dimension report** |
| B | stereo endoscope | ChArUco, ~80–120 mm, 3–5 mm squares, **matte non-glass** (coaxial light glare), rigid |
| C (later) | macro/nerve scale | chrome-on-glass or ceramic, lithographic, 0.5–1 mm squares |

Endoscopes need fisheye models (`equidistant` / Kannala–Brandt) and good
coverage of the image corners.

## 8. Evaluation (what "accurate" means here)

Report separately:

- **Shape**: best-fit reference geometry (ICP/least squares) → cloud-to-model
  distances (mean, RMS, P95, max) + a deviation heatmap.
- **Scale**: the fitted scale factor's deviation from 1, plus the scale bar
  check.
- **Placement**: fitted pose vs fixture pose (mm, degrees); marker
  check-point errors.
- **Repeatability**: repeat scans of the same object.

Phase-1 reference objects:

- precision ceramic balls (µm-level diameter tolerance),
- a machined block or gauge blocks,
- 3D-printed platonic solids (**measured** with calipers, not taken from
  CAD).

## 9. Roadmap

0. **Repo + environment** (uv, Python ≥3.10), config schema for devices,
   targets and presets.
1. **Calibration** (`calibrate`): AprilGrid now, ChArUco later.
   - RAW → linear TIFF.
   - Export to COLMAP / Kalibr formats.
   - Report: reprojection error, coverage heatmap, parameter uncertainty.
2. **SfM** (`sfm`) with known intrinsics; marker detection and frame alignment
   (`markers`).
3. **Dense** (`dense`) with OpenMVS: CPU and GPU presets.
4. **Evaluation** (`evaluate`) on simple shapes.
5. Phantom (TAPP), coating and lighting study.
6. CT registration (`register`).
7. Smaller scales: focus stacking, macro rigs; endoscope camera frame
   registration.

## 10. Working across machines

| Machine | Python | Role |
|---|---|---|
| laptop | 3.10.2 | development, small CPU runs |
| desktop | 3.12.3 | CPU runs |
| GPU server (RTX PRO 6000) | 3.12.3 | full-resolution runs |

Code is shared through git. Project knowledge for Claude Code lives in
`CLAUDE.md` (committed), so a Claude Code session on any machine starts with
the same context.

## 11. Development quick start

```sh
uv sync                          # base deps (+ dev group: pytest, ruff)
uv sync --extra sfm --extra eval # later stages: pycolmap, open3d, trimesh
uv run metrogram check-config configs/*/*.yaml
PYTHONPATH= uv run pytest -q
uv run ruff check . && uv run ruff format .
```

- Package: `metrogram` (`src/metrogram/`), CLI `metrogram <subcommand>`.
- Configs: `configs/devices/`, `configs/targets/`, `configs/presets/`. Each
  YAML file has a `kind`. Lengths are in metres, with the unit in the key name.
  Unknown keys are rejected.
- No `.python-version` is pinned: each machine uses its own interpreter
  (3.10 laptop, 3.12 desktop/server) against the same `uv.lock`.
- **ROS note:** a sourced ROS Humble puts `/opt/ros/...` on `PYTHONPATH`. Its
  pytest plugins then break test collection, and its packages can shadow the
  venv's. Clear it for this project (`PYTHONPATH= uv run ...`), or use an
  alias such as `alias uvr='PYTHONPATH= uv run'`.
