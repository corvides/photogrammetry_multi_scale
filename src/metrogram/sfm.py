"""Sparse reconstruction (Structure-from-Motion) with COLMAP through pycolmap.

Steps: prepare images (RAW decode, preset downscaling) -> SIFT features ->
matching (exhaustive or sequential) -> incremental mapping with known or
self-calibrated intrinsics -> report.

The result has an arbitrary scale and frame; the ``markers`` stage maps it to
the metric board frame.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from metrogram.config import Device, Preset
from metrogram.sources import IMAGE_EXT, RAW_EXT, read_gray

# Cameras supported by our calibration / this stage (COLMAP names).
_PARAM_COUNT = {"PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5, "OPENCV": 8, "FULL_OPENCV": 12}


@dataclass
class CameraSpec:
    """Intrinsics in COLMAP convention for a given image size."""

    model: str
    width: int
    height: int
    params: list[float]
    source: str = ""  # where the values came from

    def __post_init__(self):
        if self.model not in _PARAM_COUNT:
            raise ValueError(f"unsupported camera model {self.model}")
        if len(self.params) != _PARAM_COUNT[self.model]:
            raise ValueError(f"{self.model} needs {_PARAM_COUNT[self.model]} params")

    def scaled(self, width: int, height: int) -> CameraSpec:
        """Same camera for a resized image (COLMAP convention: scale f and c)."""
        sx, sy = width / self.width, height / self.height
        p = list(self.params)
        if self.model in ("SIMPLE_RADIAL", "RADIAL"):
            p[0] *= (sx + sy) / 2
            p[1] *= sx
            p[2] *= sy
        else:
            p[0], p[1], p[2], p[3] = p[0] * sx, p[1] * sy, p[2] * sx, p[3] * sy
        return CameraSpec(self.model, width, height, p, f"{self.source} (scaled)")

    @classmethod
    def from_calibration(cls, path: str | Path) -> CameraSpec:
        with open(path) as fh:
            c = yaml.safe_load(fh)
        if c.get("kind") != "calibration":
            raise ValueError(f"{path} is not a calibration file")
        return cls(
            c["colmap"]["model"],
            c["image_size"]["width"],
            c["image_size"]["height"],
            [float(v) for v in c["colmap"]["params"]],
            str(path),
        )

    @classmethod
    def from_device(cls, device: Device, width: int, height: int) -> CameraSpec:
        """Nominal pinhole guess (no distortion) for self-calibration."""
        f = device.focal_length_px * width / device.sensor.width_px
        model = device.colmap_model if device.colmap_model in _PARAM_COUNT else "OPENCV"
        n_extra = _PARAM_COUNT[model] - (3 if model in ("SIMPLE_RADIAL", "RADIAL") else 4)
        head = [f, width / 2, height / 2] if n_extra < 3 else [f, f, width / 2, height / 2]
        return cls(model, width, height, head + [0.0] * n_extra, f"device {device.name} nominal")


@dataclass
class SfmSettings:
    refine_intrinsics: bool = False  # False: keep calibrated intrinsics fixed
    matcher: str = "exhaustive"  # exhaustive | sequential
    sequential_overlap: int = 15
    max_num_features: int = 8192
    max_image_size_px: int | None = None
    use_gpu: bool = False
    num_threads: int = -1
    seed: int = 0


@dataclass
class SfmResult:
    out_dir: str
    image_dir: str  # images as read by COLMAP (prepared/resized if needed)
    n_input_images: int
    n_registered: int
    n_points3D: int
    mean_reproj_error_px: float
    mean_track_length: float
    mean_obs_per_image: float
    n_models: int
    camera_initial: dict
    camera_final: dict
    unregistered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def settings_from_preset(p: Preset, refine_intrinsics: bool) -> SfmSettings:
    return SfmSettings(
        refine_intrinsics=refine_intrinsics,
        matcher=p.sfm.matcher,
        max_num_features=p.sfm.max_num_features,
        max_image_size_px=p.max_image_size_px,
        use_gpu=p.sfm.use_gpu,
    )


def _list_images(image_dir: Path) -> list[Path]:
    files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXT)
    if not files:
        raise ValueError(f"no images in {image_dir}")
    return files


def _image_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() in RAW_EXT:
        h, w = read_gray(path).shape
        return w, h
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read {path}")
    return img.shape[1], img.shape[0]


def prepare_images(image_dir: Path, work: Path, max_size: int | None, log) -> tuple[Path, tuple]:
    """Return (dir COLMAP should read, (w, h)). Copies only when decoding/resizing."""
    files = _list_images(image_dir)
    w, h = _image_size(files[0])
    raw = any(f.suffix.lower() in RAW_EXT for f in files)
    scale = 1.0 if not max_size or max(w, h) <= max_size else max_size / max(w, h)
    if not raw and scale == 1.0:
        return image_dir, (w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    out = work / "images"
    out.mkdir(parents=True, exist_ok=True)
    if log:
        print(f"  preparing {len(files)} images -> {nw}x{nh} in {out}", file=log)
    for f in files:
        if f.suffix.lower() in RAW_EXT:
            import rawpy

            with rawpy.imread(str(f)) as r:
                img = cv2.cvtColor(
                    r.postprocess(use_camera_wb=True, no_auto_bright=True), cv2.COLOR_RGB2BGR
                )
        else:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img.shape[1] != w or img.shape[0] != h:
            raise ValueError(f"{f.name}: size {img.shape[1]}x{img.shape[0]} != {w}x{h}")
        if scale != 1.0:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / (f.stem + ".png")), img)
    return out, (nw, nh)


def _camera_dict(cam) -> dict:
    return {
        "model": cam.model_name if hasattr(cam, "model_name") else str(cam.model),
        "width": int(cam.width),
        "height": int(cam.height),
        "params": [float(v) for v in cam.params],
    }


def run(
    image_dir: str | Path,
    out: str | Path,
    camera: CameraSpec | None,
    s: SfmSettings,
    log=sys.stderr,
) -> SfmResult:
    import pycolmap

    image_dir, out = Path(image_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    img_path, (w, h) = prepare_images(image_dir, out, s.max_image_size_px, log)
    if camera is not None and (camera.width, camera.height) != (w, h):
        aspect_cam, aspect_img = camera.width / camera.height, w / h
        if abs(aspect_cam - aspect_img) > 1e-3:
            raise ValueError(
                f"calibration is for {camera.width}x{camera.height}, images are {w}x{h} "
                "(different aspect ratio: wrong capture mode?)"
            )
        camera = camera.scaled(w, h)

    db = out / "database.db"
    if db.exists():
        db.unlink()
    sparse = out / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()

    device = pycolmap.Device.cuda if (s.use_gpu and pycolmap.has_cuda) else pycolmap.Device.cpu
    warnings = []
    if s.use_gpu and not pycolmap.has_cuda:
        warnings.append("GPU requested but this pycolmap build has no CUDA: ran on CPU")

    reader = pycolmap.ImageReaderOptions()
    if camera is not None:
        reader.camera_model = camera.model
        reader.camera_params = ",".join(f"{v:.10g}" for v in camera.params)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = max(w, h)  # never let COLMAP downscale silently
    extraction.sift.max_num_features = s.max_num_features
    extraction.num_threads = s.num_threads

    if log:
        print(f"  features ({device.name}) ...", file=log)
    pycolmap.extract_features(
        db,
        img_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=device,
    )

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = s.num_threads
    if log:
        print(f"  matching ({s.matcher}) ...", file=log)
    if s.matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=matching, device=device)
    elif s.matcher == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = s.sequential_overlap
        pairing.loop_detection = False
        pycolmap.match_sequential(
            db, matching_options=matching, pairing_options=pairing, device=device
        )
    else:
        raise ValueError(f"unsupported matcher '{s.matcher}' (exhaustive | sequential)")

    opts = pycolmap.IncrementalPipelineOptions()
    refine = s.refine_intrinsics or camera is None
    opts.ba_refine_focal_length = refine
    opts.ba_refine_principal_point = False  # poorly constrained; calibrate it instead
    opts.ba_refine_extra_params = refine
    opts.num_threads = s.num_threads
    opts.random_seed = s.seed
    if log:
        print(f"  mapping (intrinsics {'refined' if refine else 'fixed'}) ...", file=log)
    models = pycolmap.incremental_mapping(db, img_path, sparse, options=opts)
    if not models:
        raise RuntimeError("SfM failed: no model reconstructed")

    best_idx = max(models, key=lambda k: models[k].num_reg_images())
    rec = models[best_idx]
    best_dir = sparse / str(best_idx)
    rec.export_PLY(str(out / "sparse_points.ply"))

    names = sorted(f.name for f in img_path.iterdir() if f.suffix.lower() in IMAGE_EXT)
    registered = {rec.image(i).name for i in rec.reg_image_ids()}
    unregistered = [n for n in names if n not in registered]
    cam_final = _camera_dict(next(iter(rec.cameras.values())))
    res = SfmResult(
        out_dir=str(best_dir),
        image_dir=str(img_path),
        n_input_images=len(names),
        n_registered=len(registered),
        n_points3D=rec.num_points3D(),
        mean_reproj_error_px=float(rec.compute_mean_reprojection_error()),
        mean_track_length=float(rec.compute_mean_track_length()),
        mean_obs_per_image=float(rec.compute_mean_observations_per_reg_image()),
        n_models=len(models),
        camera_initial=asdict(camera) if camera else {"source": "COLMAP default guess"},
        camera_final=cam_final,
        unregistered=unregistered,
        warnings=warnings,
    )
    if res.n_registered < 0.9 * res.n_input_images:
        warnings.append(
            f"only {res.n_registered}/{res.n_input_images} images registered: "
            "check overlap, texture, blur"
        )
    if len(models) > 1:
        sizes = sorted((m.num_reg_images() for m in models.values()), reverse=True)
        warnings.append(f"scene split into {len(models)} models (sizes {sizes}); kept the largest")
    if res.mean_reproj_error_px > 1.0:
        warnings.append(f"mean reprojection error {res.mean_reproj_error_px:.2f} px > 1 px")
    if camera is not None and refine:
        p0, p1 = np.array(camera.params), np.array(cam_final["params"])
        df = (p1[0] - p0[0]) / p0[0]
        if abs(df) > 0.01:
            warnings.append(f"refined focal length differs by {df:+.1%} from the calibration")

    (out / "sfm_report.json").write_text(json.dumps(asdict(res), indent=1))
    return res


def summary(r: SfmResult) -> str:
    lines = [
        f"registered {r.n_registered}/{r.n_input_images} images, {r.n_points3D} points, "
        f"{r.n_models} model(s)",
        f"  mean reprojection error {r.mean_reproj_error_px:.3f} px, "
        f"track length {r.mean_track_length:.1f}, {r.mean_obs_per_image:.0f} obs/image",
        f"  camera {r.camera_final['model']} params "
        + ", ".join(f"{v:.6g}" for v in r.camera_final["params"]),
    ]
    lines += [f"  WARNING: {w}" for w in r.warnings]
    return "\n".join(lines)
