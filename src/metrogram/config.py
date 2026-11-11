"""Typed configuration loaded from YAML.

Every config file has a top-level ``kind`` (``device``, ``target`` or ``preset``).
Lengths are in metres; field names carry their unit suffix.
Unknown or missing keys raise ``ConfigError`` so typos do not pass silently.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


def _build(cls, data: dict[str, Any], where: str):
    """Instantiate a (possibly nested) dataclass from a dict, strictly."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        if name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ConfigError(f"{where}: missing required key '{name}'")
            continue
        value = data[name]
        sub = _NESTED.get((cls, name))
        kwargs[name] = _build(sub, value, f"{where}.{name}") if sub else value
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: {e}") from e


# --------------------------------------------------------------------------- targets

APRILTAG_FAMILIES = {"16h5", "25h9", "36h10", "36h11"}


@dataclass(frozen=True)
class AprilGridTarget:
    """Kalibr-style AprilGrid. ``tag_spacing_ratio`` = gap / tag_size."""

    name: str
    rows: int
    cols: int
    tag_size_m: float
    tag_spacing_ratio: float
    tag_family: str = "36h11"
    # Corner of the board where tag id 0 sits; must be checked on a real image.
    id_origin: str = "bottom_left"
    id_origin_verified: bool = False
    material: str = ""
    thermal_expansion_per_K: float | None = None
    dimensions_certified: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.rows < 1 or self.cols < 1:
            raise ValueError("rows and cols must be >= 1")
        if self.tag_size_m <= 0 or self.tag_spacing_ratio < 0:
            raise ValueError("tag_size_m must be > 0 and tag_spacing_ratio >= 0")
        if self.tag_family not in APRILTAG_FAMILIES:
            raise ValueError(f"tag_family must be one of {sorted(APRILTAG_FAMILIES)}")
        if self.id_origin not in {"bottom_left", "top_left"}:
            raise ValueError("id_origin must be 'bottom_left' or 'top_left'")

    @property
    def gap_m(self) -> float:
        return self.tag_size_m * self.tag_spacing_ratio

    @property
    def pitch_m(self) -> float:
        return self.tag_size_m + self.gap_m

    @property
    def extent_m(self) -> tuple[float, float]:
        """(width, height) of the tag area, outer tag edge to outer tag edge."""
        return (
            self.cols * self.tag_size_m + (self.cols - 1) * self.gap_m,
            self.rows * self.tag_size_m + (self.rows - 1) * self.gap_m,
        )


@dataclass(frozen=True)
class CharucoTarget:
    name: str
    squares_x: int
    squares_y: int
    square_size_m: float
    marker_size_m: float
    dictionary: str = "DICT_5X5_1000"
    legacy_pattern: bool = False
    material: str = ""
    thermal_expansion_per_K: float | None = None
    dimensions_certified: bool = False
    notes: str = ""

    def __post_init__(self):
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("squares_x and squares_y must be >= 2")
        if not 0 < self.marker_size_m < self.square_size_m:
            raise ValueError("need 0 < marker_size_m < square_size_m")

    @property
    def extent_m(self) -> tuple[float, float]:
        return (self.squares_x * self.square_size_m, self.squares_y * self.square_size_m)


Target = AprilGridTarget | CharucoTarget
_TARGET_TYPES = {"aprilgrid": AprilGridTarget, "charuco": CharucoTarget}

# --------------------------------------------------------------------------- devices


@dataclass(frozen=True)
class Sensor:
    width_px: int
    height_px: int
    pixel_pitch_um: float


@dataclass(frozen=True)
class Lens:
    model: str
    focal_length_mm: float  # nominal; the calibration gives the real value
    fisheye: bool = False


@dataclass(frozen=True)
class Device:
    """A camera body + lens. Focus setting belongs to a calibration, not here."""

    name: str
    camera_make: str
    camera_model: str
    sensor: Sensor
    lens: Lens
    # COLMAP camera model used for SfM (OPENCV, FULL_OPENCV, OPENCV_FISHEYE, ...)
    colmap_model: str = "OPENCV"
    raw_extension: str | None = None
    notes: str = ""

    @property
    def focal_length_px(self) -> float:
        return self.lens.focal_length_mm * 1000.0 / self.sensor.pixel_pitch_um

    def gsd_m(self, distance_m: float) -> float:
        """Ground sampling distance (thin-lens approximation, distance from sensor)."""
        f = self.lens.focal_length_mm / 1000.0
        return (distance_m - f) * self.sensor.pixel_pitch_um * 1e-6 / f


# --------------------------------------------------------------------------- presets


@dataclass(frozen=True)
class SfmSettings:
    use_gpu: bool = False
    matcher: str = "exhaustive"  # exhaustive | sequential | vocab_tree
    max_num_features: int = 8192


@dataclass(frozen=True)
class DenseSettings:
    use_cuda: bool = False
    resolution_level: int = 1  # OpenMVS: each level halves image size
    number_views: int = 8  # 0 = all
    min_resolution_px: int = 640


@dataclass(frozen=True)
class MeshSettings:
    decimate_target_faces: int | None = None  # None = keep all
    refine: bool = False
    texture: bool = True


@dataclass(frozen=True)
class Preset:
    name: str
    max_image_size_px: int | None = None  # None = full resolution
    sfm: SfmSettings = field(default_factory=SfmSettings)
    dense: DenseSettings = field(default_factory=DenseSettings)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    notes: str = ""


_NESTED = {
    (Device, "sensor"): Sensor,
    (Device, "lens"): Lens,
    (Preset, "sfm"): SfmSettings,
    (Preset, "dense"): DenseSettings,
    (Preset, "mesh"): MeshSettings,
}

# --------------------------------------------------------------------------- loading


def load_config(path: str | Path) -> Device | Target | Preset:
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "kind" not in data:
        raise ConfigError(f"{path}: top-level 'kind' is required")
    data = dict(data)
    kind = data.pop("kind")
    where = str(path)
    if kind == "device":
        return _build(Device, data, where)
    if kind == "preset":
        return _build(Preset, data, where)
    if kind == "target":
        ttype = data.pop("type", None)
        if ttype not in _TARGET_TYPES:
            raise ConfigError(f"{where}: target 'type' must be one of {sorted(_TARGET_TYPES)}")
        return _build(_TARGET_TYPES[ttype], data, where)
    raise ConfigError(f"{where}: unknown kind '{kind}'")


def describe(cfg: Device | Target | Preset) -> str:
    """Human-readable one-block summary (used by ``metrogram check-config``)."""
    lines = [f"{type(cfg).__name__}: {cfg.name}"]
    if isinstance(cfg, AprilGridTarget):
        w, h = cfg.extent_m
        lines += [
            f"  {cfg.cols}x{cfg.rows} tags, family {cfg.tag_family}",
            f"  tag {cfg.tag_size_m * 1e3:.2f} mm, gap {cfg.gap_m * 1e3:.2f} mm, "
            f"pitch {cfg.pitch_m * 1e3:.2f} mm",
            f"  extent {w * 1e3:.1f} x {h * 1e3:.1f} mm",
            f"  id origin {cfg.id_origin} (verified: {cfg.id_origin_verified}), "
            f"certified: {cfg.dimensions_certified}",
        ]
    elif isinstance(cfg, CharucoTarget):
        w, h = cfg.extent_m
        lines += [
            f"  {cfg.squares_x}x{cfg.squares_y} squares, {cfg.dictionary}",
            f"  square {cfg.square_size_m * 1e3:.2f} mm, marker {cfg.marker_size_m * 1e3:.2f} mm",
            f"  extent {w * 1e3:.1f} x {h * 1e3:.1f} mm, certified: {cfg.dimensions_certified}",
        ]
    elif isinstance(cfg, Device):
        lines += [
            f"  {cfg.camera_make} {cfg.camera_model} + {cfg.lens.model}",
            f"  {cfg.sensor.width_px}x{cfg.sensor.height_px} px, "
            f"{cfg.sensor.pixel_pitch_um} um, f_nominal {cfg.focal_length_px:.0f} px",
            f"  COLMAP model {cfg.colmap_model}",
            "  GSD: "
            + ", ".join(
                f"{d * 100:.0f} cm -> {cfg.gsd_m(d) * 1e6:.0f} um" for d in (0.15, 0.25, 0.4)
            ),
        ]
    elif isinstance(cfg, Preset):
        lines += [
            f"  max image size: {cfg.max_image_size_px or 'full'}",
            f"  sfm: {cfg.sfm}",
            f"  dense: {cfg.dense}",
            f"  mesh: {cfg.mesh}",
        ]
    if getattr(cfg, "notes", ""):
        lines.append(f"  notes: {cfg.notes.strip()}")
    return "\n".join(lines)
