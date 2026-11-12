"""Write calibration results: YAML (main), Kalibr camchain, JSON report, plots."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from metrogram import __version__, plots
from metrogram.calibration import CalibrationResult, CalibrationSettings
from metrogram.sources import Source


def _r(x, n=6):
    return float(round(float(x), n))


def calibration_dict(res: CalibrationResult, meta: dict) -> dict:
    w, h = res.image_size
    return {
        "kind": "calibration",
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "metrogram_version": __version__,
        "opencv_version": cv2.__version__,
        **meta,
        "image_size": {"width": w, "height": h},
        "model": res.model,
        "opencv": {
            "convention": "pixel centers at integer coordinates",
            "K": [[_r(v) for v in row] for row in res.K],
            "dist": [_r(v, 9) for v in res.dist],
        },
        "colmap": {
            "model": res.model,
            "params": [_r(v, 9) for v in res.colmap_params()],
            "convention": "principal point +0.5 px (COLMAP pixel-center convention)",
        },
        "intrinsics": {
            "fx": _r(res.K[0, 0]),
            "fy": _r(res.K[1, 1]),
            "cx": _r(res.K[0, 2]),
            "cy": _r(res.K[1, 2]),
            **{k: _r(v, 9) for k, v in res.dist_used().items()},
        },
        "std": {k: _r(v, 9) for k, v in res.std.items()},
        "errors": {
            "train_rms_px": _r(res.train_rms_px, 4),
            "holdout_rms_px": None if res.holdout_rms_px is None else _r(res.holdout_rms_px, 4),
            "n_train_views": sum(v.status == "train" for v in res.views),
            "n_holdout_views": sum(v.status == "holdout" for v in res.views),
            "n_rejected_views": sum(v.status == "rejected" for v in res.views),
            "n_train_points": int(len(res.residuals["train"])),
            "coverage_fraction": _r(res.coverage["occupied_fraction"], 3),
        },
        "warnings": res.warnings,
    }


def kalibr_camchain(res: CalibrationResult) -> dict | None:
    if res.model not in ("PINHOLE", "OPENCV"):
        return None  # Kalibr's radtan = OpenCV k1, k2, p1, p2
    d = res.dist_used()
    return {
        "cam0": {
            "camera_model": "pinhole",
            "intrinsics": [_r(res.K[0, 0]), _r(res.K[1, 1]), _r(res.K[0, 2]), _r(res.K[1, 2])],
            "distortion_model": "radtan",
            "distortion_coeffs": [_r(d.get(k, 0.0), 9) for k in ("k1", "k2", "p1", "p2")],
            "resolution": list(res.image_size),
        }
    }


def view_report(res: CalibrationResult, s: CalibrationSettings) -> dict:
    counts: dict[str, int] = {}
    for v in res.views:
        counts[v.status] = counts.get(v.status, 0) + 1
    return {
        "settings": vars(s),
        "n_scanned": res.n_scanned,
        "status_counts": counts,
        "coverage": res.coverage,
        "views": [
            {
                "key": v.key,
                "index": v.index,
                "status": v.status,
                "reason": v.reason,
                "n_points": v.n_points,
                "sharpness": round(v.sharpness, 1),
                "rms_px": None if v.rms_px is None else round(v.rms_px, 4),
            }
            for v in res.views
        ],
    }


def write(
    res: CalibrationResult,
    s: CalibrationSettings,
    out: Path,
    meta: dict,
    source: Source | None = None,
    save_frames: bool = False,
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def dump_yaml(name, data):
        p = out / name
        with p.open("w") as fh:
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=None, width=100)
        written.append(p)

    dump_yaml("calibration.yaml", calibration_dict(res, meta))
    kc = kalibr_camchain(res)
    if kc:
        dump_yaml("camchain_kalibr.yaml", kc)
    p = out / "report.json"
    p.write_text(json.dumps(view_report(res, s), indent=1))
    written.append(p)
    written += [out / f for f in plots.all_plots(res, out)]

    if save_frames and source is not None:
        fdir = out / "frames"
        fdir.mkdir(exist_ok=True)
        colors = {"train": (214, 120, 42), "holdout": (52, 104, 235), "rejected": (147, 152, 154)}
        for v in res.views:
            if v.status not in colors:
                continue
            img = source.read(v.index)
            for x, y in v.img:
                cv2.circle(img, (int(round(x)), int(round(y))), 4, colors[v.status], 1)
            cv2.imwrite(str(fdir / f"{v.status}_{Path(v.key).stem}.jpg"), img)
        written.append(fdir)
    return written


def summary(res: CalibrationResult) -> str:
    i = calibration_dict(res, {})
    e = i["errors"]
    lines = [
        f"model {res.model}, image {res.image_size[0]}x{res.image_size[1]}",
        "  "
        + ", ".join(
            f"{k} {v:.6g} ± {res.std.get(k, np.nan):.2g}" for k, v in i["intrinsics"].items()
        ),
        f"  train RMS {e['train_rms_px']:.3f} px ({e['n_train_views']} views, "
        f"{e['n_train_points']} pts), holdout RMS "
        + ("n/a" if e["holdout_rms_px"] is None else f"{e['holdout_rms_px']:.3f} px")
        + f" ({e['n_holdout_views']} views), rejected {e['n_rejected_views']}",
        f"  coverage {e['coverage_fraction']:.0%}",
    ]
    lines += [f"  WARNING: {w}" for w in res.warnings]
    return "\n".join(lines)
