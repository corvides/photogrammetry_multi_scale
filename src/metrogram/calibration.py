"""Intrinsic camera calibration from a planar target.

Pipeline: scan frames -> detect target -> quality filter (point count,
sharpness) -> pose-diverse selection -> train/holdout split -> OpenCV
calibration with iterative per-view outlier rejection -> holdout validation.

Conventions
-----------
- Internally OpenCV conventions (pixel centers at integer coordinates).
- COLMAP export shifts the principal point by +0.5 px (COLMAP puts the center
  of the top-left pixel at (0.5, 0.5)).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from metrogram.sources import Source
from metrogram.targets import BoardDetector

# Model name (COLMAP naming) -> OpenCV flags, and how to pack COLMAP params.
MODELS: dict[str, int] = {
    "PINHOLE": cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "SIMPLE_RADIAL": cv2.CALIB_FIX_ASPECT_RATIO
    | cv2.CALIB_FIX_K2
    | cv2.CALIB_FIX_K3
    | cv2.CALIB_ZERO_TANGENT_DIST,
    "RADIAL": cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    "OPENCV": cv2.CALIB_FIX_K3,
    "FULL_OPENCV": cv2.CALIB_RATIONAL_MODEL,
}
_STD_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]
_MODEL_DIST = {  # which OpenCV distortion coefficients each model actually uses
    "PINHOLE": [],
    "SIMPLE_RADIAL": ["k1"],
    "RADIAL": ["k1", "k2"],
    "OPENCV": ["k1", "k2", "p1", "p2"],
    "FULL_OPENCV": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
}


@dataclass
class View:
    key: str
    index: int
    obj: np.ndarray
    img: np.ndarray
    point_ids: np.ndarray
    sharpness: float
    features: np.ndarray  # pose-diversity descriptor
    edge_px: np.ndarray | None = None  # per-point edge width (blur), if the detector gives it
    status: str = "detected"  # detected | low_quality | unselected | train | holdout | rejected
    reason: str = ""
    rms_px: float | None = None
    n_blurred: int = 0  # points dropped by the blur filter

    @property
    def n_points(self) -> int:
        return len(self.point_ids)


@dataclass
class CalibrationSettings:
    model: str = "OPENCV"
    stride: int = 1
    min_points: int = 24
    sharpness_rel: float = 0.5  # keep views with sharpness >= rel * median
    max_views: int = 60
    holdout_frac: float = 0.2
    reject_factor: float = 3.0  # reject views with rms > factor * median rms
    # Blur filter (per tag): drop points whose edge width exceeds
    # max(blur_floor_px, blur_rel x median edge width of the whole scan). 0 = off.
    blur_rel: float = 2.0
    blur_floor_px: float = 3.0
    corner_refine: str = "full"  # AprilGrid corner refinement (recorded; used by the detector)
    max_reject_rounds: int = 5
    seed: int = 0


@dataclass
class CalibrationResult:
    model: str
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray
    dist: np.ndarray  # full OpenCV vector (5 or 8 coefficients)
    std: dict[str, float]
    train_rms_px: float
    holdout_rms_px: float | None
    views: list[View]
    n_scanned: int
    coverage: dict
    # robust statistics per status ("train", "holdout"); see error_stats()
    stats: dict = field(default_factory=dict)
    blur: dict = field(default_factory=dict)
    residuals: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # per status: (N, 4) x, y, dx, dy
    warnings: list[str] = field(default_factory=list)

    def colmap_params(self) -> list[float]:
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2] + 0.5, self.K[1, 2] + 0.5
        d = dict(zip(_STD_NAMES[4:], np.pad(self.dist.ravel(), (0, 8))[:8], strict=True))
        if self.model in ("SIMPLE_RADIAL", "RADIAL"):
            head = [(fx + fy) / 2, cx, cy]
        else:
            head = [fx, fy, cx, cy]
        return [float(v) for v in head + [d[n] for n in _MODEL_DIST[self.model]]]

    def dist_used(self) -> dict[str, float]:
        d = dict(zip(_STD_NAMES[4:], np.pad(self.dist.ravel(), (0, 8))[:8], strict=True))
        return {n: float(d[n]) for n in _MODEL_DIST[self.model]}


# ----------------------------------------------------------------------------- scan


def _sharpness(gray: np.ndarray, pts: np.ndarray) -> float:
    """Variance of the Laplacian inside the detected-points bounding box."""
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    h, w = gray.shape
    crop = gray[max(y0, 0) : min(y1, h), max(x0, 0) : min(x1, w)]
    if crop.size < 100:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def _pose_features(obj: np.ndarray, img: np.ndarray, extent: np.ndarray, size) -> np.ndarray:
    """K-free descriptor of where/how the board appears in the image.

    Board outline projected with a plane homography: center (x, y), log apparent
    size, log foreshortening ratios (left/right and top/bottom edge lengths),
    which together capture position, distance and both tilt directions.
    """
    H, _ = cv2.findHomography(obj[:, :2], img, 0)
    lo, hi = extent
    outline = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    q = cv2.perspectiveTransform(outline[None].astype(np.float64), H)[0]
    w, h = size
    center = q.mean(0) / np.array([w, h])
    e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)  # edges 01, 12, 23, 30
    d1, d2 = q[2] - q[0], q[3] - q[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    return np.array(
        [
            center[0],
            center[1],
            np.log(np.sqrt(area) / np.hypot(w, h)),
            np.log(e[1] / e[3]),
            np.log(e[0] / e[2]),
        ]
    )


def scan(source: Source, detector: BoardDetector, s: CalibrationSettings, log=sys.stderr):
    all_pts = detector.all_points()
    extent = np.array([all_pts[:, :2].min(0), all_pts[:, :2].max(0)])
    views: list[View] = []
    size = None
    n = 0
    total = len(source)
    for fr in source.frames(stride=s.stride):
        n += 1
        h, w = fr.gray.shape
        if size is None:
            size = (w, h)
        elif size != (w, h):
            raise ValueError(f"{fr.key}: image size {(w, h)} differs from {size}")
        det = detector.detect(fr.gray)
        if log and n % 100 == 0:
            print(
                f"  scanned {fr.index + 1}/{total} frames, {len(views)} with detections", file=log
            )
        if det is None or len(det) < 4:
            continue
        views.append(
            View(
                fr.key,
                fr.index,
                det.obj,
                det.img,
                det.point_ids,
                _sharpness(fr.gray, det.img),
                _pose_features(det.obj, det.img, extent, size),
                det.edge_px,
            )
        )
    if size is None:
        raise ValueError("source yielded no frames")
    return views, size, n


# ------------------------------------------------------------------------ selection


def filter_blur(views: list[View], s: CalibrationSettings) -> dict:
    """Drop points of blurred tags (in place). Partial defocus of a tilted board is
    invisible to the per-view sharpness score but biases corners by pixels."""
    widths = [v.edge_px for v in views if v.edge_px is not None and len(v.edge_px)]
    if not s.blur_rel or not widths:
        return {"enabled": False}
    med = float(np.median(np.concatenate(widths)))
    thr = max(s.blur_floor_px, s.blur_rel * med)
    n_drop = 0
    for v in views:
        if v.edge_px is None:
            continue
        keep = v.edge_px <= thr
        v.n_blurred = int((~keep).sum())
        n_drop += v.n_blurred
        if v.n_blurred:
            v.obj, v.img, v.point_ids, v.edge_px = (
                v.obj[keep],
                v.img[keep],
                v.point_ids[keep],
                v.edge_px[keep],
            )
    n_all = sum(len(w) for w in widths)
    return {
        "enabled": True,
        "median_edge_px": med,
        "threshold_px": thr,
        "n_points_dropped": n_drop,
        "fraction_dropped": n_drop / n_all,
        "n_views_affected": sum(v.n_blurred > 0 for v in views),
    }


def select(views: list[View], s: CalibrationSettings) -> None:
    """Mark views as low_quality / unselected / train / holdout (in place)."""
    for v in views:
        if v.n_points < s.min_points:
            v.status, v.reason = "low_quality", f"{v.n_points} points < {s.min_points}"
    ok = [v for v in views if v.status == "detected"]
    if ok:
        med = float(np.median([v.sharpness for v in ok]))
        for v in ok:
            if v.sharpness < s.sharpness_rel * med:
                v.status = "low_quality"
                v.reason = f"sharpness {v.sharpness:.0f} < {s.sharpness_rel} x median {med:.0f}"
    ok = [v for v in views if v.status == "detected"]
    if not ok:
        return
    # Farthest-point sampling in standardized pose-feature space.
    F = np.array([v.features for v in ok])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    chosen = [int(np.argmax([v.n_points for v in ok]))]
    dmin = np.linalg.norm(F - F[chosen[0]], axis=1)
    while len(chosen) < min(s.max_views, len(ok)):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[nxt], axis=1))
    for i, v in enumerate(ok):
        v.status = "train" if i in set(chosen) else "unselected"
    n_hold = int(round(s.holdout_frac * len(chosen))) if len(chosen) >= 10 else 0
    rng = np.random.default_rng(s.seed)
    for i in rng.choice(chosen, size=n_hold, replace=False):
        ok[i].status = "holdout"


# ---------------------------------------------------------------------- calibration


def _residuals(v: View, K, dist, rvec, tvec) -> np.ndarray:
    proj, _ = cv2.projectPoints(v.obj.astype(np.float64), rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - v.img
    return np.column_stack([v.img, d])


def _coverage(points: np.ndarray, size, grid=(8, 6)) -> dict:
    w, h = size
    gx = np.clip((points[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
    gy = np.clip((points[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
    occ = np.zeros(grid[::-1], dtype=int)
    np.add.at(occ, (gy, gx), 1)
    corners = [occ[0, 0], occ[0, -1], occ[-1, 0], occ[-1, -1]]
    return {
        "grid": list(grid),
        "occupied_fraction": float((occ > 0).mean()),
        "corner_cells_points": [int(c) for c in corners],
        "counts": occ.tolist(),
    }


def error_stats(views: list[View], res: list[np.ndarray], core_px: float) -> dict:
    """Robust error statistics of a set of views (residual arrays aligned with views).

    RMS squares errors, so a few bad views dominate it. Also reported: the
    per-view RMS distribution (median, p90), the per-point error distribution
    (median, p95), and the "core" RMS over views whose RMS is <= core_px (the
    same outlier rule as in training, so core RMS is comparable to train RMS).
    """
    if not views:
        return {}
    vr = np.array([v.rms_px for v in views])
    e = np.concatenate([np.hypot(r[:, 2], r[:, 3]) for r in res])
    core = vr <= core_px
    ec = np.concatenate(
        [np.hypot(r[:, 2], r[:, 3]) for r, c in zip(res, core, strict=True) if c] or [np.zeros(0)]
    )
    return {
        "n_views": int(len(vr)),
        "n_points": int(len(e)),
        "rms_px": float(np.sqrt((e**2).mean())),
        "view_rms_median_px": float(np.median(vr)),
        "view_rms_p90_px": float(np.percentile(vr, 90)),
        "point_err_median_px": float(np.median(e)),
        "point_err_p95_px": float(np.percentile(e, 95)),
        "core_threshold_px": float(core_px),
        "n_outlier_views": int((~core).sum()),
        "outlier_views": [v.key for v, c in zip(views, core, strict=True) if not c],
        "core_rms_px": float(np.sqrt((ec**2).mean())) if len(ec) else None,
    }


def calibrate(views: list[View], size, s: CalibrationSettings, n_scanned: int, log=sys.stderr):
    if s.model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    flags = MODELS[s.model]
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10)
    warnings: list[str] = []

    for rnd in range(s.max_reject_rounds + 1):
        train = [v for v in views if v.status == "train"]
        if len(train) < 5:
            raise ValueError(f"only {len(train)} usable views; need at least 5")
        K0 = np.eye(3)
        rms, K, dist, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
            [v.obj.reshape(-1, 1, 3) for v in train],
            [v.img.reshape(-1, 1, 2) for v in train],
            size,
            K0,
            None,
            flags=flags,
            criteria=criteria,
        )
        per_view = per_view.ravel()
        for v, e in zip(train, per_view, strict=True):
            v.rms_px = float(e)
        thresh = s.reject_factor * float(np.median(per_view))
        bad = [v for v in train if v.rms_px > thresh]
        if log:
            print(
                f"  round {rnd}: {len(train)} views, RMS {rms:.3f} px, "
                f"{len(bad)} above {thresh:.3f} px",
                file=log,
            )
        if not bad or rnd == s.max_reject_rounds:
            break
        # Drop at most 10% of the views per round, worst first.
        for v in sorted(bad, key=lambda v: -v.rms_px)[: max(1, len(train) // 10)]:
            v.status, v.reason = "rejected", f"rms {v.rms_px:.3f} px > {thresh:.3f} px"

    per_train = [_residuals(v, K, dist, r, t) for v, r, t in zip(train, rvecs, tvecs, strict=True)]
    res_train = np.vstack(per_train)
    residuals = {"train": res_train}
    core_px = s.reject_factor * float(np.median([v.rms_px for v in train]))
    stats = {"train": error_stats(train, per_train, core_px)}
    train_rms = float(np.sqrt((res_train[:, 2:] ** 2).sum(1).mean()))

    holdout = [v for v in views if v.status == "holdout"]
    holdout_rms = None
    if holdout:
        res, done = [], []
        for v in holdout:
            ok, rvec, tvec = cv2.solvePnP(v.obj, v.img, K, dist, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                v.status, v.reason = "rejected", "solvePnP failed"
                continue
            rvec, tvec = cv2.solvePnPRefineLM(v.obj, v.img, K, dist, rvec, tvec)
            r = _residuals(v, K, dist, rvec, tvec)
            v.rms_px = float(np.sqrt((r[:, 2:] ** 2).sum(1).mean()))
            res.append(r)
            done.append(v)
        if res:
            residuals["holdout"] = np.vstack(res)
            stats["holdout"] = error_stats(done, res, core_px)
            holdout_rms = float(np.sqrt((residuals["holdout"][:, 2:] ** 2).sum(1).mean()))
    else:
        warnings.append("no holdout views (fewer than 10 selected): no independent error")

    std = {n: float(x) for n, x in zip(_STD_NAMES, std_int.ravel()[:12], strict=True)}
    used = {"fx", "fy", "cx", "cy", *_MODEL_DIST[s.model]}
    std = {k: v for k, v in std.items() if k in used}

    coverage = _coverage(res_train[:, :2], size)
    if coverage["occupied_fraction"] < 0.75:
        warnings.append(
            f"image coverage {coverage['occupied_fraction']:.0%} < 75%: "
            "distortion poorly constrained near the edges"
        )
    if min(coverage["corner_cells_points"]) == 0:
        warnings.append("at least one image corner cell has no points")
    hs = stats.get("holdout")
    if hs:
        # Generalization is judged on robust numbers: the core RMS (outlier views
        # excluded, same rule as training) and the median view.
        core = hs["core_rms_px"]
        if (core is not None and core > 1.5 * train_rms) or (
            hs["view_rms_median_px"] > 1.5 * stats["train"]["view_rms_median_px"]
        ):
            warnings.append(
                f"holdout core RMS {core:.3f} px / median view {hs['view_rms_median_px']:.3f} px "
                f">> train RMS {train_rms:.3f} px: overfitting or unstable intrinsics "
                "(focus moved, autofocus, stabilization?)"
            )
        if hs["n_outlier_views"]:
            warnings.append(
                f"{hs['n_outlier_views']}/{hs['n_views']} holdout views above "
                f"{core_px:.2f} px (bad views: blur, defocus, detection errors; "
                f"excluded from the core RMS): {', '.join(hs['outlier_views'])}"
            )
    n_rej = sum(v.status == "rejected" for v in views)
    if n_rej > 0.2 * (len(train) + n_rej):
        warnings.append(f"{n_rej} views rejected as outliers: inconsistent intrinsics?")

    return CalibrationResult(
        model=s.model,
        image_size=tuple(size),
        K=K,
        dist=dist.ravel(),
        std=std,
        train_rms_px=train_rms,
        holdout_rms_px=holdout_rms,
        views=views,
        n_scanned=n_scanned,
        coverage=coverage,
        stats=stats,
        residuals=residuals,
        warnings=warnings,
    )


def run(source: Source, detector: BoardDetector, s: CalibrationSettings, log=sys.stderr):
    if log:
        print(f"scanning {source.description} (stride {s.stride})", file=log)
    views, size, n = scan(source, detector, s, log)
    if log:
        print(f"  {len(views)}/{n} frames with detections, image size {size}", file=log)
    blur = filter_blur(views, s)
    if log and blur.get("enabled"):
        print(
            f"  blur filter: median edge width {blur['median_edge_px']:.2f} px, dropped "
            f"{blur['n_points_dropped']} points ({blur['fraction_dropped']:.1%}) above "
            f"{blur['threshold_px']:.2f} px in {blur['n_views_affected']} views",
            file=log,
        )
    select(views, s)
    res = calibrate(views, size, s, n, log)
    res.blur = blur
    return res
