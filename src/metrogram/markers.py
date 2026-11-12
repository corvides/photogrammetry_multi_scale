"""Metric scale and reference frame from a calibration target seen in the SfM images.

1. Detect the target in every registered image (same images COLMAP used).
2. Triangulate each target point from all its observations with the SfM poses
   (midpoint init + Gauss-Newton on pixel reprojection, outlier observations
   dropped).
3. Split target points into control (alignment) and check points (whole tags /
   cells held out, spread over the board).
4. Similarity (Umeyama, robust) SfM -> board frame from control points.
5. Errors: control residuals, check-point errors (3D, in-plane, out-of-plane),
   independent scale from check points, distance errors between check points.
6. Write the metric model (COLMAP), PLYs (points, cameras, board, markers),
   a report and a plot.

Pixel conventions: detections are OpenCV (pixel centers at integers); COLMAP
cameras expect +0.5.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from metrogram.geometry import Similarity, umeyama
from metrogram.targets import BoardDetector


@dataclass
class MarkerSettings:
    check_every: int = 4  # hold out every n-th tag/cell as check points (0: none)
    max_reproj_px: float = 2.0  # drop observations above this after triangulation
    min_views: int = 3
    min_angle_deg: float = 2.0  # min max-ray-angle for a usable triangulation
    reject_factor: float = 5.0  # alignment outliers: > factor * median residual (gross only)
    plane_px: float = 3.0  # per-image board-homography consistency threshold (pixels)


@dataclass
class MarkerPoint:
    point_id: int
    group: int  # tag (AprilGrid) or corner (ChArUco): unit of hold-out
    board: np.ndarray  # true board coordinates (m)
    sfm: np.ndarray  # triangulated, SfM frame
    n_views: int
    reproj_rms_px: float
    max_angle_deg: float
    role: str = "control"  # control | check | rejected
    error_m: np.ndarray | None = None  # aligned - board


@dataclass
class MarkerResult:
    sim: Similarity
    points: list[MarkerPoint]
    n_images: int
    n_images_with_target: int
    scale_check_rel: float | None
    distance_errors_rel: dict
    stats: dict
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- triangulation


def _poses(rec):
    out = {}
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        T = im.cam_from_world().matrix()
        out[iid] = (T[:, :3], T[:, 3], im.camera_id, im.projection_center())
    return out


def _triangulate(obs, poses, cameras, s: MarkerSettings):
    """obs: list of (image_id, uv_colmap). Returns (X, kept_obs, rms_px, max_angle) or None."""
    for _ in range(3):
        if len(obs) < s.min_views:
            return None
        # rays
        A, b, dirs = np.zeros((3, 3)), np.zeros(3), []
        for iid, uv in obs:
            R, _, cid, c = poses[iid]
            xn = cameras[cid].cam_from_img(uv[None])[0]
            d = R.T @ np.array([xn[0], xn[1], 1.0])
            d /= np.linalg.norm(d)
            dirs.append(d)
            P = np.eye(3) - np.outer(d, d)
            A += P
            b += P @ c
        X0 = np.linalg.solve(A, b)

        def resid(X, obs=obs):
            r = []
            for iid, uv in obs:
                R, t, cid, _ = poses[iid]
                xc = R @ X + t
                r.append(cameras[cid].img_from_cam(xc[None])[0] - uv)
            return np.concatenate(r)

        X = least_squares(resid, X0, method="lm").x
        e = np.linalg.norm(resid(X).reshape(-1, 2), axis=1)
        keep = e <= s.max_reproj_px
        if keep.all():
            break
        obs = [o for o, k in zip(obs, keep, strict=True) if k]
    else:
        return None
    if len(obs) < s.min_views:
        return None
    D = np.array(dirs)
    cos = np.clip(D @ D.T, -1, 1)
    max_angle = float(np.degrees(np.arccos(cos.min())))
    return X, obs, float(np.sqrt((e**2).mean())), max_angle


def _plane_inliers(det, cam, detector: BoardDetector, thresh_px: float) -> np.ndarray:
    """Mask of observations consistent with one board->image homography.

    Works in undistorted normalized coordinates (lens distortion removed with the
    SfM camera). A tag/cell with any outlier corner is dropped entirely: partial
    occlusion (e.g. by the object on the board) or a false decode.
    """
    if len(det) < 8:
        return np.ones(len(det), bool)
    xn = cam.cam_from_img(det.img.astype(np.float64) + 0.5)
    f = float(np.mean([cam.focal_length_x, cam.focal_length_y]))
    H, _ = cv2.findHomography(
        det.obj[:, :2].astype(np.float64), xn, cv2.RANSAC, thresh_px / f, maxIters=2000
    )
    if H is None:
        return np.zeros(len(det), bool)
    proj = cv2.perspectiveTransform(det.obj[None, :, :2].astype(np.float64), H)[0]
    ok = np.linalg.norm(proj - xn, axis=1) * f <= thresh_px
    groups = np.array([detector.group_of(p) for p in det.point_ids])
    bad_groups = set(groups[~ok].tolist())
    return np.array([g not in bad_groups for g in groups])


def detect_all(rec, image_dir: Path, detector: BoardDetector, s: MarkerSettings, log=sys.stderr):
    """{point_id: [(image_id, uv_colmap)]} over registered images, plus counts."""
    obs: dict[int, list] = {}
    n_with, n_dropped = 0, 0
    ids = sorted(rec.reg_image_ids())
    for k, iid in enumerate(ids):
        im = rec.images[iid]
        gray = cv2.imread(str(image_dir / im.name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OSError(f"cannot read {image_dir / im.name}")
        det = detector.detect(gray)
        if log and (k + 1) % 50 == 0:
            print(f"  detected in {k + 1}/{len(ids)} images", file=log)
        if det is None:
            continue
        keep = _plane_inliers(det, rec.cameras[im.camera_id], detector, s.plane_px)
        n_dropped += int((~keep).sum())
        if not keep.any():
            continue
        n_with += 1
        for pid, uv in zip(det.point_ids[keep], det.img[keep], strict=True):
            obs.setdefault(int(pid), []).append((iid, uv.astype(np.float64) + 0.5))
    return obs, n_with, len(ids), n_dropped


# ------------------------------------------------------------------------------ alignment


def _stats(errs: np.ndarray) -> dict:
    if len(errs) == 0:
        return {"n": 0}
    n3 = np.linalg.norm(errs, axis=1)
    return {
        "n": int(len(errs)),
        "rms_3d_mm": float(np.sqrt((n3**2).mean()) * 1e3),
        "max_3d_mm": float(n3.max() * 1e3),
        "p95_3d_mm": float(np.percentile(n3, 95) * 1e3),
        "rms_inplane_mm": float(np.sqrt((errs[:, :2] ** 2).sum(1).mean()) * 1e3),
        "rms_normal_mm": float(np.sqrt((errs[:, 2] ** 2).mean()) * 1e3),
        "mean_xyz_mm": [float(v * 1e3) for v in errs.mean(0)],
    }


def align(points: list[MarkerPoint], s: MarkerSettings) -> Similarity:
    ctrl = [p for p in points if p.role == "control"]
    for _ in range(5):
        if len(ctrl) < 3:
            raise ValueError("fewer than 3 control points: cannot align")
        sim = umeyama(np.array([p.sfm for p in ctrl]), np.array([p.board for p in ctrl]))
        r = np.linalg.norm(
            sim.apply(np.array([p.sfm for p in ctrl])) - [p.board for p in ctrl], axis=1
        )
        thresh = s.reject_factor * max(float(np.median(r)), 1e-9)
        bad = r > thresh
        if not bad.any():
            break
        for p, b in zip(ctrl, bad, strict=True):
            if b:
                p.role = "rejected"
        ctrl = [p for p in ctrl if p.role == "control"]
    return sim


def run(rec, image_dir: Path, detector: BoardDetector, s: MarkerSettings, log=sys.stderr):
    poses = _poses(rec)
    cameras = {cid: rec.cameras[cid] for cid in rec.cameras}
    obs, n_with, n_img, n_dropped = detect_all(rec, image_dir, detector, s, log)
    if log:
        print(
            f"  target seen in {n_with}/{n_img} images, {len(obs)} distinct points, "
            f"{n_dropped} observations dropped (not on the board plane)",
            file=log,
        )
    board_pts = detector.all_points()
    points: list[MarkerPoint] = []
    for pid, o in sorted(obs.items()):
        tri = _triangulate(o, poses, cameras, s)
        if tri is None:
            continue
        X, kept, rms, ang = tri
        if ang < s.min_angle_deg:
            continue
        points.append(
            MarkerPoint(pid, detector.group_of(pid), board_pts[pid], X, len(kept), rms, ang)
        )
    if len(points) < 4:
        raise ValueError(f"only {len(points)} target points triangulated: cannot align")
    groups = sorted({p.group for p in points})
    check_groups = set(groups[s.check_every // 2 :: s.check_every]) if s.check_every else set()
    for p in points:
        if p.group in check_groups:
            p.role = "check"

    sim = align(points, s)
    for p in points:
        p.error_m = sim.apply(p.sfm[None])[0] - p.board

    warnings = []
    by = {
        r: np.array([p.error_m for p in points if p.role == r]).reshape(-1, 3)
        for r in ("control", "check", "rejected")
    }
    stats = {r: _stats(e) for r, e in by.items() if r != "rejected"}
    stats["n_rejected"] = int(len(by["rejected"]))
    stats["n_obs_dropped_off_plane"] = n_dropped

    chk = [p for p in points if p.role == "check"]
    scale_check_rel, dist = None, {}
    if len(chk) >= 3:
        sc = umeyama(np.array([p.sfm for p in chk]), np.array([p.board for p in chk]))
        scale_check_rel = float(sc.scale / sim.scale - 1.0)
        est = sim.apply(np.array([p.sfm for p in chk]))
        tru = np.array([p.board for p in chk])
        i, j = np.triu_indices(len(chk), 1)
        dt = np.linalg.norm(tru[i] - tru[j], axis=1)
        de = np.linalg.norm(est[i] - est[j], axis=1)
        m = dt > 0.05 * dt.max()
        rel = (de[m] - dt[m]) / dt[m]
        k = int(np.argmax(dt))
        dist = {
            "n_pairs": int(m.sum()),
            "median_abs_rel": float(np.median(np.abs(rel))),
            "max_abs_rel": float(np.abs(rel).max()),
            "longest_true_mm": float(dt[k] * 1e3),
            "longest_error_mm": float((de[k] - dt[k]) * 1e3),
        }
        if abs(scale_check_rel) > 0.002:
            warnings.append(f"check-point scale differs by {scale_check_rel:+.2%} from control")
    else:
        warnings.append("fewer than 3 check points: no independent error estimate")
    if n_with < 0.3 * n_img:
        warnings.append(f"target visible in only {n_with}/{n_img} images")
    if stats["n_rejected"]:
        warnings.append(f"{stats['n_rejected']} control points rejected as outliers")
    return MarkerResult(sim, points, n_img, n_with, scale_check_rel, dist, stats, warnings)


# --------------------------------------------------------------------------------- output


def _report(res: MarkerResult, s: MarkerSettings, meta: dict) -> dict:
    return {
        **meta,
        "board_extent_m": meta.get("board_extent_m"),
        "settings": vars(s),
        "units": "metres (board frame); errors in mm",
        "board_from_sfm": {
            "scale_m_per_sfm_unit": res.sim.scale,
            "matrix_4x4": res.sim.matrix().tolist(),
        },
        "image_counts": {"registered": res.n_images, "with_target": res.n_images_with_target},
        "errors": res.stats,
        "scale_check_rel": res.scale_check_rel,
        "check_distances": res.distance_errors_rel,
        "warnings": res.warnings,
        "points": [
            {
                "id": p.point_id,
                "group": p.group,
                "role": p.role,
                "n_views": p.n_views,
                "reproj_rms_px": round(p.reproj_rms_px, 3),
                "max_angle_deg": round(p.max_angle_deg, 1),
                "error_mm": [round(float(v) * 1e3, 4) for v in p.error_m],
            }
            for p in res.points
        ],
    }


ROLE_RGB = {"control": (42, 120, 214), "check": (235, 104, 52), "rejected": (154, 152, 147)}


def write(res: MarkerResult, rec, detector: BoardDetector, s, out: Path, meta: dict) -> None:
    import pycolmap

    from metrogram import ply

    out.mkdir(parents=True, exist_ok=True)
    allb = detector.all_points()
    meta = {**meta, "board_extent_m": [allb[:, :2].min(0).tolist(), allb[:, :2].max(0).tolist()]}
    (out / "markers_report.json").write_text(json.dumps(_report(res, s, meta), indent=1))

    # metric COLMAP model in the board frame
    rec = pycolmap.Reconstruction(rec)  # copy
    rec.transform(pycolmap.Sim3d(res.sim.matrix()[:3]))
    model = out / "sparse_metric"
    model.mkdir(exist_ok=True)
    rec.write(str(model))

    note = "metrogram markers: board frame, metres"
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts])
    rgb = np.array([p.color for p in pts], np.uint8)
    ply.write_points(out / "points_metric.ply", xyz, rgb, comment=note)

    lo, hi = allb[:, :2].min(0), allb[:, :2].max(0)
    extent = float(np.max(hi - lo))
    w = extent / 300

    m = ply.MeshBuilder()
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        T = im.cam_from_world().matrix()
        cam = rec.cameras[im.camera_id]
        m.add_frustum(
            im.projection_center(),
            T[:, :3].T,
            0.08 * extent,
            cam.width / cam.height,
            (80, 80, 80),
            w,
        )
    m.write(out / "cameras.ply", comment=note)

    b = ply.MeshBuilder()
    corners = [
        np.array([x, y, 0.0])
        for x, y in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))
    ]
    for i in range(4):
        b.add_box_segment(corners[i], corners[(i + 1) % 4], w, (120, 120, 120))
    L = 0.4 * extent
    for axis, rgb_ in zip(np.eye(3), ((220, 40, 40), (40, 170, 40), (40, 80, 220)), strict=True):
        b.add_box_segment(np.zeros(3), L * axis, 2.5 * w, rgb_)
    b.write(out / "board_frame.ply", comment=note + "; x red, y green, z blue")

    k = ply.MeshBuilder()
    cube = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    faces = [
        [0, 1, 3],
        [0, 3, 2],
        [4, 6, 7],
        [4, 7, 5],
        [0, 4, 5],
        [0, 5, 1],
        [2, 3, 7],
        [2, 7, 6],
        [0, 2, 6],
        [0, 6, 4],
        [1, 5, 7],
        [1, 7, 3],
    ]
    for p in res.points:
        k.add(res.sim.apply(p.sfm[None])[0] + cube * 1.5 * w, faces, ROLE_RGB[p.role])
    k.write(out / "markers.ply", comment=note + "; triangulated target points by role")

    from metrogram import plots

    plots.markers(res, out / "markers.png")


def summary(res: MarkerResult) -> str:
    st = res.stats
    lines = [
        f"target in {res.n_images_with_target}/{res.n_images} images; "
        f"{st['control']['n']} control, {st.get('check', {}).get('n', 0)} check, "
        f"{st['n_rejected']} rejected points",
        f"  scale {res.sim.scale:.6g} m per SfM unit",
    ]
    for r in ("control", "check"):
        x = st.get(r, {})
        if x.get("n"):
            lines.append(
                f"  {r:7s}: RMS 3D {x['rms_3d_mm']:.3f} mm (in-plane {x['rms_inplane_mm']:.3f}, "
                f"normal {x['rms_normal_mm']:.3f}), max {x['max_3d_mm']:.3f} mm"
            )
    if res.scale_check_rel is not None:
        d = res.distance_errors_rel
        lines.append(
            f"  scale from check points: {res.scale_check_rel * 1e6:+.0f} ppm; "
            f"check distances: median |rel| {d['median_abs_rel'] * 1e6:.0f} ppm, "
            f"longest {d['longest_true_mm']:.1f} mm off by {d['longest_error_mm']:+.3f} mm"
        )
    lines += [f"  WARNING: {w}" for w in res.warnings]
    return "\n".join(lines)
