"""Dense reconstruction with OpenMVS, from the metric (board-frame) COLMAP model.

Steps: COLMAP undistortion -> InterfaceCOLMAP -> DensifyPointCloud ->
ReconstructMesh -> [RefineMesh] -> [TextureMesh]. Because the input model is
the one written by ``markers`` (board frame, metres), every output is metric
and in the board frame.

Also reports a first dense-level accuracy check: dense points on the board
must lie on z = 0 (plane-fit RMS, offset, tilt).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from metrogram import ply
from metrogram.config import Preset

TOOLS = ("InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh")


def find_openmvs() -> Path:
    """$OPENMVS_BIN, then ~/opt/openmvs/bin/OpenMVS (scripts/build_openmvs.sh), then PATH."""
    cands = []
    if os.environ.get("OPENMVS_BIN"):
        cands.append(Path(os.environ["OPENMVS_BIN"]))
    cands.append(Path.home() / "opt/openmvs/bin/OpenMVS")
    for c in cands:
        if (c / "DensifyPointCloud").is_file():
            return c
    w = shutil.which("DensifyPointCloud")
    if w:
        return Path(w).parent
    raise FileNotFoundError("OpenMVS not found: run scripts/build_openmvs.sh or set OPENMVS_BIN")


@dataclass
class DenseSettings:
    resolution_level: int = 1
    number_views: int = 5
    min_resolution_px: int = 640
    target_faces: int | None = None
    refine: bool = False
    texture: bool = True
    max_threads: int = 0  # 0 = all cores


def settings_from_preset(p: Preset) -> DenseSettings:
    return DenseSettings(
        resolution_level=p.dense.resolution_level,
        number_views=p.dense.number_views,
        min_resolution_px=p.dense.min_resolution_px,
        target_faces=p.mesh.decimate_target_faces,
        refine=p.mesh.refine,
        texture=p.mesh.texture,
    )


@dataclass
class DenseResult:
    out_dir: str
    n_images: int
    n_dense_points: int
    n_mesh_faces: int
    files: dict[str, str]
    timings_s: dict[str, float]
    board_plane: dict
    warnings: list[str] = field(default_factory=list)


def _run(tool: Path, args: list[str], cwd: Path, logs: Path, log) -> float:
    t0 = time.time()
    cmd = [str(tool), *args]
    if log:
        print(f"  {tool.name} ...", file=log, flush=True)
    with open(logs / f"{tool.name}.log", "w") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{tool.name} failed (exit {r.returncode}); see {logs / tool.name}.log")
    if log:
        print(f"    done in {dt:.0f} s", file=log)
    return dt


def board_plane_check(points_ply: Path, extent, inset: float = 0.05) -> dict:
    """Dense points over the board (xy inset, |z| < 1 % of board size) vs plane z = 0."""
    v = ply.read_vertices(points_ply)
    P = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    (x0, y0), (x1, y1) = extent
    size = max(x1 - x0, y1 - y0)
    dx, dy = inset * (x1 - x0), inset * (y1 - y0)
    m = (
        (P[:, 0] > x0 + dx)
        & (P[:, 0] < x1 - dx)
        & (P[:, 1] > y0 + dy)
        & (P[:, 1] < y1 - dy)
        & (np.abs(P[:, 2]) < 0.01 * size)
    )
    Q = P[m]
    if len(Q) < 100:
        return {"n_points": int(len(Q)), "note": "too few dense points on the board"}
    c = Q.mean(0)
    _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
    n = Vt[2] if Vt[2, 2] > 0 else -Vt[2]
    d = (Q - c) @ n
    return {
        "n_points": int(len(Q)),
        "z_gate_mm": 10 * size,
        "rms_to_fitted_plane_mm": float(np.sqrt((d**2).mean()) * 1e3),
        "p95_abs_to_fitted_plane_mm": float(np.percentile(np.abs(d), 95) * 1e3),
        "rms_z_mm": float(np.sqrt((Q[:, 2] ** 2).mean()) * 1e3),
        "mean_z_mm": float(Q[:, 2].mean() * 1e3),
        "tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n[2]))))),
    }


def run(
    model_dir: Path,
    image_dir: Path,
    out: Path,
    s: DenseSettings,
    board_extent=None,
    log=sys.stderr,
) -> DenseResult:
    import pycolmap

    bin_dir = find_openmvs()
    tools = {t: bin_dir / t for t in TOOLS}
    out = out.resolve()
    work, logs = out / "work", out / "logs"
    for d in (work, logs):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    timings: dict[str, float] = {}

    # 1. undistort (pinhole images + COLMAP model for OpenMVS)
    if log:
        print("  undistorting images (COLMAP) ...", file=log, flush=True)
    t0 = time.time()
    und = work / "undistorted"
    opts = pycolmap.UndistortCameraOptions()
    opts.max_image_size = -1  # keep SfM resolution; OpenMVS scales via resolution-level
    pycolmap.undistort_images(str(und), str(model_dir), str(image_dir), undistort_options=opts)
    timings["undistort"] = time.time() - t0
    n_images = len(list((und / "images").iterdir()))

    th = ["--max-threads", str(s.max_threads)]
    scene = work / "scene.mvs"
    timings["InterfaceCOLMAP"] = _run(
        tools["InterfaceCOLMAP"],
        ["-i", str(und), "-o", str(scene), "--image-folder", str(und / "images"), *th],
        work,
        logs,
        log,
    )
    dense = work / "scene_dense.mvs"
    timings["DensifyPointCloud"] = _run(
        tools["DensifyPointCloud"],
        [
            "-i", str(scene), "-o", str(dense),
            "--resolution-level", str(s.resolution_level),
            "--number-views", str(s.number_views),
            "--min-resolution", str(s.min_resolution_px),
            *th,
        ],
        work,
        logs,
        log,
    )  # fmt: skip
    dense_ply = work / "scene_dense.ply"
    mesh = work / "scene_mesh.ply"
    timings["ReconstructMesh"] = _run(
        tools["ReconstructMesh"],
        ["-i", str(dense), "-o", str(mesh), "--target-face-num", str(s.target_faces or 0), *th],
        work,
        logs,
        log,
    )
    final_mesh = mesh
    if s.refine:
        refined = work / "scene_mesh_refined.ply"
        timings["RefineMesh"] = _run(
            tools["RefineMesh"],
            ["-i", str(dense), "-m", str(mesh), "-o", str(refined), *th],
            work,
            logs,
            log,
        )
        final_mesh = refined

    for f in (dense_ply, final_mesh):
        if not f.exists():
            raise RuntimeError(f"expected output missing: {f}")
    # Clean dense cloud (xyz + rgb): OpenMVS's PLY carries per-point view lists
    # that many viewers do not need; the original stays in work/.
    v = ply.read_vertices(dense_ply)
    ply.write_points(
        out / "dense_points.ply",
        np.column_stack([v["x"], v["y"], v["z"]]),
        np.column_stack([v["red"], v["green"], v["blue"]]),
        comment="metrogram dense: board frame, metres",
    )
    shutil.copy2(final_mesh, out / "mesh.ply")
    files = {"dense_points.ply": str(out / "dense_points.ply"), "mesh.ply": str(out / "mesh.ply")}
    warnings = []
    if s.texture:
        tex_dir = out / "textured"
        if tex_dir.exists():
            shutil.rmtree(tex_dir)
        tex_dir.mkdir()
        # PLY export: v2.3.0's OBJ exporter segfaults after texturing.
        timings["TextureMesh"] = _run(
            tools["TextureMesh"],
            [
                "-i", str(dense), "-m", str(final_mesh),
                "-o", str(tex_dir / "mesh_textured.ply"), "--export-type", "ply", *th,
            ],
            work,
            logs,
            log,
        )  # fmt: skip
        files["textured"] = str(tex_dir / "mesh_textured.ply")
        try:  # OBJ copy for viewers that ignore PLY textures (e.g. Open3D)
            import trimesh

            obj_dir = tex_dir / "obj"
            obj_dir.mkdir()
            trimesh.load(tex_dir / "mesh_textured.ply", process=False).export(
                obj_dir / "mesh_textured.obj"
            )
            files["textured_obj"] = str(obj_dir / "mesh_textured.obj")
        except ImportError:
            warnings.append("trimesh not installed (uv sync --extra eval): no OBJ copy")

    n_pts = ply.read_header(out / "dense_points.ply")[0].get("vertex", 0)
    n_faces = ply.read_header(out / "mesh.ply")[0].get("face", 0)
    plane = {}
    if board_extent is not None:
        plane = board_plane_check(out / "dense_points.ply", board_extent)
        if plane.get("rms_to_fitted_plane_mm") is None:
            warnings.append("board plane check skipped: too few dense points on the board")
    res = DenseResult(str(out), n_images, n_pts, n_faces, files, timings, plane, warnings)
    (out / "dense_report.json").write_text(json.dumps(asdict(res), indent=1))
    return res


def summary(r: DenseResult) -> str:
    total = sum(r.timings_s.values())
    lines = [
        f"{r.n_images} images -> {r.n_dense_points} dense points, mesh {r.n_mesh_faces} faces "
        f"({total:.0f} s: " + ", ".join(f"{k} {v:.0f}" for k, v in r.timings_s.items()) + ")",
    ]
    bp = r.board_plane
    if bp.get("rms_to_fitted_plane_mm") is not None:
        lines.append(
            f"  board plane ({bp['n_points']} pts): RMS {bp['rms_to_fitted_plane_mm']:.3f} mm "
            f"to fitted plane (p95 {bp['p95_abs_to_fitted_plane_mm']:.3f}), "
            f"offset {bp['mean_z_mm']:+.3f} mm, tilt {bp['tilt_deg']:.3f} deg"
        )
    lines += [f"  {k}: {v}" for k, v in r.files.items()]
    lines += [f"  WARNING: {w}" for w in r.warnings]
    return "\n".join(lines)
