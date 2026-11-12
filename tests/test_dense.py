"""Synthetic dense test: board + 80 mm cube; the dense cloud must be metric, in the board frame."""

import cv2
import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from synthetic_scene import board_scene, render  # noqa: E402
from test_markers import SIZE, TARGET, K, cameras  # noqa: E402

from metrogram import dense, markers, ply, sfm  # noqa: E402
from metrogram.targets import AprilGridDetector  # noqa: E402

CUBE = 0.08


@pytest.mark.slow
def test_synthetic_dense(tmp_path):
    try:
        dense.find_openmvs()
    except FileNotFoundError:
        pytest.skip("OpenMVS not installed (scripts/build_openmvs.sh)")
    scene, cams = board_scene(TARGET, cube_size=CUBE), cameras()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i, (_, R, t) in enumerate(cams):
        cv2.imwrite(str(img_dir / f"cam_{i:02d}.png"), render(scene, K, R, t, SIZE))

    cam = sfm.CameraSpec("PINHOLE", *SIZE, [1100.0, 1100.0, 640.5, 480.5], "truth")
    r = sfm.run(img_dir, tmp_path / "sfm", cam, sfm.SfmSettings(), log=None)
    rec = pycolmap.Reconstruction(r.out_dir)
    det = AprilGridDetector(TARGET)
    ms = markers.MarkerSettings()
    mres = markers.run(rec, img_dir, det, ms, log=None)
    markers.write(mres, rec, det, ms, tmp_path / "markers", {})

    allb = det.all_points()
    extent = [allb[:, :2].min(0).tolist(), allb[:, :2].max(0).tolist()]
    s = dense.DenseSettings(resolution_level=0, texture=False, target_faces=100000)
    res = dense.run(
        tmp_path / "markers" / "sparse_metric", img_dir, tmp_path / "dense", s, extent, None
    )

    bp = res.board_plane
    assert bp["n_points"] > 1000
    # observed +0.30 mm (~0.5 px): a small upward bias of the board in the dense cloud,
    # not of the cube top (+0.09 mm). Open item; bound kept loose to avoid flakiness.
    assert abs(bp["mean_z_mm"]) < 0.5
    assert bp["tilt_deg"] < 0.1

    # Top face of the cube: z = 80 mm, footprint centred on the board.
    v = ply.read_vertices(tmp_path / "dense" / "dense_points.ply")
    P = np.column_stack([v["x"], v["y"], v["z"]])
    c = np.array([(extent[0][0] + extent[1][0]) / 2, (extent[0][1] + extent[1][1]) / 2])
    top = P[(np.abs(P[:, :2] - c).max(1) < 0.4 * CUBE) & (P[:, 2] > CUBE / 2)]
    assert len(top) > 500
    assert np.median(top[:, 2]) == pytest.approx(CUBE, abs=0.5e-3)
