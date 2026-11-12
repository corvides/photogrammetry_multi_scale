"""Synthetic markers test: board + cube, known cameras; metric frame must match the truth."""

import cv2
import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from synthetic_scene import board_scene, look_at, render  # noqa: E402

from metrogram import markers, sfm  # noqa: E402
from metrogram.config import AprilGridTarget  # noqa: E402
from metrogram.targets import AprilGridDetector  # noqa: E402

SIZE = (1280, 960)
K = np.array([[1100.0, 0, 640.0], [0, 1100.0, 480.0], [0, 0, 1]])
# Round dimensions: whole pixels at the 4000 px/m board texture (no quantization).
TARGET = AprilGridTarget(name="synthetic", rows=6, cols=6, tag_size_m=0.04, tag_spacing_ratio=0.25)


def cameras(n_az=12, elevations=(40, 65), radius=0.6):
    w, h = TARGET.extent_m
    c0 = np.array([w / 2, h / 2, 0.03])
    out = []
    for el in np.deg2rad(elevations):
        for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            c = c0 + radius * np.array(
                [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
            )
            out.append((c, *look_at(c, c0)))
    return out


@pytest.mark.slow
def test_synthetic_markers(tmp_path):
    scene, cams = board_scene(TARGET), cameras()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i, (_, R, t) in enumerate(cams):
        cv2.imwrite(str(img_dir / f"cam_{i:02d}.png"), render(scene, K, R, t, SIZE))

    cam = sfm.CameraSpec("PINHOLE", *SIZE, [1100.0, 1100.0, 640.5, 480.5], "truth")
    r = sfm.run(img_dir, tmp_path / "sfm", cam, sfm.SfmSettings(), log=None)
    assert r.n_registered == len(cams)

    rec = pycolmap.Reconstruction(r.out_dir)
    det = AprilGridDetector(TARGET)
    s = markers.MarkerSettings()
    res = markers.run(rec, img_dir, det, s, log=None)

    assert res.stats["n_rejected"] == 0
    assert res.stats["check"]["n"] >= 16
    # floor: ~0.2 px tag-corner detection noise ~ 0.11 mm at 0.6 m, f = 1100 px
    assert res.stats["check"]["rms_3d_mm"] < 0.15
    assert abs(res.scale_check_rel) < 1e-3

    # World frame of the renderer == board frame: aligned cameras must match the truth.
    est = res.sim.apply(np.array([rec.images[i].projection_center() for i in rec.reg_image_ids()]))
    true = np.array([cams[int(rec.images[i].name[4:6])][0] for i in rec.reg_image_ids()])
    err = np.linalg.norm(est - true, axis=1)
    # Gross check only: cameras sit 0.6 m away, so ~0.3 mrad of alignment rotation plus
    # per-camera depth uncertainty is amplified by the lever arm (observed ~0.3 mm mean).
    assert err.max() < 2e-3, f"camera centers off by up to {err.max() * 1e3:.3f} mm"

    out = tmp_path / "markers"
    markers.write(res, rec, det, s, out, {"source": "synthetic"})
    for f in ("points_metric.ply", "cameras.ply", "board_frame.ply", "markers.ply", "markers.png"):
        assert (out / f).stat().st_size > 0
    m = pycolmap.Reconstruction(str(out / "sparse_metric"))
    assert m.num_reg_images() == len(cams)
