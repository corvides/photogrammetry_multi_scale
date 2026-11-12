"""Synthetic SfM test: known cameras around a textured cube; recover poses up to similarity."""

import cv2
import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from synthetic_scene import make_scene, render, ring_cameras  # noqa: E402

from metrogram import sfm  # noqa: E402
from metrogram.geometry import umeyama  # noqa: E402

SIZE = (800, 600)
K = np.array([[700.0, 0, 400.0], [0, 700.0, 300.0], [0, 0, 1]])


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R, _ = cv2.Rodrigues(np.array([0.3, -0.2, 0.9]))
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    sim = umeyama(src, dst)
    assert sim.scale == pytest.approx(2.5)
    np.testing.assert_allclose(sim.R, R, atol=1e-10)
    np.testing.assert_allclose(sim.apply(src), dst, atol=1e-10)


def test_camera_spec_scaling():
    c = sfm.CameraSpec("OPENCV", 4000, 3000, [3000, 3001, 2000.5, 1500.5, 0.1, -0.2, 0, 0])
    h = c.scaled(2000, 1500)
    assert h.params[:4] == [1500, 1500.5, 1000.25, 750.25]
    assert h.params[4:] == c.params[4:]  # distortion is in normalized coordinates


@pytest.mark.slow
def test_synthetic_sfm(tmp_path):
    quads = make_scene()
    cams = ring_cameras()
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i, (_, R, t) in enumerate(cams):
        cv2.imwrite(str(img_dir / f"cam_{i:02d}.png"), render(quads, K, R, t, SIZE))

    # OpenCV -> COLMAP principal point: +0.5 px
    cam = sfm.CameraSpec("PINHOLE", *SIZE, [700.0, 700.0, 400.5, 300.5], "synthetic truth")
    res = sfm.run(img_dir, tmp_path / "out", cam, sfm.SfmSettings(), log=None)

    assert res.n_registered == len(cams)
    assert res.mean_reproj_error_px < 0.5

    rec = pycolmap.Reconstruction(res.out_dir)
    est, true = [], []
    for img in rec.images.values():
        est.append(img.projection_center())
        true.append(cams[int(img.name[4:6])][0])
    est, true = np.array(est), np.array(true)
    sim = umeyama(est, true)
    err = np.linalg.norm(sim.apply(est) - true, axis=1)
    # Ring radius 0.45 m: require camera centers within 1 mm (0.2 % of the radius).
    assert err.max() < 1e-3, f"camera center errors (m): max {err.max():.2e}"
