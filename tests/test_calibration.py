"""Synthetic end-to-end test: render an AprilGrid through a known camera, recover K."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from metrogram import calibration, calibration_io
from metrogram.config import load_config
from metrogram.sources import Source
from metrogram.targets import AprilGridDetector

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SIZE = (1280, 960)
K_TRUE = np.array([[1000.0, 0, 642.0], [0, 1000.0, 478.0], [0, 0, 1]])
PPM = 5000.0  # board rendering resolution, px per metre
MARGIN = 0.02  # white margin around the tags, metres


@pytest.fixture(scope="module")
def target():
    return load_config(CONFIGS / "targets/aprilgrid_6x6_acrylic.yaml")


def render_board(t) -> np.ndarray:
    """Kalibr-like board: tags rotated by t.tag_rotation_deg, y up, tag 0 bottom-left."""
    w, h = t.extent_m
    W, H = int((w + 2 * MARGIN) * PPM), int((h + 2 * MARGIN) * PPM)
    canvas = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side = int(round(t.tag_size_m * PPM))
    for tag in range(t.rows * t.cols):
        r, c = divmod(tag, t.cols)
        img = cv2.aruco.generateImageMarker(d, tag, side, borderBits=1)
        img = np.rot90(img, k=t.tag_rotation_deg // 90)  # CCW, as seen with y up
        x0 = MARGIN + c * t.pitch_m
        y_top = MARGIN + r * t.pitch_m + t.tag_size_m  # board y of the tag's top edge
        u, v = int(round(x0 * PPM)), int(round((h + 2 * MARGIN - y_top) * PPM))
        canvas[v : v + side, u : u + side] = img
    return canvas


def canvas_to_board(t) -> np.ndarray:
    """Homography canvas pixel -> board metres (x right, y up, origin at tag 0)."""
    h = t.extent_m[1]
    return np.array([[1 / PPM, 0, -MARGIN], [0, -1 / PPM, h + MARGIN], [0, 0, 1]], dtype=np.float64)


def synth_views(t, n=16, seed=1):
    rng = np.random.default_rng(seed)
    canvas = render_board(t)
    A = canvas_to_board(t)
    w, h = t.extent_m
    center = np.array([w / 2, h / 2, 0])
    imgs = []
    for _ in range(n):
        rx, ry, rz = np.deg2rad(rng.uniform([-35, -35, -20], [35, 35, 20]))
        R, _ = cv2.Rodrigues(np.array([np.pi + rx, ry, rz]))  # board faces the camera
        dist = rng.uniform(0.45, 0.8)
        shift = rng.uniform([-0.12, -0.08], [0.12, 0.08])
        t_vec = np.array([shift[0], shift[1], dist]) - R @ center
        Hb = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], t_vec])  # board -> image
        # Supersample 4x then area-downsample: a plain perspective warp from a much
        # finer canvas aliases the tag edges and biases the detected corners.
        ss = 4
        S = np.diag([ss, ss, 1.0])
        # pixel-center alignment: big pixel (ss*u + (ss-1)/2) -> small pixel u
        T = np.array([[1, 0, (ss - 1) / 2], [0, 1, (ss - 1) / 2], [0, 0, 1]])
        big = cv2.warpPerspective(
            canvas,
            T @ S @ Hb @ A,
            (SIZE[0] * ss, SIZE[1] * ss),
            flags=cv2.INTER_LINEAR,
            borderValue=128,
        )
        img = cv2.resize(big, SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        img = np.clip(img + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        imgs.append(img)
    return imgs


def test_tag_corner_order(target):
    det = AprilGridDetector(target)
    s = target.tag_size_m
    # 180° rotation: OpenCV corners 0..3 = BR, BL, TL, TR of the tag (verified on the real board)
    np.testing.assert_allclose(
        det.tag_corners(0)[:, :2], [[s, 0], [0, 0], [0, s], [s, s]], atol=1e-12
    )
    p = target.pitch_m
    np.testing.assert_allclose(det.tag_corners(7)[1, :2], [p, p])  # row 1, col 1, BL corner


def test_synthetic_calibration(target, tmp_path):
    for i, img in enumerate(synth_views(target)):
        cv2.imwrite(str(tmp_path / f"view_{i:02d}.png"), img)
    src = Source(tmp_path)
    s = calibration.CalibrationSettings(model="OPENCV", min_points=16, max_views=60)
    res = calibration.run(src, AprilGridDetector(target), s, log=None)

    assert res.image_size == SIZE
    assert res.K[0, 0] == pytest.approx(1000.0, rel=5e-3)
    assert res.K[1, 1] == pytest.approx(1000.0, rel=5e-3)
    assert res.K[0, 2] == pytest.approx(642.0, abs=3.0)
    assert res.K[1, 2] == pytest.approx(478.0, abs=3.0)
    assert np.abs(res.dist[:4]).max() < 0.02
    assert res.train_rms_px < 0.3  # ~0.2 px: residual rendering quantization
    assert res.holdout_rms_px is not None and res.holdout_rms_px < 0.3

    params = res.colmap_params()
    assert params[2] == pytest.approx(res.K[0, 2] + 0.5)  # COLMAP pixel-center shift

    out = tmp_path / "out"
    calibration_io.write(res, s, out, {"source": "synthetic"})
    for f in ("calibration.yaml", "camchain_kalibr.yaml", "report.json", "coverage.png"):
        assert (out / f).exists()
