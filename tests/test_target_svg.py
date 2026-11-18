"""Printable ChArUco SVG: same geometry as OpenCV's board, detectable, mm units."""

import numpy as np

from metrogram.config import CharucoTarget
from metrogram.target_svg import charuco_svg, rasterize
from metrogram.targets import CharucoDetector

T = CharucoTarget(
    name="t", squares_x=7, squares_y=5, square_size_m=0.012, marker_size_m=0.009,
    dictionary="DICT_5X5_1000",
)  # fmt: skip


def test_svg_units_and_size():
    s = charuco_svg(T, page="a4")
    assert 'width="297mm" height="210mm" viewBox="0 0 297 210"' in s
    n_markers = 7 * 5 // 2  # the top-left square is black: 17 markers
    assert s.count('fill-rule="evenodd"') == n_markers  # one path per marker


def test_rasterized_board_is_detected_at_nominal_positions():
    k, m = 8.0, 12.0  # px per mm, quiet-zone margin [mm]
    img = rasterize(T, k, margin_mm=m)
    d = CharucoDetector(T).detect(img)
    assert d is not None and len(d) == (7 - 1) * (5 - 1)
    pred = (d.obj[:, :2] * 1e3 + m) * k - 0.5
    assert np.abs(pred - d.img).max() < 0.25
