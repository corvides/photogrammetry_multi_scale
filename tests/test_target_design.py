"""target-design: board sizing rules and the written YAML."""

import pytest

from metrogram.config import CharucoTarget, load_config
from metrogram.target_design import design_charuco, marker_bits

SENSOR = (6000, 4000)


def test_far_setup_uses_ten_squares_on_short_side():
    d = design_charuco("far", SENSOR, 70e-6)  # field 420 x 280 mm
    assert d.squares_y == 10 and d.square_size_m == pytest.approx(
        0.0165
    )  # 168 mm / 10, rounded down
    bw, bh = d.board_m
    assert bw <= 0.6 * 0.42 + 1e-9 and bh <= 0.6 * 0.28 + 1e-9
    assert d.marker_size_m == pytest.approx(0.75 * d.square_size_m)


def test_close_setup_is_limited_by_print_resolution():
    d = design_charuco("close", SENSOR, 20e-6, min_bit_mm=0.8)
    bit_mm = d.marker_size_m * 1e3 / marker_bits(d.dictionary)
    assert bit_mm >= 0.8 - 1e-9
    assert d.squares_y < 10  # fewer squares rather than unprintable ones
    # a finer print process allows smaller squares
    fine = design_charuco("close", SENSOR, 20e-6, min_bit_mm=0.2)
    assert fine.square_size_m < d.square_size_m


def test_square_aspect_and_errors():
    d = design_charuco("sq", SENSOR, 70e-6, aspect="square")
    assert d.squares_x == d.squares_y
    with pytest.raises(ValueError, match="too small"):
        design_charuco("tiny", SENSOR, 2e-6)


def test_yaml_round_trip(tmp_path):
    d = design_charuco("rt", SENSOR, 50e-6)
    p = tmp_path / "t.yaml"
    p.write_text(d.yaml("Setup: test."))
    t = load_config(p)
    assert isinstance(t, CharucoTarget)
    assert (t.squares_x, t.squares_y) == (d.squares_x, d.squares_y)
    assert t.square_size_m == pytest.approx(d.square_size_m)
    assert t.marker_size_m == pytest.approx(d.marker_size_m)
