from pathlib import Path

import pytest

from metrogram.cli import main
from metrogram.config import AprilGridTarget, ConfigError, Device, Preset, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path):
    load_config(path)


def test_aprilgrid_geometry():
    t = load_config(CONFIGS / "targets/aprilgrid_6x6_acrylic.yaml")
    assert isinstance(t, AprilGridTarget)
    assert t.gap_m == pytest.approx(0.04313 * 0.391)
    w, h = t.extent_m
    assert w == h == pytest.approx(6 * 0.04313 + 5 * 0.04313 * 0.391)


def test_device_gsd():
    d = load_config(CONFIGS / "devices/canon_r6_rf35macro.yaml")
    assert isinstance(d, Device)
    # thin lens: (0.30 - 0.035) * 6.56e-6 / 0.035
    assert d.gsd_m(0.30) == pytest.approx(49.67e-6, rel=1e-3)


def test_preset_nested():
    p = load_config(CONFIGS / "presets/gpu-full.yaml")
    assert isinstance(p, Preset) and p.dense.use_cuda and p.max_image_size_px is None


def test_unknown_key_rejected(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("kind: preset\nname: x\nmax_imge_size_px: 10\n")
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(f)


def test_cli_check_config(capsys):
    assert main(["check-config", str(CONFIGS / "targets/aprilgrid_6x6_acrylic.yaml")]) == 0
    assert "extent" in capsys.readouterr().out
