"""PLY writer <-> viewer loading and framing (no rendering: CI may be headless)."""

import numpy as np
import pytest

pytest.importorskip("open3d")

from metrogram import ply, viewer  # noqa: E402


def test_layers_roi_and_crop(tmp_path):
    rng = np.random.default_rng(0)
    near = rng.uniform(0, 0.3, (500, 3))
    far = rng.uniform(5, 6, (20, 3))  # e.g. the room behind the board
    ply.write_points(tmp_path / "points_metric.ply", np.vstack([near, far]), (10, 20, 30))
    b = ply.MeshBuilder()
    b.add_box_segment([0, 0, 0], [0.3, 0, 0], 0.002, (255, 0, 0))
    b.add_box_segment([0, 0, 0], [0, 0.3, 0], 0.002, (0, 255, 0))
    b.write(tmp_path / "board_frame.ply")

    layers = viewer.load([tmp_path])
    kinds = {ly.name: ly.is_mesh for ly in layers}
    assert kinds == {"board_frame": True, "points_metric": False}
    pts = next(ly for ly in layers if ly.name == "points_metric")
    assert len(pts.points()) == 520
    np.testing.assert_allclose(np.asarray(pts.geo.colors)[0], np.array([10, 20, 30]) / 255)

    lo, hi = viewer.roi(layers, margin=0.5)
    assert hi.max() < 1.0  # framed on the board, not the far points
    viewer.crop(layers, lo, hi)
    assert len(pts.points()) == 500


def test_crop_keeps_texture():
    import open3d as o3d

    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5], [6, 5, 5], [5, 6, 5]], float)
    T = np.array([[0, 1, 2], [3, 4, 5]])
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T))
    m.triangle_uvs = o3d.utility.Vector2dVector(np.random.default_rng(0).random((6, 2)))
    m.triangle_material_ids = o3d.utility.IntVector([0, 0])
    m.textures = [o3d.geometry.Image(np.zeros((4, 4, 3), np.uint8))]
    layer = viewer.Layer("mesh", m, True)
    viewer.crop([layer], [-1, -1, -1], [2, 2, 2])
    c = layer.geo
    assert len(c.triangles) == 1
    assert c.has_textures() and len(c.triangle_uvs) == 3
    np.testing.assert_allclose(np.asarray(c.triangle_uvs), np.asarray(m.triangle_uvs)[:3])
