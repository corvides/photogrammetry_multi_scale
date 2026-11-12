"""Minimal PLY writers (binary little-endian, viewer-agnostic).

- point clouds: x y z (float32) + red green blue (uchar)
- triangle meshes with per-vertex colors (used for overlays: camera frustums,
  board outline, axes), since edge elements are poorly supported by viewers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _header(n_vertex: int, n_face: int = 0, comment: str = "") -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    if comment:
        lines += [f"comment {c}" for c in comment.splitlines()]
    lines += [
        f"element vertex {n_vertex}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if n_face:
        lines += [f"element face {n_face}", "property list uchar int vertex_indices"]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode()


def _vertices(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.empty(len(xyz), dt)
    v["x"], v["y"], v["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rgb = np.broadcast_to(np.asarray(rgb, np.uint8), (len(xyz), 3))
    v["r"], v["g"], v["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return v.tobytes()


def write_points(path: str | Path, xyz: np.ndarray, rgb=(200, 200, 200), comment: str = ""):
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    with open(path, "wb") as fh:
        fh.write(_header(len(xyz), comment=comment))
        fh.write(_vertices(xyz, rgb))


class MeshBuilder:
    """Accumulate colored triangles, then write one PLY."""

    def __init__(self):
        self.v: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.f: list[np.ndarray] = []
        self.n = 0

    def add(self, vertices: np.ndarray, faces: np.ndarray, rgb) -> None:
        vertices = np.asarray(vertices, float).reshape(-1, 3)
        self.v.append(vertices)
        self.c.append(np.broadcast_to(np.asarray(rgb, np.uint8), (len(vertices), 3)))
        self.f.append(np.asarray(faces, int).reshape(-1, 3) + self.n)
        self.n += len(vertices)

    def add_box_segment(self, a, b, width: float, rgb) -> None:
        """A thin square prism from a to b (visible 'line' in any mesh viewer)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        d /= np.linalg.norm(d)
        u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0])
        u *= width / 2 / np.linalg.norm(u)
        w = np.cross(d, u)
        ring = [u + w, u - w, -u - w, -u + w]
        verts = [a + r for r in ring] + [b + r for r in ring]
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            faces += [[i, j, 4 + j], [i, 4 + j, 4 + i]]
        faces += [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
        self.add(verts, faces, rgb)

    def add_frustum(self, center, R_world_from_cam, size: float, aspect: float, rgb, width):
        """Camera pyramid: apex at the center, base at distance `size` along +z."""
        hw, hh = size * 0.5 * aspect, size * 0.5
        base = [np.array([x, y, size]) for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
        corners = [center + R_world_from_cam @ p for p in base]
        for i in range(4):
            self.add_box_segment(center, corners[i], width, rgb)
            self.add_box_segment(corners[i], corners[(i + 1) % 4], width, rgb)
        # "up" tick marks image top (camera -y)
        top = (corners[0] + corners[1]) / 2
        self.add_box_segment(
            top, top + R_world_from_cam @ np.array([0, -size * 0.25, 0]), width, rgb
        )

    def write(self, path: str | Path, comment: str = "") -> None:
        v, c, f = np.vstack(self.v), np.vstack(self.c), np.vstack(self.f)
        fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
        faces = np.empty(len(f), fdt)
        faces["n"], faces["i"] = 3, f
        with open(path, "wb") as fh:
            fh.write(_header(len(v), len(f), comment))
            fh.write(_vertices(v, c))
            fh.write(faces.tobytes())
