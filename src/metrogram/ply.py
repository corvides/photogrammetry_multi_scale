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


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}  # fmt: skip


def read_header(path: str | Path) -> tuple[dict[str, int], str, int]:
    """({element: count}, format, header byte length)."""
    counts, fmt = {}, ""
    with open(path, "rb") as fh:
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            t = line.decode("ascii", "replace").split()
            if t[:1] == ["format"]:
                fmt = t[1]
            elif t[:1] == ["element"]:
                counts[t[1]] = int(t[2])
            elif t[:1] == ["end_header"]:
                return counts, fmt, fh.tell()


def read_vertices(path: str | Path, chunk: int = 500_000) -> np.ndarray:
    """Scalar properties of the vertex element of a binary little-endian PLY.

    Handles list properties after the scalars (e.g. OpenMVS's per-point
    ``view_indices`` / ``view_weights``): they are skipped. The vertex element
    must come first (true for COLMAP, OpenMVS and our writers).
    """
    scalars: list[tuple[str, str]] = []
    lists: list[tuple[np.dtype, np.dtype]] = []  # (count type, item type)
    in_vertex, n = False, 0
    with open(path, "rb") as fh:
        while True:
            t = fh.readline().decode("ascii", "replace").split()
            if t[:1] == ["format"] and t[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary_little_endian PLY supported")
            if t[:1] == ["element"]:
                in_vertex = t[1] == "vertex" and n == 0
                if in_vertex:
                    n = int(t[2])
            elif t[:1] == ["property"] and in_vertex:
                if t[1] == "list":
                    lists.append(
                        (np.dtype("<" + _PLY_TYPES[t[2]]), np.dtype("<" + _PLY_TYPES[t[3]]))
                    )
                elif lists:
                    raise ValueError(f"{path}: scalar property after a list is not supported")
                else:
                    scalars.append((t[2], "<" + _PLY_TYPES[t[1]]))
            elif t[:1] == ["end_header"]:
                break
        dt = np.dtype(scalars)
        if not lists:
            return np.frombuffer(fh.read(n * dt.itemsize), dtype=dt, count=n)
        buf = fh.read()
    # Variable-length records: one pass to find each record start.
    raw = np.frombuffer(buf, np.uint8)
    fixed = dt.itemsize
    starts = np.empty(n, np.int64)
    off = 0
    for i in range(n):
        starts[i] = off
        off += fixed
        for ct, it in lists:
            cnt = int.from_bytes(buf[off : off + ct.itemsize], "little")
            off += ct.itemsize + cnt * it.itemsize
    out = np.empty(n, dt)
    cols = np.arange(fixed)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        out[a:b] = raw[starts[a:b, None] + cols].copy().view(dt).ravel()
    return out
