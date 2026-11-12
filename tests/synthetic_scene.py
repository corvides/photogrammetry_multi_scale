"""Tiny renderer for synthetic SfM tests: textured planar quads, known cameras.

Scene: a textured floor (z = 0) and a textured cube standing on it. Convex
objects drawn floor-first then the camera-facing cube faces give correct
occlusion without a z-buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def noise_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    """Multi-scale random texture (good SIFT features at several scales)."""
    tex = np.zeros((size, size), np.float64)
    for sigma, w in ((1.0, 0.5), (3.0, 1.0), (8.0, 1.0), (20.0, 0.7)):
        n = cv2.GaussianBlur(rng.normal(0, 1, (size, size)), (0, 0), sigma)
        tex += w * n / n.std()
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    return (40 + 180 * tex).astype(np.uint8)


@dataclass
class Quad:
    origin: np.ndarray  # 3D corner
    u: np.ndarray  # 3D edge vector (texture x)
    v: np.ndarray  # 3D edge vector (texture y)
    texture: np.ndarray

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u, self.v)
        return n / np.linalg.norm(n)

    @property
    def center(self) -> np.ndarray:
        return self.origin + (self.u + self.v) / 2


def make_scene(cube_size=0.10, floor_size=0.45, tex_px=1024, seed=0) -> list[Quad]:
    rng = np.random.default_rng(seed)
    a, f = cube_size, floor_size
    ex, ey, ez = np.eye(3)
    quads = [Quad(np.array([-f / 2, -f / 2, 0]), f * ex, f * ey, noise_texture(tex_px, rng))]
    o = np.array([-a / 2, -a / 2, 0])
    faces = [  # outward normals: u x v
        (o + a * ez, a * ex, a * ey),  # top (+z)
        (o, a * ez, a * ex),  # -y
        (o + a * ey, a * ex, a * ez),  # +y
        (o, a * ey, a * ez),  # -x
        (o + a * ex, a * ez, a * ey),  # +x
    ]
    for org, u, v in faces:
        quads.append(Quad(org, u, v, noise_texture(tex_px // 2, rng)))
    return quads


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)):
    """World->camera rotation R and translation t (OpenCV camera: x right, y down, z fwd)."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


def ring_cameras(radius=0.45, elevations_deg=(30, 55), n_az=12, target=(0, 0, 0.04)):
    cams = []
    for el in np.deg2rad(elevations_deg):
        for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            c = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            cams.append((c, *look_at(c, np.asarray(target, float))))
    return cams


def render(quads: list[Quad], K: np.ndarray, R, t, size, ss: int = 2) -> np.ndarray:
    W, H = size
    S = np.diag([ss, ss, 1.0])
    T = np.array([[1, 0, (ss - 1) / 2], [0, 1, (ss - 1) / 2], [0, 0, 1]])
    img = np.full((H * ss, W * ss), 128, np.uint8)
    cam_center = -R.T @ t
    for q in quads:
        if np.dot(q.normal, cam_center - q.center) <= 0:
            continue  # back-facing
        th, tw = q.texture.shape
        # texture pixel (x, y) -> 3D origin + u*x/tw + v*y/th -> image
        Hq = K @ np.column_stack([R @ q.u / tw, R @ q.v / th, R @ q.origin + t])
        Hq = T @ S @ Hq
        warped = cv2.warpPerspective(q.texture, Hq, (W * ss, H * ss), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(q.texture, 255), Hq, (W * ss, H * ss))
        img[mask > 127] = warped[mask > 127]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
