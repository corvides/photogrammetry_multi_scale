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


PPM = 5000.0  # board rendering resolution, px per metre
MARGIN = 0.02  # white margin around the tags, metres


def render_board(t, ppm=PPM) -> np.ndarray:
    """Kalibr-like board: tags rotated by t.tag_rotation_deg, y up, tag 0 bottom-left."""
    w, h = t.extent_m
    W, H = int((w + 2 * MARGIN) * ppm), int((h + 2 * MARGIN) * ppm)
    canvas = np.full((H, W), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    side = int(round(t.tag_size_m * ppm))
    for tag in range(t.rows * t.cols):
        r, c = divmod(tag, t.cols)
        img = cv2.aruco.generateImageMarker(d, tag, side, borderBits=1)
        img = np.rot90(img, k=t.tag_rotation_deg // 90)  # CCW, as seen with y up
        x0 = MARGIN + c * t.pitch_m
        y_top = MARGIN + r * t.pitch_m + t.tag_size_m  # board y of the tag's top edge
        u, v = int(round(x0 * ppm)), int(round((h + 2 * MARGIN - y_top) * ppm))
        canvas[v : v + side, u : u + side] = img
    return canvas


def canvas_to_board(t, ppm=PPM) -> np.ndarray:
    """Homography canvas pixel -> board metres (x right, y up, origin at tag 0)."""
    h = t.extent_m[1]
    return np.array([[1 / ppm, 0, -MARGIN], [0, -1 / ppm, h + MARGIN], [0, 0, 1]], dtype=np.float64)


@dataclass
class Quad:
    origin: np.ndarray  # 3D corner
    u: np.ndarray  # 3D edge vector (texture x)
    v: np.ndarray  # 3D edge vector (texture y)
    texture: np.ndarray
    two_sided: bool = False

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
        if not q.two_sided and np.dot(q.normal, cam_center - q.center) <= 0:
            continue  # back-facing
        th, tw = q.texture.shape
        # texture pixel center (x, y) -> 3D origin + u*(x+0.5)/tw + v*(y+0.5)/th
        # (pixel edges, not centers, coincide with the quad's edges)
        o3 = q.origin + 0.5 * q.u / tw + 0.5 * q.v / th
        Hq = K @ np.column_stack([R @ q.u / tw, R @ q.v / th, R @ o3 + t])
        Hq = T @ S @ Hq
        warped = cv2.warpPerspective(q.texture, Hq, (W * ss, H * ss), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(q.texture, 255), Hq, (W * ss, H * ss))
        img[mask > 127] = warped[mask > 127]
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def board_quad(t, ppm=4000.0) -> Quad:
    """The AprilGrid as a quad in its own board frame (z = 0).

    For exact geometry, tag size, pitch and MARGIN must be whole pixels at ``ppm``
    (e.g. tag 40 mm, gap 10 mm at 4000 px/m).
    """
    canvas = render_board(t, ppm)
    h = t.extent_m[1]
    Hc, Wc = canvas.shape
    origin = np.array([-MARGIN, h + MARGIN, 0.0])  # canvas pixel (0, 0)
    return Quad(origin, np.array([Wc / ppm, 0, 0]), np.array([0, -Hc / ppm, 0]), canvas, True)


def board_scene(t, cube_size=0.08, seed=0) -> list[Quad]:
    """AprilGrid on the floor with a textured cube standing at its center."""
    rng = np.random.default_rng(seed)
    w, h = t.extent_m
    c = np.array([w / 2, h / 2, 0])
    a = cube_size
    ex, ey, ez = np.eye(3)
    o = c - np.array([a / 2, a / 2, 0])
    faces = [
        (o + a * ez, a * ex, a * ey),
        (o, a * ez, a * ex),
        (o + a * ey, a * ex, a * ez),
        (o, a * ey, a * ez),
        (o + a * ex, a * ez, a * ey),
    ]
    return [board_quad(t)] + [Quad(org, u, v, noise_texture(512, rng)) for org, u, v in faces]
