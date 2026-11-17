"""Calibration-target detection: image -> (board 3D points, image 2D points).

Board frames
------------
- AprilGrid: x right, y up, z = x × y (out of the board, towards the camera when
  the board faces it); origin at the outer bottom-left corner of the tag at
  grid position (row 0, col 0) counted from the bottom.
- ChArUco: OpenCV's convention, origin at the top-left chessboard corner, x
  right, y down, z into the board.

Units are metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from metrogram.config import AprilGridTarget, CharucoTarget, Target


@dataclass
class Detection:
    obj: np.ndarray  # (N, 3) float32, board frame, metres
    img: np.ndarray  # (N, 2) float32, pixels (OpenCV convention: pixel centers at integers)
    point_ids: np.ndarray  # (N,) int, unique id of each point on the board
    # (N,) edge width in px around each point (blur measure; AprilGrid only):
    # step height / max slope across the tag edges, ~1.5-2 px when in focus.
    edge_px: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.point_ids)


class AprilGridDetector:
    """Kalibr-style AprilGrid detected with OpenCV's AprilTag dictionaries.

    Tags are found on a copy downscaled to at most ``detect_max_px`` (OpenCV's
    adaptive threshold and contour steps fail on tags hundreds of pixels wide,
    as in close-range full-frame shots), then every corner is refined with
    cornerSubPix on the full-resolution image. On Kalibr boards each tag corner
    touches a small black square, so it is an X-junction: an ideal subpixel
    feature. Refinement runs twice: a wide window (15 % of the shortest tag
    side; on oblique, slightly blurred views the coarse corner can sit tens of
    pixels inside the tag) then a narrow one (5 %). Both stay below the 2-bit
    border and the corner squares (20 % of the side), so the only structure in
    the window is the junction itself.
    """

    # OpenCV corner order of a tag in its canonical upright orientation, as
    # unit-square positions in a y-up frame: TL, TR, BR, BL.
    _CANONICAL = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)

    def __init__(self, target: AprilGridTarget, detect_max_px: int = 2000):
        self.t = target
        self.detect_max_px = detect_max_px
        family = getattr(cv2.aruco, f"DICT_APRILTAG_{target.tag_family}")
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = target.border_bits
        params.adaptiveThreshWinSizeMax = 63  # large tags need wide windows
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE  # done below
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(family), params)
        # Printed tags may be rotated w.r.t. OpenCV's canonical orientation
        # (Kalibr boards: 180°). Rotating CCW by k*90° shifts the corner roles.
        k = (target.tag_rotation_deg // 90) % 4
        self._corner_unit = np.roll(self._CANONICAL, k, axis=0)
        self.n_tags = target.rows * target.cols

    @staticmethod
    def _refine(gray: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Refine a tag's 4 corners on the full-resolution image; None if unreliable.

        Returns (corners, edge width in px): the width (step height / max slope
        of the edge profiles) measures defocus and motion blur of this tag.

        1. Each tag edge is re-located from subpixel edge points on its middle 60 %
           (clean: away from the corner squares, which touch the tag at the
           corners and can drag coarse vertices along the merged "neck").
        2. Corners = intersections of adjacent edge lines.
        3. Final small-window cornerSubPix on the X-junction (absorbs the bias
           of straight-line fits on edges curved by lens distortion).
        """
        from scipy.ndimage import map_coordinates

        q = q.astype(np.float64)
        side = float(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1).min())
        if side < 12:
            return None
        h, w = gray.shape
        # Search half-length across each edge. Inside a correct edge lies the
        # 2-bit (20 %) black border, outside the white gap, so the tag edge is
        # the dark->bright transition (going outward) nearest to the coarse
        # edge; the reach tolerates coarse errors of ~1 bit.
        reach = max(3.0, 0.18 * side)
        center = q.mean(0)
        lines, widths = [], []
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            t = (b - a) / np.linalg.norm(b - a)
            n = np.array([-t[1], t[0]])
            if (a + b) / 2 @ n - center @ n < 0:
                n = -n  # outward normal
            m = int(np.clip(np.linalg.norm(b - a) / 4, 8, 60))
            ss = np.linspace(0.2, 0.8, m)
            offs = np.arange(-reach, reach + 0.25, 0.5)
            P = a + ss[:, None, None] * (b - a) + offs[None, :, None] * n  # (m, o, 2)
            if P[..., 0].min() < 1 or P[..., 1].min() < 1:
                return None
            if P[..., 0].max() > w - 2 or P[..., 1].max() > h - 2:
                return None
            prof = map_coordinates(gray, [P[..., 1].ravel(), P[..., 0].ravel()], order=1)
            prof = cv2.GaussianBlur(prof.reshape(len(ss), -1).astype(np.float32), (5, 1), 1.0)
            g = np.diff(prof, axis=1)  # > 0: dark -> bright going outward
            gmax = g.max(axis=1)
            i = np.empty(len(ss), int)
            # mid-point of gradient sample k, relative to the coarse edge (outward > 0)
            mid = offs[:-1] + 0.25
            cost = np.where(mid > 0, mid, -1.5 * mid)  # nearest, slight outward bias
            for j in range(len(ss)):  # strong dark->bright local maxima: nearest wins
                gj = g[j]
                peak = np.flatnonzero(
                    (gj[1:-1] >= 0.5 * gmax[j]) & (gj[1:-1] >= gj[:-2]) & (gj[1:-1] >= gj[2:])
                )
                i[j] = peak[np.argmin(cost[peak + 1])] + 1 if len(peak) else 1
            i = np.clip(i, 1, g.shape[1] - 2)
            r = np.arange(len(ss))
            g0, g1, g2 = g[r, i - 1], g[r, i], g[r, i + 1]
            den = g0 - 2 * g1 + g2
            d = np.where(np.abs(den) > 1e-9, 0.5 * (g0 - g2) / np.where(den == 0, 1, den), 0)
            o = offs[0] + 0.5 * (i + 0.5 + d)  # gradient sample i lies between offs i, i+1
            pts = a + ss[:, None] * (b - a) + o[:, None] * n
            strong = gmax > 0.25 * np.median(gmax)  # rows with a clear edge
            step = np.percentile(prof, 95, axis=1) - np.percentile(prof, 5, axis=1)
            widths.append((step / (np.maximum(gmax, 1e-6) / 0.5))[strong])  # samples 0.5 px
            keep = strong.copy()
            for _ in range(3):  # line fit with residual-based rejection
                c = pts[keep].mean(0)
                _, _, vt = np.linalg.svd(pts[keep] - c)
                nn = vt[1]
                res = np.abs((pts - c) @ nn)
                keep = strong & (res < max(1.0, 3 * np.median(res[keep])))
            if keep.sum() < max(5, len(pts) // 2):
                return None
            lines.append((c, nn))
        out = np.empty((4, 2))
        for k in range(4):  # vertex k = edge (k-1 -> k) x edge (k -> k+1)
            (c1, n1), (c2, n2) = lines[k - 1], lines[k]
            A = np.array([n1, n2])
            if abs(np.linalg.det(A)) < 1e-6:
                return None
            out[k] = np.linalg.solve(A, [n1 @ c1, n2 @ c2])
        if np.any(np.linalg.norm(out - q, axis=1) > 0.15 * side):
            return None  # edge fit far from the coarse quad: not this tag
        half = max(2, int(round(0.04 * side)))
        if np.any(out < half + 1) or np.any(out[:, 0] > w - half - 2):
            return None
        if np.any(out[:, 1] > h - half - 2):
            return None
        q1 = out.astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.01)
        cv2.cornerSubPix(gray, q1, (half, half), (-1, -1), crit)
        if np.any(np.linalg.norm(q1 - out, axis=1) > max(1.0, 0.5 * half)):
            return None  # no clean junction near the line intersection
        return q1, float(np.median(np.concatenate(widths)))

    def tag_corners(self, tag_id: int) -> np.ndarray:
        """(4, 3) board coordinates of a tag's corners, in OpenCV detection order."""
        r, c = divmod(int(tag_id), self.t.cols)
        if self.t.id_origin == "top_left":
            r = self.t.rows - 1 - r
        origin = np.array([c * self.t.pitch_m, r * self.t.pitch_m])
        xy = origin + self._corner_unit * self.t.tag_size_m
        return np.column_stack([xy, np.zeros(4)])

    def all_points(self) -> np.ndarray:
        return np.vstack([self.tag_corners(i) for i in range(self.n_tags)])

    @staticmethod
    def group_of(point_id: int) -> int:
        """Hold-out unit: the tag (its 4 corners are not independent)."""
        return int(point_id) // 4

    def detect(self, gray: np.ndarray) -> Detection | None:
        scale = min(1.0, self.detect_max_px / max(gray.shape))
        small = (
            gray
            if scale == 1.0
            else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        )
        corners, ids, _ = self._det.detectMarkers(small)
        if ids is None:
            return None
        obj, img, pid, edge = [], [], [], []
        for c, tag_id in zip(corners, ids.ravel(), strict=True):
            if not 0 <= tag_id < self.n_tags:
                continue  # stray tag from another board
            # pixel centres at integers: small (u + 0.5) / scale - 0.5 = full u
            q = ((c.reshape(4, 2).astype(np.float64) + 0.5) / scale - 0.5).astype(np.float32)
            r = self._refine(gray, q)
            if r is None:
                continue
            q, width = r
            edge.append(np.full(4, width))
            obj.append(self.tag_corners(tag_id))
            img.append(q)
            pid.append(tag_id * 4 + np.arange(4))
        if not obj:
            return None
        return Detection(
            np.vstack(obj).astype(np.float32),
            np.vstack(img).astype(np.float32),
            np.concatenate(pid),
            np.concatenate(edge),
        )


class CharucoDetector:
    def __init__(self, target: CharucoTarget):
        self.t = target
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, target.dictionary))
        self.board = cv2.aruco.CharucoBoard(
            (target.squares_x, target.squares_y),
            target.square_size_m,
            target.marker_size_m,
            dictionary,
        )
        self.board.setLegacyPattern(target.legacy_pattern)
        self._det = cv2.aruco.CharucoDetector(self.board)
        self._corners = np.asarray(self.board.getChessboardCorners(), dtype=np.float64)

    def all_points(self) -> np.ndarray:
        return self._corners

    @staticmethod
    def group_of(point_id: int) -> int:
        return int(point_id)

    def detect(self, gray: np.ndarray) -> Detection | None:
        cc, cid, _, _ = self._det.detectBoard(gray)
        if cid is None or len(cid) == 0:
            return None
        cid = cid.ravel().astype(int)
        return Detection(
            self._corners[cid].astype(np.float32),
            cc.reshape(-1, 2).astype(np.float32),
            cid,
        )


BoardDetector = AprilGridDetector | CharucoDetector


def make_detector(target: Target) -> BoardDetector:
    if isinstance(target, AprilGridTarget):
        return AprilGridDetector(target)
    if isinstance(target, CharucoTarget):
        return CharucoDetector(target)
    raise TypeError(f"unsupported target {type(target).__name__}")
