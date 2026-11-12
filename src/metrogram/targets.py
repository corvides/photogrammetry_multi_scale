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

    def __len__(self) -> int:
        return len(self.point_ids)


class AprilGridDetector:
    """Kalibr-style AprilGrid detected with OpenCV's AprilTag dictionaries."""

    # OpenCV corner order of a tag in its canonical upright orientation, as
    # unit-square positions in a y-up frame: TL, TR, BR, BL.
    _CANONICAL = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)

    def __init__(self, target: AprilGridTarget):
        self.t = target
        family = getattr(cv2.aruco, f"DICT_APRILTAG_{target.tag_family}")
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(family), params)
        # Printed tags may be rotated w.r.t. OpenCV's canonical orientation
        # (Kalibr boards: 180°). Rotating CCW by k*90° shifts the corner roles.
        k = (target.tag_rotation_deg // 90) % 4
        self._corner_unit = np.roll(self._CANONICAL, k, axis=0)
        self.n_tags = target.rows * target.cols

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
        corners, ids, _ = self._det.detectMarkers(gray)
        if ids is None:
            return None
        obj, img, pid = [], [], []
        for c, tag_id in zip(corners, ids.ravel(), strict=True):
            if not 0 <= tag_id < self.n_tags:
                continue  # stray tag from another board
            obj.append(self.tag_corners(tag_id))
            img.append(c.reshape(4, 2))
            pid.append(tag_id * 4 + np.arange(4))
        if not obj:
            return None
        return Detection(
            np.vstack(obj).astype(np.float32),
            np.vstack(img).astype(np.float32),
            np.concatenate(pid),
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
