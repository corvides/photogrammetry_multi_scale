"""Image sources: a video file, a directory of images (incl. RAW), or a list of files.

Frames are yielded as 8-bit grayscale. RAW files are decoded with LibRaw (rawpy)
WITHOUT any lens correction (LibRaw applies none), so the calibration models
the real lens distortion. Images are always in SENSOR orientation (EXIF/RAW
rotation flags ignored: intrinsics are per sensor, not per display), and RAW
frames are cropped to the maker's default area (e.g. Canon R6: 5472x3648).

A directory holding RAW files plus other images (e.g. CR3 + preview JPG pairs)
is read as RAW only: in-camera JPEGs are resized and lens-corrected.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts"}
RAW_EXT = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"} | RAW_EXT


@dataclass
class Frame:
    key: str  # stable identifier: "f000368" for video, file name for images
    index: int  # position in the source
    gray: np.ndarray


def list_images(directory: str | Path) -> list[Path]:
    """Sorted image files of a directory; RAW only if any RAW is present."""
    files = sorted(f for f in Path(directory).iterdir() if f.suffix.lower() in IMAGE_EXT)
    raw = [f for f in files if f.suffix.lower() in RAW_EXT]
    return raw if raw else files


def decode_raw(path: str | Path, output_bps: int = 8) -> np.ndarray:
    """RGB, no lens correction, sensor orientation, maker default crop."""
    import rawpy  # local import: only needed for RAW

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True, no_auto_bright=True, output_bps=output_bps, user_flip=0
        )
        s = raw.sizes
    # LibRaw's output starts at (left_margin, top_margin) of the raw frame; the
    # maker crop (crop_*) is given in raw coordinates.
    x0, y0 = s.crop_left_margin - s.left_margin, s.crop_top_margin - s.top_margin
    cw, ch = s.crop_width, s.crop_height
    if cw and ch and x0 >= 0 and y0 >= 0 and x0 + cw <= rgb.shape[1] and y0 + ch <= rgb.shape[0]:
        rgb = rgb[y0 : y0 + ch, x0 : x0 + cw]
    return np.ascontiguousarray(rgb)


IMREAD_GRAY = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION
IMREAD_COLOR = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def read_gray(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        gray16 = cv2.cvtColor(decode_raw(path, 16), cv2.COLOR_RGB2GRAY)
        return (gray16 / 257.0).round().astype(np.uint8)
    img = cv2.imread(str(path), IMREAD_GRAY)
    if img is None:
        raise OSError(f"cannot read image {path}")
    return img


class Source:
    """Uniform access to frames from a video or a set of images."""

    def __init__(self, spec: str | Path | Sequence[str | Path]):
        if isinstance(spec, (str, Path)):
            p = Path(spec)
            if p.is_dir():
                self.files = list_images(p)
                self.video = None
            elif p.suffix.lower() in VIDEO_EXT:
                self.files, self.video = [], p
            elif p.suffix.lower() in IMAGE_EXT:
                self.files, self.video = [p], None
            else:
                raise ValueError(f"unsupported source {p}")
        else:
            self.files, self.video = [Path(f) for f in spec], None
        if self.video is None and not self.files:
            raise ValueError(f"no images found in {spec}")
        if self.video is not None and not self.video.exists():
            raise FileNotFoundError(self.video)
        self.description = (
            str(spec) if isinstance(spec, (str, Path)) else f"{len(self.files)} files"
        )

    def __len__(self) -> int:
        if self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n
        return len(self.files)

    def frames(self, stride: int = 1) -> Iterator[Frame]:
        if self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            i = 0
            while True:
                if i % stride:
                    if not cap.grab():
                        break
                else:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    yield Frame(f"f{i:06d}", i, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
                i += 1
            cap.release()
        else:
            for i, f in enumerate(self.files[::stride]):
                yield Frame(f.name, i * stride, read_gray(f))

    def read(self, index: int) -> np.ndarray:
        """Random access to one frame (BGR for video, gray->BGR for images)."""
        if self.video is not None:
            cap = cv2.VideoCapture(str(self.video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            cap.release()
            if not ok:
                raise OSError(f"cannot read frame {index} of {self.video}")
            return bgr
        return cv2.cvtColor(read_gray(self.files[index]), cv2.COLOR_GRAY2BGR)
