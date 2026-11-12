"""Select sharp, evenly spread frames from a video for SfM.

The video is split into ``n`` equal segments; in each segment the sharpest frame
(variance of the Laplacian on a downscaled copy) is kept, unless it is much
blurrier than the video's typical frame. Frames are written as lossless PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from metrogram.sources import Source


def _sharpness(gray: np.ndarray, max_side: int = 960) -> float:
    s = max_side / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(
    video: str | Path,
    out: str | Path,
    n: int = 120,
    sharpness_rel: float = 0.5,
    log=sys.stderr,
) -> dict:
    src = Source(video)
    if src.video is None:
        raise ValueError(f"{video} is not a video")
    total = len(src)
    sharp = np.zeros(total)
    for fr in src.frames():
        if fr.index < total:
            sharp[fr.index] = _sharpness(fr.gray)
        if log and fr.index % 200 == 0:
            print(f"  sharpness {fr.index}/{total}", file=log)
    thresh = sharpness_rel * float(np.median(sharp[sharp > 0]))
    bounds = np.linspace(0, total, n + 1).astype(int)
    chosen, skipped = [], 0
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if b <= a:
            continue
        i = a + int(np.argmax(sharp[a:b]))
        if sharp[i] >= thresh:
            chosen.append(int(i))
        else:
            skipped += 1
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for i in chosen:
        cv2.imwrite(str(out / f"f{i:06d}.png"), src.read(i))
    return {
        "video": str(video),
        "n_frames": total,
        "n_written": len(chosen),
        "n_segments_skipped_blurry": skipped,
        "sharpness_threshold": thresh,
        "frames": chosen,
    }
