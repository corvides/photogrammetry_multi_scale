"""Image listing: RAW files take precedence over preview JPEGs in the same folder."""

import cv2
import numpy as np

from metrogram.sources import Source, list_images


def test_raw_preferred_over_jpeg(tmp_path):
    for stem in ("a", "b"):
        (tmp_path / f"{stem}.CR3").write_bytes(b"")
        cv2.imwrite(str(tmp_path / f"{stem}.JPG"), np.zeros((4, 6), np.uint8))
    (tmp_path / "notes.txt").write_text("x")
    assert [f.name for f in list_images(tmp_path)] == ["a.CR3", "b.CR3"]


def test_images_only_folder(tmp_path):
    for stem in ("b", "a"):
        cv2.imwrite(str(tmp_path / f"{stem}.png"), np.zeros((4, 6), np.uint8))
    assert [f.name for f in Source(tmp_path).files] == ["a.png", "b.png"]
