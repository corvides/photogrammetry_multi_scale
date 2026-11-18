"""Size a ChArUco target for a camera setup (``metrogram target-design``).

Inputs: the sensor size in pixels and the ground sampling distance (GSD,
metres per pixel on the board) at the working distance. Output: a target YAML
(``kind: target, type: charuco``) that `target-svg` turns into a printable SVG
and `calibrate` uses for detection.

Rules (all of them in pixels or in printed millimetres, so the same rules hold
from the endoscope to the dm scale):

- **Board size** = ``fill`` x the field of view on each axis (sensor aspect),
  or ``fill`` x the short side of the field on both axes (``aspect="square"``).
  Below 1, the board can be moved into the image corners (§7.6 of the README).
- **Square size**: as large as ``board short side / squares_short`` (a few
  big squares: blur-tolerant), but never smaller than
  - the size at which one marker bit spans ``min_bit_px`` pixels at the
    working distance (OpenCV needs about 4 px per bit; a 45° tilt costs 30 %),
  - the size at which one marker bit is ``min_bit_mm`` on the print (ink
    spread eats about 0.2 mm per edge on ordinary prints; 0.8 mm leaves a
    white bit of about 0.4 mm).
  It is rounded to ``step_mm``: down for the squares wish (keeps N squares),
  up for the two minima.
- The square counts are the largest that fit in the board size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class CharucoDesign:
    name: str
    squares_x: int
    squares_y: int
    square_size_m: float
    marker_size_m: float
    dictionary: str
    # for the report
    gsd_m: float
    field_m: tuple[float, float]
    limits_mm: dict  # square size wanted by each rule [mm]

    @property
    def board_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_size_m, self.squares_y * self.square_size_m

    @property
    def bits(self) -> int:
        return marker_bits(self.dictionary)

    @property
    def n_markers(self) -> int:
        return self.squares_x * self.squares_y // 2

    def summary(self) -> str:
        bw, bh = (v * 1e3 for v in self.board_m)
        fw, fh = (v * 1e3 for v in self.field_m)
        sq, mk = self.square_size_m * 1e3, self.marker_size_m * 1e3
        bit_mm = mk / self.bits
        rule = max(self.limits_mm, key=self.limits_mm.get)  # the binding one
        lim = ", ".join(f"{k} {v:.2f}" for k, v in self.limits_mm.items())
        return "\n".join(
            [
                f"field of view   {fw:.1f} x {fh:.1f} mm  (GSD {self.gsd_m * 1e6:.1f} um/px)",
                f"board           {bw:.1f} x {bh:.1f} mm  "
                f"({100 * bw / fw:.0f} % x {100 * bh / fh:.0f} % of the field)",
                f"squares         {self.squares_x} x {self.squares_y} of {sq:g} mm "
                f"({sq / (self.gsd_m * 1e3):.0f} px), "
                f"{(self.squares_x - 1) * (self.squares_y - 1)} corners",
                f"markers         {mk:g} mm, {self.n_markers} of {self.dictionary}, "
                f"bit {bit_mm:.2f} mm = {bit_mm / (self.gsd_m * 1e3):.1f} px",
                f"square size set by: {rule}  (wanted [mm]: {lim})",
            ]
        )

    def yaml(self, notes: str = "") -> str:
        bw, bh = (v * 1e3 for v in self.board_m)
        fw, fh = (v * 1e3 for v in self.field_m)
        sq = self.square_size_m * 1e3
        text = (
            f"Designed by metrogram target-design for a field of {fw:.0f} x {fh:.0f} mm "
            f"(GSD {self.gsd_m * 1e6:.1f} um/px): board {bw:g} x {bh:g} mm. Nominal "
            "dimensions: after printing, measure between same-side edges (first and "
            f"last columns: {(self.squares_x - 1) * sq:g} mm; rows: "
            f"{(self.squares_y - 1) * sq:g} mm) and scale square_size_m and "
            "marker_size_m by the same factor. " + notes
        ).strip()
        return "\n".join(
            [
                "kind: target",
                "type: charuco",
                f"name: {self.name}",
                f"squares_x: {self.squares_x}",
                f"squares_y: {self.squares_y}",
                f"square_size_m: {self.square_size_m:.6g}   # NOMINAL until measured",
                f"marker_size_m: {self.marker_size_m:.6g}",
                f"dictionary: {self.dictionary}",
                "legacy_pattern: false",
                "material: ''",
                "dimensions_certified: false",
                "notes: >",
                *_wrap(text, 78, "  "),
                "",
            ]
        )


def marker_bits(dictionary: str) -> int:
    """Cells across one marker: data bits + a 1-bit black border on each side."""
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)).markerSize + 2


def dictionary_size(dictionary: str) -> int:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return int(d.bytesList.shape[0])


def design_charuco(
    name: str,
    sensor_px: tuple[int, int],
    gsd_m: float,
    *,
    fill: float = 0.6,
    aspect: str = "sensor",
    squares_short: int = 10,
    marker_ratio: float = 0.75,
    dictionary: str = "DICT_5X5_1000",
    min_bit_px: float = 6.0,
    min_bit_mm: float = 0.8,
    step_mm: float = 0.5,
) -> CharucoDesign:
    if not 0 < fill <= 1.5:
        raise ValueError("fill must be in (0, 1.5]")
    if aspect not in ("sensor", "square"):
        raise ValueError("aspect must be 'sensor' or 'square'")
    if not 0.5 <= marker_ratio < 1:
        raise ValueError("marker_ratio must be in [0.5, 1)")
    fw, fh = sensor_px[0] * gsd_m, sensor_px[1] * gsd_m
    short = min(fw, fh)
    if aspect == "square":
        bw = bh = fill * short
    else:
        bw, bh = fill * fw, fill * fh
    bits = marker_bits(dictionary)
    limits_mm = {
        f"{squares_short} squares on the short side": 1e3 * min(bw, bh) / squares_short,
        f"bit >= {min_bit_px:g} px": 1e3 * min_bit_px * bits * gsd_m / marker_ratio,
        f"bit >= {min_bit_mm:g} mm printed": min_bit_mm * bits / marker_ratio,
    }
    soft, *hard = limits_mm.values()  # wish (rounded down: keep N squares) / minima (up)
    sq_mm = max(
        step_mm * max(1, math.floor(soft / step_mm + 1e-9)),
        step_mm * math.ceil(max(hard) / step_mm - 1e-9),
    )
    nx, ny = int(1e3 * bw // sq_mm), int(1e3 * bh // sq_mm)
    if min(nx, ny) < 3:
        raise ValueError(
            f"only {nx} x {ny} squares of {sq_mm:g} mm fit in {1e3 * bw:.0f} x {1e3 * bh:.0f} mm: "
            "the field is too small for these print / pixel limits (a finer print process, "
            "i.e. a smaller --min-bit-mm, or a larger --fill)"
        )
    if nx * ny // 2 > dictionary_size(dictionary):
        raise ValueError(f"{nx * ny // 2} markers needed, {dictionary} has fewer")
    return CharucoDesign(
        name=name,
        squares_x=nx,
        squares_y=ny,
        square_size_m=sq_mm / 1e3,
        marker_size_m=round(marker_ratio * sq_mm, 4) / 1e3,
        dictionary=dictionary,
        gsd_m=gsd_m,
        field_m=(fw, fh),
        limits_mm=limits_mm,
    )


def _wrap(text: str, width: int, indent: str) -> list[str]:
    lines, cur = [], indent
    for word in text.split():
        if len(cur) + len(word) + 1 > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += word + " "
    if cur.strip():
        lines.append(cur.rstrip())
    return lines
