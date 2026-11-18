"""Printable vector (SVG) calibration targets.

The layout comes from OpenCV's own ``CharucoBoard`` built from the target
config (marker ids, positions and the legacy / new pattern), so the printed
board is exactly the board the detector expects.

Units: the SVG user unit is the millimetre (``width="..mm"`` with a matching
``viewBox``), so Inkscape and PDF export keep the true size. Black areas are
drawn as one path per marker (evenodd fill) plus one rectangle per black
square: no abutting same-colour shapes, hence no anti-aliasing hairlines
between marker bits.

A 100 mm scale bar and a caption are added outside the board, to check the
printed scale with a caliper (printers are often 0.1-1 % off, and differently
along and across the paper feed).
"""

from __future__ import annotations

from datetime import date

import cv2
import numpy as np

from metrogram.config import CharucoTarget

PAGES_MM = {"a4": (297.0, 210.0), "a3": (420.0, 297.0), "letter": (279.4, 215.9)}


def parse_page(s: str) -> tuple[float, float] | None:
    """Page size in mm from a name (a4, a3, letter: landscape), ``WxH`` in mm
    (e.g. ``300x200``, as given), or ``fit`` (None: board + margins + caption)."""
    s = s.strip().lower()
    if s == "fit":
        return None
    if s in PAGES_MM:
        return PAGES_MM[s]
    try:
        w, h = (float(v) for v in s.split("x"))
    except ValueError:
        raise ValueError(f"page {s!r}: use a4, a3, letter, fit or WxH in mm (e.g. 300x200)")
    if w <= 0 or h <= 0:
        raise ValueError(f"page {s!r}: sizes must be positive")
    return w, h


def charuco_board(t: CharucoTarget) -> cv2.aruco.CharucoBoard:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, t.dictionary))
    b = cv2.aruco.CharucoBoard((t.squares_x, t.squares_y), t.square_size_m, t.marker_size_m, d)
    b.setLegacyPattern(t.legacy_pattern)
    return b


def charuco_shapes(t: CharucoTarget) -> tuple[list[tuple], list[list[list[tuple]]]]:
    """Black geometry of a ChArUco board, in mm, origin top-left, y down.

    Returns (squares, markers): squares = [(x, y, w, h)]; markers = one list of
    closed loops [(x, y), ...] per marker, to be filled with the evenodd rule.
    """
    b = charuco_board(t)
    sq = t.square_size_m * 1e3
    d = b.getDictionary()
    nbits = d.markerSize + 2  # 1-bit black border on each side
    has_marker = np.zeros((t.squares_y, t.squares_x), bool)
    markers = []
    for mid, obj in zip(b.getIds().ravel(), b.getObjPoints(), strict=True):
        p = np.asarray(obj).reshape(-1, 3)[:, :2] * 1e3  # TL, TR, BR, BL in mm
        x0, y0 = p[0]
        bit = (p[1, 0] - p[0, 0]) / nbits
        cx, cy = p.mean(0)
        has_marker[int(cy // sq), int(cx // sq)] = True
        cells = cv2.aruco.generateImageMarker(d, int(mid), nbits, borderBits=1) == 0
        markers.append(_cell_loops(cells, x0, y0, bit))
    squares = [
        (i * sq, j * sq, sq, sq)
        for j in range(t.squares_y)
        for i in range(t.squares_x)
        if not has_marker[j, i]
    ]
    return squares, markers


def _cell_loops(cells: np.ndarray, x0: float, y0: float, s: float) -> list[list[tuple]]:
    """Boundary loops of the True cells of a grid (cell size s, origin x0, y0).

    Every cell edge between a black and a white (or outside) cell is used once;
    chaining them into closed loops in any order gives the right evenodd fill.
    """
    h, w = cells.shape
    edges: dict[tuple, list[tuple]] = {}
    for r in range(h):
        for c in range(w):
            if not cells[r, c]:
                continue
            # clockwise in y-down coordinates: top, right, bottom, left
            for (a, b), (dr, dc) in (
                (((c, r), (c + 1, r)), (-1, 0)),
                (((c + 1, r), (c + 1, r + 1)), (0, 1)),
                (((c + 1, r + 1), (c, r + 1)), (1, 0)),
                (((c, r + 1), (c, r)), (0, -1)),
            ):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and cells[rr, cc]:
                    continue
                edges.setdefault(a, []).append(b)
    loops = []
    while edges:
        start = next(iter(edges))
        loop, v = [start], start
        while True:
            nxt = edges[v].pop()
            if not edges[v]:
                del edges[v]
            if nxt == start:
                break
            loop.append(nxt)
            v = nxt
        loops.append([(x0 + x * s, y0 + y * s) for x, y in _drop_collinear(loop)])
    return loops


def _drop_collinear(loop: list[tuple]) -> list[tuple]:
    n = len(loop)
    out = []
    for i in range(n):
        a, b, c = loop[i - 1], loop[i], loop[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) != (b[1] - a[1]) * (c[0] - b[0]):
            out.append(b)
    return out


def _f(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


CHAR_W = 0.6  # average glyph width / font size (DejaVu Sans; generous)


def charuco_svg(
    t: CharucoTarget, page: str | tuple[float, float] | None = "a4", margin_mm: float = 8.0
) -> str:
    """SVG document of the board, centred on ``page``: a name or ``WxH`` string
    (see `parse_page`), a (width, height) in mm, or None / "fit" for a page just
    large enough (board + margin + caption).

    The caption font is 3 mm, reduced (down to 2 mm) when the caption is wider
    than the board; on a "fit" page the page is widened if it still does not fit.
    """
    squares, markers = charuco_shapes(t)
    bw, bh = (v * 1e3 for v in t.extent_m)
    sq, mk = t.square_size_m * 1e3, t.marker_size_m * 1e3
    lines = [
        f"{t.name}: ChArUco {t.squares_x} x {t.squares_y} squares of {_f(sq)} mm, "
        f"markers {_f(mk)} mm, {t.dictionary}, legacy_pattern={t.legacy_pattern}, "
        f"board {_f(bw)} x {_f(bh)} mm",
        "Print at 100 % (no fit-to-page). Check the scale with a caliper between SAME-SIDE "
        f"edges (e.g. left edges of the first and last columns: {_f((t.squares_x - 1) * sq)} mm),",
        "not the outer extent: printers enlarge black areas. "
        f"Generated {date.today().isoformat()} by metrogram target-svg.",
    ]
    text_w3 = max(len(line) for line in lines) * CHAR_W * 3.0  # width at 3 mm font
    size = parse_page(page) if isinstance(page, str) else page
    room = bw if size is None else size[0] - 2 * margin_mm
    fs = min(3.0, max(2.0, 3.0 * room / text_w3))
    text_w = text_w3 * fs / 3.0
    cap_h = 11.0 + fs * (1 + 5 * (len(lines) - 1) / 3.0)  # scale bar + text lines
    if size is None:
        pw, ph = max(bw, text_w) + 2 * margin_mm, bh + 2 * margin_mm + cap_h
    else:
        pw, ph = size
    if max(bw, text_w) + 2 * margin_mm > pw or bh + cap_h + 2 * margin_mm > ph:
        raise ValueError(
            f"board {_f(bw)}x{_f(bh)} mm + {cap_h:.0f} mm caption + 2 x {margin_mm:g} mm "
            f"margin does not fit on a {_f(pw)}x{_f(ph)} mm page"
        )
    ox = (pw - bw) / 2
    oy = (ph - bh - cap_h) / 2
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_f(pw)}mm" height="{_f(ph)}mm" '
        f'viewBox="0 0 {_f(pw)} {_f(ph)}">',
        f"<title>{t.name}</title>",
        f'<rect x="0" y="0" width="{_f(pw)}" height="{_f(ph)}" fill="#ffffff"/>',
        f'<g id="board" transform="translate({_f(ox)} {_f(oy)})" fill="#000000" stroke="none">',
    ]
    for x, y, w, h in squares:
        out.append(f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}"/>')
    for loops in markers:
        d = " ".join(
            "M " + " L ".join(f"{_f(x)} {_f(y)}" for x, y in loop) + " Z" for loop in loops
        )
        out.append(f'<path fill-rule="evenodd" d="{d}"/>')
    out.append("</g>")
    # Scale bar: 100 mm (or the board width rounded down to 10 mm), ticks every 10 mm.
    L = 100.0 if bw >= 100 else max(10.0, 10.0 * int(bw // 10))
    sx = min(ox, (pw - text_w) / 2)
    sy = oy + bh + 6.0
    n = int(L // 10)
    out.append('<g id="scale" fill="#000000" stroke="none">')
    out.append(f'<rect x="{_f(sx)}" y="{_f(sy)}" width="{_f(L)}" height="0.5"/>')
    for k in range(n + 1):
        th = 3.0 if k in (0, n) else 1.5
        x = sx + 10 * k - (0.25 if k == n else 0)
        out.append(f'<rect x="{_f(x)}" y="{_f(sy - th)}" width="0.25" height="{_f(th + 0.5)}"/>')
    out.append("</g>")
    font = f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{_f(fs)}" fill="#000000"'
    out.append(f'<text x="{_f(sx + L + 3)}" y="{_f(sy + 0.5)}" {font}>{L:.0f} mm</text>')
    for i, line in enumerate(lines):
        y = sy + 4 + fs * (1 + 5 * i / 3.0)
        out.append(f'<text x="{_f(sx)}" y="{_f(y)}" {font}>{line}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def rasterize(
    t: CharucoTarget, px_per_mm: float, margin_mm: float = 0.0, ss: int = 8
) -> np.ndarray:
    """Render the same geometry to a uint8 image (for tests): board only.

    Supersampled ``ss`` x then area-averaged. cv2.fillPoly fills the pixels
    whose centres lie inside a polygon or on its boundary, so vertices are
    shifted by -0.5 fine pixel: an edge at x covers fine pixels j < x.
    """
    squares, markers = charuco_shapes(t)
    bw, bh = (v * 1e3 for v in t.extent_m)
    k = px_per_mm * ss
    W, H = round((bw + 2 * margin_mm) * k), round((bh + 2 * margin_mm) * k)
    black = np.zeros((H, W), np.uint8)

    def pts(poly):
        return np.round((np.asarray(poly, float) + margin_mm) * k - 0.5).astype(np.int32)

    for x, y, w, h in squares:
        cv2.fillPoly(black, [pts([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])], 1)
    for loops in markers:
        m = np.zeros_like(black)
        for loop in loops:  # evenodd: XOR the loops
            one = np.zeros_like(black)
            cv2.fillPoly(one, [pts(loop)], 1)
            m ^= one
        black |= m
    fine = (255 * (1 - black)).astype(np.float32)
    return cv2.resize(fine, (W // ss, H // ss), interpolation=cv2.INTER_AREA).astype(np.uint8)
