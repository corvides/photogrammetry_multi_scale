"""Compare AprilGrid corner-refinement modes (README section 7.2 figure).

Inputs: the RAW/image directory of a calibration set, and the outputs of
`metrogram calibrate --corner-refine full|lines|none` on it (one run dir per
mode). Writes one PNG with three panels:

1. per-view RMS for each mode (log scale);
2. offset of the edge-line corner from the X-junction corner, decomposed along
   the two tag edges meeting at the corner (outward > 0), in mm on the board;
3. a full-resolution crop of one junction with both corners.

Usage:
    PYTHONPATH= uv run python scripts/plot_corner_refine.py data/raw/r6_calib_40cm \\
        runs/calib/r6_calib_40cm_refine_{full,lines,none} -o out.png [--every 5]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from metrogram import plots  # noqa: F401  (house style: rcParams)
from metrogram.config import load_config
from metrogram.sources import Source, read_gray
from metrogram.targets import AprilGridDetector

plt = plots.plt
MODES = {
    "full": (".", plots.C_TRAIN),
    "lines": ("x", plots.C_HOLDOUT),
    "none": ("^", plots.C_MUTED),
}
TARGET = "configs/targets/aprilgrid_6x6_acrylic.yaml"


def offsets(path) -> list[tuple]:
    """Per matched corner: (offset along edge a, along edge b) [mm], px offset,
    file, point id, lines xy, full xy, mm per px."""
    t = load_config(TARGET)
    g = read_gray(path)
    a = AprilGridDetector(t, corner_refine="full").detect(g)
    b = AprilGridDetector(t, corner_refine="lines").detect(g)
    if a is None or b is None:
        return []
    full = {int(p): xy for p, xy in zip(a.point_ids, a.img, strict=True)}
    out = []
    for tag in np.unique(b.point_ids // 4):
        m = b.point_ids // 4 == tag
        if m.sum() != 4:
            continue
        q, pid = b.img[m].astype(np.float64), b.point_ids[m]
        mm_per_px = t.tag_size_m * 1e3 / np.mean(np.linalg.norm(q - np.roll(q, 1, 0), axis=1))
        for k in range(4):
            if int(pid[k]) not in full:
                continue
            d = q[k] - full[int(pid[k])]  # lines - junction [px]
            # outward directions along the two edges meeting at corner k
            ea = q[k] - q[(k + 1) % 4]
            eb = q[k] - q[k - 1]
            ea, eb = ea / np.linalg.norm(ea), eb / np.linalg.norm(eb)
            ca, cb = np.linalg.solve(np.column_stack([ea, eb]), d) * mm_per_px
            out.append(
                (
                    ca,
                    cb,
                    np.linalg.norm(d),
                    Path(path).name,
                    int(pid[k]),
                    q[k],
                    full[int(pid[k])],
                    mm_per_px,
                )
            )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("images")
    ap.add_argument("runs", nargs=3, help="run dirs for modes full, lines, none (in that order)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--every", type=int, default=5, help="use every n-th frame for panels 2-3")
    a = ap.parse_args()

    files = Source(a.images).files[:: a.every]
    with ProcessPoolExecutor(6) as ex:
        rows = [r for o in ex.map(offsets, files) for r in o]
    off = np.array([(r[0], r[1]) for r in rows])
    dist_px = np.array([r[2] for r in rows])

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(15, 4.8), constrained_layout=True, width_ratios=[1.1, 1, 0.9]
    )
    # 1. per-view RMS per mode
    for i, (mode, run) in enumerate(zip(MODES, a.runs, strict=True)):
        rep = json.loads((Path(run) / "report.json").read_text())
        y = np.array([v["rms_px"] for v in rep["views"] if v["status"] in ("train", "holdout")])
        x = i + np.random.default_rng(0).uniform(-0.18, 0.18, len(y))
        mk, col = MODES[mode]
        a1.scatter(x, y, marker=mk, color=col, s=40 if mk == "." else 22, linewidths=0.9)
        a1.text(i + 0.24, np.median(y), f"median\n{np.median(y):.2f} px", va="center", fontsize=8)
    a1.set_yscale("log")
    a1.set_xticks(range(3), [f"{m}" for m in MODES])
    a1.set_xlim(-0.5, 2.8)
    a1.set_ylabel("view RMS [px] (train + holdout views)")
    a1.set_title("Reprojection error per view, by --corner-refine", loc="left")
    a1.grid(axis="x", visible=False)

    # 2. corner offset along the two tag edges
    a2.scatter(off[:, 0], off[:, 1], s=6, marker=".", color=plots.C_TRAIN, alpha=0.5, lw=0)
    med = np.median(off, axis=0)
    a2.plot(*med, marker="x", color=plots.C_HOLDOUT, ms=12, mew=2.5)
    a2.axhline(0, color=plots.INK_2, lw=0.8)
    a2.axvline(0, color=plots.INK_2, lw=0.8)
    lim = max(0.5, np.percentile(np.abs(off), 99) * 1.1)
    a2.set_xlim(-lim, lim)
    a2.set_ylim(-lim, lim)
    a2.set_aspect("equal")
    a2.set_xlabel("along edge 1, outward [mm on the board]")
    a2.set_ylabel("along edge 2, outward [mm]")
    a2.set_title(
        f"Edge-line corner minus X-junction corner\n{len(off)} corners; median (x) "
        f"{med[0]:.2f} / {med[1]:.2f} mm per edge ({np.median(dist_px):.1f} px)",
        loc="left",
    )

    # 3. crop of one junction: the corner whose offset is closest to the median
    r = rows[int(np.argmin(np.linalg.norm(off - med, axis=1)))]
    g = read_gray(Path(a.images) / r[3])
    c = np.round(r[6]).astype(int)
    h = 22
    crop = g[c[1] - h : c[1] + h + 1, c[0] - h : c[0] + h + 1]
    a3.imshow(
        crop, cmap="gray", extent=(c[0] - h - 0.5, c[0] + h + 0.5, c[1] + h + 0.5, c[1] - h - 0.5)
    )
    a3.plot(*r[6], marker=".", color=plots.C_TRAIN, ms=14, ls="none", label="X-junction (full)")
    a3.plot(
        *r[5],
        marker="x",
        color=plots.C_HOLDOUT,
        ms=11,
        mew=2.5,
        ls="none",
        label="edge lines (lines)",
    )
    a3.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
    a3.grid(False)
    a3.set_xlabel("x [px]")
    a3.set_ylabel("y [px]")
    a3.set_title(f"{r[3]}, point {r[4]} (1 px ≈ {r[7] * 1e3:.0f} µm)", loc="left")
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}: {len(off)} corners, median offset {med.round(3)} mm")


if __name__ == "__main__":
    main()
