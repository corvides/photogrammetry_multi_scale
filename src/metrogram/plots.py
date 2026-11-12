"""Diagnostic plots (static PNG, headless)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import matplotlib.lines
import matplotlib.patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots (light mode) + neutral inks.
C_TRAIN = "#2a78d6"
C_HOLDOUT = "#eb6834"
C_MUTED = "#9a9893"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 9,
    }
)


def _image_axes(ax, size):
    w, h = size
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")


def coverage(result, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    gx, gy = result.coverage["grid"]
    w, h = result.image_size
    counts = np.array(result.coverage["counts"])
    ax.imshow(
        counts > 0,
        extent=(0, w, h, 0),
        cmap=matplotlib.colors.ListedColormap(["#f6e3dc", "#eef4fc"]),
        alpha=1.0,
        zorder=0,
    )
    for name, color in (("train", C_TRAIN), ("holdout", C_HOLDOUT)):
        r = result.residuals.get(name)
        if r is not None:
            ax.scatter(r[:, 0], r[:, 1], s=2, color=color, label=f"{name} ({len(r)} pts)", lw=0)
    for i in range(1, gx):
        ax.axvline(i * w / gx, color=INK_2, lw=0.4, alpha=0.4)
    for j in range(1, gy):
        ax.axhline(j * h / gy, color=INK_2, lw=0.4, alpha=0.4)
    ax.grid(False)
    _image_axes(ax, result.image_size)
    ax.set_title(
        f"Target point coverage: {result.coverage['occupied_fraction']:.0%} of cells "
        "(pale red = empty cell)",
        loc="left",
    )
    ax.legend(loc="upper right", markerscale=4)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def residuals(result, path: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    cx, cy = result.K[0, 2], result.K[1, 2]
    rmax = np.hypot(*result.image_size) / 2
    lim = 0.0
    for name, color in (("train", C_TRAIN), ("holdout", C_HOLDOUT)):
        r = result.residuals.get(name)
        if r is None:
            continue
        a1.scatter(r[:, 2], r[:, 3], s=3, color=color, alpha=0.5, lw=0, label=name)
        lim = max(lim, np.percentile(np.abs(r[:, 2:]), 99.5))
        rad = np.hypot(r[:, 0] - cx, r[:, 1] - cy) / rmax
        a2.scatter(rad, np.hypot(r[:, 2], r[:, 3]), s=3, color=color, alpha=0.5, lw=0, label=name)
    lim = max(lim, 0.1) * 1.1
    a1.set_xlim(-lim, lim)
    a1.set_ylim(lim, -lim)
    a1.set_aspect("equal")
    a1.set_xlabel("dx [px]")
    a1.set_ylabel("dy [px]")
    a1.set_title("Reprojection residuals (projected - detected)", loc="left")
    a1.legend(markerscale=4)
    a2.set_xlabel("distance from principal point / half diagonal")
    a2.set_ylabel("|residual| [px]")
    a2.set_title(
        "Residual vs radius (a trend means the distortion model is too simple)", loc="left"
    )
    a2.set_ylim(bottom=0)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_view(result, path: Path):
    vs = [v for v in result.views if v.rms_px is not None]
    vs.sort(key=lambda v: v.index)
    colors = {"train": C_TRAIN, "holdout": C_HOLDOUT, "rejected": C_MUTED}
    fig, ax = plt.subplots(figsize=(max(6, 0.12 * len(vs) + 2), 4), constrained_layout=True)
    x = np.arange(len(vs))
    ax.bar(x, [v.rms_px for v in vs], color=[colors[v.status] for v in vs], width=0.8)
    ax.axhline(result.train_rms_px, color=INK_2, lw=1, ls="--")
    handles = [matplotlib.patches.Patch(color=c, label=n) for n, c in colors.items()]
    handles.append(matplotlib.lines.Line2D([], [], color=INK_2, ls="--", label="train RMS"))
    step = max(1, len(vs) // 25)
    ax.set_xticks(x[::step], [v.key for v in vs][::step], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("view RMS [px]")
    ax.set_title("Per-view reprojection error", loc="left")
    ax.legend(handles=handles, ncols=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def distortion(result, path: Path):
    w, h = result.image_size
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(0, h - 1, 48))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    und = cv2.undistortPoints(pts[:, None], result.K, result.dist, P=result.K).reshape(-1, 2)
    disp = (pts - und).reshape(48, 64, 2)
    mag = np.hypot(disp[..., 0], disp[..., 1])
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    im = ax.imshow(mag, extent=(0, w, h, 0), cmap="Blues", origin="upper")
    cs = ax.contour(xs, ys, mag, levels=8, colors=INK_2, linewidths=0.6)
    ax.clabel(cs, fmt="%.0f px", fontsize=7)
    s = 6
    # True-scale arrows (1 px displacement = 1 px in image coordinates).
    ax.quiver(
        xs[::s, ::s],
        ys[::s, ::s],
        disp[::s, ::s, 0],
        disp[::s, ::s, 1],
        color=INK,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002,
    )
    # Outline of the region actually covered by target points: outside it the
    # model is extrapolated.
    r = result.residuals.get("train")
    if r is not None and len(r) >= 3:
        hull = cv2.convexHull(r[:, :2].astype(np.float32)).reshape(-1, 2)
        hull = np.vstack([hull, hull[:1]])
        ax.plot(hull[:, 0], hull[:, 1], color=C_HOLDOUT, lw=1.2, ls="--", label="data region")
        ax.legend(loc="upper right")
    ax.grid(False)
    _image_axes(ax, result.image_size)
    fig.colorbar(im, ax=ax, label="distortion displacement [px]", shrink=0.8)
    ax.set_title(
        f"Lens distortion ({result.model}), arrows true scale;"
        " extrapolated outside the dashed hull",
        loc="left",
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def all_plots(result, out: Path) -> list[str]:
    files = {
        "coverage.png": coverage,
        "residuals.png": residuals,
        "per_view_error.png": per_view,
        "distortion.png": distortion,
    }
    for name, fn in files.items():
        fn(result, out / name)
    return list(files)
