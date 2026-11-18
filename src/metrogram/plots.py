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


# Status is encoded by marker shape (readable in greyscale / for colour-blind
# readers); colour repeats it.
MARK = {"train": ".", "holdout": "x", "rejected": "v"}
COLOR = {"train": C_TRAIN, "holdout": C_HOLDOUT, "rejected": C_MUTED}


def _image_axes(ax, size):
    w, h = size
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")


def _scatter(ax, x, y, name, size=12, **kw):
    """Status-coded scatter: '.' train, 'x' holdout (x needs a line width)."""
    lw = 0.8 if MARK[name] in "x+" else 0
    return ax.scatter(x, y, s=size, marker=MARK[name], color=COLOR[name], linewidths=lw, **kw)


def coverage(result, path: Path):
    """Where on the sensor the training points fall.

    The sensor is cut into a coarse grid of equal cells (default 8 x 6, i.e.
    684 x 608 px on a 5472 x 3648 sensor). The cells are image regions, not
    board tags. A cell is "occupied" when at least one training point falls in
    it; the number in each cell is its point count. Distortion is only
    constrained where there are points, so empty cells (hatched), especially
    the corners, mean extrapolated distortion there.
    """
    fig, ax = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    gx, gy = result.coverage["grid"]
    w, h = result.image_size
    counts = np.array(result.coverage["counts"])
    ax.imshow(
        counts,
        extent=(0, w, h, 0),
        cmap="Greys",
        vmin=0,
        vmax=max(counts.max(), 1) * 2.5,  # keep cells pale so the points stay visible
        zorder=0,
    )
    cw, ch = w / gx, h / gy
    for j in range(gy):
        for i in range(gx):
            n = int(counts[j, i])
            if n == 0:
                ax.add_patch(
                    matplotlib.patches.Rectangle(
                        (i * cw, j * ch), cw, ch, fill=False, hatch="///", ec=C_HOLDOUT, lw=0
                    )
                )
            ax.text(
                (i + 0.5) * cw,
                (j + 0.5) * ch,
                str(n),
                ha="center",
                va="center",
                fontsize=11,
                color=INK_2,
                alpha=0.8,
                zorder=1,
            )
    for name in ("train", "holdout"):
        r = result.residuals.get(name)
        if r is not None:
            _scatter(ax, r[:, 0], r[:, 1], name, size=10, label=f"{name} ({len(r)} pts)", zorder=2)
    for i in range(1, gx):
        ax.axvline(i * cw, color=INK_2, lw=0.5, alpha=0.5)
    for j in range(1, gy):
        ax.axhline(j * ch, color=INK_2, lw=0.5, alpha=0.5)
    ax.grid(False)
    _image_axes(ax, result.image_size)
    ax.set_title(
        f"Sensor coverage: {gx}x{gy} image cells of {cw:.0f}x{ch:.0f} px; number = train "
        f"points per cell; {result.coverage['occupied_fraction']:.0%} occupied "
        "(hatched = empty)",
        loc="left",
        fontsize=8.5,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2, markerscale=1.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def residuals(result, path: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    cx, cy = result.K[0, 2], result.K[1, 2]
    rmax = np.hypot(*result.image_size) / 2
    lim = 0.0
    for name in ("train", "holdout"):
        r = result.residuals.get(name)
        if r is None:
            continue
        _scatter(a1, r[:, 2], r[:, 3], name, alpha=0.6, label=name)
        lim = max(lim, np.percentile(np.abs(r[:, 2:]), 99.5))
        rad = np.hypot(r[:, 0] - cx, r[:, 1] - cy) / rmax
        _scatter(a2, rad, np.hypot(r[:, 2], r[:, 3]), name, alpha=0.6, label=name)
    lim = max(lim, 0.1) * 1.1
    a1.set_xlim(-lim, lim)
    a1.set_ylim(lim, -lim)
    a1.set_aspect("equal")
    a1.set_xlabel("dx [px]")
    a1.set_ylabel("dy [px]")
    a1.set_title("Reprojection residuals (projected - detected), 99.5 % shown", loc="left")
    a1.legend(markerscale=2)
    a2.set_xlabel("distance from principal point / half diagonal")
    a2.set_ylabel("|residual| [px]")
    a2.set_title(
        "Residual vs radius (a trend means the distortion model is too simple)", loc="left"
    )
    a2.set_ylim(bottom=0)
    a2.legend(markerscale=2)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_view(result, path: Path):
    vs = [v for v in result.views if v.rms_px is not None]
    vs.sort(key=lambda v: v.index)
    fig, ax = plt.subplots(figsize=(max(9, 0.12 * len(vs) + 2), 4.3), constrained_layout=True)
    x = np.arange(len(vs))
    y = np.array([v.rms_px for v in vs])
    ax.vlines(x, 0, y, color="#d6d5d0", lw=1, zorder=1)
    for name in ("train", "holdout", "rejected"):
        m = np.array([v.status == name for v in vs])
        if m.any():
            _scatter(ax, x[m], y[m], name, size=60 if name == "train" else 28, label=name, zorder=3)
    ax.axhline(result.train_rms_px, color=INK_2, lw=1, ls="--", label="train RMS")
    core = (result.stats.get("train") or {}).get("core_threshold_px")
    if core:
        ax.axhline(core, color=INK_2, lw=1, ls=":", label="outlier threshold")
    step = max(1, len(vs) // 25)
    ax.set_xticks(x[::step], [v.key for v in vs][::step], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("view RMS [px]")
    ax.set_ylim(0, max(1.3 * float(y.max()), 1.1 * (core or 0)))  # legend headroom
    ax.set_title(
        "Per-view reprojection error\n(train: fitted views; holdout: intrinsics frozen, "
        "pose only; rejected: outliers dropped from the fit)",
        loc="left",
        fontsize=8.5,
    )
    ax.legend(ncols=5, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _nice(x: float) -> float:
    """Round down to 1, 2 or 5 x 10^n."""
    e = 10 ** np.floor(np.log10(x))
    return float(max(m for m in (1, 2, 5) if m * e <= x) * e)


def distortion(result, path: Path):
    """Lens distortion of the fitted model, in sensor pixels.

    For a pixel grid, each arrow starts at the ideal pinhole position (where a
    distortion-free lens with the same K would image the point) and ends at
    the actual (distorted) position. Arrow lengths are magnified (gain in the
    key); the colour map gives the true displacement in pixels. Right panel:
    the radial profile, in px and in % of the radius.
    """
    w, h = result.image_size
    K, dist = result.K, result.dist
    cx, cy = K[0, 2], K[1, 2]
    xs, ys = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(0, h - 1, 48))
    pts = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    und = cv2.undistortPoints(pts[:, None], K, dist, P=K).reshape(-1, 2)
    mag = np.hypot(*(pts - und).T).reshape(48, 64)

    fig, (ax, a2) = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True, width_ratios=[1.55, 1]
    )
    im = ax.imshow(mag, extent=(0, w, h, 0), cmap="Blues", origin="upper")
    cs = ax.contour(xs, ys, mag, levels=8, colors=INK_2, linewidths=0.6)
    ax.clabel(cs, fmt="%.0f px", fontsize=7)
    # Arrows on a coarse grid: ideal -> distorted, magnified.
    ax_, ay_ = np.meshgrid(np.linspace(0.04, 0.96, 12) * w, np.linspace(0.06, 0.94, 8) * h)
    q = np.column_stack([ax_.ravel(), ay_.ravel()])
    qd = cv2.projectPoints(
        np.column_stack([(q - [cx, cy]) / [K[0, 0], K[1, 1]], np.ones(len(q))]),
        np.zeros(3),
        np.zeros(3),
        K,
        dist,
    )[0].reshape(-1, 2)
    d = qd - q
    dmax = float(np.hypot(*d.T).max()) or 1.0
    gain = _nice(0.07 * w / dmax) if dmax > 0 else 1.0
    Q = ax.quiver(
        q[:, 0],
        q[:, 1],
        d[:, 0] * gain,
        d[:, 1] * gain,
        color=INK,
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        zorder=3,
    )
    ref = _nice(dmax)
    ax.quiverkey(
        Q,
        0.80,
        1.035,
        ref * gain,
        f"{ref:g} px (arrows x{gain:g})",
        labelpos="E",
        fontproperties={"size": 8},
    )
    # Convex hull of the training points: outside it the model is extrapolated.
    r = result.residuals.get("train")
    rdata = None
    if r is not None and len(r) >= 3:
        hull = cv2.convexHull(r[:, :2].astype(np.float32)).reshape(-1, 2)
        rdata = float(np.hypot(r[:, 0] - cx, r[:, 1] - cy).max())
        hull = np.vstack([hull, hull[:1]])
        ax.plot(
            hull[:, 0], hull[:, 1], color=C_HOLDOUT, lw=1.4, ls="--", label="hull of train points"
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.plot(cx, cy, marker="+", color=C_HOLDOUT, ms=10, mew=1.5)
    ax.grid(False)
    _image_axes(ax, result.image_size)
    fig.colorbar(im, ax=ax, label="|distorted - ideal| [px]", shrink=0.8)
    ax.set_title(f"Lens distortion ({result.model}): ideal -> imaged position", loc="left")

    # Radial profile along the diagonal to the farthest image corner.
    corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], float)
    far = corners[np.argmax(np.hypot(corners[:, 0] - cx, corners[:, 1] - cy))]
    t = np.linspace(1e-3, 1, 200)[:, None]
    ideal = np.array([cx, cy]) + t * (far - [cx, cy])
    dd = cv2.projectPoints(
        np.column_stack([(ideal - [cx, cy]) / [K[0, 0], K[1, 1]], np.ones(len(ideal))]),
        np.zeros(3),
        np.zeros(3),
        K,
        dist,
    )[0].reshape(-1, 2)
    rad_i = np.hypot(*(ideal - [cx, cy]).T)
    rad_d = np.hypot(*(dd - [cx, cy]).T)
    a2.plot(rad_i, rad_d - rad_i, color=C_TRAIN, lw=2)
    a2.set_xlabel("ideal radius from principal point [px]")
    a2.set_ylabel("radial displacement [px] (< 0: barrel)")
    a2.axhline(0, color=INK_2, lw=0.6)
    if rdata is not None:
        a2.axvspan(rdata, rad_i.max(), color=C_HOLDOUT, alpha=0.12, lw=0)
        a2.axvline(rdata, color=C_HOLDOUT, lw=1.4, ls="--")
        a2.text(rdata, a2.get_ylim()[1], " extrapolated", color=C_HOLDOUT, va="top", fontsize=8)
    sec = a2.secondary_yaxis(
        "right", functions=(lambda y: 100 * y / rad_i.max(), lambda p: p * rad_i.max() / 100)
    )
    sec.set_ylabel("% of corner radius")
    a2.set_title("Radial profile (centre -> farthest corner)", loc="left")
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


def markers(res, path: Path):
    from metrogram.markers import ROLE_RGB

    col = {r: "#{:02x}{:02x}{:02x}".format(*c) for r, c in ROLE_RGB.items()}
    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(17, 5.4), constrained_layout=True, width_ratios=[1, 1.15, 0.9]
    )
    pts = res.points
    B = np.array([p.board for p in pts]) * 1e3
    E = np.array([p.error_m for p in pts]) * 1e3
    span = np.ptp(B[:, :2], axis=0).max()
    emax = max(np.linalg.norm(E[:, :2], axis=1).max(), 1e-9)
    gain = 0.08 * span / emax  # longest arrow = 8 % of the board
    for role in ("control", "check", "rejected"):
        m = np.array([p.role == role for p in pts])
        if not m.any():
            continue
        a1.scatter(B[m, 0], B[m, 1], s=10, color=col[role], label=role, zorder=3)
        a1.quiver(
            B[m, 0],
            B[m, 1],
            E[m, 0] * gain,
            E[m, 1] * gain,
            color=col[role],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
        )
    a1.set_title(f"In-plane error, arrows x{gain:.0f} (longest {emax:.2f} mm)", loc="left")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)

    zmax = max(np.abs(E[:, 2]).max(), 1e-9)
    sc = a2.scatter(
        B[:, 0],
        B[:, 1],
        c=E[:, 2],
        cmap="coolwarm",
        vmin=-zmax,
        vmax=zmax,
        s=28,
        edgecolors=[col[p.role] for p in pts],
        linewidths=0.8,
    )
    fig.colorbar(sc, ax=a2, label="out-of-plane error z [mm]", shrink=0.85)
    a2.set_title("Out-of-plane error (pattern = bowl/tilt, noise = random)", loc="left")
    for a in (a1, a2):
        a.set_aspect("equal")
        a.set_xlabel("board x [mm]")
        a.set_ylabel("board y [mm]")

    bins = np.linspace(0, np.linalg.norm(E, axis=1).max() * 1.05, 25)
    for role in ("control", "check"):
        e = [np.linalg.norm(p.error_m) * 1e3 for p in pts if p.role == role]
        if e:
            a3.hist(
                e,
                bins=bins,
                histtype="step",
                linewidth=2,
                color=col[role],
                label=f"{role} ({len(e)})",
            )
    a3.set_xlabel("3D error [mm]")
    a3.set_ylabel("points")
    a3.set_title("3D error (check = not used to align)", loc="left")
    a3.legend()
    a3.grid(axis="x", visible=False)
    fig.savefig(path, dpi=130)
    plt.close(fig)
