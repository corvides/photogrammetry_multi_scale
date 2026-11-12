"""Small geometry helpers shared by several stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Similarity:
    """dst ≈ scale * R @ src + t"""

    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, pts: np.ndarray) -> np.ndarray:
        return self.scale * pts @ self.R.T + self.t

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Similarity:
    """Least-squares similarity (or rigid) transform src -> dst (Umeyama 1991)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or len(src) < 3:
        raise ValueError("need two (N, d) arrays with N >= 3")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(src.shape[1])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return Similarity(scale, R, t)
