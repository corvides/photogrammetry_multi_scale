"""Render PLY layers (point clouds and meshes) as a Z-up turntable of PNGs, or interactively.

Design adapted from the user's Isaac Sim R&D viewer (view_recon.py): Open3D
offscreen rendering, fixed framing, black-frame sanity check. Added here:
several overlaid layers (a metrogram output dir loads all its PLYs), framing on
a region of interest instead of the whole (often room-sized) point cloud, and
optional cropping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Layers that define the region of interest when present (markers outputs).
ROI_LAYERS = ("board_frame", "markers")


@dataclass
class Layer:
    name: str
    geo: object  # open3d PointCloud or TriangleMesh
    is_mesh: bool

    def points(self) -> np.ndarray:
        return np.asarray(self.geo.vertices if self.is_mesh else self.geo.points)


def load(paths: list[str | Path]) -> list[Layer]:
    import open3d as o3d

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.ply")) if p.is_dir() else [p]
    if not files:
        raise ValueError(f"no .ply files in {paths}")
    bad = [f for f in files if f.suffix.lower() not in (".ply", ".obj")]
    if bad:
        raise ValueError(f"unsupported files (PLY or OBJ only): {bad}")
    layers = []
    for f in files:
        # OBJ: textured mesh from TextureMesh (texture loaded through the .mtl)
        mesh = o3d.io.read_triangle_mesh(str(f), enable_post_processing=f.suffix == ".obj")
        if len(mesh.triangles):
            mesh.compute_vertex_normals()
            if mesh.has_textures() and mesh.has_triangle_uvs():
                # OBJ/OpenGL UVs have V from the bottom; Open3D's renderer samples
                # from the top (verified: unflipped UVs hit only empty atlas).
                uv = np.asarray(mesh.triangle_uvs).copy()
                uv[:, 1] = 1.0 - uv[:, 1]
                mesh.triangle_uvs = o3d.utility.Vector2dVector(uv)
            layers.append(Layer(f.stem, mesh, True))
        else:
            pc = o3d.io.read_point_cloud(str(f))
            if len(pc.points) == 0:
                raise ValueError(f"{f} is empty")
            layers.append(Layer(f.stem, pc, False))
    return layers


def roi(layers: list[Layer], margin: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Region of interest: ROI layers' bbox (extended upward by the board size, for
    objects standing on it) grown by `margin` x its size; else the 1-99 %
    percentile box of everything (robust to stray far points)."""
    sel = [ly for ly in layers if ly.name in ROI_LAYERS]
    if sel:
        P = np.vstack([ly.points() for ly in sel])
        lo, hi = P.min(0), P.max(0)
        size = (hi - lo).max()
        # Objects stand ON the (flat) board: leave room above it, up to its size.
        hi[2] = max(hi[2], lo[2] + size)
        return lo - margin * size, hi + margin * size
    P = np.vstack([ly.points() for ly in layers])
    return np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)


def _crop_mesh(mesh, lo, hi):
    """Keep triangles with all vertices in the box, preserving UVs and textures
    (Open3D's TriangleMesh.crop drops the texture images)."""
    import open3d as o3d

    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles)
    inside = np.all((V >= lo) & (V <= hi), axis=1)
    keep = inside[T].all(axis=1)
    out = o3d.geometry.TriangleMesh(mesh.vertices, o3d.utility.Vector3iVector(T[keep]))
    if mesh.has_vertex_normals():
        out.vertex_normals = mesh.vertex_normals
    if mesh.has_vertex_colors():
        out.vertex_colors = mesh.vertex_colors
    if mesh.has_triangle_uvs():
        uv = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)[keep].reshape(-1, 2)
        out.triangle_uvs = o3d.utility.Vector2dVector(uv)
    if len(mesh.triangle_material_ids):
        ids = np.asarray(mesh.triangle_material_ids)[keep]
        out.triangle_material_ids = o3d.utility.IntVector(ids.astype(np.int32))
    out.textures = mesh.textures
    return out


def crop(layers: list[Layer], lo, hi) -> None:
    import open3d as o3d

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    box = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    for ly in layers:
        # never crop the overlays that define the frame (board, markers, cameras)
        if ly.name in ROI_LAYERS or ly.name == "cameras":
            continue
        ly.geo = _crop_mesh(ly.geo, lo, hi) if ly.is_mesh else ly.geo.crop(box)


def show(layers: list[Layer], title: str = "metrogram") -> None:
    import open3d as o3d

    o3d.visualization.draw_geometries([ly.geo for ly in layers], window_name=title)


def turntable(
    layers: list[Layer],
    out: Path,
    base: str,
    lo,
    hi,
    azimuths_deg,
    elevations_deg,
    size=(1280, 960),
    zoom: float = 1.9,
    point_size: float = 2.0,
) -> list[Path]:
    import open3d as o3d

    out.mkdir(parents=True, exist_ok=True)
    r = o3d.visualization.rendering.OffscreenRenderer(*size)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    for i, ly in enumerate(layers):
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit" if ly.is_mesh else "defaultUnlit"
        mat.point_size = point_size
        if ly.is_mesh and ly.geo.has_textures():
            mat.albedo_img = o3d.geometry.Image(np.asarray(ly.geo.textures[0]))
            mat.shader = "defaultUnlit"  # show the photo texture as captured
        r.scene.add_geometry(f"{i}_{ly.name}", ly.geo, mat)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre = (lo + hi) / 2
    dist = zoom * float(np.linalg.norm(hi - lo)) / 2
    written = []
    for i, (az, el) in enumerate(zip(azimuths_deg, elevations_deg, strict=True)):
        a, e = math.radians(az), math.radians(el)
        eye = centre + dist * np.array(
            [math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]
        )
        up = [0, 0, 1] if el < 89.9 else [0, 1, 0]
        r.setup_camera(50.0, centre, eye, up)
        img = r.render_to_image()
        f = out / f"{base}_view{i:02d}.png"
        o3d.io.write_image(str(f), img)
        if float(np.asarray(img).mean()) > 254.5:
            print(f"  WARNING: {f.name} is blank (nothing in view?)")
        written.append(f)
    return written
