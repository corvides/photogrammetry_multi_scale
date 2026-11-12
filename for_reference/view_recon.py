"""Render a .ply (mesh or point cloud) to PNGs, headlessly.

    /isaac-sim/python.sh view_recon.py sweep_49/recon/mesh.ply
    /isaac-sim/python.sh view_recon.py sweep_49/recon/cloud.ply --views 12

This server has no display, so there is no interactive viewer. Open3D's
OffscreenRenderer works here via Vulkan, so we render a turntable of fixed
viewpoints to PNG instead. To inspect interactively, copy the .ply to a local
machine (see --help epilogue).
"""

import argparse
import math
import os
import shutil
import subprocess
import time

import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser(
        epilog="Interactive alternative: scp the .ply locally and open it in "
               "MeshLab, CloudCompare, Blender, or python -c \"import open3d "
               "as o3d; o3d.visualization.draw_geometries([o3d.io."
               "read_triangle_mesh('mesh.ply')])\"")
    ap.add_argument("ply")
    ap.add_argument("--views", type=int, default=6,
                    help="evenly spaced turntable positions (ignored if "
                         "--azimuths is given)")
    ap.add_argument("--azimuths", default=None,
                    help="explicit azimuths in deg, e.g. 0,45,90,270 -- use this "
                         "when you want a specific viewpoint rather than a sweep")
    ap.add_argument("--elev", type=float, default=35.0,
                    help="elevation, deg. 90 = straight down, 0 = side-on")
    ap.add_argument("--elevs", default=None,
                    help="explicit elevations in deg, paired with --azimuths")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--zoom", type=float, default=1.9,
                    help="camera distance as a multiple of the object radius")
    ap.add_argument("--frame-from", default=None, metavar="PLY",
                    help="take the camera framing from THIS mesh instead of the "
                         "one being rendered; use it to render a growing "
                         "sequence from a fixed viewpoint")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window instead of rendering PNGs "
                         "(OpenGL viewer; works without the Vulkan offscreen path)")
    ap.add_argument("--out", default=None, help="output dir (default: next to the ply)")
    ap.add_argument("--record", nargs="?", const="auto", default=None, metavar="MP4",
                    help="record an interactive session to video. Opens the "
                         "viewer (implies --show), grabs a frame every "
                         "1/--record-fps seconds WHILE YOU DRAG, and assembles "
                         "an mp4 with ffmpeg when you close the window. Give a "
                         "filename, or omit it for <ply>_turn.mp4")
    ap.add_argument("--record-fps", type=float, default=12.0,
                    help="capture rate. Frames are PNGs at window resolution, "
                         "so 12 fps at 1280x960 is roughly 1 GB per minute -- "
                         "drop --width/--height for a long session")
    ap.add_argument("--record-max-s", type=float, default=300.0,
                    help="stop recording after this many seconds no matter "
                         "what. Without a display `poll_events()` never returns "
                         "False, so the loop would otherwise spin forever -- "
                         "which it did, on a headless test")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the PNGs after assembling the mp4")
    a = ap.parse_args()

    base = os.path.splitext(os.path.basename(a.ply))[0]
    out = a.out or os.path.dirname(os.path.abspath(a.ply))
    os.makedirs(out, exist_ok=True)

    geo = o3d.io.read_triangle_mesh(a.ply)
    is_mesh = len(geo.triangles) > 0
    if is_mesh:
        geo.compute_vertex_normals()
        n = len(geo.vertices)
        kind = f"mesh, {n} verts / {len(geo.triangles)} tris"
    else:
        geo = o3d.io.read_point_cloud(a.ply)
        n = len(geo.points)
        kind = f"point cloud, {n} pts"
    if n == 0:
        raise SystemExit(f"{a.ply} is empty")

    if a.record:
        if is_mesh:
            geo.compute_vertex_normals()
        mp4 = (a.record if a.record != "auto"
               else os.path.join(out, base + "_turn.mp4"))
        tmp = os.path.join(out, "_rec_" + base)
        os.makedirs(tmp, exist_ok=True)
        print(f"{a.ply}: {kind}")
        print(f"RECORDING to {mp4} at {a.record_fps:g} fps")
        print("  drag to rotate, scroll to zoom, CLOSE THE WINDOW (or q) to stop")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=os.path.basename(a.ply),
                          width=a.width, height=a.height)
        vis.add_geometry(geo)
        n, last, t0 = 0, 0.0, time.time()
        stopped = "window closed"
        try:
            while vis.poll_events():
                vis.update_renderer()
                now = time.time()
                if now - t0 > a.record_max_s:
                    stopped = f"--record-max-s ({a.record_max_s:g}s) reached"
                    break
                if now - last >= 1.0 / a.record_fps:
                    # do_render=True. The cheaper do_render=False reuses
                    # whatever is in the framebuffer, which depending on buffer
                    # swap timing can be an empty one -- it produced a wholly
                    # BLACK video, and the only hint was the file size.
                    vis.capture_screen_image(
                        os.path.join(tmp, f"rec_{n:05d}.png"), do_render=True)
                    n += 1
                    last = now
        except KeyboardInterrupt:
            stopped = "interrupted"
        finally:
            vis.destroy_window()
        print(f"  stopped: {stopped}")
        if n == 0:
            raise SystemExit("  no frames captured")
        print(f"  captured {n} frames ({n / a.record_fps:.1f} s)")
        # Is there anything IN them? A valid mp4 of uniform black is still a
        # valid mp4, and that is exactly what shipped the first time.
        _probe = np.asarray(o3d.io.read_image(
            os.path.join(tmp, f"rec_{n // 2:05d}.png")))
        _mean = float(_probe.mean())
        if _mean < 5.0:
            print(f"  *** WARNING: frames are essentially BLACK (mean pixel "
                  f"{_mean:.2f}/255). The renderer drew nothing -- on a headless "
                  f"machine this is expected; on a desktop check the window was "
                  f"actually visible and not minimised or occluded.")
        else:
            print(f"  frame brightness OK (mean pixel {_mean:.1f}/255)")
        cmd = ["ffmpeg", "-y", "-framerate", str(a.record_fps),
               "-i", os.path.join(tmp, "rec_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4]
        if shutil.which("ffmpeg"):
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"-> {mp4}")
            if not a.keep_frames:
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                print(f"   frames kept in {tmp}")
        else:
            print("  ffmpeg not found. Frames are in " + tmp + "; assemble with:")
            print("   " + " ".join(cmd))
        return

    if a.show:
        if is_mesh:
            geo.compute_vertex_normals()
        print(f"{a.ply}: {kind}")
        print("interactive viewer -- drag to rotate, scroll to zoom, q to quit")
        o3d.visualization.draw_geometries([geo], window_name=os.path.basename(a.ply))
        return

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit" if is_mesh else "defaultUnlit"
    mat.point_size = a.point_size

    lo = geo.get_min_bound()
    hi = geo.get_max_bound()
    centre = (lo + hi) / 2
    if a.frame_from:
        # Fixed framing from a reference mesh, so a SEQUENCE of growing
        # reconstructions is rendered from one unmoving camera. Auto-framing
        # each mesh separately would make the camera lurch every frame.
        ref = o3d.io.read_triangle_mesh(a.frame_from)
        rv = np.asarray(ref.vertices)
        if len(rv) == 0:
            rv = np.asarray(o3d.io.read_point_cloud(a.frame_from).points)
        lo, hi = rv.min(0), rv.max(0)
        centre = (lo + hi) / 2
        print(f"  framing fixed from {os.path.basename(a.frame_from)}")
    radius = float(np.linalg.norm(hi - lo)) / 2
    dist = radius * a.zoom
    print(f"{a.ply}: {kind}")
    print(f"  bounds {np.round(lo, 4)} .. {np.round(hi, 4)}  (radius {radius * 1000:.1f} mm)")

    r = o3d.visualization.rendering.OffscreenRenderer(a.width, a.height)
    r.scene.add_geometry("g", geo, mat)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])

    if a.azimuths:
        az_list = [math.radians(float(x)) for x in a.azimuths.split(",")]
    else:
        az_list = [2 * math.pi * i / a.views for i in range(a.views)]
    if a.elevs:
        el_list = [math.radians(float(x)) for x in a.elevs.split(",")]
        if len(el_list) == 1:
            el_list *= len(az_list)
    else:
        el_list = [math.radians(a.elev)] * len(az_list)
    if len(el_list) != len(az_list):
        raise SystemExit("--azimuths and --elevs must have the same length")

    for i, (az, el) in enumerate(zip(az_list, el_list)):
        eye = centre + dist * np.array([math.cos(az) * math.cos(el),
                                        math.sin(az) * math.cos(el),
                                        math.sin(el)])
        r.setup_camera(50.0, centre, eye, [0, 0, 1])
        name = f"{base}_view{i:02d}.png"
        o3d.io.write_image(os.path.join(out, name), r.render_to_image())
        print(f"  {name}  azimuth {math.degrees(az):6.1f}  elev {math.degrees(el):5.1f} deg")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
