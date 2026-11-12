"""Command-line entry point: ``metrogram <subcommand>``."""

from __future__ import annotations

import argparse
import sys

from metrogram import __version__
from metrogram.config import ConfigError, Device, describe, load_config


def _cmd_check_config(args: argparse.Namespace) -> int:
    status = 0
    for path in args.paths:
        try:
            print(describe(load_config(path)))
        except (ConfigError, OSError) as e:
            print(f"ERROR {e}", file=sys.stderr)
            status = 1
    return status


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from metrogram import calibration, calibration_io
    from metrogram.sources import Source
    from metrogram.targets import make_detector

    target = load_config(args.target)
    device = load_config(args.device) if args.device else None
    if device is not None and not isinstance(device, Device):
        raise ConfigError(f"{args.device} is not a device config")
    model = args.model or (device.colmap_model if device else "OPENCV")
    s = calibration.CalibrationSettings(
        model=model,
        stride=args.stride,
        min_points=args.min_points,
        sharpness_rel=args.sharpness_rel,
        max_views=args.max_views,
        holdout_frac=args.holdout,
        reject_factor=args.reject_factor,
        seed=args.seed,
    )
    source = Source(args.source)
    log = None if args.quiet else sys.stderr
    res = calibration.run(source, make_detector(target), s, log=log)
    if device is not None and res.image_size != (device.sensor.width_px, device.sensor.height_px):
        res.warnings.append(
            f"image size {res.image_size} != sensor {device.sensor.width_px}x"
            f"{device.sensor.height_px} of device '{device.name}' (video crop/scaling?): "
            "this calibration is only valid for this capture mode"
        )
    out = Path(args.out or f"runs/calib/{Path(args.source).stem}_{model.lower()}")
    meta = {
        "source": str(args.source),
        "device": device.name if device else None,
        "target": target.name,
        "focus_note": args.focus_note,
    }
    calibration_io.write(res, s, out, meta, source, save_frames=args.save_frames)
    print(calibration_io.summary(res))
    print(f"outputs in {out}/")
    return 0


def _cmd_extract_frames(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from metrogram import frames

    out = Path(args.out or f"data/frames/{Path(args.video).stem}")
    info = frames.extract(
        args.video,
        out,
        n=args.n,
        sharpness_rel=args.sharpness_rel,
        log=None if args.quiet else sys.stderr,
    )
    (out.parent / f"{out.name}_frames.json").write_text(json.dumps(info, indent=1))
    print(
        f"wrote {info['n_written']} frames to {out}/ "
        f"({info['n_segments_skipped_blurry']} segments skipped as blurry)"
    )
    return 0


def _cmd_sfm(args: argparse.Namespace) -> int:
    from pathlib import Path

    from metrogram import sfm
    from metrogram.config import Preset

    preset = load_config(args.preset)
    if not isinstance(preset, Preset):
        raise ConfigError(f"{args.preset} is not a preset config")
    camera = None
    if args.calibration:
        camera = sfm.CameraSpec.from_calibration(args.calibration)
    elif args.device:
        device = load_config(args.device)
        if not isinstance(device, Device):
            raise ConfigError(f"{args.device} is not a device config")
        files = sfm._list_images(Path(args.images))
        w, h = sfm._image_size(files[0])
        camera = sfm.CameraSpec.from_device(device, w, h)
    refine = args.refine_intrinsics or camera is None or not args.calibration
    s = sfm.settings_from_preset(preset, refine_intrinsics=refine)
    if args.matcher:
        s.matcher = args.matcher
    out = Path(args.out or f"runs/sfm/{Path(args.images).name}_{preset.name}")
    res = sfm.run(args.images, out, camera, s, log=None if args.quiet else sys.stderr)
    print(sfm.summary(res))
    print(f"outputs in {out}/ (model in {res.out_dir})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metrogram", description=__doc__)
    p.add_argument("--version", action="version", version=f"metrogram {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check-config", help="validate and summarize config files")
    c.add_argument("paths", nargs="+")
    c.set_defaults(func=_cmd_check_config)

    c = sub.add_parser(
        "calibrate",
        help="intrinsic calibration from a planar target (video, image dir, or RAW)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    c.add_argument("source", help="video file, directory of images/RAW, or single image")
    c.add_argument("-t", "--target", required=True, help="target config YAML")
    c.add_argument("-d", "--device", help="device config YAML (default model, sanity checks)")
    c.add_argument(
        "-m",
        "--model",
        choices=sorted(calibration_models()),
        help="camera model (COLMAP naming); default: device's colmap_model or OPENCV",
    )
    c.add_argument("-o", "--out", help="output dir (default runs/calib/<source>_<model>)")
    c.add_argument("--stride", type=int, default=1, help="use every n-th video frame / image")
    c.add_argument("--max-views", type=int, default=60, help="views kept after pose selection")
    c.add_argument("--min-points", type=int, default=24, help="min target points per view")
    c.add_argument(
        "--sharpness-rel", type=float, default=0.5, help="drop views below this x median sharpness"
    )
    c.add_argument("--holdout", type=float, default=0.2, help="fraction of views held out")
    c.add_argument(
        "--reject-factor", type=float, default=3.0, help="reject views above this x median RMS"
    )
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--focus-note", default="", help="focus setting/distance, stored in output")
    c.add_argument("--save-frames", action="store_true", help="write annotated used frames")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=_cmd_calibrate)

    c = sub.add_parser(
        "extract-frames",
        help="pick sharp, evenly spread frames from a video (for SfM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    c.add_argument("video")
    c.add_argument("-n", type=int, default=120, help="number of frames (segments)")
    c.add_argument("-o", "--out", help="output dir (default data/frames/<video>)")
    c.add_argument("--sharpness-rel", type=float, default=0.5)
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=_cmd_extract_frames)

    c = sub.add_parser(
        "sfm",
        help="sparse reconstruction (COLMAP) with calibrated or self-calibrated intrinsics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    c.add_argument("images", help="directory of images (PNG/JPEG/TIFF/RAW)")
    cam = c.add_mutually_exclusive_group()
    cam.add_argument("-c", "--calibration", help="calibration.yaml from 'metrogram calibrate'")
    cam.add_argument("-d", "--device", help="device config: nominal focal, self-calibration")
    c.add_argument("-p", "--preset", default="configs/presets/cpu-fast.yaml")
    c.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="let bundle adjustment refine focal length and distortion (always on without -c)",
    )
    c.add_argument("--matcher", choices=["exhaustive", "sequential"], help="override preset")
    c.add_argument("-o", "--out", help="output dir (default runs/sfm/<images>_<preset>)")
    c.add_argument("-q", "--quiet", action="store_true")
    c.set_defaults(func=_cmd_sfm)
    return p


def calibration_models():
    from metrogram.calibration import MODELS

    return MODELS


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError, OSError, RuntimeError) as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
