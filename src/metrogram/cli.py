"""Command-line entry point: ``metrogram <subcommand>``."""

from __future__ import annotations

import argparse
import sys

from metrogram import __version__
from metrogram.config import ConfigError, describe, load_config


def _cmd_check_config(args: argparse.Namespace) -> int:
    status = 0
    for path in args.paths:
        try:
            print(describe(load_config(path)))
        except (ConfigError, OSError) as e:
            print(f"ERROR {e}", file=sys.stderr)
            status = 1
    return status


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="metrogram", description=__doc__)
    p.add_argument("--version", action="version", version=f"metrogram {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check-config", help="validate and summarize config files")
    c.add_argument("paths", nargs="+")
    c.set_defaults(func=_cmd_check_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
