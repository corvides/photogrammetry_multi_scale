#!/usr/bin/env bash
# Build OpenMVS (+ CGAL headers + VCG) into a dedicated prefix. No sudo, no system
# changes: it writes only to --src and --prefix, never touches shell rc files,
# and its environment changes die with the script's own process.
# Mirrors upstream's docker/buildInDocker.sh of the chosen release, pinned to
# exact commits.
#
#   scripts/build_openmvs.sh                      # CPU build (default v2.3.0)
#   scripts/build_openmvs.sh --cuda --arch 120    # GPU server (Blackwell, CUDA >= 12.8)
#
# Options: --version v2.3.0|v2.4.0  --prefix DIR (default ~/opt/openmvs)
#          --src DIR (default ~/src/openmvs)  --jobs N (default nproc)
#
# Versions:
#   v2.3.0 (default)  upstream-built on Ubuntu 22.04: CGAL 5.4, system OpenCV 4.5,
#                     no JPEG XL, old nanoflann API. Matches the laptop.
#   v2.4.0            needs OpenCV >= 4.11 (unconditional cv::IMWRITE_JPEGXL_QUALITY)
#                     and nanoflann >= 1.5 (SearchParameters): NOT Ubuntu 22.04's
#                     apt packages (nor 24.04's OpenCV 4.6).
#
# Why a dedicated prefix (not ~/.local): CMake searches <dir>/.. of every $PATH
# entry ending in /bin, so a CGAL installed in ~/.local could be picked up by
# unrelated builds (e.g. colcon/ROS). ~/opt/openmvs is searched by nobody, and
# CGAL goes to its own sub-prefix passed explicitly via CGAL_DIR.
#
# Uninstall: rm -rf <prefix> <src>
#
# apt prerequisites (Ubuntu 22.04; all already present on the laptop):
#   build-essential git cmake libeigen3-dev(>=3.4) libopencv-dev
#   libboost-{iostreams,program-options,system,serialization,thread}-dev
#   libgmp-dev libmpfr-dev zlib1g-dev libpng-dev libjpeg-dev libtiff-dev
#   libglu1-mesa-dev libglew-dev libglfw3-dev
# Binaries end up in <prefix>/bin/OpenMVS (metrogram looks there, then in
# $OPENMVS_BIN, then on $PATH).

# Refuse to be sourced: the environment edits below must not leak into your shell.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "run this script, do not source it" >&2
  return 1
fi
set -euo pipefail

VERSION=v2.3.0 CUDA=OFF ARCH=native PREFIX="$HOME/opt/openmvs" SRC="$HOME/src/openmvs"
JOBS=$(nproc)
while [[ $# -gt 0 ]]; do
  case $1 in
    --version) VERSION=$2; shift 2 ;;
    --cuda) CUDA=ON; shift ;;
    --arch) ARCH=$2; shift 2 ;;
    --prefix) PREFIX=$2; shift 2 ;;
    --src) SRC=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

# Pinned sources per release: url ref commit
case $VERSION in
  v2.3.0)
    OPENMVS=(https://github.com/cdcseacave/openMVS.git v2.3.0 ce03889e2055a3d6cbcae9dd5a93b5eeea94b909)
    CGAL=(https://github.com/CGAL/cgal.git v5.4 c58ac97e93c838ebfb1e8adaf23ff4fd185dc8e4)
    # last VCG commit before the v2.3.0 release (2024-01-07)
    VCG=(https://github.com/cdcseacave/VCG.git 2022-07-05 10d73fc962f99bf5e3ca2c26ee1c8d417b0c3c9a)
    ;;
  v2.4.0)
    OPENMVS=(https://github.com/cdcseacave/openMVS.git v2.4.0 58117204c86bbb11a0b25b26a8987676cf11274d)
    CGAL=(https://github.com/CGAL/cgal.git v6.0.1 50cfbde3b84dbeae8338268db2d78fe4fcb522de)
    VCG=(https://github.com/cdcseacave/VCG.git 2025-08-24 658ba36d0a5666650da6e066b4794efc5a463407)
    ;;
  *) echo "unsupported --version $VERSION (v2.3.0 | v2.4.0)" >&2; exit 1 ;;
esac

# Keep your CMAKE_PREFIX_PATH, minus ROS entries (ROS ships its own copies of
# some packages). Only this process and its children see the change.
CMAKE_PREFIX_PATH=$(tr ':' '\n' <<<"${CMAKE_PREFIX_PATH:-}" | grep -v '^/opt/ros/' | paste -sd: - || true)
export CMAKE_PREFIX_PATH

mkdir -p "$SRC" "$PREFIX"
[[ -w "$PREFIX" ]] || { echo "prefix $PREFIX is not writable (this script never uses sudo)" >&2; exit 1; }
cd "$SRC"
echo ">> OpenMVS $VERSION, sources in $SRC, install prefix $PREFIX, CUDA=$CUDA (arch $ARCH)"
echo ">> CMAKE_PREFIX_PATH for this build: '${CMAKE_PREFIX_PATH}'"

# Clone a pinned commit into <dir>; an existing checkout must be at that commit.
fetch() {
  local dir=$1 url=$2 ref=$3 sha=$4
  if [[ ! -d $dir ]]; then
    git init -q "$dir"
    git -C "$dir" fetch -q --depth 1 "$url" "$sha"
    git -C "$dir" checkout -q FETCH_HEAD
  fi
  local head
  head=$(git -C "$dir" rev-parse HEAD)
  [[ $head == "$sha" ]] || { echo "$dir is at $head, expected $sha ($ref)" >&2; exit 1; }
  echo ">> $dir @ ${ref} (${sha:0:12}) verified"
}

CGAL_DIR_SRC="cgal-${CGAL[1]}"
VCG_DIR="vcglib-${VCG[2]:0:12}"
OMVS_DIR="openMVS-$VERSION"
CGAL_PREFIX="$PREFIX/deps/cgal-${CGAL[1]}"

# 1. CGAL (header-only: configure + install into its own sub-prefix)
fetch "$CGAL_DIR_SRC" "${CGAL[@]}"
cmake -S "$CGAL_DIR_SRC" -B "$CGAL_DIR_SRC-build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CGAL_PREFIX" >/dev/null
cmake --install "$CGAL_DIR_SRC-build" >/dev/null
CGAL_CONFIG_DIR=$(dirname "$(find "$CGAL_PREFIX" -name CGALConfig.cmake | head -1)")

# 2. VCG (header-only, used in place)
fetch "$VCG_DIR" "${VCG[@]}"

# 3. OpenMVS
fetch "$OMVS_DIR" "${OPENMVS[@]}"
# v2.4.0 only: JPEG XL is meant to be optional (libs/IO/CMakeLists.txt has a
# "support will be disabled" branch, JXL code is behind #ifdef _IMAGE_JXL), but
# the macro passes REQUIRED. Patch the working tree only when libjxl is absent.
IO_CMAKE="$OMVS_DIR/libs/IO/CMakeLists.txt"
if grep -q libjxl "$IO_CMAKE" && ! pkg-config --exists libjxl; then
  sed -i 's/pkg_check_modules(${PREFIX} REQUIRED IMPORTED_TARGET/pkg_check_modules(${PREFIX} IMPORTED_TARGET/' "$IO_CMAKE"
  grep -q 'pkg_check_modules(${PREFIX} IMPORTED_TARGET' "$IO_CMAKE" \
    || { echo "libjxl patch did not apply to $IO_CMAKE" >&2; exit 1; }
  echo ">> libjxl not found: patched $IO_CMAKE to make it optional (JPEG XL disabled)"
fi

CUDA_ARGS=(-DOpenMVS_USE_CUDA="$CUDA")
[[ $CUDA == ON ]] && CUDA_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
cmake -S "$OMVS_DIR" -B "$OMVS_DIR-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCGAL_DIR="$CGAL_CONFIG_DIR" \
  -DVCG_ROOT="$SRC/$VCG_DIR" \
  -DOpenMVS_USE_PYTHON=OFF \
  "${CUDA_ARGS[@]}"
cmake --build "$OMVS_DIR-build" -j "$JOBS"
cmake --install "$OMVS_DIR-build"

echo ">> done. Binaries in $PREFIX/bin/OpenMVS:"
ls "$PREFIX/bin/OpenMVS"
echo ">> uninstall: rm -rf '$PREFIX' '$SRC'"
