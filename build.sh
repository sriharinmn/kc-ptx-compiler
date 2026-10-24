#!/usr/bin/env bash
# Configure and build the project. Run this inside WSL.
#
# The build folder lives in the Linux filesystem (~/kc-build), not next to the
# sources: building on /mnt/c is slow, and it keeps build junk out of OneDrive.
set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${KC_BUILD_DIR:-$HOME/kc-build}"

cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCUDAToolkit_ROOT=/usr/local/cuda > /dev/null
cmake --build "$BUILD"
