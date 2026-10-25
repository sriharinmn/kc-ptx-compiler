#!/usr/bin/env bash
# Build, run the GPU tests, then compile every example and check it with the
# offline PTX assembler (ptxas). Run this inside WSL. Pass -v to see the JIT logs.
set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${KC_BUILD_DIR:-$HOME/kc-build}"

"$SRC/build.sh"
echo
"$BUILD/kc-tests" "$@"

# Every example must assemble with ptxas. cubin-info then reads the register
# count back out of the cubin's ELF sections; it must agree with ptxas -v.
echo
echo "ptxas check of examples/ (registers per thread on sm_89, from ptxas -v and from cubin-info):"
mkdir -p "$BUILD/examples"
for f in "$SRC"/examples/*.kc; do
  out="$BUILD/examples/$(basename "${f%.kc}")"
  "$BUILD/kc" "$f" -o "$out.ptx"
  fromPtxas=$(ptxas -arch=sm_89 -v "$out.ptx" -o "$out.cubin" 2>&1 | grep -o "Used [0-9]* registers" | tail -1 | grep -o "[0-9]*")
  fromElf=$("$BUILD/cubin-info" "$out.cubin" | grep "kernel" | grep -o "[0-9]* registers" | grep -o "[0-9]*")
  printf "  %-18s %2s registers (cubin-info: %s)\n" "$(basename "$f")" "$fromPtxas" "$fromElf"
  [ "$fromPtxas" = "$fromElf" ] || { echo "  cubin-info disagrees with ptxas"; exit 1; }
done

# Random programs: each one must give the same output at -O0 and -O1.
# (A failing program is saved as fuzz_fail_<seed>.kc in the build folder.)
echo
echo "random program testing (-O0 and -O1 must agree):"
(cd "$BUILD" && ./kc-fuzz 100 | sed "s/^/  /")

# compute-sanitizer re-runs the tests while watching every memory access the
# GPU makes, and reports out-of-bounds or misaligned reads/writes.
echo
echo "compute-sanitizer memcheck:"
compute-sanitizer --tool memcheck "$BUILD/kc-tests" 2>&1 | grep "ERROR SUMMARY" | sed "s/^=* */  /"
