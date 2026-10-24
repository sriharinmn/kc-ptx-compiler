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

echo
echo "ptxas check of examples/ (registers each kernel needs on sm_89):"
mkdir -p "$BUILD/examples"
for f in "$SRC"/examples/*.kc; do
  out="$BUILD/examples/$(basename "${f%.kc}").ptx"
  "$BUILD/kc" "$f" -o "$out"
  regs=$(ptxas -arch=sm_89 -v "$out" -o /dev/null 2>&1 | grep -o "Used [0-9]* registers" | head -1)
  printf "  %-18s %s\n" "$(basename "$f")" "$regs"
done

# compute-sanitizer re-runs the tests while watching every memory access the
# GPU makes, and reports out-of-bounds or misaligned reads/writes.
echo
echo "compute-sanitizer memcheck:"
compute-sanitizer --tool memcheck "$BUILD/kc-tests" 2>&1 | grep "ERROR SUMMARY" | sed "s/^=* */  /"
