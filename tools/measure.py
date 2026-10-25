#!/usr/bin/env python3
"""What does kc's optimizer change, after ptxas has done its own optimizing?

For every example, at -O0 and -O1, prints:
  - PTX instructions   (what kc writes)
  - SASS instructions  (the real GPU machine code ptxas makes from it)
  - registers          (per thread, from ptxas -v; fewer = more threads can run at once)
  - calls              (CALL instructions left in the SASS)

Run inside WSL after building:  python3 tools/measure.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KC = Path.home() / "kc-build" / "kc"
CUDA = Path("/usr/local/cuda/bin")
ARCH = "sm_89"


def run(*args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True, text=True)


def ptx_instructions(ptx):
    # An instruction line starts with a tab and ends with ';' (".reg" etc. are declarations).
    return sum(1 for line in ptx.splitlines()
               if line.startswith("\t") and line.endswith(";") and not line.startswith("\t."))


def sass_stats(cubin):
    sass = run(CUDA / "cuobjdump", "-sass", cubin).stdout
    count = calls = 0
    for m in re.finditer(r"/\*([0-9a-f]{4})\*/\s+([^;]*);", sass):
        addr, text = int(m.group(1), 16), m.group(2).strip()
        if text.startswith("NOP"):
            continue  # padding
        jump = re.match(r"BRA 0x([0-9a-f]+)", text)
        if jump and int(jump.group(1), 16) == addr:
            continue  # "jump to itself" padding after the last EXIT
        count += 1
        calls += text.startswith("CALL")
    return count, calls


def measure(example, level, tmp):
    ptx = run(KC, example, f"-O{level}", "-o", "-").stdout
    ptx_file = tmp / f"{example.stem}.O{level}.ptx"
    cubin = tmp / f"{example.stem}.O{level}.cubin"
    ptx_file.write_text(ptx)
    info = run(CUDA / "ptxas", f"-arch={ARCH}", "-v", ptx_file, "-o", cubin).stderr
    # ptxas -v prints one "Used N registers" line per function; the kernel's comes after "entry function".
    kernel_info = info[info.find("entry function"):]
    regs = int(re.search(r"Used (\d+) registers", kernel_info).group(1))
    sass, calls = sass_stats(cubin)
    return ptx_instructions(ptx), sass, regs, calls


def main():
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for example in sorted((ROOT / "examples").glob("*.kc")):
            o0 = measure(example, 0, Path(d))
            o1 = measure(example, 1, Path(d))
            rows.append((example.name, o0, o1))

    print(f"| example | PTX -O0 | PTX -O1 | SASS -O0 | SASS -O1 | registers -O0 | registers -O1 |")
    print(f"|---|---:|---:|---:|---:|---:|---:|")
    for name, o0, o1 in rows:
        sass0 = f"{o0[1]}" + (f" ({o0[3]} calls)" if o0[3] else "")
        sass1 = f"{o1[1]}" + (f" ({o1[3]} calls)" if o1[3] else "")
        print(f"| {name} | {o0[0]} | {o1[0]} | {sass0} | {sass1} | {o0[2]} | {o1[2]} |")
    total = [sum(r[i][j] for r in rows) for i in (1, 2) for j in (0, 1)]
    print(f"| **total** | {total[0]} | {total[2]} | {total[1]} | {total[3]} | | |")


if __name__ == "__main__":
    sys.exit(main())
