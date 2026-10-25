#!/usr/bin/env python3
"""Turns docs/report.md into docs/report.pdf (run inside WSL: python3 docs/build_report.py).

Same Markdown subset as the guide (it reuses build_guide.py), plus two charts,
drawn as vector graphics with reportlab and placed with <!-- chart: name -->.
The numbers are the measured benchmark results (bench/, kc-bench).
"""

import sys
from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.graphics.shapes import Path as ShapePath  # (pathlib.Path is used for files)
from reportlab.lib import colors
from reportlab.lib.units import cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_guide as g  # noqa: E402

HERE = Path(__file__).resolve().parent

# Colors: categorical slots 1-3 of the validated reference palette (blue, orange, aqua),
# a neutral gray for the "before" bars, and quiet ink for text, axes and grid.
BLUE, ORANGE, AQUA = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a")
BEFORE = colors.HexColor("#b9b7b0")
INK, MUTED, GRID = colors.HexColor("#1f2328"), colors.HexColor("#5b636b"), colors.HexColor("#e3e6ea")

# Benchmark: ms per launch on an RTX 4060 Laptop GPU (kc-bench, final run).
KERNELS = ["saxpy", "row_sum", "matmul", "block_sum", "warp_reduce", "histogram", "adjust"]
KC_O0 = [0.923, 0.811, 0.499, 0.655, 0.285, 1.424, 0.580]
KC_O1 = [0.927, 0.517, 0.340, 0.561, 0.283, 1.422, 0.595]
NVCC = [0.920, 0.520, 0.361, 0.560, 0.285, 1.423, 0.592]

# Time relative to nvcc in the same run, at each step of closing the gaps.
STEPS = ["session 3 optimizer", "+ loop unrolling", "+ division by 2^k"]
GAPS = [("row_sum", BLUE, [1.485, 1.040, 0.994]),
        ("matmul", ORANGE, [1.455, 0.939, 0.942]),
        ("block_sum", AQUA, [1.164, 1.182, 1.002])]


def text(x, y, s, size=7.5, color=MUTED, anchor="start", bold=False):
    return String(x, y, s, fontName="Sans-Bold" if bold else "Sans", fontSize=size, fillColor=color, textAnchor=anchor)


def legend(d, x, y, items):
    """A row of swatches with labels, above the plot."""
    for label, color in items:
        d.add(Rect(x, y - 1, 9, 9, rx=2, ry=2, fillColor=color, strokeColor=None))
        d.add(text(x + 13, y, label, 8, INK))
        x += 13 + 6 * len(label) + 18


def bar(d, x0, y, length, height, color):
    """A horizontal bar anchored at x0: square at the baseline, its data end rounded (4 px)."""
    r = min(4, length / 2, height / 2)
    x1, k = x0 + length, 0.45 * r  # k: control-point distance that makes a quarter circle
    p = ShapePath(fillColor=color, strokeColor=None)
    p.moveTo(x0, y)
    p.lineTo(x1 - r, y)
    p.curveTo(x1 - r + k, y, x1, y + r - k, x1, y + r)
    p.lineTo(x1, y + height - r)
    p.curveTo(x1, y + height - r + k, x1 - r + k, y + height, x1 - r, y + height)
    p.lineTo(x0, y + height)
    p.closePath()
    d.add(p)


def relative_chart(width):
    """Grouped bars: kc -O0 and kc -O1 time, relative to nvcc (1.0 = as fast as nvcc)."""
    rows, row_h, bar_h, gap = len(KERNELS), 22, 7, 2
    left, right, top, bottom = 78, 40, 34, 26
    h = top + rows * row_h + bottom
    d = Drawing(width, h)
    plot_w = width - left - right
    xmax = 1.6
    sx = lambda v: left + v / xmax * plot_w  # noqa: E731

    legend(d, left, h - 14, [("kc -O0", BEFORE), ("kc -O1", BLUE)])
    y_axis = bottom
    for v in [0, 0.4, 0.8, 1.2, 1.6]:  # recessive grid
        d.add(Line(sx(v), y_axis, sx(v), h - top + 6, strokeColor=GRID, strokeWidth=0.6))
        d.add(text(sx(v), y_axis - 11, f"{v:.1f}", 7, MUTED, "middle"))
    d.add(text(left + plot_w / 2, 2, "time relative to nvcc (lower is faster; 1.0 = same as nvcc)", 7.5, MUTED, "middle"))

    for i, name in enumerate(KERNELS):
        y = h - top - (i + 1) * row_h + (row_h - 2 * bar_h - gap) / 2
        d.add(text(left - 8, y + bar_h - 1, name, 8, INK, "end"))
        before, after = KC_O0[i] / NVCC[i], KC_O1[i] / NVCC[i]
        bar(d, sx(0), y + bar_h + gap, sx(before) - sx(0), bar_h, BEFORE)
        bar(d, sx(0), y, sx(after) - sx(0), bar_h, BLUE)
        # Label the result (not every bar), just right of the bar or of the nvcc line, whichever is further.
        d.add(text(max(sx(after), sx(1)) + 5, y + 0.5, f"{after:.2f}", 7, INK))
        if before > 1.1:
            d.add(text(sx(before) + 4, y + bar_h + gap + 1, f"{before:.2f}", 7, MUTED))

    d.add(Line(sx(1), y_axis, sx(1), h - top + 6, strokeColor=INK, strokeWidth=1, strokeDashArray=[3, 2]))
    d.add(text(sx(1) + 3, h - top + 8, "nvcc", 7.5, INK))
    return d


def gaps_chart(width):
    """Lines: the three kernels where nvcc was faster, after each fix (relative to nvcc)."""
    left, right, top, bottom = 44, 70, 30, 30
    h = 190
    d = Drawing(width, h)
    plot_w, plot_h = width - left - right, h - top - bottom
    ymin, ymax = 0.8, 1.6
    sx = lambda i: left + i / (len(STEPS) - 1) * plot_w  # noqa: E731
    sy = lambda v: bottom + (v - ymin) / (ymax - ymin) * plot_h  # noqa: E731

    legend(d, left + 90, h - 14, [(name, color) for name, color, _ in GAPS])
    for v in [0.8, 1.0, 1.2, 1.4, 1.6]:
        d.add(Line(left, sy(v), left + plot_w, sy(v), strokeColor=GRID, strokeWidth=0.6))
        d.add(text(left - 6, sy(v) - 2.5, f"{v:.1f}", 7, MUTED, "end"))
    d.add(Line(left, sy(1), left + plot_w, sy(1), strokeColor=INK, strokeWidth=1, strokeDashArray=[3, 2]))
    d.add(text(left + 4, sy(1) + 4, "nvcc", 7.5, INK))
    for i, step in enumerate(STEPS):
        d.add(text(sx(i), bottom - 14, step, 7.5, INK, "middle"))

    for name, color, values in GAPS:
        points = []
        for i, v in enumerate(values):
            points += [sx(i), sy(v)]
        d.add(PolyLine(points, strokeColor=color, strokeWidth=2))
        for i, v in enumerate(values):  # 8 px markers with a surface ring
            d.add(Circle(sx(i), sy(v), 4, fillColor=color, strokeColor=colors.white, strokeWidth=1.5))
    # Direct labels at the right end (the end values are close together, so they are spread out vertically).
    offsets = {"block_sum": 10, "row_sum": -1, "matmul": -9}
    for name, color, values in GAPS:
        d.add(text(sx(2) + 8, sy(values[2]) + offsets[name] - 2.5, f"{name} {values[2]:.2f}", 7, INK))
    d.add(text(left - 34, sy(1.6) + 8, "time relative to nvcc", 7, MUTED))
    return d


def main():
    g.build(HERE / "report.md", HERE / "report.pdf", "kc: technical report",
            charts={"relative": relative_chart, "gaps": gaps_chart}, title_gap=0.6 * cm)


if __name__ == "__main__":
    sys.exit(main())
