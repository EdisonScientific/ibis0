"""One way to write a figure, shared by every renderer in this repo.

A save loop duplicated per renderer drifts silently (dpi, `bbox_inches`, `facecolor`), and nothing tells
you which figure got which treatment. This module owns the decision once.

**PDF is the paper artifact; PNG is for quick viewing.** The PDF is vector, so `PNG_DPI` changes nothing
about what ends up in the paper — it only controls how legible the raster copy is when reviewing.

**`tight` is off by default, deliberately.** `bbox_inches="tight"` crops or expands the canvas to fit
whatever the content happens to occupy, which is wrong for a figure laid out in absolute inches to match
`\\textwidth`: a figure authored at 6.7 in comes out at 6.74, LaTeX scales it by 0.994, and every font
shrinks with it. Figures that size themselves (the legacy panels, which are included at whatever width
fits) pass `tight=True`.

When `tight` is off, `save_figure` also checks that nothing overhangs the canvas and says so if it does.
That check matters precisely because these figures do NOT get rescued by a tight bbox: an over-wide title
is simply cut off at the page edge, and a figure checked from its JSON sidecar rather than by opening
the render would hide the crop.
"""

from __future__ import annotations

import os
from pathlib import Path

# Raster resolution for the .png companion. The .pdf is vector and unaffected.
PNG_DPI = 300

FORMATS = ("pdf", "png")

# ---------------------------------------------------------------------------------------------------
# The page. ONE definition, inherited by every renderer.
# ---------------------------------------------------------------------------------------------------
# The paper is single-column `article` with 0.9 in side margins on US Letter, so \textwidth is
# 8.5 - 1.8 = 6.7 in and the text block is 9.25 in tall.
#
# Author figures at exactly TEXT_WIDTH_IN and include them at `width=\textwidth`, so LaTeX applies NO
# scale factor and a declared 8 pt label renders at 8 pt. A figure authored wider is silently shrunk
# (9.3 in comes in at x0.72, turning 8 pt ticks into 5.8 pt), so the declared size is not what the
# reader sees.
#
# **Change these two numbers if the paper changes template** (a two-column venue, A4, different margins)
# and every renderer follows. Nothing else should hardcode a page dimension.
TEXT_WIDTH_IN = 6.7
TEXT_HEIGHT_IN = 9.25

# Shared type scale, in points. Because every figure is authored at TEXT_WIDTH_IN and never rescaled,
# these are also the sizes the READER sees — which is the whole reason a shared scale is possible at all.
FONT = {
    "title": 9.5,  # panel title
    "label": 8.0,  # axis label
    "tick": 7.0,  # tick label
    "legend": 7.0,
    "note": 6.2,  # in-figure caption / method note
    "annot": 5.6,  # smallest: in-panel reference labels, rotated tick labels
}

# Shared neutrals. Model colours come from `roster_palette`; these are the non-data ink, defined once so
# renderers do not each pick their own near-identical greys.
GREY = {
    "ink": "#252525",  # primary text / whisker
    "ref": "#9e9e9e",  # reference lines (identity, chance, zero)
    "note": "#6f6f6f",  # secondary text
    "rule": "#cfcfcf",  # hairline separators
    "fill": "#ececec",  # shaded regions (sub-chance wedge)
    "accent": "#e6550d",  # the constant-p reference rule
    "flag": "#a63603",  # detached-interval dagger
}


def new_figure(height_in: float, width_in: float = TEXT_WIDTH_IN):
    """A figure at exactly `width_in` x `height_in` inches, Agg backend.

    Use with `panel_row` and `save_figure(..., tight=False)`: the three together are the absolute-inch
    idiom that guarantees the PDF is `\\textwidth` wide and therefore never rescaled by LaTeX.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(width_in, height_in))


def panel_row(
    fig,
    n: int,
    *,
    y: float,
    h: float,
    left: float = 0.58,
    right: float = 0.10,
    gap: float = 0.30,
    widths: list[float] | None = None,
):
    """`n` axes in a row, positioned in INCHES from the figure's bottom-left. Returns them left to right.

    `y`/`h` are the row's bottom and height; `left`/`right` are the margins that hold tick labels and the
    axis label; `gap` separates panels. Panel width is whatever is left over, split equally — or per
    `widths` when panels should differ (e.g. a wider first column).

    Sized in inches rather than via gridspec because a gridspec's `tight_layout` reflows unpredictably
    around aspect-locked axes, and `adjustable="box"` silently letterboxes a panel whose cell is the wrong
    shape. Both bit this repo; see the comments in `render_reliability`.
    """
    W, H = fig.get_size_inches()
    if widths is None:
        pw = (W - left - right - (n - 1) * gap) / n
        widths = [pw] * n
    out = []
    x = left
    for w in widths:
        out.append(fig.add_axes((x / W, y / H, w / W, h / H)))
        x += w + gap
    return out


def panel_grid(
    fig,
    nrow: int,
    ncol: int,
    *,
    y: float,
    h: float,
    row_gap: float,
    left: float = 0.58,
    right: float = 0.10,
    gap: float = 0.30,
):
    """`nrow` x `ncol` axes sized in INCHES, returned flat in row-major order with the TOP row first.

    `y` is the bottom of the lowest row and `h` each panel's height, so the block occupies
    `y .. y + nrow*h + (nrow-1)*row_gap`. Row-major-from-the-top matches how callers enumerate panels.
    """
    out = []
    for r in range(nrow):
        row_y = y + (nrow - 1 - r) * (h + row_gap)
        out.extend(panel_row(fig, ncol, y=row_y, h=h, left=left, right=right, gap=gap))
    return out


def overflow(fig) -> list[str]:
    """Which canvas edges the drawn content overhangs, in inches. Empty means nothing is clipped."""
    fig.canvas.draw()
    W, H = fig.get_size_inches()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    edges = (("left", -tb.x0), ("bottom", -tb.y0), ("right", tb.x1 - W), ("top", tb.y1 - H))
    return [f"{name} by {over:.3f}in" for name, over in edges if over > 0.005]


def save_figure(
    fig,
    out_base: str | Path,
    *,
    tag: str = "figure",
    tight: bool = False,
    facecolor: str | None = None,
    close: bool = True,
    report_size: bool | None = None,
) -> list[str]:
    """Write `<out_base>.pdf` and `<out_base>.png`. Returns the `overflow` list (empty when fine).

    `tag` prefixes the log line so a multi-figure renderer says which one it wrote. `report_size` defaults
    to "whenever the size is meaningful", i.e. when `tight` is off and the authored size is what ships.
    """
    import sys

    import matplotlib.pyplot as plt

    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    if report_size is None:
        report_size = not tight

    if STYLE == "edison":
        import edison_style

        edison_style.finalize(fig)

    over: list[str] = []
    if not tight:
        over = overflow(fig)
        if over:
            print(f"[{tag}] !! content overhangs the canvas: {', '.join(over)}", file=sys.stderr)

    kw: dict = {}
    if tight:
        kw["bbox_inches"] = "tight"
    if facecolor is not None:
        kw["facecolor"] = facecolor
    for ext in FORMATS:
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=PNG_DPI, **kw)

    size = ""
    if report_size:
        w, h = fig.get_size_inches()
        size = f"  [{w:.2f} x {h:.2f} in]"
    print(f"[{tag}] -> {out_base.with_suffix('.pdf')} (+ .png){size}")
    if close:
        plt.close(fig)
    return over


# ---------------------------------------------------------------------------------------------------
# House style. Every figure that goes through this module renders in the Edison house style by default (see
# scripts/lib/edison_style.py) -- the style the paper uses. Fonts, rcParams and the GREY
# neutrals are swapped HERE, at import, because matplotlib Text objects capture their font family and math
# fontset when they are created, and renderers capture GREY values into module constants at their own
# import. `save_figure` then runs the whole-figure pass. Data colours never change.
# `IBIS_FIG_STYLE=classic` opts out and renders every figure in plain matplotlib style.
# ---------------------------------------------------------------------------------------------------
STYLE = os.environ.get("IBIS_FIG_STYLE", "edison")
if STYLE not in ("classic", "edison"):
    raise ValueError(f"IBIS_FIG_STYLE must be 'classic' or 'edison', got {STYLE!r}")
if STYLE == "edison":
    import edison_style as _edison

    _edison.install()
