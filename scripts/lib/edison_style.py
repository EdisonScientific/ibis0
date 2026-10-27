"""Edison house style for the paper's matplotlib figures.

Matches the visual style of the Fig. 1 schematic, so the data figures share one visual language with it:

- **Type.** IBM Plex Mono for every plot string (ticks, labels, legend, eyebrows); Computer Modern for math
  (`mathtext.fontset = "cm"`, so $\\theta$ matches the body). Plex Mono is vendored under `fonts/` (SIL OFL
  1.1, see `fonts/ibm-plex-mono/OFL.txt`) because the cluster has no copy and a silent DejaVu fallback would
  make the figures disagree with Fig. 1 without any error.
- **Ink.** One dark ink (`#1e2433`) for spines, ticks, text and error bars; `muted` for secondary text and
  reference lines; `hairline` for grids. No pure black anywhere.
- **Eyebrows.** Panel names are uppercase tracked mono over a 2 px ink rule, left-aligned -- never titles.
- **Data colours are NOT changed.** The roster palette (`roster_palette.py`) and the constant-p / flag
  accents stay as they are; this module restyles type, neutrals and structure only.

Usage: call `install()` before drawing (it swaps the shared neutrals in place, so every drawing primitive
that reads them follows), then `style_axes(ax)` per panel and
`eyebrow(...)` for headers.
"""

from __future__ import annotations

from pathlib import Path

_FONTS = Path(__file__).resolve().parent / "fonts"
FONT_DIR = _FONTS / "ibm-plex-mono"
MONO = "IBM Plex Mono"
# Fig. 1 sets words next to math in Helvetica Neue (the fallback for the house font, DIN 2014). Helvetica
# Neue is commercial and cannot be vendored, so TeX Gyre Heros (a free Helvetica clone, GUST Font License,
# from TeX Live) is vendored instead. A licensed Helvetica Neue wins automatically if matplotlib can see
# one, e.g. via IBIS_EXTRA_FONT_DIR pointing at a local, NOT committed, font directory.
SANS_DIR = _FONTS / "tex-gyre-heros"
# DIN 2014 is the house font proper (Adobe Fonts / ParaType, licensed, never committed). It is first so
# that a machine with a licensed copy on IBIS_EXTRA_FONT_DIR renders the true house style.
SANS_CANDIDATES = ("DIN 2014", "din-2014", "Helvetica Neue", "TeX Gyre Heros")

# House-style colour tokens.
INK = "#1e2433"
MUTED = "#5c6370"
HAIRLINE = "#c4cad6"
BORDER_STRONG = "#a8b0bd"  # reference diagonals: visible, but lighter than any data line
CARD = "#f5f6f8"
BRAND = "#ffcd00"  # fill only, never lines

# figure_io.GREY keys, re-pointed at Feathers roles. Structural neutrals ONLY: the data colours (roster
# palette, the constant-p `accent` rule, the detached-interval `flag`) are deliberately left as they are.
GREY = {
    "ink": INK,
    "ref": BORDER_STRONG,
    "note": MUTED,
    "rule": HAIRLINE,  # figure_io: "hairline separators" -- also used for gridlines; eyebrow rules are drawn in INK separately
    "fill": CARD,
}

BAR_EDGE = MUTED
BAR_EDGE_LW = 0.5
ERR_LW = 0.8
TRACKING = 0.14  # eyebrow letter-spacing, em
_MONO_ADVANCE = 0.6  # Plex Mono advance width, em (every glyph)


def register_fonts() -> None:
    import os

    from matplotlib import font_manager as fm

    dirs = [FONT_DIR, SANS_DIR]
    if extra := os.environ.get("IBIS_EXTRA_FONT_DIR"):
        dirs.append(Path(extra))
    have = {f.fname for f in fm.fontManager.ttflist}
    for d in dirs:
        for p in sorted([*d.glob("*.ttf"), *d.glob("*.otf"), *d.glob("*.ttc")]):
            if str(p) not in have:
                fm.fontManager.addfont(str(p))


def sans() -> str:
    """The sans family Fig. 1 pairs with math: Helvetica Neue if available, else TeX Gyre Heros."""
    from matplotlib import font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    return next(n for n in SANS_CANDIDATES if n in available)


def rc() -> dict:
    return {
        "font.family": MONO,
        "font.monospace": [MONO],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # embed TrueType so the PDF text stays selectable and exact
    }


def install() -> None:
    """Fonts + rcParams + shared structural neutrals, in place. Roster colours are untouched."""
    import figure_io
    import matplotlib as mpl

    register_fonts()
    mpl.rcParams.update(rc())
    figure_io.GREY.update(GREY)


def style_axes(ax, *, tick_pt: float, grid: bool = True) -> None:
    """Ink spines and outward ticks; hairline y-grid behind the data."""
    ax.tick_params(colors=INK, labelcolor=INK, labelsize=tick_pt, length=2.5, width=0.6, direction="out")
    for s in ("left", "bottom"):
        if ax.spines[s].get_visible():
            ax.spines[s].set_color(INK)
            ax.spines[s].set_linewidth(0.7)
    if grid:
        ax.grid(axis="y", color=HAIRLINE, linewidth=0.5, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)


def edge_bars(ax) -> None:
    """Feathers bars: muted hairline edge."""
    from matplotlib.patches import Rectangle

    for p in ax.patches:
        if isinstance(p, Rectangle) and p is not ax.patch:
            p.set_edgecolor(BAR_EDGE)
            p.set_linewidth(BAR_EDGE_LW)
    for ln in ax.lines:
        if ln.get_color() == INK:  # whiskers + caps
            ln.set_linewidth(ERR_LW)


def eyebrow(fig, x_in: float, y_in: float, w_in: float, label: str, *, size: float = 7.5) -> None:
    """Ruled mono eyebrow: uppercase, tracked, left-aligned, 2 px ink rule beneath.

    Matplotlib has no letter-spacing, and Plex Mono has no hair space to fake it with, so the label is set
    one glyph at a time. Mono advance is a constant 0.6 em, which makes the placement exact rather than
    measured. Math-mode labels ("$...$") are set untracked.
    """
    from matplotlib.lines import Line2D

    FW, FH = fig.get_size_inches()
    # Uppercase only OUTSIDE math: "$p(\\theta)$".upper() is "$P(\\THETA)$", which mathtext rejects.
    parts = label.split("$")
    s = "$".join(part if i % 2 else part.upper() for i, part in enumerate(parts))
    if "$" in s:
        fig.text(x_in / FW, y_in / FH, s, fontsize=size, weight="semibold", color=INK, va="bottom")
    else:
        step = (_MONO_ADVANCE + TRACKING) * size / 72.0
        for i, ch in enumerate(s):
            fig.text((x_in + i * step) / FW, y_in / FH, ch, fontsize=size, weight="semibold", color=INK, va="bottom")
    yr = (y_in - 0.05) / FH
    fig.add_artist(Line2D([x_in / FW, (x_in + w_in) / FW], [yr, yr], color=INK, lw=1.0, transform=fig.transFigure))


# ---------------------------------------------------------------------------------------------------------
# Whole-figure pass, applied by figure_io.save_figure when IBIS_FIG_STYLE=edison.
# ---------------------------------------------------------------------------------------------------------
#
# Every paper renderer saves through `figure_io.save_figure`, so restyling there converts all of them without
# editing each one. What it can do generically: ink/sans on axes, bar edges, hairline grids, and turning
# panel titles and suptitles into ruled eyebrows. Bespoke headers drawn with `fig.text` are left alone and
# handled per renderer. Idempotent: running it on an already-styled figure (the compact headline styles
# itself) changes nothing visible.

EYEBROW_PT = 7.5
_MIN_EYEBROW_W_IN = 1.0  # narrower axes (insets) keep an ordinary title
_MAX_EDGED_BARS = 24  # histograms have many thin bars; a hairline edge on each turns them grey


def _eyebrow_size(label: str, width_in: float) -> float:
    need = len(label) * (_MONO_ADVANCE + TRACKING) * EYEBROW_PT / 72.0
    return EYEBROW_PT if need <= width_in else max(5.5, EYEBROW_PT * width_in / need)


def _title_to_eyebrow(fig, renderer, text_artist, x0_in: float, width_in: float) -> None:
    label = text_artist.get_text()
    if not label.strip():
        return
    bb = text_artist.get_window_extent(renderer)
    y_in = bb.y0 / fig.dpi
    text_artist.set_visible(False)
    s = label if "$" in label else label.replace("\n", " ")
    eyebrow(fig, x0_in, y_in + 0.02, width_in, s, size=_eyebrow_size(s, width_in))


def finalize(fig) -> None:
    from matplotlib.collections import QuadMesh
    from matplotlib.container import BarContainer

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    family = sans()
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        bb = ax.get_window_extent(r)
        x0_in, w_in = bb.x0 / fig.dpi, bb.width / fig.dpi
        is_map = bool(ax.images) or any(isinstance(c, QuadMesh) for c in ax.collections)

        # Tick MARKS only. Label colours are left as the renderer set them: some carry meaning (the
        # prob-hist summary greys the rows that have no histogram), and the rc default is already INK.
        ax.tick_params(color=INK)
        for sp in ax.spines.values():
            if sp.get_visible():
                sp.set_color(INK)
        for lab in (ax.xaxis.label, ax.yaxis.label):
            lab.set_fontfamily(family)
            lab.set_color(INK)
        # Tick labels: numbers stay mono (Fig. 1's belief-chart axes), words go sans (Fig. 1's values and
        # glosses). Long category names in mono run ~40% wider and clip or collide once rotated.
        for t in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            if any(ch.isalpha() for ch in t.get_text().replace("$", "")):
                t.set_fontfamily(family)

        for c in ax.containers:
            if isinstance(c, BarContainer) and len(c) <= _MAX_EDGED_BARS:
                for p in c:
                    p.set_edgecolor(BAR_EDGE)
                    p.set_linewidth(BAR_EDGE_LW)

        has_data = bool(ax.lines or ax.patches or ax.collections)
        if not is_map and has_data:
            # Grid on the VALUE axis: y for vertical bars and curves, x for horizontal bars (a y-grid there
            # runs through the bar centres, at the category ticks).
            horizontal = any(
                isinstance(c, BarContainer) and getattr(c, "orientation", "vertical") == "horizontal"
                for c in ax.containers
            )
            axis = ax.xaxis if horizontal else ax.yaxis
            if not any(gl.get_visible() for gl in axis.get_gridlines()):
                ax.grid(visible=False)
                ax.grid(axis="x" if horizontal else "y", color=HAIRLINE, linewidth=0.5)
                ax.set_axisbelow(True)

        if w_in >= _MIN_EYEBROW_W_IN:
            for t in (ax.title, ax._left_title, ax._right_title):
                if t.get_visible() and t.get_text().strip():
                    _title_to_eyebrow(fig, r, t, x0_in, w_in)

    for lab in (getattr(fig, "_supxlabel", None), getattr(fig, "_supylabel", None)):
        if lab is not None:  # figure-level axis labels follow the per-axes rule: words in sans
            lab.set_fontfamily(family)
            lab.set_color(INK)

    # Free-standing figure-level words that are not headers (hand-placed axis labels, notes) are sans too:
    # Fig. 1 sets its glosses and values in the sans face and reserves mono for eyebrows, chips and ticks.
    for t in fig.texts:
        w = t.get_fontweight()
        bold = w in ("bold", "semibold", "demibold", "heavy") or (isinstance(w, (int, float)) and w >= 600)
        if t.get_visible() and len(t.get_text().strip()) > 1 and not bold:
            t.set_fontfamily(family)

    _ruled_headers_to_eyebrows(fig, r)
    _plain_headers_to_eyebrows(fig, r)
    _fit_legends(fig)
    _clear_xlabels(fig)

    st = getattr(fig, "_suptitle", None)
    if st is not None and st.get_visible() and st.get_text().strip():
        boxes = [a.get_window_extent(r) for a in fig.axes if a.get_visible() and a.axison]
        if boxes:
            x0 = min(b.x0 for b in boxes) / fig.dpi
            x1 = max(b.x1 for b in boxes) / fig.dpi
            _title_to_eyebrow(fig, r, st, x0, x1 - x0)


def _ruled_headers_to_eyebrows(fig, renderer) -> None:
    """Hand-drawn "bold text over a horizontal rule" headers -> eyebrows at the same place.

    Several renderers (render_headline._block_header, the BED and WoE figures' panel_header) draw a bold
    figure-level label with a hairline rule just below it. That is structurally an eyebrow already, so it is
    re-set as one: the rule's x-extent becomes the eyebrow width, and the label sits just above the rule,
    left-aligned. Single glyphs are skipped -- they are panel letters, or the per-glyph pieces of an eyebrow
    this module already drew.
    """
    from matplotlib.lines import Line2D

    dpi = fig.dpi
    rules = []
    for a in [*fig.lines, *fig.artists]:
        if isinstance(a, Line2D) and a.get_visible() and a.get_transform() == fig.transFigure:
            ys = a.get_ydata()
            if len(ys) == 2 and ys[0] == ys[1]:
                rules.append(a)
    FW, FH = fig.get_size_inches()
    for t in list(fig.texts):
        label = t.get_text().strip()
        if not t.get_visible() or len(label) <= 1:
            continue
        w = t.get_fontweight()
        if w not in ("bold", "semibold", "demibold", "heavy") and not (isinstance(w, (int, float)) and w >= 600):
            continue
        bb = t.get_window_extent(renderer)
        cx, y0 = (bb.x0 + bb.x1) / 2 / dpi, bb.y0 / dpi
        best = None
        for ln in rules:
            x0, x1 = sorted(v * FW for v in ln.get_xdata())
            ry = ln.get_ydata()[0] * FH
            if x0 - 0.05 <= cx <= x1 + 0.05 and 0 <= y0 - ry <= 0.35:
                if best is None or ry > best[2]:
                    best = (x0, x1, ry, ln)
        if best is None:
            continue
        x0, x1, ry, ln = best
        t.set_visible(False)
        ln.set_visible(False)
        s = label if "$" in label else label.replace("\n", " ")
        eyebrow(fig, x0, ry + 0.05, x1 - x0, s, size=_eyebrow_size(s, x1 - x0))


def _fit_legends(fig, min_pt: float = 5.5) -> None:
    """Shrink a figure-level legend that no longer fits the canvas, in 0.25 pt steps.

    Plex Mono sets wider than the proportional face these legends were laid out for, so a legend that
    used to span most of the width can now overhang both edges. Only overflowing legends change; the rest
    keep their authored size.
    """
    W = fig.get_size_inches()[0] * fig.dpi
    for leg in fig.legends:
        texts = leg.get_texts()
        if not texts:
            continue
        size = texts[0].get_fontsize()
        while size > min_pt:
            fig.canvas.draw()
            bb = leg.get_window_extent(fig.canvas.get_renderer())
            if bb.x0 >= 0 and bb.x1 <= W:
                break
            size -= 0.25
            for t in texts:
                t.set_fontsize(size)


def _plain_headers_to_eyebrows(fig, renderer, min_pt: float = 9.0) -> None:
    """Large, non-bold figure-level text sitting above an axes -> an eyebrow over that axes.

    Catches headers drawn as a plain `fig.text` at title size with no rule (render_transfer_matrix's panel
    headers). The eyebrow starts at the axes' left edge, or just after a bold one-letter panel label
    ("A", "B") that shares its line, which is kept.
    """
    dpi = fig.dpi
    axes = [a for a in fig.axes if a.get_visible() and a.axison]
    letters = [
        t
        for t in fig.texts
        if t.get_visible() and len(t.get_text().strip()) == 1 and t.get_fontweight() in ("bold", 700)
    ]
    for t in list(fig.texts):
        label = t.get_text().strip()
        if not t.get_visible() or len(label) <= 1 or t.get_fontsize() < min_pt:
            continue
        w = t.get_fontweight()
        if w in ("bold", "semibold", "demibold", "heavy") or (isinstance(w, (int, float)) and w >= 600):
            continue
        bb = t.get_window_extent(renderer)
        cx = (bb.x0 + bb.x1) / 2
        below = [a for a in axes if a.get_window_extent(renderer).x0 <= cx <= a.get_window_extent(renderer).x1]
        below = [a for a in below if a.get_window_extent(renderer).y1 <= bb.y0 + 2]
        if not below:
            continue
        ax = max(below, key=lambda a: a.get_window_extent(renderer).y1)
        ab = ax.get_window_extent(renderer)
        x0, x1 = ab.x0 / dpi, ab.x1 / dpi
        for L in letters:
            lb = L.get_window_extent(renderer)
            same_line = abs((lb.y0 + lb.y1) / 2 - (bb.y0 + bb.y1) / 2) < 0.15 * dpi
            at_start = x0 - 0.3 <= lb.x1 / dpi and lb.x0 / dpi <= x0 + 0.1  # the letter heading THIS panel
            if same_line and at_start:
                x0 = max(x0, lb.x1 / dpi + 0.08)
        t.set_visible(False)
        y_rule = ab.y1 / dpi + 0.12
        eyebrow(fig, x0, y_rule + 0.05, x1 - x0, label, size=_eyebrow_size(label, x1 - x0))


def _clear_xlabels(fig, gap_in: float = 0.03) -> None:
    """Push an x-axis label down until it clears its own tick labels."""
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison or not ax.xaxis.label.get_text():
            continue
        for _ in range(12):
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            ticks = [t.get_window_extent(r) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
            if not ticks:
                break
            lab = ax.xaxis.label.get_window_extent(r)
            if lab.y1 <= min(b.y0 for b in ticks) - gap_in * fig.dpi:
                break
            if ax.xaxis._autolabelpos:
                ax.xaxis.labelpad += 1.0
            else:  # pinned with set_label_coords: labelpad is ignored, so move the pinned position
                x, y = ax.xaxis.label.get_position()
                ax.xaxis.set_label_coords(x, y - 1.0 / 72.0 * fig.dpi / ax.get_window_extent(r).height)
