"""BED paper figures, drawn through the shared house style (scripts/lib/figure_io.py + roster_palette.py)
so they match Figures 2--3: authored at TEXT_WIDTH_IN, shared FONT scale, roster colours (Qwen family =
blues, frontier = muted brand colours). Produces:
  bed_per_domain           -- belief in the true hypothesis vs #experiments, 3 domains, shared y
  bed_entropy_per_domain   -- first-step information gain H(p0)-H(p1), 3 domains, shared x (95% CI)

  python scripts/analysis/rq5_bed/bed_paper_figs.py --results <final_*.jsonl> --outdir <dir>
"""

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager

_LIB = Path(__file__).resolve().parents[2] / "lib"  # .../scripts/lib
sys.path.insert(0, str(_LIB))
font_manager.fontManager.addfont(_LIB / "fonts" / "KaTeX_AMS-Regular.ttf")
matplotlib.rcParams.update(
    {
        "mathtext.fontset": "custom",
        "mathtext.rm": "KaTeX_AMS",
        "mathtext.it": "STIXGeneral:italic",
        "mathtext.bf": "STIXGeneral:bold",
        "mathtext.cal": "STIXGeneral",
        "mathtext.fallback": "stix",
    }
)
from figure_io import FONT, GREY, TEXT_WIDTH_IN, new_figure, panel_row, save_figure  # noqa: E402
from roster_palette import ROSTER, plot_color  # noqa: E402

# BED selection policies, coloured by the MODEL that drives them so every legend in the paper agrees:
# both EIG methods run on the Ibis0 classifier (belief model + simulator), so both take its colour; the naive
# choosers take their own chooser's roster colour (base Qwen3 non-thinking, rl-direct Ibis0, GPT, Opus).
# Line style still separates EIG (solid, marked) from naive (dashed); random = neutral dotted.
STYLE = {  # policy: (label, colour, linestyle, marker, linewidth)
    "theta_2pole": ("EIG-simulate", plot_color(ROSTER["classifier"]), "-", "o", 2.1),
    "exemplar_eig": ("EIG-exemplar", plot_color(ROSTER["classifier"]), "-", "s", 2.1),
    "naive_opus5": ("Opus 5", plot_color(ROSTER["opus5"]), "--", "^", 1.3),
    "naive_gpt56": ("GPT-5.6-sol", plot_color(ROSTER["gpt56sol"]), "--", "D", 1.3),
    "naive_base": ("Qwen3 (non-thinking)", plot_color(ROSTER["base_direct"]), "--", "v", 1.3),
    "naive_rld": ("Ibis0 (non-thinking)", plot_color(ROSTER["direct_rl"]), "--", "P", 1.3),
    "random": ("random", "#bdbdbd", ":", None, 1.3),
}
ORDER = ["theta_2pole", "exemplar_eig", "naive_opus5", "naive_gpt56", "naive_base", "naive_rld", "random"]
DOM_LABEL = {"depmap": "DepMap", "gwas": "GWAS", "tahoe": "Tahoe"}
SHORT = {
    "theta_2pole": "EIG-simulate",
    "exemplar_eig": "EIG-exemplar",
    "naive_opus5": "Opus 5",
    "naive_gpt56": "GPT-5.6-sol",
    "naive_base": "Qwen3 (non-thinking)",
    "naive_rld": "Ibis0 (non-thinking)",
    "random": "random",
}
BELIEF_POLICIES = ["theta_2pole", "naive_opus5", "random"]


def H(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load(results):
    return [json.loads(l) for pat in results for f in glob.glob(pat) for l in open(f)]


def _clean_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=FONT["tick"], colors=GREY["ink"])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GREY["rule"])


def belief_curves(rows, doms, present, out):
    from matplotlib.lines import Line2D

    # per (domain, policy): t -> (mean, sem)
    cur = {}
    ymin, ymax = 1.0, 0.0
    for d in doms:
        for p in present:
            byt = collections.defaultdict(list)
            for r in rows:
                if r["domain"] != d or r["policy"] != p:
                    continue
                for pt in r["traj"]:
                    byt[pt["t"]].append(pt["p"] if r["label"] == 1 else 1 - pt["p"])
            ts = np.array(sorted(byt))
            m = np.array([np.mean(byt[t]) for t in ts])
            s = np.array([np.std(byt[t]) / np.sqrt(len(byt[t])) for t in ts])
            cur[(d, p)] = (ts, m, s)
            ymin, ymax = min(ymin, (m - s).min()), max(ymax, (m + s).max())
    pad = 0.02
    ylo, yhi = ymin - pad, ymax + pad

    L, R, GAP = 0.56, 0.12, 0.28
    PH, TOP, BOT, LEG = 1.95, 0.32, 0.44, 0.42
    fig = new_figure(BOT + PH + TOP + LEG)
    axes = panel_row(fig, len(doms), y=BOT, h=PH, left=L, right=R, gap=GAP)
    for i, (ax, d) in enumerate(zip(axes, doms)):
        cap = max(t.max() for (dd, _), (t, _, _) in cur.items() if dd == d)
        for p in present:
            lab, col, ls, mk, lw = STYLE[p]
            ts, m, s = cur[(d, p)]
            ax.fill_between(ts, m - s, m + s, color=col, alpha=0.14, lw=0, zorder=1)
            ax.plot(
                ts,
                m,
                color=col,
                ls=ls,
                lw=lw,
                marker=mk,
                ms=3.2,
                mfc=col,
                mec=col,
                markevery=2,
                zorder=5 if p in ("theta_2pole", "exemplar_eig") else 3,
                clip_on=True,
            )
        ax.set_xlim(0, cap)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(range(0, int(cap) + 1, 4))
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_axisbelow(True)
        _clean_ax(ax)
        ax.set_title(DOM_LABEL[d], fontsize=FONT["title"], color=GREY["ink"])
        ax.set_xlabel("experiments performed", fontsize=FONT["label"])
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"belief in true hypothesis  $P(\mathrm{true}\mid\mathrm{ev.})$", fontsize=FONT["label"])
    handles = [
        Line2D(
            [],
            [],
            color=STYLE[p][1],
            ls=STYLE[p][2],
            lw=STYLE[p][4] + 0.3,
            marker=STYLE[p][3],
            ms=3.6,
            label=STYLE[p][0],
        )
        for p in present
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=FONT["legend"],
        frameon=False,
        handlelength=1.9,
        columnspacing=1.3,
        borderaxespad=0.2,
    )
    save_figure(fig, out, tag="bed_per_domain")


def entropy_bars(rows, doms, present, out):
    stat = {}
    xmax = 0.0
    for d in doms:
        for p in present:
            v = []
            for r in rows:
                if r["domain"] != d or r["policy"] != p:
                    continue
                tt = {pt["t"]: pt["p"] for pt in r["traj"]}
                if 0 in tt and 1 in tt:
                    v.append(H(tt[0]) - H(tt[1]))
            v = np.array(v)
            m, ci = v.mean(), 1.96 * v.std() / np.sqrt(len(v))
            stat[(d, p)] = (m, ci)
            xmax = max(xmax, m + ci)
    xhi = xmax * 1.14

    L, R, GAP = 1.42, 0.10, 0.20
    PH, TOP, BOT = 2.05, 0.32, 0.44
    fig = new_figure(BOT + PH + TOP)
    axes = panel_row(fig, len(doms), y=BOT, h=PH, left=L, right=R, gap=GAP)
    ypos = np.arange(len(present))[::-1]
    for i, (ax, d) in enumerate(zip(axes, doms)):
        means = [stat[(d, p)][0] for p in present]
        cis = [stat[(d, p)][1] for p in present]
        cols = [STYLE[p][1] for p in present]
        ax.barh(ypos, means, xerr=cis, color=cols, height=0.68, error_kw=dict(lw=0.8, ecolor=GREY["ink"], capsize=2))
        ax.axvline(0, color=GREY["ref"], lw=0.6, zorder=0)
        for y, m, c in zip(ypos, means, cis):
            ax.text(
                m + c + xhi * 0.02, y, f"{m:.3f}", va="center", ha="left", fontsize=FONT["annot"], color=GREY["note"]
            )
        ax.set_xlim(-0.035, xhi)
        ax.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])  # shared across panels
        ax.set_yticks(ypos)
        _clean_ax(ax)
        ax.set_title(DOM_LABEL[d], fontsize=FONT["title"], color=GREY["ink"])
        ax.set_xlabel(r"$H(p_0)-H(p_1)$  (bits)", fontsize=FONT["label"])
        if i == 0:
            ax.set_yticklabels([STYLE[p][0] for p in present], fontsize=FONT["tick"])
        else:
            ax.tick_params(labelleft=False)
    save_figure(fig, out, tag="bed_entropy_per_domain")


def _pooled_curve(rows, present, cap):
    out = {}
    for p in present:
        byt = collections.defaultdict(list)
        for r in rows:
            if r["policy"] != p:
                continue
            for pt in r["traj"]:
                if pt["t"] <= cap:
                    byt[pt["t"]].append(pt["p"] if r["label"] == 1 else 1 - pt["p"])
        ts = np.array(sorted(byt))
        out[p] = (
            ts,
            np.array([np.mean(byt[t]) for t in ts]),
            np.array([np.std(byt[t]) / np.sqrt(len(byt[t])) for t in ts]),
        )
    return out


def _dh(rows, present, dom=None):
    st = {}
    for p in present:
        v = [
            H(t0) - H(t1)
            for r in rows
            if r["policy"] == p and (dom is None or r["domain"] == dom)
            for t0, t1 in [
                ({pt["t"]: pt["p"] for pt in r["traj"]}.get(0), {pt["t"]: pt["p"] for pt in r["traj"]}.get(1))
            ]
            if t0 is not None and t1 is not None
        ]
        v = np.array(v)
        st[p] = (v.mean(), 1.96 * v.std() / np.sqrt(len(v)))
    return st


def combined(rows, doms, present, out):
    from matplotlib.lines import Line2D

    cap = min(max(pt["t"] for r in rows if r["domain"] == d for pt in r["traj"]) for d in doms)
    belief_policies = [p for p in BELIEF_POLICIES if p in present]
    curve = _pooled_curve(rows, belief_policies, cap)
    dh_all = _dh(rows, present)
    info_policies = sorted(present, key=lambda p: dh_all[p][0], reverse=True)
    xhi = max(m + c for m, c in dh_all.values()) * 1.14
    ypos = np.arange(len(info_policies))[::-1]

    W = TEXT_WIDTH_IN
    L, R, GAP = 0.62, 0.12, 1.20  # holds "Qwen3 (non-thinking)", the longest bar label
    LEG, TOP, PH, BOT = 0.0, 0.42, 1.95, 0.48  # no legend strip: the legend sits inside the belief panel
    figh = BOT + LEG + PH + TOP
    fig = new_figure(figh, W)
    a_cv, a_dh = panel_row(fig, 2, y=BOT + LEG, h=PH, left=L, right=R, gap=GAP)

    def panel_header(ax, text):
        pos = ax.get_position()
        title_y = (BOT + LEG + PH + 0.24) / figh
        rule_y = (BOT + LEG + PH + 0.13) / figh
        fig.text(
            (pos.x0 + pos.x1) / 2,
            title_y,
            text,
            ha="center",
            va="bottom",
            fontsize=FONT["title"],
            weight="bold",
            color=GREY["ink"],
        )
        fig.add_artist(
            Line2D(
                [pos.x0, pos.x1],
                [rule_y, rule_y],
                transform=fig.transFigure,
                color=GREY["rule"],
                lw=0.7,
            )
        )

    panel_header(a_cv, "BELIEF IN HELD-OUT CONCLUSION")
    panel_header(a_dh, "INFORMATION GAIN")

    def bars(ax, st, labels):
        m = [st[p][0] for p in info_policies]
        c = [st[p][1] for p in info_policies]
        bars_ = ax.barh(
            ypos,
            m,
            xerr=c,
            color=[STYLE[p][1] for p in info_policies],
            height=0.66,
            error_kw=dict(lw=0.7, ecolor=GREY["ink"], capsize=1.6),
        )
        # EIG-exemplar shares EIG-simulate's colour (both run on the classifier); hatching tells them apart.
        for bar, p in zip(bars_, info_policies):
            if p == "exemplar_eig":
                bar.set_hatch("////")
        ax.axvline(0, color=GREY["ref"], lw=0.6, zorder=0)
        ax.set_xlim(0, xhi)
        ax.set_yticks(ypos)
        _clean_ax(ax)
        if labels:
            ax.set_yticklabels([SHORT[p] for p in info_policies], fontsize=FONT["tick"])
        else:
            ax.tick_params(labelleft=False)
        return m, c

    # ---- Left: aggregate belief curve, restricted to the requested comparison ----
    ymin = min((mm - ss).min() for (_, mm, ss) in curve.values())
    ymax = max((mm + ss).max() for (_, mm, ss) in curve.values())
    for p in belief_policies:
        lab, col, ls, mk, lw = STYLE[p]
        ts, mm, ss = curve[p]
        a_cv.fill_between(ts, mm - ss, mm + ss, color=col, alpha=0.13, lw=0, zorder=1)
        a_cv.plot(
            ts,
            mm,
            color=col,
            ls=ls,
            lw=lw,
            marker=mk,
            ms=3.0,
            mfc=col,
            mec=col,
            markevery=2,
            zorder=5 if p == "theta_2pole" else 3,
        )
    a_cv.set_xlim(0, cap)
    a_cv.set_ylim(ymin - 0.02, ymax + 0.02)
    a_cv.set_xticks(range(0, cap + 1, 2))  # = markevery
    _clean_ax(a_cv)
    a_cv.set_xlabel(r"$k$ experiments performed", fontsize=FONT["label"])
    a_cv.set_ylabel(r"$p(\theta = \theta^\ast\,\vert\,I_k)$", fontsize=FONT["label"])

    # ---- Right: aggregate first-step information gain ----
    m, c = bars(a_dh, dh_all, labels=True)
    for y, mm, cc in zip(ypos, m, c):
        a_dh.text(
            mm + cc + xhi * 0.02, y, f"{mm:.3f}", va="center", ha="left", fontsize=FONT["annot"], color=GREY["note"]
        )
    a_dh.set_xticks([0.0, 0.05, 0.10, 0.15, 0.20])
    a_dh.set_xlabel(
        "Info gained from first experiment (bits)",
        fontsize=FONT["label"],
    )

    # The information-gain bars are directly labelled, so the legend only needs to identify the three
    # belief curves. It sits small in the belief panel's empty lower-right corner (every curve rises from
    # ~0.6 toward ~0.72, so the bottom right is clear), which removes the figure-wide legend strip.
    handles = [
        Line2D([], [], color=STYLE[p][1], ls=STYLE[p][2], lw=STYLE[p][4], marker=STYLE[p][3], ms=2.6, label=STYLE[p][0])
        for p in belief_policies
    ]
    a_cv.legend(
        handles=handles,
        loc="lower right",
        fontsize=FONT["annot"],
        frameon=False,
        handlelength=2.2,
        handletextpad=0.5,
        labelspacing=0.3,
        borderaxespad=0.4,
    )
    save_figure(fig, out, tag="bed_combined")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--outdir", default="ibis_data/bed_sim/bed_choosers512")
    args = ap.parse_args()
    rows = load(args.results)
    doms = [d for d in ["depmap", "gwas", "tahoe"] if d in {r["domain"] for r in rows}]
    present = [p for p in ORDER if p in {r["policy"] for r in rows}]
    combined(rows, doms, present, f"{args.outdir}/bed_combined")


if __name__ == "__main__":
    main()
