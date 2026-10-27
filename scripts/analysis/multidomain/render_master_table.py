"""The master table for the published multi-domain TEST set, in a compact and a complete form.

Outputs under `--out-dir`:

- `master_table.{md,tex}` — the COMPACT table the paper's body carries: macro-averages, point estimates
  only, one row per roster entry plus the constant-`p` reference.
- `master_table_full.{md,tex}` — the COMPLETE table for the appendix: `n`, parse-fail rate, an interval
  under every point estimate, and each row broken out by domain.
- `master_table.json` — both forms' numbers, for anything that wants to read rather than typeset them.
- `recalibration.{md,tex}` — the cross-fit Platt diagnostic, ONLY with `--recalibrate`: every system
  as reported and again after scaling, macro only.
- `recalibration_per_domain.{md,tex}` — the same recalibrated rows broken out by domain, likewise only
  with `--recalibrate`.

Two tables rather than one because the two jobs conflict. The body table is read at a glance and has to
fit `\\textwidth`; the appendix table is where a reader checks a number, and every column it adds is a
column the body table does not want. Both are generated from the same rows, so they cannot disagree.

This module emits TABLES only, so editing it never invalidates a cached bootstrap. Macro numbers come
from `macro_metrics.py`, the
per-domain intervals from `per_domain_metrics.py` (a separate cache namespace, so the appendix breakdown
does not invalidate the macro one), and the figure from `render_grouped_metrics.py`.

**Recalibration is a diagnostic, not a condition.** A two-parameter cross-fit Platt fit on the logit
removes most of the Brier gap for base and frontier models while barely moving the trained ones (they
have no reliability left to recover). Genuinely informative, but it doubles the row count everywhere it
appears, so it gets its own table and never enters the primary comparison or the figure. The trained
rows are scored under it anyway, as the control: comparing recalibrated untrained systems against
unrecalibrated trained ones would be a comparison of two treatments. `recal_diagnostic` builds that
roster, from outside the bootstrap cache boundary — read its docstring before moving any of it.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import display_names as dn  # noqa: E402
import per_domain_metrics  # noqa: E402
import recal_diagnostic  # noqa: E402
from macro_metrics import METRICS, _outside_ci  # noqa: E402
from multidomain_cells import DOMAINS  # noqa: E402
from shared_args import _common_args, resolve, rows_and_reference  # noqa: E402


def _header_row(label: str) -> dict:
    """A block-heading pseudo-row, spanning the table width instead of carrying numbers.

    Carries `missing` because the note builders inspect it on every row; everything else is absent by
    design, so a renderer that forgets to special-case `_hdr` fails loudly rather than printing zeros.
    """
    return {"key": "_hdr", "label": label, "missing": []}


def _macro_only(rows: list[dict]) -> list[dict]:
    """Shallow copies with the per-domain breakdown dropped, for a table that reports macro rows only.

    Copies rather than mutates: `per_domain_metrics.attach` set `per_domain_rows` on the very objects
    the complete table renders from, and stripping them in place would empty that table too.
    """
    return [{k: v for k, v in r.items() if k != "per_domain_rows"} for r in rows]


def _fmt_cell(pt: float, lo: float, hi: float) -> tuple[str, bool]:
    """(text, flagged) for one metric cell in the markdown table."""
    if np.isnan(pt):
        return "—", False
    if np.isnan(lo) or np.isnan(hi):
        return f"{pt:.4f}", False
    flag = _outside_ci(pt, lo, hi)
    return f"{pt:.4f} [{lo:.4f}, {hi:.4f}]" + (" †" if flag else ""), flag


def _constant_note(rows: list[dict]) -> list[str]:
    """The constant-row explanation, only when that row is actually on the page.

    The recalibration table has no constant row, and explaining a row the reader cannot see is worse
    than saying nothing.
    """
    if not any(r["key"] == "_const" for r in rows):
        return []
    return [
        "The constant row ignores all evidence; the test set is balanced ~50:50 by construction, so "
        "it scores ≈0.25 Brier. Rows above that line do not beat guessing."
    ]


def _incomplete_note(rows: list[dict]) -> list[str]:
    incomplete = [r for r in rows if r["missing"]]
    if not incomplete:
        return []
    return [
        "Macro omitted (incomplete domains): "
        + "; ".join(f"{r['label']} missing {', '.join(dn.domain(d) for d in r['missing'])}" for r in incomplete)
    ]


def _notes(rows: list[dict], n_boot: int, smece_boot: int, flagged: bool, per_domain: bool = False) -> list[str]:
    notes = [
        "Macro-average over "
        + ", ".join(dn.domain(d) for d in DOMAINS)
        + f"; 95% CI from a within-domain bootstrap resampling episodes ({n_boot:,} draws"
        + (f"; sMECE {smece_boot:,}, it is ~35x costlier per draw)." if smece_boot else "; sMECE point-only)."),
    ]
    if per_domain:
        notes.append(
            "The unindented row of each block is that macro-average; the indented rows below it are the "
            "single domains, bootstrapped over that domain's episodes alone."
        )
    notes += _constant_note(rows)
    if any((r.get("attempted", 0) - r.get("parsed", 0)) for r in rows):
        # Only worth a line when it happened. Every row currently returns a readable probability on
        # every episode, and a column of zeros with a paragraph explaining it spends the reader's
        # attention on nothing.
        notes += [
            "Episodes whose probability could not be read are excluded from every metric rather than "
            "scored as wrong, so a row with fewer than the full n was graded on a subset.",
        ]
    if flagged:
        notes.append(
            "† point estimate outside its bootstrap interval — expected for the positively-biased binned "
            "metrics (Reliability, sMECE) on rows near the zero floor. Compare rows by these intervals; "
            "do not read them as covering the point."
        )
    return notes + _incomplete_note(rows)


def _notes_compact(rows: list[dict], flagged: bool) -> list[str]:
    """Notes for the point-estimate table. It shows no intervals, so it must not describe any."""
    notes = [
        "Macro-average over "
        + ", ".join(dn.domain(d) for d in DOMAINS)
        + ". Point estimates only; n, parse-fail rates, 95% bootstrap intervals and the per-domain "
        "breakdown are in the complete version of this table.",
    ]
    notes += _constant_note(rows)
    notes.append("Bold marks the best value in each column, over the model rows only.")
    if flagged:
        notes.append(
            "† point estimate outside its own bootstrap interval (shown in the complete table) — expected "
            "for the positively-biased binned metrics on rows near the zero floor."
        )
    return notes + _incomplete_note(rows)


def _best_keys(rows: list[dict], metrics: tuple[str, ...]) -> dict[str, str]:
    """{metric: key of the winning row}. The constant reference is a level, not a competitor."""
    best = {}
    for m in metrics:
        scored = [(r[m][0], r["key"]) for r in rows if r["key"] != "_const" and not np.isnan(r[m][0])]
        if scored:
            best[m] = (min if dn.lower_is_better(m) else max)(scored)[1]
    return best


def markdown(
    rows: list[dict],
    n_boot: int,
    smece_boot: int,
    metrics: tuple[str, ...] = METRICS,
    per_domain: bool = False,
) -> str:
    head = ["Model", "n scored"] + [f"{dn.metric(m)} ({dn.direction_arrow(m)})" for m in metrics]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    flagged = False

    def emit(r: dict, label: str) -> None:
        nonlocal flagged
        cells = [label, f"{r['n']:,}"]
        for m in metrics:
            txt, f = _fmt_cell(*r[m])
            flagged |= f
            cells.append(txt)
        out.append("| " + " | ".join(cells) + " |")

    for r in rows:
        if r["key"] == "_hdr":
            out.append(f"| **{r['label']}** |" + " |" * (len(head) - 1))
            continue
        emit(r, r["label"])
        if per_domain:
            for sub in r.get("per_domain_rows", []):
                emit(sub, f"↳ {sub['label']}")
    notes = _notes(rows, n_boot, smece_boot, flagged, per_domain=per_domain)
    return "\n".join(out) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"


def _tex_escape(s: str) -> str:
    """Escape for LaTeX, including the non-ASCII glyphs the notes use.

    The prose carries ≈, em dashes, arrows and daggers; pdflatex without a unicode engine chokes on those
    even though they render fine in the markdown twin. Mapping them to math/ligature equivalents keeps one
    source of note text for both outputs.
    """
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("≈", r"$\approx$"),
        ("—", "---"),
        ("†", r"$\dagger$"),
        ("↓", r"$\downarrow$"),
        ("↑", r"$\uparrow$"),
    ):
        s = s.replace(a, b)
    return s


_HEADER = "% Generated by scripts/analysis/multidomain/render_master_table.py -- do not edit by hand."
_REQUIRES = "% Requires \\usepackage{booktabs}."
_REQUIRES_STACK = "% Requires \\usepackage{booktabs} and \\usepackage{xcolor}."

# Point estimate over its interval, so a column is only as wide as one number. \providecommand, not
# \newcommand: the compact and the complete table are both \input into the same document.
#
# The interval is set in black!45, matching \tmstack in render_transfer_matrix.py. These tables are
# dense enough that equal-weight intervals compete with the point estimates for attention; greying them
# lets a reader scan the column of numbers first and consult the interval second. Same hue as the
# transfer table on purpose -- the two appendices sit close together and a different grey reads as a
# different kind of number.
_STACK_DEF = (
    "\\providecommand{\\mtstack}[2]{\\begin{tabular}[t]{@{}r@{}}"
    "#1\\\\[-2pt]{\\scriptsize\\color{black!45}#2}\\end{tabular}}"
)


def _tex_metric_head(m: str) -> str:
    return f"{_tex_escape(dn.metric(m))} ($\\{'downarrow' if dn.lower_is_better(m) else 'uparrow'}$)"


def _tex_float(lines: list[str], note: str, caption: str, label: str) -> str:
    """Close a tabular into a complete `table` float.

    Emitted as a float with caption and label rather than a bare tabular so it drops into the paper
    unmodified. The notes go below the tabular rather than into a `%` comment: they define the CI
    procedure and the dagger, so a reader of the compiled PDF needs them.
    """
    return (
        "\n".join(
            lines
            + [
                "\\bottomrule",
                "\\end{tabular}",
                "\\\\[2pt]",
                f"\\parbox{{\\linewidth}}{{\\footnotesize\\raggedright {note}}}",
                f"\\caption{{{_tex_escape(caption)}}}",
                f"\\label{{{label}}}",
                "\\end{table}",
            ]
        )
        + "\n"
    )


def latex_compact(
    rows: list[dict],
    metrics: tuple[str, ...] = METRICS,
    caption: str = "Macro-averaged calibration and discrimination on the held-out multi-domain test set.",
    label: str = "tab:multidomain-master",
) -> str:
    """The body table: one macro point estimate per metric, best value per column in bold.

    No `n`, no parse-fail rate and no intervals — those are what `latex_full` is for. Dropping them is
    what lets this fit `\\textwidth` at `\\small` with room to spare, and the columns that remain are the
    ones the surrounding prose actually compares.
    """
    best = _best_keys(rows, metrics)
    lines = [
        _HEADER,
        _REQUIRES,
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "r" * len(metrics) + "}",
        "\\toprule",
        " & ".join(["Model"] + [_tex_metric_head(m) for m in metrics]) + " \\\\",
        "\\midrule",
    ]
    flagged = False
    for r in rows:
        cells = [_tex_escape(r["label"])]
        for m in metrics:
            pt, lo, hi = r[m]
            if np.isnan(pt):
                cells.append("--")
                continue
            txt = f"{pt:.4f}"
            if best.get(m) == r["key"]:
                txt = f"\\textbf{{{txt}}}"
            if _outside_ci(pt, lo, hi):
                txt, flagged = f"{txt}$^{{\\dagger}}$", True
            cells.append(txt)
        # The constant reference is a level, not a competitor: rule it off from the models above it.
        if r["key"] == "_const":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    note = " ".join(_tex_escape(n) for n in _notes_compact(rows, flagged))
    return _tex_float(lines, note, caption, label)


def latex_full(
    rows: list[dict],
    n_boot: int,
    smece_boot: int,
    metrics: tuple[str, ...] = METRICS,
    caption: str = "Calibration and discrimination on the held-out multi-domain test set, in full.",
    label: str = "tab:multidomain-master-full",
) -> str:
    """The appendix table: `n`, parse-fail rate, an interval under every value, broken out by domain.

    A row's `per_domain_rows` (set by `per_domain_metrics.attach`, or by `recal_diagnostic` for the
    recalibrated roster) become indented sub-rows under it. Rows without them render macro-only, which
    is what `_macro_only` is for. A `_hdr` row from `_header_row` spans the width as a block heading.
    """
    ncol = 2 + len(metrics)  # model, n, then one per metric
    lines = [
        _HEADER,
        _REQUIRES_STACK,
        _STACK_DEF,
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        # Four lines per model at two typeset lines each: the float is too tall for the page at the
        # default leading. 0.9 buys ~60pt, which is what keeps the block separators affordable.
        "\\renewcommand{\\arraystretch}{0.9}",
        "\\begin{tabular}{l" + "r" * (ncol - 1) + "}",
        "\\toprule",
        " & ".join(["Model", "$n$"] + [_tex_metric_head(m) for m in metrics]) + " \\\\",
        "\\midrule",
    ]
    flagged = False

    def emit(r: dict, label_tex: str) -> None:
        nonlocal flagged
        cells = [
            label_tex,
            f"{r['n']:,}",
        ]
        for m in metrics:
            pt, lo, hi = r[m]
            if np.isnan(pt):
                cells.append("--")
            elif np.isnan(lo):
                cells.append(f"{pt:.4f}")
            else:
                mark = ""
                if _outside_ci(pt, lo, hi):
                    mark, flagged = "$^{\\dagger}$", True
                cells.append(f"\\mtstack{{{pt:.4f}{mark}}}{{[{lo:.4f}, {hi:.4f}]}}")
        lines.append(" & ".join(cells) + " \\\\")

    blocked = any(r.get("per_domain_rows") for r in rows)
    for i, r in enumerate(rows):
        if r["key"] == "_hdr":
            if i:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{_tex_escape(r['label'])}}}}} \\\\")
            continue
        if r["key"] == "_const":
            lines.append("\\midrule")
        elif i and blocked:
            lines.append("\\addlinespace")
        emit(r, _tex_escape(r["label"]))
        for sub in r.get("per_domain_rows", []):
            emit(sub, f"\\quad {_tex_escape(sub['label'])}")
    note = " ".join(_tex_escape(n) for n in _notes(rows, n_boot, smece_boot, flagged, per_domain=blocked))
    return _tex_float(lines, note, caption, label)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _common_args(ap)
    ap.add_argument("--out-dir", default="outputs/master_table")
    ap.add_argument(
        "--recalibrate",
        action="store_true",
        help="also write recalibration.{md,tex}: a cross-fit Platt diagnostic for the untrained rows. "
        "Never enters the master table or the figure — it is a diagnostic, not a condition.",
    )
    args = ap.parse_args()

    slug, root, keys = resolve(args)
    # recalibrate=False, even under --recalibrate: `build_rows` recalibrates only the untrained rows,
    # and the diagnostic now covers the full roster. `recal_diagnostic` builds that set instead, from
    # outside the bootstrap cache boundary -- see its docstring.
    rows, _, const = rows_and_reference(args, slug, root, keys, recalibrate=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_rows = rows + ([const] if const else [])
    per_domain_metrics.attach(slug, table_rows, n_boot=args.n_boot, smece_boot=args.smece_boot, root=root)
    recal_rows = (
        recal_diagnostic.build_recal_rows(
            slug,
            keys,
            n_boot=args.n_boot,
            smece_boot=args.smece_boot,
            allow_partial=args.allow_partial,
            root=root,
        )
        if args.recalibrate
        else []
    )

    md = markdown(table_rows, args.n_boot, args.smece_boot)
    (out_dir / "master_table.md").write_text(md)
    (out_dir / "master_table.tex").write_text(latex_compact(table_rows))
    (out_dir / "master_table_full.md").write_text(markdown(table_rows, args.n_boot, args.smece_boot, per_domain=True))
    (out_dir / "master_table_full.tex").write_text(latex_full(table_rows, args.n_boot, args.smece_boot))
    (out_dir / "master_table.json").write_text(
        json.dumps(
            [
                {
                    "key": r["key"],
                    "label": r["label"],
                    "n": r["n"],
                    "attempted": r.get("attempted"),
                    "parsed": r.get("parsed"),
                    "parse_fail": r.get("parse_fail"),
                    "missing": r["missing"],
                    "recalibrated": r["recalibrated"],
                    "macro": {m: r[m] for m in METRICS},
                    "per_domain": r["per_domain"],
                    "per_domain_ci": {
                        sub["key"]: {k: v for k, v in sub.items() if k != "key"} for sub in r.get("per_domain_rows", [])
                    },
                }
                for r in table_rows + recal_rows
            ],
            indent=2,
        )
    )
    print(md)
    print(f"[master] compact  -> {out_dir / 'master_table.tex'}  (+ master_table.md, master_table.json)")
    print(f"[master] complete -> {out_dir / 'master_table_full.tex'}  (+ master_table_full.md)")

    if recal_rows:
        # Paired: as-reported above, recalibrated below, macro only. Every system appears in both
        # blocks, including the trained ones -- recalibrating only the untrained rows and comparing
        # them against unrecalibrated trained rows would contrast two different treatments.
        cap = (
            "Cross-fit Platt recalibration of every evaluated system. A two-parameter affine fit on "
            "the logit, fitted out-of-fold and separately within each domain; it needs labelled "
            "target-domain data, so it bounds how much of each gap is a post-hoc fix rather than "
            "being a deployable method. The trained systems are recalibrated too, as the control that "
            "makes the two blocks comparable."
        )
        paired = [
            _header_row("As reported"),
            *_macro_only(rows),
            _header_row("After cross-fit Platt scaling"),
            *_macro_only(recal_rows),
        ]
        rmd = markdown(paired, args.n_boot, args.smece_boot)
        (out_dir / "recalibration.md").write_text(f"# Recalibration diagnostic\n\n{cap}\n\n{rmd}")
        (out_dir / "recalibration.tex").write_text(
            latex_full(paired, args.n_boot, args.smece_boot, caption=cap, label="tab:multidomain-recal")
        )

        # The breakdown behind the lower block. Its as-reported counterpart is already in the complete
        # master table, so only the recalibrated side is emitted here rather than doubling 28 rows.
        pd_cap = (
            "Cross-fit Platt recalibration by domain, with the map fitted separately within each of "
            "Tahoe, DepMap and GWAS. The as-reported breakdown is in the complete master table."
        )
        (out_dir / "recalibration_per_domain.md").write_text(
            f"# Recalibration diagnostic, by domain\n\n{pd_cap}\n\n"
            f"{markdown(recal_rows, args.n_boot, args.smece_boot, per_domain=True)}"
        )
        (out_dir / "recalibration_per_domain.tex").write_text(
            latex_full(
                recal_rows, args.n_boot, args.smece_boot, caption=pd_cap, label="tab:multidomain-recal-per-domain"
            )
        )
        print(f"[master] recalibration diagnostic -> {out_dir / 'recalibration.md'}  (+ .tex)")
        print(f"[master] recalibration by domain  -> {out_dir / 'recalibration_per_domain.md'}  (+ .tex)")


if __name__ == "__main__":
    main()
