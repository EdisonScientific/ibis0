"""Single source of truth for the multi-domain roster's labels, colors, and line styles.

Every figure script imports this instead of hand-authoring parallel `--colors`/`--labels` strings, so
the palette is consistent and reconfigurable in one place:

- **Qwen3-derived family -> blues** (the `qwen` group: untrained Qwen3 plus the Ibis0 models trained
  from it), deepening with capability: base is the darkest navy (both prompt styles share the hue,
  distinguished by line style), then the trained models lighten (direct-RL -> thinking-RL ->
  classifier). The group name and the `base_*`/`direct_rl`/`thinking_rl` keys are internal
  identifiers; the display labels carry the Ibis0 names.
- **Frontier models -> their vendors' brand hues** -- Claude the Anthropic accent orange, GPT the
  ChatGPT sage teal. Both are far from the Qwen blues in hue, which is what keeps them legible as
  separate rows (blue, the usual GPT colour, is taken by the Qwen family). Trade-off: both rows sit
  in a narrow luminance band (0.334 and 0.349) and merge in a greyscale print; greys #525252 and
  #969696 are a greyscale-safe alternative.

Each entry also carries `eval_key`, the directory its results live under in the eval tree, so a figure
script resolves "which rows to draw" and "where to read each row" from one place. Note the roster here
is the **primary** roster (7 rows); the leave-one-out classifiers belong only to the transfer matrix and
are declared in render_transfer_matrix.py.

Reconfigure by editing `ROSTER` here, or at runtime by pointing `IBIS_PALETTE_JSON` at a JSON file of
`{key: {label?, color?, linestyle?, group?}}` overrides (partial entries merge over the defaults).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    key: str
    label: str
    color: str
    linestyle: str = "-"
    group: str = "qwen"  # "qwen" | "frontier"
    # Directory this roster row's results live under in the eval tree (see scripts/lib/eval_paths.py).
    # For a trained model that is `<run_name>__step_<N>`; for the base model `<model>__<prompt_style>`;
    # for a frontier model its API id. Kept here so "which row" and "where its numbers are" are defined
    # together -- a figure that knows a row exists but not where to read it is useless.
    eval_key: str = ""


# Canonical roster order (legend order). Qwen family in deepening-capability order, then frontier.
_R = "gwas-depmap-tahoe__qwen3-30b-a3b"  # the all-3 training runs behind the primary roster
_DEFAULT: list[Entry] = [
    Entry("base_direct", "Qwen3 (non-thinking)", "#08306b", "-", "qwen", "Qwen3-30B-A3B__direct"),
    Entry("base_thinking", "Qwen3 (thinking)", "#08306b", "--", "qwen", "Qwen3-30B-A3B__reasoning"),
    Entry("direct_rl", "Ibis0 (non-thinking)", "#2171b5", "-", "qwen", f"{_R}__rl-direct__r2.1.0__step_4200"),
    Entry("thinking_rl", "Ibis0 (thinking)", "#4292c6", "-", "qwen", f"{_R}__rl-thinking__r2.1.0__step_2400"),
    Entry("classifier", "Ibis0 (classifier)", "#6baed6", "-", "qwen", f"{_R}__cls__r2.1.0__step_2400"),
    # #74aa9c is the ChatGPT logo teal; #dd8669 is Anthropic's accent orange #d97757 (verified in
    # github.com/anthropics/skills, skills/brand-guidelines/SKILL.md) lightened to sit beside the teal.
    Entry("gpt56sol", "GPT-5.6-sol", "#74aa9c", "--", "frontier", "gpt-5.6-sol"),
    Entry("opus5", "Claude Opus 5", "#dd8669", "--", "frontier", "claude-opus-5"),
]


def _apply_overrides(entries: list[Entry]) -> list[Entry]:
    """Merge `IBIS_PALETTE_JSON` overrides (partial per-key dicts) over the defaults, if set."""
    path = os.environ.get("IBIS_PALETTE_JSON")
    if not path:
        return entries
    ov = json.loads(open(path).read())
    out = []
    for e in entries:
        o = ov.get(e.key, {})
        out.append(
            Entry(
                e.key,
                o.get("label", e.label),
                o.get("color", e.color),
                o.get("linestyle", e.linestyle),
                o.get("group", e.group),
                o.get("eval_key", e.eval_key),
            )
        )
    return out


ROSTER: dict[str, Entry] = {e.key: e for e in _apply_overrides(_DEFAULT)}
ORDER: list[str] = list(ROSTER)


def plot_color(entry: Entry) -> str:
    """Fill/line colour for one row, disambiguating same-hue variants without using linestyle.

    `linestyle` separates rows that SHARE a colour (base-direct vs base-thinking, both #08306b). That
    works on a line plot but not on a bar, and on a reliability diagram it actively misleads: dashed grey
    is the perfect-calibration reference there, so a dashed grey model curve reads as the reference.

    So: a row whose colour is shared with another row and which carries the dashed style gets a lightened
    shade of the same hue; every other row keeps its declared colour exactly. Rows with a unique colour
    (the frontier models) are returned untouched even if they are declared dashed -- lightening those
    would silently make a figure disagree with the palette it claims to use.
    """
    shared = sum(1 for e in ROSTER.values() if e.color == entry.color) > 1
    if not shared or getattr(entry, "linestyle", "-") == "-":
        return entry.color
    r, g, b = (int(entry.color[i : i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * 0.38) for v in (r, g, b))


def entries_for(keys: list[str] | None = None) -> list[Entry]:
    """Roster entries for `keys` (default: full roster in canonical order). Unknown key -> KeyError."""
    keys = ORDER if keys is None else keys
    return [ROSTER[k] for k in keys]


def labels_for(keys: list[str] | None = None) -> list[str]:
    return [e.label for e in entries_for(keys)]


def colors_for(keys: list[str] | None = None) -> list[str]:
    return [e.color for e in entries_for(keys)]


def linestyles_for(keys: list[str] | None = None) -> list[str]:
    return [e.linestyle for e in entries_for(keys)]


def csv(keys: list[str] | None = None) -> dict[str, str]:
    """Comma-joined strings for scripts that take `--labels`/`--colors` CLI args."""
    e = entries_for(keys)
    return {
        "labels": ",".join(x.label for x in e),
        "colors": ",".join(x.color for x in e),
        "linestyles": ",".join(x.linestyle for x in e),
    }


if __name__ == "__main__":  # quick dump for shell consumption / eyeballing
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=None, help="comma-separated subset (default: full roster)")
    ap.add_argument("--field", choices=["labels", "colors", "linestyles"], default=None)
    a = ap.parse_args()
    ks = a.keys.split(",") if a.keys else None
    if a.field:
        print(csv(ks)[a.field])
    else:
        for e in entries_for(ks):
            print(f"{e.key:14s} {e.color}  {e.linestyle:2s}  {e.group:8s}  {e.label}")
