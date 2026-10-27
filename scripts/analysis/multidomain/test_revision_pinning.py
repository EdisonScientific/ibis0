"""Every renderer pins the test set through ONE constant, and pins its stub metadata to the same one.

Both properties fail silently, which is why they are tested rather than documented:

- If each renderer carried its own copy of the revision literal, one could fall behind the others. A
  renderer reading a stale revision does not error; it renders a plausible figure from a different test
  set.
- `render_woe_structure.py` reads per-stub metadata from the dataset as well as cells. If the metadata
  came from the default branch while the cells came from the pinned revision, episodes would lose their
  metadata and WoE increments would vanish — and the figure would still render.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIB = next(p for p in _HERE.parents if p.name == "scripts") / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_HERE))

import eval_paths  # noqa: E402

# A 40-hex-char string, i.e. a git sha someone pasted in rather than importing the constant.
SHA_LITERAL = re.compile(r"[\"'][0-9a-f]{40}[\"']")

# Scripts whose `--revision` must come from the shared constant. `test_*` files are excluded: they use
# opaque sha-shaped strings as fixtures for slug computation and never read a real dataset.
RENDERERS = sorted(p for p in _HERE.glob("render_*.py") if not p.name.startswith("test_")) + [_HERE / "verify_cells.py"]


def test_renderers_exist():
    """Guard the glob itself — a rename that empties it would make every test below vacuous."""
    assert len(RENDERERS) >= 9, [p.name for p in RENDERERS]


def test_no_renderer_hardcodes_a_revision_sha():
    offenders = {}
    for p in RENDERERS:
        hits = [
            f"{p.name}:{i}"
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if SHA_LITERAL.search(line) and "DEFAULT_TESTSET_REVISION" not in line
        ]
        if hits:
            offenders[p.name] = hits
    assert not offenders, (
        f"hardcoded revision sha(s) found: {offenders}. Use eval_paths.DEFAULT_TESTSET_REVISION — "
        "per-script copies drift, and a stale one changes which data a figure is built from without failing."
    )


def test_every_renderer_defaults_to_the_shared_constant():
    """Assert the RESOLVED `--revision` default, not the presence of the constant's name in the source.

    Grepping for `DEFAULT_TESTSET_REVISION` would break whenever `--revision` moves into shared plumbing
    such as `shared_args._common_args`, though the property still holds. Introspecting the built parser tests what actually matters and survives that kind of refactor: whatever plumbing a
    renderer uses, the default a user gets must be the one shared constant.
    """
    import argparse
    import importlib

    want = eval_paths.DEFAULT_TESTSET_REVISION
    bad = {}
    for p in RENDERERS:
        mod = importlib.import_module(p.stem)
        if not hasattr(mod, "main"):
            continue
        seen: dict[str, object] = {}

        hub: dict[str, object] = {}

        def spy(self, *a, _seen=seen, _hub=hub, **k):
            for act in self._actions:
                if act.dest in ("revision", "testset_revision"):
                    _seen[act.dest] = act.default
                # Whether this renderer reads the Hub at all. A revision only means anything for an
                # `hf://` path: `eval_paths.testset_slug` REQUIRES one there and REFUSES one for a local
                # directory, so a renderer that only reads local slugs has nothing to pin.
                if act.dest.endswith("testset") and str(act.default).startswith("hf://"):
                    _hub[act.dest] = act.default
            raise SystemExit(0)

        orig = argparse.ArgumentParser.parse_args
        argparse.ArgumentParser.parse_args = spy
        try:
            mod.main()
        except SystemExit:
            pass
        finally:
            argparse.ArgumentParser.parse_args = orig
        if not hub:
            # Reads only local eval-tree slugs (the conflict and position probes build their own sets and
            # write them under `local-*`). There is no publish to be confused with, so nothing to pin.
            continue
        if not seen:
            bad[p.name] = "no --revision argument"
        elif any(v != want for v in seen.values()):
            bad[p.name] = seen
    assert not bad, f"renderers whose --revision default is not eval_paths.DEFAULT_TESTSET_REVISION: {bad}"


def test_default_revision_is_a_sha():
    rev = eval_paths.DEFAULT_TESTSET_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", rev), f"not a full git sha: {rev!r}"


def test_stub_meta_accepts_and_forwards_a_revision():
    """`render_woe_structure.stub_meta` must pin its dataset load, not read the default branch.

    Asserted at the call boundary rather than by inspecting source: what matters is that the revision
    reaches `load_dataset_split`, since that is the argument that decides which episodes come back.
    """
    import render_woe_structure as woe

    seen = []

    def fake_load(path, split, columns=None, revision=None):
        seen.append(revision)
        raise RuntimeError("stop after the first load — only the revision argument is under test")

    orig = woe.load_dataset_split
    woe.load_dataset_split = fake_load
    try:
        try:
            woe.stub_meta("hf://org/repo/config/{domain}", revision="deadbeef" * 5)
        except RuntimeError:
            pass
    finally:
        woe.load_dataset_split = orig

    assert seen == ["deadbeef" * 5], f"stub_meta did not forward its revision to the loader: {seen}"


def test_local_path_ignores_a_revision(tmp_path):
    """A local directory has no revisions; the render pipeline always threads one through, so it is ignored."""
    import pandas as pd
    from ibis_env_utils import Split, load_dataset_split

    pd.DataFrame({"episode_id": ["e0"]}).to_parquet(tmp_path / "test.parquet")
    df = load_dataset_split(str(tmp_path), Split.TEST, revision="abc123")
    assert df["episode_id"].tolist() == ["e0"]
