"""Canonical layout for TEST-set eval results — one scheme for classifier, generative, and frontier.

Every eval writer and every figure reader goes through this module, so the layout cannot drift between
the thing that produces results and the thing that consumes them.

    <root>/<testset>/<model_key>/
        <domain>_<tag>.json     tahoe_primary.json, gwas_ctx7.json, …
        run.json                provenance for this model's cells

`root` defaults to `$IBIS_DATA/eval` (`ibis_data/eval` when `IBIS_DATA` is unset) and is overridable with `IBIS_EVAL_ROOT`. It lives
outside the repo on purpose: results outlive the worktree that produced them.

**`<testset>` — the published set's identity, in the path.** `EDB-Bio@df43070d` for a
Hub revision, `local-<dirname>` for a local directory. Putting the revision in the path (rather than only
in a sidecar) makes it *structurally impossible* to render a figure that mixes two revisions, and means
re-publishing the set never overwrites results collected against the old one.

**`<model_key>` — what produced the numbers.**

| Model kind | Key | Example |
|---|---|---|
| trained checkpoint | `<run_name>__step_<N>` | `gwas-depmap-tahoe__qwen3-30b-a3b__rl-direct__r2.1.0__step_4200` |
| untrained served model | `<model>__<prompt_style>` | `Qwen3-30B-A3B__direct` |
| frontier API | `<api_model_id>` | `gpt-5.6-sol` |

The run name is the key for anything trained, so a result directory names the training run that produced
it without needing a curated lookup table. The step is appended because the run name does not carry one,
and evaluating a second checkpoint of the same run must not overwrite the first.

Prompt style is in the key only where it is the *sole* distinguishing feature — the base model, which has
no run name and contributes two roster rows. For a trained run the style is implied (cls and rl-direct are
`direct`, rl-thinking is `reasoning`), so it stays out of an already-long key. It is always recorded in
`run.json`, and `write_run_json` **refuses to overwrite a directory whose provenance disagrees**, so an
accidental off-style re-run is a loud error rather than a silent mix.

Filenames carry no mode and no family prefix: the directory already says which model, so
`classifier_tahoe_primary.json` would be saying it twice. `tag` is `primary` or `ctx<k>`, which puts the
sensitivity sweep alongside the primary cell in the same directory — a dose/WoE renderer globs `ctx*`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9._@-]+")


# The published multi-domain TEST set revision every renderer defaults to. ONE definition on purpose:
# a renderer reads BOTH its cells and (for the per-stub figures) its stub metadata through this revision,
# so a stale per-renderer copy would silently change which data a figure is built from rather than
# failing. Bump this when a new test set is published; never fork it.
# EDB-Bio tag v1.0.0: the paper test set, N=2000 primary / 256 sensitivity per domain.
DEFAULT_TESTSET_REVISION = "df43070d4a1338210ed33c368d49e58da60471cb"


def eval_root() -> Path:
    """Root of the eval tree. `IBIS_EVAL_ROOT` overrides; otherwise `$IBIS_DATA/eval` (default `ibis_data/eval`)."""
    if env := os.environ.get("IBIS_EVAL_ROOT"):
        return Path(env)
    return Path(os.environ.get("IBIS_DATA", "ibis_data")) / "eval"


def _slug(s: str) -> str:
    """Collapse anything path-hostile so a key is always a single safe directory name."""
    return _SAFE.sub("-", s).strip("-")


def testset_slug(testset: str, revision: str | None = None) -> str:
    """Identity of the test set a result was produced against.

    `testset` is the path or `{domain}` template handed to the eval. A Hub path becomes
    `<repo>@<rev8>`; a local directory becomes `local-<dirname>`. `revision` is required for Hub paths —
    an unpinned Hub result cannot be told apart from one collected against a later publish, which is the
    exact failure this scheme exists to prevent.
    """
    if testset.startswith("hf://"):
        parts = testset[len("hf://") :].strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"hf:// testset must be hf://<org>/<repo>/<config>, got {testset!r}")
        repo = parts[1]
        if not revision:
            raise ValueError(
                f"a Hub test set needs an explicit revision so results cannot be confused across "
                f"publishes; got testset={testset!r} with revision=None"
            )
        return f"{_slug(repo)}@{_slug(revision)[:8]}"
    # Local directory: strip the {domain} placeholder and any config segment to name the set root.
    p = testset.replace("{domain}", "").rstrip("/")
    parts = [x for x in Path(p).parts if x not in ("/", "")]
    # .../testset_smoke/primary -> testset_smoke ; .../testset_smoke -> testset_smoke
    for cand in reversed(parts):
        if cand not in ("primary", "sensitivity"):
            return f"local-{_slug(cand)}"
    raise ValueError(f"could not derive a test-set name from {testset!r}")


def checkpoint_key(run_name: str, step: int | str) -> str:
    """Key for a trained checkpoint: `<run_name>__step_<N>`."""
    return f"{_slug(run_name)}__step_{int(step)}"


def served_key(model: str, prompt_style: str) -> str:
    """Key for an untrained served model, e.g. `Qwen3-30B-A3B__direct`.

    Prompt style is part of the key here because it is the only thing separating the base model's two
    roster rows. `model` may be an HF id (`Qwen/Qwen3-30B-A3B`) or a path; only the basename is used.
    """
    if prompt_style not in ("direct", "reasoning"):
        raise ValueError(f"prompt_style must be 'direct' or 'reasoning', got {prompt_style!r}")
    return f"{_slug(Path(model.rstrip('/')).name)}__{prompt_style}"


def api_key(model_id: str) -> str:
    """Key for a frontier API model, e.g. `gpt-5.6-sol`."""
    return _slug(model_id)


def result_dir(testset_slug_: str, model_key: str, root: Path | None = None) -> Path:
    return (root or eval_root()) / testset_slug_ / model_key


def cell_path(testset_slug_: str, model_key: str, domain: str, tag: str, root: Path | None = None) -> Path:
    """Path of one result cell: `<root>/<testset>/<model_key>/<domain>_<tag>.json`."""
    return result_dir(testset_slug_, model_key, root) / f"{_slug(domain)}_{_slug(tag)}.json"


def out_prefix(testset_slug_: str, model_key: str, domain: str, root: Path | None = None) -> str:
    """`--out-prefix` for the eval scripts, which append `_<tag>.json` themselves."""
    return str(result_dir(testset_slug_, model_key, root) / _slug(domain))


def _testsets_of(prov: dict) -> set[str]:
    """Every test-set config a provenance dict mentions, tolerating the pre-`testsets` scalar form."""
    out = set(prov.get("testsets") or [])
    if t := prov.get("testset"):
        out.add(t)
    return out


def write_run_json(
    testset_slug_: str,
    model_key: str,
    provenance: dict,
    root: Path | None = None,
) -> Path:
    """Record how this model's cells were produced, refusing to silently reuse a directory.

    If `run.json` exists and disagrees on any identity field (`model_source`, `prompt_style`,
    `checkpoint_step`), this raises: writing different-provenance results into one directory is exactly
    how a figure ends up mixing two models. Anything else merges, so a second run appends cells to a
    matching run (e.g. adding `ctx*` cells next to `primary`).

    **Which test-set configs the directory covers accumulates.** One directory legitimately holds cells
    from more than one config — `primary` and `sensitivity` share a slug, because the slug keys on repo
    plus revision, and the per-cell `tag` is what separates them. A plain `{**prev, **new}` merge would
    leave the scalar `testset` naming only the most recent config while the directory held both, so
    `run.json` — the first file a reader opens — would describe half its own directory. Every config
    seen is therefore unioned into `testsets`, and the scalar `testset` is kept only while that union
    has exactly one member: a missing scalar beside a two-entry list is unambiguous, where a scalar
    that silently went stale is not.
    """
    d = result_dir(testset_slug_, model_key, root)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "run.json"
    if p.exists():
        prev = json.loads(p.read_text())
        keys = ("model_source", "prompt_style", "checkpoint_step")
        clash = {k: (prev.get(k), provenance.get(k)) for k in keys if prev.get(k) != provenance.get(k)}
        if clash:
            raise ValueError(
                f"{p} already records different provenance {clash}; refusing to mix results in one "
                f"directory. Use a distinct model_key, or clear the directory first."
            )
        merged = {**prev, **provenance}
        merged["cells_written"] = sorted(set(prev.get("cells_written", [])) | set(provenance.get("cells_written", [])))
        merged["testsets"] = sorted(_testsets_of(prev) | _testsets_of(provenance))
        provenance = merged
    else:
        provenance = {**provenance, "testsets": sorted(_testsets_of(provenance))}
    # Keep the scalar only while it is the whole truth; drop it once the directory spans two configs.
    if len(provenance.get("testsets", [])) == 1:
        provenance["testset"] = provenance["testsets"][0]
    else:
        provenance.pop("testset", None)
    p.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return p


def read_run_json(testset_slug_: str, model_key: str, root: Path | None = None) -> dict | None:
    p = result_dir(testset_slug_, model_key, root) / "run.json"
    return json.loads(p.read_text()) if p.exists() else None


def iter_model_keys(testset_slug_: str, root: Path | None = None) -> list[str]:
    """Model keys present for a test set — the reader side, so figures never hardcode paths."""
    d = (root or eval_root()) / testset_slug_
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


# --- CLI, so bash launchers resolve the layout through this same module ----------------------------
# Without this, each launcher would reimplement the slug/key rules in shell and drift from the readers.
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Resolve or record canonical eval output paths.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--testset", required=True, help="test-set path or {domain} template")
        q.add_argument("--revision", default=None, help="required for an hf:// test set")
        q.add_argument("--root", default=None)
        g = q.add_mutually_exclusive_group(required=True)
        g.add_argument("--run-name", help="trained run name; use with --step")
        g.add_argument("--served", help="untrained served model id/path; use with --style")
        g.add_argument("--api", help="frontier API model id")
        q.add_argument("--step", type=int, default=None)
        q.add_argument("--style", choices=["direct", "reasoning"], default=None)

    r = sub.add_parser("resolve", help="print shell-evalable slug/key/dir")
    common(r)
    c = sub.add_parser("record", help="write run.json from a JSON blob")
    common(c)
    c.add_argument("--json", required=True, help="provenance fields as a JSON object")

    a = ap.parse_args()
    # Normalize an empty --revision to None: a local test set has no revision, and recording ""
    # would put a meaningless empty string in run.json.
    a.revision = a.revision or None
    slug = testset_slug(a.testset, a.revision)
    if a.run_name:
        if a.step is None:
            ap.error("--run-name requires --step")
        key = checkpoint_key(a.run_name, a.step)
    elif a.served:
        if not a.style:
            ap.error("--served requires --style")
        key = served_key(a.served, a.style)
    else:
        key = api_key(a.api)
    root = Path(a.root) if a.root else None

    if a.cmd == "resolve":
        print(f"IBIS_TESTSET_SLUG={slug}")
        print(f"IBIS_MODEL_KEY={key}")
        print(f"IBIS_RESULT_DIR={result_dir(slug, key, root)}")
        return
    prov = json.loads(a.json)
    prov.setdefault("testset", a.testset)
    prov.setdefault("testset_revision", a.revision)
    prov.setdefault("model_key", key)
    if a.step is not None:
        prov.setdefault("checkpoint_step", a.step)
    if a.style:
        prov.setdefault("prompt_style", a.style)
    print(write_run_json(slug, key, prov, root))


if __name__ == "__main__":
    _main()
