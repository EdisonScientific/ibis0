"""Biology-blind baseline for the Evidence Deliberation task (construct validity).

Trains a gradient-boosted tree (and logistic regression) on NUMERIC-ONLY pool features --- no gene,
drug, or cell identities, no biology, just the DESeq2 statistics and counts --- on the same
dataset `split` partition the LLMs see, and reports calibration (Brier/ECE/AUROC). If an LLM does not
beat this blind baseline, its apparent "biological reasoning" adds nothing over reading the table.

Reproducible: deterministic feature extraction + fixed seed; writes a predictions JSON in the
format scripts/lib/render_calibration_report.py reads, so it drops in alongside the model JSONs.

## Scoring against the frozen test set

`--train-data-path` exists because the published test set
(`hf://EdisonScientific/EDB-Bio/testsets/primary/<domain>`) publishes only `test.parquet`, so the
training rows have to come from the source dataset instead of the same repo.

`--train-context` matters more than it looks. A published test-set row has already been truncated to its
own `k`, whereas a source-dataset row carries the full 32-slot pool, so training on the full pool and
scoring on truncated rows silently mismatches two families of features:

  - **count features** (`n_stubs`, `n_<verdict>`, `n_<context_type>`, `n_sig_aligned/opposed`) come out
    about 1.8x larger in training. `n_stubs` is worse than shifted: at a fixed `--context-size` it is
    *constant* in training, so the tree never learns to use it, and then sees 1..16 at eval.
  - **aggregate features** (`agg_nsig`, `agg_meanabslfc`) are the real hazard. Every source row has a
    genome-wide aggregate stub, but it sits at an arbitrary pool index, so truncation to `k` keeps it in
    only ~28% of test rows. `design_matrix` fills the rest with 0.0 -- a value the model learned to read as
    "no genome-wide effect" (training `agg_nsig` reaches 18,767 and is never 0), not as "absent". That
    understates the baseline, which is the dangerous direction: it would make the LLMs look like they clear
    the blind floor more comfortably than they do.

`match-eval` and `augment` both fix this by truncating each training pool the way
`build_static_testset.py` does -- a prefix slice of the whole pool, aggregate included, so `k` counts pool
slots rather than evidence stubs. `fixed` preserves the original behaviour and stays the default.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
# Resolve both first-party imports against THIS checkout, matching the convention in
# scripts/analysis/multidomain/. A bare `import ibis_env_utils` picks up whatever is installed, which may
# come from another checkout -- so the script could silently run against a loader that does not match
# the code around it.
sys.path.insert(0, str(_SCRIPTS.parent / "envs" / "ibis_env_utils"))
sys.path.insert(0, str(_SCRIPTS / "lib"))
from calibration_metrics import compute_calibration  # noqa: E402
from ibis_env_utils import load_dataset_split  # noqa: E402

CTX_TYPES = ["same_drug_diff_cell", "same_moa_evidence", "diff_drug_same_cell", "negative_control"]
VERDICTS = ["supported", "refuted", "inconclusive"]


def featurize(row, k, thresh):
    """Numeric-only features over the first k evidence stubs + the genome-wide aggregate stub.
    No identities (gene/drug/cell names, MOA, SMILES) ever enter."""
    up = str(row["target_direction"]).lower().startswith("up")
    ev = [
        s
        for s in row["context_pool"]
        if s.get("kind") == "evidence" and (s.get("y") or {}).get("log2FoldChange") is not None
    ][:k]
    agg = [s for s in row["context_pool"] if s.get("kind") == "aggregate"]
    f = {"n_stubs": len(ev), "target_up": int(up)}
    if ev:
        y = [s["y"] for s in ev]
        lfc = np.array([v["log2FoldChange"] for v in y], float)
        padj = np.array([v.get("padj", 1.0) for v in y], float)
        absl = np.abs(lfc)
        sig = (padj < 0.05) & (absl > thresh)
        f.update(
            mean_lfc=float(lfc.mean()),
            max_lfc=float(lfc.max()),
            min_lfc=float(lfc.min()),
            mean_abslfc=float(absl.mean()),
            max_abslfc=float(absl.max()),
            frac_sig=float(sig.mean()),
            mean_neglog10padj=float(np.mean(-np.log10(np.clip(padj, 1e-300, 1.0)))),
            mean_lfcSE=float(np.mean([v.get("lfcSE", 0.0) for v in y])),
            mean_logbasemean=float(np.mean([np.log1p(v.get("baseMean", 0.0)) for v in y])),
            mean_ncells=float(np.mean([v.get("n_cells_trt", 0) + v.get("n_cells_ctrl", 0) for v in y])),
            n_sig_aligned=int(
                sum(
                    1
                    for v in y
                    if v.get("padj", 1) < 0.05
                    and abs(v["log2FoldChange"]) > thresh
                    and ((v["log2FoldChange"] > 0) == up)
                )
            ),
            n_sig_opposed=int(
                sum(
                    1
                    for v in y
                    if v.get("padj", 1) < 0.05
                    and abs(v["log2FoldChange"]) > thresh
                    and ((v["log2FoldChange"] > 0) != up)
                )
            ),
        )
        for v in VERDICTS:
            f[f"n_{v}"] = sum(1 for s in ev if s.get("verdict") == v)
        for ct in CTX_TYPES:
            f[f"n_{ct}"] = sum(1 for s in ev if s.get("context_type") == ct)
    if agg:
        ay = agg[0]["y"]
        f.update(
            agg_strength=float(ay.get("perturbation_strength_percentile") or 0.0),
            agg_nsig=float(ay.get("n_sig_genes_lfc05") or 0.0),
            agg_meanabslfc=float(ay.get("mean_abs_lfc") or 0.0),
        )
    return f


def design_matrix(rows, k, thresh):
    feats = [featurize(r, k, thresh) for r in rows]
    X = pd.DataFrame(feats).fillna(0.0)
    y = np.array([int(r["label"]) for r in rows])
    return X, y


# Featurize with no further slicing: the pool has already been cut to k, so `featurize`'s own `ev[:k]`
# must not cut it a second time (k counts pool slots, but that slice counts evidence stubs).
NO_SLICE = 1 << 30


def truncate_pool(row, k):
    """Cut a source-dataset row's pool to k slots, matching `build_static_testset.materialize`.

    A plain prefix slice of the pre-shuffled pool, aggregate stub included. That is what makes the
    aggregate survive at the same rate as in the published test set instead of always surviving.
    """
    out = dict(row)
    out["context_pool"] = list(row["context_pool"])[:k]
    return out


def build_train_rows(rows, mode, context_size, eval_ks, seed):
    """Expand/truncate the training rows so their features match the pre-truncated eval rows."""
    if mode == "fixed":
        return rows, context_size
    ks = sorted({int(k) for k in eval_ks})
    if not ks:
        raise SystemExit("--train-context match-eval/augment needs a `k` column on the eval rows")
    if mode == "augment":
        # One row per k per episode: full k coverage and the most data, so the blind floor is as high as
        # it fairly can be. Rows are nested prefixes of the same episode, hence correlated -- fine here,
        # since train and eval episodes are disjoint splits.
        return [truncate_pool(r, k) for r in rows for k in ks], NO_SLICE
    # match-eval: one k per episode, drawn from the eval set's own k distribution.
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.asarray(eval_ks, dtype=int), size=len(rows), replace=True)
    return [truncate_pool(r, int(k)) for r, k in zip(rows, draws, strict=True)], NO_SLICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True, help="Dataset to EVALUATE on.")
    ap.add_argument("--split", required=True, help="dataset split to evaluate on (e.g. test)")
    ap.add_argument(
        "--train-data-path",
        default=None,
        help="Dataset to TRAIN on; defaults to --data-path. Needed when evaluating on a published "
        "test-set repo, which publishes only test.parquet.",
    )
    ap.add_argument("--revision", default=None, help="Pin the eval data path to a commit.")
    ap.add_argument("--train-revision", default=None, help="Pin the train data path to a commit.")
    ap.add_argument(
        "--train-context",
        choices=["fixed", "match-eval", "augment"],
        default="fixed",
        help="'fixed': slice the first --context-size evidence stubs (original behaviour). "
        "'match-eval': truncate each training pool to a k drawn from the eval set's k distribution. "
        "'augment': one training row per k present in the eval set. Use match-eval/augment whenever the "
        "eval rows are pre-truncated, or the aggregate and count features silently mismatch.",
    )
    ap.add_argument("--context-size", type=int, default=16)
    ap.add_argument("--lfc-threshold", type=float, default=0.5)
    ap.add_argument("--model", choices=["gbm", "logreg"], default="gbm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cal/gbm_blind_2048.json")
    args = ap.parse_args()

    train_path = args.train_data_path or args.data_path
    eval_frame = load_dataset_split(args.data_path, args.split, revision=args.revision)
    train_frame = load_dataset_split(train_path, "train", revision=args.train_revision)
    eval_rows = eval_frame.to_dict("records")
    src_train_rows = train_frame.to_dict("records")
    if not src_train_rows or not eval_rows:
        raise SystemExit(f"empty split: train={len(src_train_rows)} {args.split}={len(eval_rows)} in {train_path}")

    eval_ks = [int(k) for k in eval_frame["k"]] if "k" in eval_frame.columns else []
    train_rows, train_k = build_train_rows(src_train_rows, args.train_context, args.context_size, eval_ks, args.seed)

    Xtr, ytr = design_matrix(train_rows, train_k, args.lfc_threshold)
    Xev, yev = design_matrix(eval_rows, args.context_size, args.lfc_threshold)
    Xev = Xev.reindex(columns=Xtr.columns, fill_value=0.0)  # align columns
    print(
        f"train {len(ytr)} (rate {ytr.mean():.3f}) | eval {len(yev)} (rate {yev.mean():.3f}) | {Xtr.shape[1]} features"
    )
    print(f"train-context={args.train_context} ({len(src_train_rows)} episodes -> {len(train_rows)} rows)")
    # Surface the mismatch this flag exists to prevent, rather than leaving it to be discovered downstream.
    for col in ("n_stubs", "agg_nsig"):
        if col in Xtr and col in Xev:
            print(
                f"  {col:10s} train mean={Xtr[col].mean():9.3f} nonzero={100 * (Xtr[col] != 0).mean():5.1f}%"
                f"  | eval mean={Xev[col].mean():9.3f} nonzero={100 * (Xev[col] != 0).mean():5.1f}%"
            )

    if args.model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(random_state=args.seed)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=args.seed))
    clf.fit(Xtr.values, ytr)
    p = clf.predict_proba(Xev.values)[:, 1]

    rep = compute_calibration(p.tolist(), yev.tolist(), n_bins=10, adaptive=True)
    print(f"\nBIOLOGY-BLIND {args.model.upper()} (numeric-only):")
    # Both, always, explicitly named. The paper's baselines table quotes Brier *reward* (1-BS, higher
    # better) while the score (lower better) is the more common convention, and the two are easy to
    # mistake for each other.
    print(f"  Brier score  = {rep.brier:.4f}  (lower is better)")
    print(f"  Brier reward = {1.0 - rep.brier:.4f}  (1 - score; this is what the baselines table quotes)")
    print(f"  BSS={rep.brier_skill_score:.3f}  ECE={rep.ece:.4f}  AUROC={rep.auroc:.4f}")
    print("  (compare to the LLMs in results/cal/*.json — if they don't beat this, they're reading the table)")

    results = [
        {"episode_id": r["episode_id"], "parsed_prob": float(pi), "label": int(yi)}
        for r, pi, yi in zip(eval_rows, p, yev)
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": f"biology-blind {args.model}",
                "provider": "sklearn",
                "prompt_style": "-",
                "n_samples": 1,
                "context_size": args.context_size,
                "provenance": {
                    "eval_data_path": args.data_path,
                    "eval_split": args.split,
                    "eval_revision": args.revision,
                    "train_data_path": train_path,
                    "train_revision": args.train_revision,
                    "train_context": args.train_context,
                    "n_train_episodes": len(src_train_rows),
                    "n_train_rows": len(train_rows),
                    "lfc_threshold": args.lfc_threshold,
                    "seed": args.seed,
                },
                "metrics": {
                    "brier": rep.brier,
                    "brier_reward": float(1.0 - rep.brier),
                    "brier_skill_score": rep.brier_skill_score,
                    "ece": rep.ece,
                    "auroc": rep.auroc,
                    "n": int(len(yev)),
                },
                "results": results,
            }
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
