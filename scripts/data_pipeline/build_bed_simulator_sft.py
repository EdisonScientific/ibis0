"""Build the SFT dataset for the BED conditional evidence simulator.

Given a target hypothesis, an information set of already-observed evidence stubs, and a CANDIDATE
experiment's setup ``x`` (its measured values hidden), the simulator predicts the candidate's outcome
as JSON — conditioned on whether the outcome SUPPORTS the hypothesis and on the weight of evidence
(WoE, decibans) it would contribute. "Support" is defined w.r.t. the target hypothesis via the
trained belief model: the per-stub incremental WoE annotated offline by ``annotate_stub_woe.py``.

The completion is a fixed-order JSON object ``{supports_hypothesis, weight_of_evidence_decibans, y}``
and we apply **conditioning-dropout**: a per-example REQUESTED-OUTCOME block reveals nothing / the
support flag / the support flag + WoE, so one model serves every mode (generate support+woe+y;
generate woe+y given support; generate y given support+woe). Loss is masked to the assistant JSON.

x/y split: every stub is a dict whose observation is the nested ``y`` sub-dict; ``x`` is the rest.
The completion ``y`` whitelists only the keys relevant to the stub's schema (selected by domain +
``modality``/``kind``) — the stored ``y`` carries the union of all schemas' keys (nulls elsewhere).
All stub kinds are in scope, including Tahoe genome-wide ``aggregate`` stubs.

Output: one ``messages``-format parquet per split under --out-dir, columns
``[messages, chat_template_kwargs, domain]``. Train on the ``train`` split, monitor on ``val`` — the
dataset's clean train/val/test file separation makes cross-split leakage unrepresentable.

Example:
    python scripts/data_pipeline/build_bed_simulator_sft.py \
        --woe-root ibis_data/bed_sim/woe --data-root ibis_data/bed_sim/data \
        --out-dir ibis_data/bed_sim/sft --splits train val
"""

import argparse
import importlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "envs" / "ibis_env_utils").is_dir())
for _m in ("ibis_env_utils", "gwas_meta", "depmap_meta", "tahoe_meta"):
    sys.path.insert(0, str(_REPO / "envs" / _m))

from ibis_env_utils import Split, load_dataset_split  # noqa: E402

# ── Completion y schema per (domain / modality / kind). Only these keys are emitted; the stored y
#    carries the union of every schema's keys with nulls elsewhere, so we must whitelist. ──
SCHEMAS = {
    "gwas": ["p_value", "or_beta", "ci_text", "direction", "direction_phrase"],
    # depmap, keyed by stub["modality"]
    "crispr": ["chronos", "p_dependency"],
    "expression": ["log2_tpm", "lineage_z"],
    "copy_number": ["relative_cn", "call"],
    "mutation": ["status", "hotspot", "damaging", "protein_change"],
    "pharmacology": ["viability_log2fc", "sensitive"],
    # tahoe, keyed by stub["kind"]
    "evidence": ["log2FoldChange", "lfcSE", "padj", "pvalue", "baseMean", "stat", "n_cells_trt", "n_cells_ctrl"],
    "aggregate": [
        "n_sig_genes_lfc05",
        "n_sig_genes_lfc10",
        "mean_abs_lfc",
        "median_abs_lfc",
        "perturbation_strength_percentile",
    ],
}
INT_KEYS = {"direction", "n_cells_trt", "n_cells_ctrl", "n_sig_genes_lfc05", "n_sig_genes_lfc10"}
SCI_KEYS = {"p_value", "padj", "pvalue"}
STR_KEYS = {"ci_text", "direction_phrase", "call", "status", "protein_change"}
BOOL_KEYS = {"hotspot", "damaging", "sensitive"}

SIM_SYSTEM = (
    "You are a simulator of experimental outcomes in biology. You are given a target hypothesis, an "
    "information set of already-observed measurements, and a candidate experiment whose result is "
    "hidden. Predict the candidate's outcome. First state whether that outcome would SUPPORT the "
    "target hypothesis and the weight of evidence (in decibans, signed: positive supports, negative "
    "refutes) it would contribute to a rational observer's belief, then the measured values. When a "
    "REQUESTED OUTCOME is given, produce an outcome consistent with it. Respond with a single JSON "
    "object and nothing else."
)


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def fmt_val(key, v):
    if _is_missing(v):
        return None
    if key in INT_KEYS:
        return int(v)
    if key in SCI_KEYS:
        return float(f"{float(v):.2e}")
    if key in BOOL_KEYS:
        return bool(v)
    if key in STR_KEYS:
        return str(v)
    return round(float(v), 3)


def schema_key(domain, stub):
    if domain == "gwas":
        return "gwas"
    if domain == "depmap":
        return stub["modality"]
    return stub.get("kind") or "evidence"  # tahoe


def y_completion(domain, stub):
    """Ordered, whitelisted, formatted y dict for the stub's schema, plus the key list (for the prompt)."""
    keys = SCHEMAS[schema_key(domain, stub)]
    y = stub.get("y") or {}
    return {k: fmt_val(k, y.get(k)) for k in keys}, keys


def requested_block(mode, support, woe):
    """The REQUESTED OUTCOME (conditioning) block for the chosen dropout mode."""
    head = ["REQUESTED OUTCOME", "================="]
    if mode == "none":
        return "\n".join(
            head
            + [
                "(unconstrained — predict the most likely outcome, and report the support "
                "and weight of evidence you expect)"
            ]
        )
    sup = "yes" if support else "no"
    if mode == "support":
        return "\n".join(head + [f"supports hypothesis: {sup}"])
    return "\n".join(head + [f"supports hypothesis: {sup}", f"weight of evidence: {woe:+.1f} decibans"])


def build_example(env, domain, row, i, support, woe, mode, rng, drop_info_set=False):
    """One SFT example: (target, [information set I=pool[:i]], candidate=pool[i] with y hidden,
    REQUESTED OUTCOME per `mode`) -> JSON completion {supports_hypothesis, weight_of_evidence, y}.

    ``drop_info_set=True`` omits the information set entirely (context-free simulator): the prompt is
    just target hypothesis + candidate setup + the requested BF conditioning. Pairs with singleton
    (prior-invariant) WoE labels."""
    pool = list(row["context_pool"])
    info_set, candidate = pool[:i], pool[i]
    lines = ["TARGET HYPOTHESIS", "=================", env.render_target(row), ""]
    if not drop_info_set:
        lines += ["INFORMATION SET (already-observed evidence)", "==========================================="]
        if info_set:
            for j, stub in enumerate(info_set, 1):
                lines.extend(env.render_evidence_stub(j, stub))
        else:
            lines.append("(no evidence observed yet)")
        lines.append("")
    lines += ["CANDIDATE EXPERIMENT (predict its outcome)", "=========================================="]
    lines.extend(env.render_evidence_stub(1, candidate, hide_result=True))
    y, keys = y_completion(domain, candidate)
    lines += [
        "",
        requested_block(mode, support, woe),
        "",
        'Respond with a single JSON object with keys "supports_hypothesis" ("yes"/"no"), '
        '"weight_of_evidence_decibans" (number), and "y" (an object with keys ' + ", ".join(keys) + ").",
    ]
    completion = json.dumps(
        {"supports_hypothesis": "yes" if support else "no", "weight_of_evidence_decibans": round(float(woe), 1), "y": y}
    )
    return {
        "messages": [
            {"role": "system", "content": SIM_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
            {"role": "assistant", "content": completion},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "domain": domain,
    }


def load_woe(woe_root, domain, split):
    """{(episode_id, stub_index): (support, woe)} from annotate_stub_woe's output, and per-episode index."""
    path = Path(woe_root) / f"{domain}.{split}.jsonl"
    by_ep = defaultdict(list)
    labels = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            key = (d["episode_id"], d["stub_index"])
            labels[key] = (bool(d["support"]), float(d["woe_decibans"]))
            by_ep[d["episode_id"]].append(d["stub_index"])
    return labels, by_ep


def build_split(domain, split, args, rng):
    env = importlib.import_module(f"{domain}_meta")
    df = load_dataset_split(f"{args.data_root}/{domain}", Split(split))
    rows = {r["episode_id"]: r for r in df.to_dict("records")}
    labels, by_ep = load_woe(args.woe_root, domain, split)
    modes = ["none", "support", "support_woe"]
    weights = [args.w_none, args.w_support, args.w_support_woe]
    out = []
    for eid, stub_idxs in by_ep.items():
        row = rows.get(eid)
        if row is None:
            continue
        cands = sorted(stub_idxs)
        if args.filter_support == "pos":
            cands = [i for i in cands if labels[(eid, i)][0]]
        elif args.filter_support == "neg":
            cands = [i for i in cands if not labels[(eid, i)][0]]
        if not cands:
            continue
        if len(cands) > args.candidates_per_episode:
            cands = rng.sample(cands, args.candidates_per_episode)
        for i in cands:
            support, woe = labels[(eid, i)]
            mode = "support" if args.sign_only else rng.choices(modes, weights=weights, k=1)[0]
            out.append(build_example(env, domain, row, i, support, woe, mode, rng, drop_info_set=args.drop_info_set))
    rng.shuffle(out)
    if args.max_per_domain_split and len(out) > args.max_per_domain_split:
        out = out[: args.max_per_domain_split]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woe-root", default="ibis_data/bed_sim/woe")
    ap.add_argument("--data-root", default="ibis_data/bed_sim/data")
    ap.add_argument("--out-dir", default="ibis_data/bed_sim/sft")
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--candidates-per-episode", type=int, default=6)
    ap.add_argument(
        "--max-per-domain-split", type=int, default=60000, help="cap examples per (domain, split); 0 = no cap"
    )
    ap.add_argument("--w-none", type=float, default=0.25)
    ap.add_argument("--w-support", type=float, default=0.40)
    ap.add_argument("--w-support-woe", type=float, default=0.35)
    # Context-free, sign-conditioned simulator:
    ap.add_argument("--drop-info-set", action="store_true", help="omit the information set from the prompt")
    ap.add_argument("--sign-only", action="store_true", help="condition on BF sign only (mode='support' for all)")
    ap.add_argument(
        "--filter-support",
        choices=["all", "pos", "neg"],
        default="all",
        help="keep only supporting (pos) / refuting (neg) stubs -> single-class model",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rng = random.Random(args.seed)
        rows = []
        for domain in args.domains:
            r = build_split(domain, split, args, rng)
            print(f"[{split}] {domain}: {len(r)} examples")
            rows.extend(r)
        rng.shuffle(rows)
        df = pd.DataFrame(
            {
                "messages": [r["messages"] for r in rows],
                "chat_template_kwargs": [r["chat_template_kwargs"] for r in rows],
                "domain": [r["domain"] for r in rows],
            }
        )
        dst = out / f"{split}.parquet"
        df.to_parquet(dst, index=False)
        print(f"[{split}] wrote {len(df)} examples -> {dst}")
        if rows:
            ex = rows[0]
            print(f"  sample ({ex['domain']}) user[-600:]:\n{ex['messages'][1]['content'][-600:]}")
            print(f"  sample completion: {ex['messages'][2]['content']}")


if __name__ == "__main__":
    main()
