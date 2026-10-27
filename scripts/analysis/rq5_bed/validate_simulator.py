"""Check that the theta' outcome simulator produces controllable outcomes.

For held-out candidate experiments (stubs from the EDB-Bio test split, same subset the simulator was
trained on), request BOTH valences for the same setup — compelling support for theta' and a clear null —
decode greedily, and judge each generated outcome with the domain's significance rule (GWAS
p < 5e-8; DepMap CRISPR Chronos <= -0.5 and P(dependency) >= 0.5; Tahoe padj < 0.05 and |log2FC| > 0.5).
The headline check is paired: for the same setup, the support-pole outcome should be stronger than the
null-pole outcome (GWAS -log10 p, DepMap -Chronos, Tahoe |log2FC|). The simulator was trained on
belief-model valence labels, not these thresholds, so the rule pass rates are reported for comparison
with the real training outcomes rather than as a pass/fail criterion.

The paper's controllability measure scores each generated outcome with the belief model instead:
the realized weight of evidence for theta', compared between the two poles of each candidate. Pass
``--synth-out`` to write the generated outcomes as one-stub episodes, then score and report them:

    uv run python scripts/analysis/rq5_bed/validate_simulator.py EdisonScientific/Ibis0-BED-Simulator \
        --synth-out ibis_data/bed_sim/tp_eval
    for d in gwas depmap tahoe; do
      ANNOT_MODE=theta_prime ANNOT_SPLIT=train ANNOT_OUT=ibis_data/bed_sim/tp_eval_woe/$d.train.jsonl \
        uv run torchrun --standalone --nproc-per-node 8 scripts/data_pipeline/annotate_stub_woe.py \
        @ train/_common/classification-base.toml --domain $d --data-path ibis_data/bed_sim/tp_eval/$d \
        --output-dir <classifier-run-dir> --model.seq-len 16384 --ckpt.resume-step 2400
    done
    uv run python scripts/analysis/rq5_bed/theta_prime_eval_report.py

With the released simulator and classifier, 83% of candidates (100 per domain) get a higher realized
weight of evidence for the support pole than for the null pole.
"""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from ibis_env_utils import Split, load_dataset_split
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "data_pipeline"))
sys.path.insert(0, str(_HERE))
import build_theta_prime_sft as T  # noqa: E402
from theta_prime_validate import significant, strength  # noqa: E402


def in_subset(domain: str, stub: dict) -> bool:
    if domain == "depmap":
        return stub.get("modality") == "crispr"
    if domain == "tahoe":
        return (stub.get("kind") or "evidence") == "evidence"
    return True


def parse_y(text: str) -> dict | None:
    try:
        y = json.loads(text[text.find("{") : text.rfind("}") + 1])["y"]
    except (ValueError, KeyError, TypeError):
        return None
    return y if isinstance(y, dict) else None


def emitted_support(text: str) -> str | None:
    try:
        return json.loads(text[text.find("{") : text.rfind("}") + 1]).get("supports_hypothesis")
    except (ValueError, AttributeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="simulator: Hub repo id or local directory")
    ap.add_argument("--data-root", default="hf://EdisonScientific/EDB-Bio")
    ap.add_argument("--revision", default="v1.0.0")
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    ap.add_argument("--n-per-domain", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="optional JSONL of every generation")
    ap.add_argument("--synth-out", type=Path, default=None, help="write one-stub episodes for belief scoring")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        .to("cuda")
        .eval()
    )
    log = args.out.open("w") if args.out else None
    if args.synth_out:
        args.synth_out.mkdir(parents=True, exist_ok=True)
        gen_meta = (args.synth_out / "gen_meta.jsonl").open("w")

    for domain in args.domains:
        env = importlib.import_module(f"{domain}_meta")
        df = load_dataset_split(f"{args.data_root}/{domain}", Split.TEST, revision=args.revision)
        cands = [(r, i) for r in df.to_dict("records") for i, s in enumerate(r["context_pool"]) if in_subset(domain, s)]
        rng.shuffle(cands)
        tasks = [(r, i, support) for r, i in cands[: args.n_per_domain] for support in (True, False)]

        outcome: dict[tuple[str, int], dict[bool, bool | None]] = {}
        strengths: dict[tuple[str, int], dict[bool, float | None]] = {}
        synth = []
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            prompts = [
                tok.apply_chat_template(
                    T.build_example(env, domain, r["context_pool"][i], support, 0.0, rng)["messages"][:2],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for r, i, support in batch
            ]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=220, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for (r, i, support), text in zip(batch, texts):
                y = parse_y(text)
                sig = significant(domain, y) if y is not None else None
                outcome.setdefault((r["episode_id"], i), {})[support] = sig
                strengths.setdefault((r["episode_id"], i), {})[support] = strength(domain, y) if y else None
                if args.synth_out:
                    synth_id = f"{r['episode_id']}:{i}__{'P' if support else 'N'}"
                    gen_meta.write(
                        json.dumps(
                            {
                                "synth_id": synth_id,
                                "domain": domain,
                                "valence": support,
                                "emitted_support": emitted_support(text),
                            }
                        )
                        + "\n"
                    )
                    if y is not None:
                        stub = {**r["context_pool"][i], "y": {**(r["context_pool"][i].get("y") or {}), **y}}
                        synth.append(
                            {
                                **{k: v for k, v in r.items() if k != "context_pool"},
                                "episode_id": synth_id,
                                "context_pool": [stub],
                            }
                        )
                if log:
                    log.write(
                        json.dumps(
                            {
                                "domain": domain,
                                "episode_id": r["episode_id"],
                                "stub": i,
                                "support": support,
                                "significant": sig,
                                "text": text,
                            }
                        )
                        + "\n"
                    )

        if args.synth_out:
            (args.synth_out / domain).mkdir(exist_ok=True)
            pd.DataFrame(synth).to_parquet(args.synth_out / domain / "train.parquet", index=False)
        scored = [v for v in strengths.values() if v.get(True) is not None and v.get(False) is not None]
        stronger = sum(v[True] > v[False] for v in scored)
        pairs = list(outcome.values())
        parsed = [p for p in pairs if p.get(True) is not None and p.get(False) is not None]
        controlled = sum(p[True] and not p[False] for p in parsed)
        print(
            f"{domain}: {len(pairs)} candidates, {len(parsed)} with both outcomes parsed; "
            f"support pole significant {sum(p[True] for p in parsed) / max(len(parsed), 1):.0%}, "
            f"null pole significant {sum(p[False] for p in parsed) / max(len(parsed), 1):.0%}, "
            f"both rule-significant/null {controlled / max(len(parsed), 1):.0%}; "
            f"support stronger than null in {stronger / max(len(scored), 1):.0%} of pairs",
            flush=True,
        )


if __name__ == "__main__":
    main()
