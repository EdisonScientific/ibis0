"""Score the published test set with an exported (Hugging Face) Ibis0 classifier.

Builds each record exactly as the DCP evaluation does (``scripts/analysis/multidomain/
eval_classification_testset.py``: the env's ``build_record`` + ``system_prompt_for("direct")``, the
chat template without thinking) and writes one ``<domain>_<pass>.json`` per pass in the same format.
``--compare <dir>`` checks the probabilities against an earlier evaluation of the same checkpoint.

    uv run python scripts/export/predict_classifier.py EdisonScientific/Ibis0-30B-A3B-Classifier \\
        --out-dir outputs/classifier-hf --domains gwas depmap tahoe
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ibis_env_utils import Split, load_dataset_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from testset_io import testset_passes  # noqa: E402

TESTSET = "hf://EdisonScientific/EDB-Bio/testsets/{config}/{domain}"
TESTSET_REV = "df43070d4a1338210ed33c368d49e58da60471cb"


def records_for(domain: str, config: str) -> list[tuple[str, list[dict]]]:
    env = importlib.import_module(f"{domain}_meta")
    system_prompt = env.system_prompt_for("direct")
    df = load_dataset_split(TESTSET.format(config=config, domain=domain), Split.TEST, revision=TESTSET_REV)
    passes = []
    for tag, rows in testset_passes(df):
        records = []
        for r in rows:
            br = env.build_record(r, len(r["context_pool"]))
            records.append(
                {
                    "episode_id": br["info"]["episode_id"],
                    "domain": domain,
                    "label": br["info"]["label"],
                    "n_stubs": len(r["context_pool"]),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": br["question"]},
                    ],
                }
            )
        passes.append((tag, records))
    return passes


@torch.no_grad()
def predict(model, tokenizer, messages: list[dict]) -> float:
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True
    )["input_ids"].to(model.device)
    return torch.sigmoid(model(input_ids=input_ids).logits.float()).item()


def compare(results: list[dict], reference: Path) -> None:
    ref = {r["episode_id"]: r["parsed_prob"] for r in json.loads(reference.read_text())["results"]}
    diffs = np.array([abs(r["parsed_prob"] - ref[r["episode_id"]]) for r in results if r["episode_id"] in ref])
    print(
        f"  vs {reference}: n={len(diffs)} max|Δp|={diffs.max():.2e} mean|Δp|={diffs.mean():.2e} "
        f"(missing from reference: {len(results) - len(diffs)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="exported classifier: Hub repo id or local directory")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--domains", nargs="+", default=["gwas", "depmap", "tahoe"])
    ap.add_argument("--testset-config", choices=["primary", "sensitivity"], default="primary")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N records of each pass")
    ap.add_argument("--compare", type=Path, default=None, help="directory of reference <domain>_<pass>.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            args.model, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to("cuda")
        .eval()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for domain in args.domains:
        for tag, records in records_for(domain, args.testset_config):
            records = records[: args.limit]
            results = [
                {k: r[k] for k in ("episode_id", "domain", "label", "n_stubs")}
                | {"parsed_prob": predict(model, tokenizer, r["messages"])}
                for r in records
            ]
            out = args.out_dir / f"{domain}_{tag}.json"
            out.write_text(json.dumps({"model": args.model, "domain": domain, "pass": tag, "results": results}))
            print(f"{domain}/{tag}: {len(results)} records -> {out}")
            if args.compare:
                compare(results, args.compare / out.name)


if __name__ == "__main__":
    main()
