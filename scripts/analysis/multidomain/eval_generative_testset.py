"""Evaluate one generative model on a published multi-domain TEST set, writing per-episode JSON cells
into the eval tree defined by scripts/lib/eval_paths.py, which every renderer reads.

Reads a config of the frozen test set (`testsets/` in `EdisonScientific/EDB-Bio`), where every row is
already **one forward pass**: `context_pool` pre-truncated, `k`/`n_stubs` resolved. There is therefore no
episode sampling and no context flags here — the published set *is* the definition, identical for every
model in the roster. (`--seed` seeds *decoding*, not episode selection, since sampling is not greedy.)
A `primary` config yields one pass; a `sensitivity` config yields one pass per `k` (the dense nested
sweep). Each pass draws `--n-samples` completions per episode and reports their mean.

Domain-agnostic: `--domain gwas|depmap|tahoe` selects the `<domain>_meta` env module (all expose the same
`parse_prob` / `render_user` / `system_prompt_for` surface).

**Local served weights only.** This script talks to a vLLM OpenAI-compatible endpoint (`--base-url`).
Frontier API models go through `eval_frontier_testset.py`, which sets reasoning effort and output budget
explicitly and symmetrically across providers; keeping one code path per measurement avoids two paths
that silently disagree.
"""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = next(p for p in Path(__file__).resolve().parents if p.name == "scripts")
_ENVS = _SCRIPTS.parent / "envs"
sys.path.insert(0, str(_SCRIPTS / "lib"))

from ibis_env_utils import Split, load_dataset_split  # noqa: E402
from testset_io import testset_passes  # noqa: E402


def _load_env(domain: str):
    """Import the `<domain>_meta` env module (parse_prob / render_user / system_prompt_for)."""
    sys.path.insert(0, str(_ENVS / f"{domain}_meta"))
    return importlib.import_module(f"{domain}_meta")


def _testset_passes(df) -> list[tuple[str, list[dict]]]:
    """Thin alias for the shared reader — see scripts/lib/testset_io.testset_passes."""
    return testset_passes(df)


async def _chat(client, model, user, sem, sysprompt, enable_thinking=False, temperature=0.7, seed=0, n=4):
    """One episode against the served vLLM endpoint. Returns `n` completions.

    Decoded at the temperature the policy was trained and validated at, not greedily. An RL policy
    sampled at 0.7 and a greedy decode of the same weights are different forecasters, so scoring the
    second tells you little about the first -- and the base model, which is the comparison, has no
    training temperature to match, which is exactly why it has to be held equal to the trained rows.

    Sampling is not free of noise, so each episode is scored from the mean of `n` draws rather than one.
    The draws are requested as a single `n=` call so the prompt is prefilled once and only decoding is
    repeated; issuing `n` separate requests would re-prefill a context of up to 16 evidence stubs each
    time, which is most of the cost.
    """
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": sysprompt}, {"role": "user", "content": user}],
                    temperature=temperature,
                    seed=seed,
                    n=n,
                    # Reasoning models need native thinking on + a large budget; direct models get
                    # thinking off + a short budget (they emit the probability immediately).
                    max_tokens=24000 if enable_thinking else 2048,
                    extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
                )
                return [c.message.content or "" for c in r.choices]
            except Exception as e:
                if attempt == 3:
                    print(f"[gen-testset] chat failed: {e}", file=sys.stderr)
                    return [""] * n
                await asyncio.sleep(3 * (attempt + 1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="env module <domain>_meta to render/parse episodes")
    ap.add_argument(
        "--data-path",
        required=True,
        help="published test-set config, e.g. "
        "hf://EdisonScientific/EDB-Bio/testsets/primary/gwas (a local directory of the same "
        "shape also works — load_dataset_split treats the two identically)",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help="Hub revision of an hf:// test set (the launchers pass the pinned TESTSET_REV); ignored for a local directory",
    )
    ap.add_argument("--base-url", required=True, help="served vLLM OpenAI-compatible endpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="x", help="dummy; vLLM requires the header but ignores the value")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument(
        "--out-dir",
        required=True,
        help="result directory for THIS model, from scripts/lib/eval_paths.result_dir(). Cells land at "
        "<out-dir>/<domain>_<tag>.json; the launcher owns the layout and writes run.json.",
    )
    ap.add_argument("--enable-thinking", action="store_true", help="local reasoning models: native thinking on")
    ap.add_argument(
        "--prior-note", action="store_true", help="append the shared PRIOR_NOTE (0.5-prior anchor) to the system prompt"
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="decoding temperature. Defaults to the 0.7 the RL policies were trained and validated "
        "at, so the benchmark scores the policy rather than a greedy decode of it. Held equal "
        "across every generative row, trained or not.",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed, so a temperature>0 run reproduces")
    ap.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="draws per episode; the reported probability is their mean. At temperature>0 a single "
        "draw is itself a random variable, and this benchmark is about calibration, so one draw per "
        "episode would put sampling noise into every reported spread.",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["direct", "reasoning"],
        default="reasoning",
        help="system prompt: 'direct' (answer-only) or 'reasoning' (think-first). Set per model — "
        "base/direct-RL use 'direct'; reasoning-RL/frontier use 'reasoning'.",
    )
    args = ap.parse_args()

    env = _load_env(args.domain)  # gwas_meta / depmap_meta / tahoe_meta (identical public surface)
    parse_prob, render_user, system_prompt_for = env.parse_prob, env.render_user, env.system_prompt_for

    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    sysprompt = system_prompt_for(
        args.prompt_style, prior_note=args.prior_note
    )  # 'direct' -> answer-only, 'reasoning' -> think-first

    # The launcher expands a {domain} template into this path, so leftover braces mean the expansion
    # went wrong (e.g. bash's ${TESTSET:-...{domain}} mis-parse, which appends a stray "}").
    if "{" in args.data_path or "}" in args.data_path:
        raise ValueError(f"--data-path still contains braces (unexpanded/mis-expanded template?): {args.data_path!r}")
    # A published test set publishes only `test`, so the split is not a parameter here.
    df = load_dataset_split(args.data_path, Split.TEST, revision=args.revision or None)
    if "testset" not in df.columns:
        raise ValueError(
            f"{args.data_path} is not a published test set (no `testset` column); build one "
            "with build_static_testset.py."
        )
    passes = _testset_passes(df)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for tag, kr in passes:
        outs = await asyncio.gather(
            *[
                _chat(
                    client,
                    args.model,
                    render_user(r),
                    sem,
                    sysprompt,
                    args.enable_thinking,
                    args.temperature,
                    args.seed,
                    args.n_samples,
                )
                for r in kr
            ]
        )
        results, n_parsed = [], 0
        for r, os_ in zip(kr, outs):
            draws = [parse_prob(o) for o in os_]
            got = [d for d in draws if d is not None]
            # The episode's forecast is the mean of the draws that parsed. It counts as unreadable only
            # when every draw failed: one malformed sample out of n is a decoding hiccup, not a refusal
            # to answer, and discarding the whole episode for it would bias the retained set toward
            # prompts the model finds easy to format.
            p = float(np.mean(got)) if got else None
            n_parsed += p is not None
            # n_stubs is the ACTUAL pool-capped count, carried on the row rather than re-derived.
            results.append(
                {
                    "episode_id": r["episode_id"],
                    "domain": args.domain,
                    "label": int(r["label"]),
                    "parsed_prob": p,
                    "n_draws": len(draws),
                    "n_draws_parsed": len(got),
                    "draw_sd": (float(np.std(got, ddof=1)) if len(got) > 1 else None),
                    "ground_truth": r.get("ground_truth"),
                    "n_stubs": int(r["n_stubs"]),
                    "raw": (os_[0][:1200] if p is None else None),
                }
            )
        path = str(Path(args.out_dir) / f"{args.domain}_{tag}.json")
        Path(path).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "domain": args.domain,
                    "testset": args.data_path,
                    "context_size": tag,
                    "split": "test",
                    "temperature": args.temperature,
                    "seed": args.seed,
                    "n_samples": args.n_samples,
                    "n_eval": len(results),
                    "results": results,
                }
            )
        )
        probs = [x["parsed_prob"] for x in results if x["parsed_prob"] is not None]
        labs = [x["label"] for x in results if x["parsed_prob"] is not None]
        brier = float(np.mean([(p - y) ** 2 for p, y in zip(probs, labs)])) if probs else float("nan")
        print(
            f"  {tag:>7}: parsed {n_parsed}/{len(kr)}  Brier={brier:.4f}  "
            f"mean_p={np.mean(probs) if probs else float('nan'):.3f}  -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
