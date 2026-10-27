"""Bayesian experimental design (BED) proof-of-concept for tahoe_meta.

Sequential experiment selection: starting from an empty evidence pool, reveal stubs one at a time and
track how fast the belief model's P(supported) approaches the truth. Compares three policies for
choosing the next stub to reveal:

  random        — uniform random unrevealed stub
  model_select  — ask the belief model which candidate would most change its belief (cheap heuristic)
  eig           — expected information gain: for each candidate, sample plausible values for its
                  outcome, score each with the BELIEF model, EIG = H(p0) - E[H(p_i)], reveal argmax.

The EIG value-sampler has two modes:
  empirical (default) — a candidate's plausible outcomes are the real DE values already observed for
                        the target gene across the corpus (bed_value_population.py); no second model,
                        so --n-samples can be large for sharp EIG estimates.
  sft                 — an SFT generator model samples the candidate's values (--generator-url).

The belief model is always served as an OpenAI-compatible vLLM endpoint (--belief-url); the SFT
generator endpoint (--generator-url) is only needed for --generator-mode sft. Run via SLURM.
"""

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

from episodes import (
    SYSTEM_PROMPT_DIRECT,
    PromptRenderer,
    TahoeEvidenceEpisode,
    TargetHypothesis,
    _stub_to_context,
)
from ibis_env_utils import load_dataset_split, parse_prob

# bed_value_population + build_stub_generator_sft live under scripts/ and aren't packaged — add their dirs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "scripts") / "data_pipeline"))
from bed_value_population import ValuePopulation
from build_stub_generator_sft import build_example  # generator prompt, byte-identical to training

RENDERER = PromptRenderer(mode="compact", lfc_threshold=0.5, prompt_style="direct")
H = lambda p: 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def evidence_stubs(row):
    cp = row["context_pool"]
    if cp is None or isinstance(cp, float):
        return []
    return [s for s in cp if s.get("kind") == "evidence" and (s.get("y") or {}).get("log2FoldChange") is not None]


def render_belief_prompt(row, revealed_stubs, target=None):
    """Belief-model prompt for a revealed subset. `target` overrides the hypothesis being asked about
    (dict: drug, cell, gene, direction, drug_meta, cell_meta, gene_meta); default = the row's target."""
    if target is None:
        target = dict(
            drug=row["target_drug"],
            cell=row["target_cell"],
            gene=row["target_gene"],
            direction=row["target_direction"],
            drug_meta=row.get("target_drug_meta"),
            cell_meta=row.get("target_cell_meta"),
            gene_meta=row.get("target_gene_meta"),
        )
    ep = TahoeEvidenceEpisode(
        episode_id=row["episode_id"],
        split="test",
        target=TargetHypothesis(
            target_id="TARGET_1",
            drug=target["drug"],
            cell_line=target["cell"],
            gene=target["gene"],
            direction=target["direction"],
        ),
        context=[_stub_to_context(s) for s in revealed_stubs],
        label=int(row["label"]),
        metadata={
            "lfc_threshold": float(row.get("lfc_threshold", 0.5) or 0.5),
            "target_drug_meta": dict(target["drug_meta"] or {}),
            "target_cell_meta": dict(target["cell_meta"] or {}),
            "target_gene_meta": dict(target["gene_meta"] or {}),
        },
    )
    return RENDERER.render(ep)


_CONSEC_FAIL = 0  # consecutive fully-failed chat calls; a dead server trips this fast
_FAIL_ABORT = 300  # abort the run if this many calls fail back-to-back (partial results are saved)


async def _chat(
    client, model, system, user, sem, *, n=1, temperature=0.0, max_tokens=512, thinking=False, openai_reasoning=False
):
    global _CONSEC_FAIL
    async with sem:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        for attempt in range(4):
            try:
                if openai_reasoning:  # GPT-5.x reasoning model via the OpenAI API
                    r = await client.chat.completions.create(model=model, messages=msgs, max_completion_tokens=16000)
                else:
                    r = await client.chat.completions.create(
                        model=model,
                        n=n,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=msgs,
                        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
                    )
                _CONSEC_FAIL = 0
                return [c.message.content for c in r.choices]
            except Exception as e:  # deliberate fault tolerance for flaky serving
                if attempt == 3:
                    _CONSEC_FAIL += 1
                    if _CONSEC_FAIL >= _FAIL_ABORT:
                        raise SystemExit(
                            f"[bed] aborting: belief server unreachable for {_CONSEC_FAIL} "
                            f"consecutive calls (last error: {e})"
                        )
                    print(f"[bed] chat failed: {e}", file=sys.stderr)
                    return [""] * n
                await asyncio.sleep(2 * (attempt + 1))


def _mname(args, key):
    return {"belief": args.belief_model, "base": args.base_model, "gpt55": args.gpt_model}[key]


async def belief_prob(clients, args, sem, row, revealed, model_key="belief", target=None):
    """P(supported) for `target` (default = row's hypothesis) given `revealed`, per the chosen model.
    model_key in {belief (RL'd, also the trajectory tracker), base, gpt55}."""
    out = await _chat(
        clients[model_key],
        _mname(args, model_key),
        SYSTEM_PROMPT_DIRECT,
        render_belief_prompt(row, revealed, target),
        sem,
        temperature=0.0,
        max_tokens=64,
        thinking=False,
        openai_reasoning=(model_key == "gpt55"),
    )
    p = parse_prob(out[0])
    return 0.5 if p is None else p


VAL_KEYS = ["log2FoldChange", "lfcSE", "padj", "baseMean", "n_cells_trt", "n_cells_ctrl"]


async def gen_candidate_values(clients, args, sem, row, revealed, cand):
    """Sample n plausible value-dicts for candidate stub `cand` from the generator."""
    prompt = build_example(row, cand, revealed, random.Random(0))["messages"][0]["content"]
    outs = await _chat(
        clients["gen"],
        args.gen_model,
        None,
        prompt,
        sem,
        n=args.n_samples,
        temperature=args.gen_temp,
        max_tokens=160,
        thinking=False,
    )
    vals = []
    for o in outs:
        try:
            d = json.loads(o[o.find("{") : o.rfind("}") + 1])
            if all(k in d for k in VAL_KEYS):
                vals.append(d)
        except Exception:
            continue
    return vals


def _candidate_dc(cand):
    return ((cand.get("drug_metadata") or {}).get("drug"), (cand.get("cell_metadata") or {}).get("cell_name"))


async def candidate_values(clients, args, sem, row, revealed, cand):
    """Sample plausible outcome value-dicts for `cand`: empirically from the target gene's observed
    responses (default) or from the SFT generator (--generator-mode sft)."""
    if args.generator_mode == "empirical":
        rng = random.Random(hash((row["episode_id"], _candidate_dc(cand))) & 0xFFFFFFFF)
        return clients["population"].sample(
            row["target_gene"],
            args.n_samples,
            rng,
            exclude={(row["target_drug"], row["target_cell"]), _candidate_dc(cand)},
        )
    return await gen_candidate_values(clients, args, sem, row, revealed, cand)


async def eig_of(clients, args, sem, row, revealed, cand, p0):
    """Expected information gain of revealing `cand`: sample its plausible values, score each with the
    belief model, EIG = H(p0) - E[H(posterior)]."""
    vals = await candidate_values(clients, args, sem, row, revealed, cand)
    if not vals:
        return 0.0

    async def score(v):
        hyp = dict(cand)
        hyp["y"] = {**(cand.get("y") or {}), **v}
        return await belief_prob(clients, args, sem, row, revealed + [hyp])

    posts = await asyncio.gather(*[score(v) for v in vals])
    return H(p0) - sum(H(p) for p in posts) / len(posts)


def outcome_exemplars(population, gene, direction, lfc_threshold):
    """For gene G: a representative SUPPORTED-in-direction value-dict and a representative NULL one,
    drawn from G's empirical responses (median |log2FC| within each subset)."""
    want_up = str(direction).lower().startswith("up")
    pos, null = [], []
    for _, _, v in population.by_gene.get(gene, []):
        lfc, padj = v.get("log2FoldChange"), v.get("padj")
        if lfc is None or padj is None:
            continue
        if padj < 0.05 and abs(lfc) > lfc_threshold and ((lfc > 0) == want_up):
            pos.append(v)
        elif padj >= 0.05 or abs(lfc) <= lfc_threshold:
            null.append(v)

    def med(lst):
        return sorted(lst, key=lambda v: abs(v["log2FoldChange"]))[len(lst) // 2] if lst else None

    return med(pos), (min(null, key=lambda v: abs(v["log2FoldChange"])) if null else None)


def _cand_target(cand, row):
    """Candidate `cand` framed as its own hypothesis, in the target's claimed direction."""
    return dict(
        drug=(cand.get("drug_metadata") or {}).get("drug"),
        cell=(cand.get("cell_metadata") or {}).get("cell_name"),
        gene=row["target_gene"],
        direction=row["target_direction"],
        drug_meta=cand.get("drug_metadata"),
        cell_meta=cand.get("cell_metadata"),
        gene_meta=row.get("target_gene_meta"),
    )


async def eig_self(clients, args, sem, row, revealed, cand, p0, model_key):
    """Self-predictive EIG (3 belief calls): p = P(cand supported | I_t); then the belief about the
    target under the imagined positive/negative outcomes of cand; EIG = H(p0) - [p H(q+) + (1-p) H(q-)].
    The selecting model is `model_key`; the imagined outcome values are outcome-split empirical exemplars."""
    thresh = float(row.get("lfc_threshold", 0.5) or 0.5)
    pos_v, null_v = outcome_exemplars(clients["population"], row["target_gene"], row["target_direction"], thresh)
    if pos_v is None or null_v is None:
        return 0.0
    p = await belief_prob(clients, args, sem, row, revealed, model_key=model_key, target=_cand_target(cand, row))
    hyp_pos = dict(cand)
    hyp_pos["y"] = {**(cand.get("y") or {}), **pos_v}
    hyp_neg = dict(cand)
    hyp_neg["y"] = {**(cand.get("y") or {}), **null_v}
    qpos, qneg = await asyncio.gather(
        belief_prob(clients, args, sem, row, revealed + [hyp_pos], model_key=model_key),
        belief_prob(clients, args, sem, row, revealed + [hyp_neg], model_key=model_key),
    )
    return H(p0) - (p * H(qpos) + (1 - p) * H(qneg))


async def pick_next(clients, args, sem, row, revealed, candidates, policy, rng, p0):
    if policy == "random":
        return rng.randrange(len(candidates))
    naive_key = {"model_select": "belief", "naive_base": "base", "naive_gpt55": "gpt55"}.get(policy)
    if naive_key is not None:
        # the selecting model is asked which candidate to reveal; the belief tracker is always the RL'd
        # model, so the y-axis isolates selection quality regardless of which model chooses.
        lines = [
            f"[{i}] {(_stub_to_context(c).drug_metadata or {}).get('drug', '?')} in "
            f"{(_stub_to_context(c).cell_metadata or {}).get('cell_name', '?')}"
            for i, c in enumerate(candidates)
        ]
        user = (
            render_belief_prompt(row, revealed)
            + "\n\nCANDIDATE NEXT EXPERIMENTS (values hidden):\n"
            + "\n".join(lines)
            + "\n\nWhich single candidate experiment, once its result is revealed, would most change your "
            "belief about the target hypothesis? Answer with just its index as <choice>i</choice>."
        )
        out = await _chat(
            clients[naive_key],
            _mname(args, naive_key),
            SYSTEM_PROMPT_DIRECT,
            user,
            sem,
            max_tokens=64,
            openai_reasoning=(naive_key == "gpt55"),
        )
        import re

        m = re.findall(r"<choice>\s*(\d+)\s*</choice>", out[0])
        return int(m[-1]) % len(candidates) if m else rng.randrange(len(candidates))
    if policy == "eig":
        cand_idx = list(range(len(candidates)))
        if len(cand_idx) > args.max_candidates:
            cand_idx = rng.sample(cand_idx, args.max_candidates)
        eigs = await asyncio.gather(*[eig_of(clients, args, sem, row, revealed, candidates[i], p0) for i in cand_idx])
        return cand_idx[max(range(len(eigs)), key=lambda j: eigs[j])]
    eig_key = {"eig_self_rld": "belief", "eig_self_base": "base", "eig_self_gpt55": "gpt55"}.get(policy)
    if eig_key is not None:
        cand_idx = list(range(len(candidates)))
        if len(cand_idx) > args.max_candidates:
            cand_idx = rng.sample(cand_idx, args.max_candidates)
        eigs = await asyncio.gather(
            *[eig_self(clients, args, sem, row, revealed, candidates[i], p0, eig_key) for i in cand_idx]
        )
        return cand_idx[max(range(len(eigs)), key=lambda j: eigs[j])]


async def run_episode(clients, args, sem, row, policy):
    rng = random.Random(hash((row["episode_id"], policy)) & 0xFFFF)
    pool = evidence_stubs(row)
    rng.shuffle(pool)
    revealed, remaining = [], list(pool)
    label = int(row["label"])
    traj = [{"t": 0, "p": await belief_prob(clients, args, sem, row, revealed), "y": label}]
    for t in range(1, min(args.budget, len(remaining)) + 1):
        p0 = traj[-1]["p"]
        j = await pick_next(clients, args, sem, row, revealed, remaining, policy, rng, p0)
        revealed.append(remaining.pop(j))
        traj.append({"t": t, "p": await belief_prob(clients, args, sem, row, revealed), "y": label})
    return {"episode_id": row["episode_id"], "policy": policy, "label": label, "traj": traj}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--belief-url", required=True)
    ap.add_argument("--belief-model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--base-model", default=None)  # for *_base policies
    ap.add_argument("--gpt-model", default=None)
    ap.add_argument("--gpt-url", default=None)
    ap.add_argument("--gpt-key", default=None)  # OpenAI key for *_gpt55 policies (defaults to env)
    ap.add_argument("--generator-mode", choices=["empirical", "sft"], default="empirical")
    ap.add_argument("--generator-url", default=None)
    ap.add_argument("--gen-model", default=None)
    ap.add_argument("--split", required=True, help="dataset split to evaluate on (e.g. test)")
    ap.add_argument(
        "--max-episodes",
        type=int,
        required=True,
        help="cap on episodes run (truncates eval set after split/stub filter)",
    )
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=128)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--policies", default="random,model_select,eig")
    ap.add_argument("--out", default="outputs/bed/bed_results.jsonl")
    args = ap.parse_args()

    from openai import AsyncOpenAI

    clients = {"belief": AsyncOpenAI(base_url=args.belief_url, api_key="x")}
    if args.base_url:
        assert args.base_model, "--base-model required with --base-url"
        clients["base"] = AsyncOpenAI(base_url=args.base_url, api_key="x")
    if args.gpt_model:
        clients["gpt55"] = (
            AsyncOpenAI(base_url=args.gpt_url, api_key=args.gpt_key)
            if args.gpt_url
            else AsyncOpenAI(api_key=args.gpt_key)
        )
    if args.generator_mode == "sft":
        assert args.generator_url and args.gen_model, "--generator-url/--gen-model required for --generator-mode sft"
        clients["gen"] = AsyncOpenAI(base_url=args.generator_url, api_key="x")
    # the empirical generator and the self-predictive EIG both need the per-gene value population
    if args.generator_mode == "empirical" or any(p.startswith("eig_self") for p in args.policies.split(",")):
        clients["population"] = ValuePopulation(args.data_path)
        print(f"empirical value population: {clients['population'].stats()}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)

    # Load exactly the split to evaluate on (must be passed explicitly); every row is that split.
    rows = load_dataset_split(args.data_path, args.split).to_dict("records")
    eval_rows = [r for r in rows if len(evidence_stubs(r)) >= args.budget][: args.max_episodes]
    if not eval_rows:
        raise SystemExit(f"no episodes with split=={args.split!r} and >= {args.budget} stubs in {args.data_path}")
    policies = args.policies.split(",")
    print(f"BED: {len(eval_rows)} episodes (split={args.split}) x {policies} x budget {args.budget}")

    tasks = [run_episode(clients, args, sem, row, pol) for row in eval_rows for pol in policies]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # write each episode as it completes so a mid-run server crash still leaves usable partial results
    results = []
    with open(out, "w") as f:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()

    # belief Brier vs #experiments, per policy
    print(f"\n{'step':>5}" + "".join(f"{p:>16}" for p in policies))
    for t in range(args.budget + 1):
        rowstr = f"{t:>5}"
        for pol in policies:
            briers = [(s["p"] - s["y"]) ** 2 for r in results if r["policy"] == pol for s in r["traj"] if s["t"] == t]
            rowstr += f"{(sum(briers) / len(briers) if briers else float('nan')):>16.4f}"
        print(rowstr)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
