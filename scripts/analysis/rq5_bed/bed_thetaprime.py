"""Sequential BED with the theta' evidence simulator — the headline experiment.

At each step, for every candidate experiment x in the pool, estimate the expected information gain
(EIG) about the GLOBAL target hypothesis theta, pick argmax-EIG, REVEAL the real held-out outcome,
update belief P(theta|I), repeat for a horizon of `budget` steps. Only the *selection* differs across
arms; the revealed evidence is always the real measurement.

Arms (EIG estimators), all using ONE simulator (the theta' model) + the belief model:
  random     : uniform pick (no model calls).
  oracle     : peeks at the real outcome — pick the candidate whose real y moves belief toward truth
               the most (upper bound).
  theta_2pole: p' = P(theta'|I); y+ = sim(x, support), y- = sim(x, null);
               EIG = H(p0) - [p'*H(q+) + (1-p')*H(q-)]   (2 structured sim calls).
  mc<N>      : N unconditioned samples from the SAME model via valence-marginalization —
               v_i ~ Bernoulli(p'), y_i ~ sim(x, v_i) at temp; EIG = H(p0) - mean_i H(q_i).

theta' (the hypothesis x directly tests) is rendered by reusing the domain's render_target on a
remapped stub (direction-agnostic), so P(theta'|I) is just belief_prob with the target swapped.

Faithfulness-in-loop (theta_2pole only): logs the realized theta' valence of y+/y- (belief on theta'
after appending) vs context depth |I| — does simulation quality hold as the info set grows?

Served endpoints (OpenAI-compatible): --belief-url (eval_classification_head --serve) and --sim-url
(vLLM-served theta' checkpoint). Streams per-(episode,policy) trajectories to --out JSONL.
"""

import argparse
import asyncio
import collections
import contextvars
import importlib
import json
import math
import random
import sys
from pathlib import Path

from ibis_env_utils import PRIOR_NOTE

_REPO = next(p for p in Path(__file__).resolve().parents if (p / "envs" / "ibis_env_utils").is_dir())
sys.path.insert(0, str(_REPO / "scripts" / "data_pipeline"))
for _m in ("ibis_env_utils", "gwas_meta", "depmap_meta", "tahoe_meta"):
    sys.path.insert(0, str(_REPO / "envs" / _m))
import build_bed_simulator_sft as B  # schema_key (category) for exemplar-EIG
import build_theta_prime_sft as T  # theta' simulator prompt + theta_prime_row remap
from ibis_env_utils import Split, load_dataset_split, parse_prob

# Per-episode counts of calls and of the silent fallbacks below (unparsable chooser -> random candidate,
# unparsable belief -> 0.5, unparsable simulation -> candidate scored -1, failed request -> ""). Recorded
# in each trajectory's "fallbacks" so their rate can be audited per arm.
_FALLBACKS: contextvars.ContextVar = contextvars.ContextVar("fallbacks", default=None)


def _count(key: str) -> None:
    counter = _FALLBACKS.get()
    if counter is not None:
        counter[key] += 1


EXEMPLARS = {}  # category -> {"support": [y,...], "refute": [y,...]} for the exemplar-EIG baseline


def H(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def in_subset(domain, stub):
    if domain == "depmap":
        return stub.get("modality") == "crispr"
    if domain == "tahoe":
        return (stub.get("kind") or "evidence") == "evidence"
    return True


EXT_SEM = None  # separate semaphore for external (frontier API) calls — set in main; keeps slow/hanging
# OpenAI calls from draining the shared local-server semaphore and starving every other arm.


async def _chat(client, model, system, user, sem, *, max_tokens, temperature=0.0):
    async with sem:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        is_anthropic = model.startswith("claude")  # Anthropic Messages API (frontier chooser)
        is_oai = model.startswith("gpt-") or model.startswith("o")  # OpenAI reasoning API (frontier chooser)
        for attempt in range(4):
            try:
                if is_anthropic:  # Messages API: system is a top-level arg, content is a list of blocks
                    r = await client.messages.create(
                        model=model,
                        max_tokens=2048,
                        system=system or "",
                        messages=[{"role": "user", "content": user}],
                        timeout=90,
                    )
                    return "".join(b.text for b in r.content if b.type == "text")
                if is_oai:  # reasoning models: max_completion_tokens, no temperature / no extra_body; HARD timeout
                    r = await client.chat.completions.create(
                        model=model, messages=msgs, max_completion_tokens=8000, timeout=90
                    )
                else:
                    r = await client.chat.completions.create(
                        model=model,
                        messages=msgs,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=120,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                return r.choices[0].message.content
            except Exception as e:
                if attempt == 3:
                    print(f"[bed] chat failed: {e}", file=sys.stderr)
                    _count("chat_failed")
                    return ""
                await asyncio.sleep(1.0 * (attempt + 1))


async def belief_prob(clients, args, env, target_row, stubs, sem):
    """P(target_row's hypothesis | stubs). target_row supplies the TARGET; stubs are the evidence."""
    row2 = {**target_row, "context_pool": list(stubs)}
    br = env.build_record(row2, len(stubs))
    out = await _chat(
        random.choice(clients["belief"]),
        args.belief_model,
        env.system_prompt_for("direct"),
        br["question"],
        sem,
        max_tokens=16,
    )
    p = parse_prob(out)
    _count("belief_calls")
    if p is None:
        _count("belief_default")
    return 0.5 if p is None else p


async def simulate(clients, args, env, cand, support, sem, temp=0.0):
    """Generate a candidate outcome y at the requested valence (theta' simulator)."""
    ex = T.build_example(env, args.domain, cand, support, 0.0, random.Random(0))
    raw = await _chat(
        random.choice(clients["sim"]),
        args.sim_model,
        ex["messages"][0]["content"],
        ex["messages"][1]["content"],
        sem,
        max_tokens=256,
        temperature=temp,
    )
    _count("sim_calls")
    try:
        obj = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
        y = obj["y"] if isinstance(obj["y"], dict) else None
    except Exception:
        y = None
    if y is None:
        _count("sim_unparsed")
    return y


def with_y(cand, y):
    return {**cand, "y": {**(cand.get("y") or {}), **y}}


def _choice_prompt(env, row, revealed, remaining):
    """Ask-a-model-to-choose prompt: belief render (target + revealed) + candidate setups (results
    hidden) + the pick-the-most-informative instruction (matches bed_experiment's model_select)."""
    base_q = env.build_record({**row, "context_pool": list(revealed)}, len(revealed))["question"]
    lines = []
    for i, c in enumerate(remaining):
        setup = " ".join(s.strip() for s in env.render_evidence_stub(i, c, hide_result=True) if s.strip())
        lines.append(f"[{i}] {setup[:400]}")
    return (
        base_q
        + "\n\nCANDIDATE NEXT EXPERIMENTS (results hidden):\n"
        + "\n".join(lines)
        + "\n\nWhich single candidate experiment, once its result is revealed, you expect would most change"
        "your belief about the target hypothesis? Answer with just its index as <choice>i</choice>."
    )


# Chooser system prompt: a SELECTION task, NOT the belief model's "output a probability" prompt.
# With the belief prompt, format-strict models (GPT-5.6-sol) emit <probability> instead of <choice>;
# a dedicated selection prompt keeps the instruction unambiguous.
CHOOSER_SYS = (
    "You are an expert scientist selecting which experiment to run next. You are shown a target "
    "hypothesis, the evidence gathered so far, and a numbered list of candidate experiments whose "
    "results are hidden. Do NOT output a probability. Your only task is to choose the single candidate "
    "whose hidden result you expect would most change your belief about the target hypothesis. Respond with just "
    "its index in the exact form <choice>i</choice>."
)


async def choose_model(clients, args, env, row, revealed, remaining, chooser_key, rng, sem):
    """model_select: a chooser LM names the most-informative candidate index. The belief TRACKER stays
    the CLS classifier for all arms, so the y-axis isolates selection quality regardless of chooser."""
    import re

    model = {"base": args.base_model, "rld": args.rld_model, "gpt56": args.gpt_model, "opus5": args.opus_model}[
        chooser_key
    ]
    use_sem = (
        EXT_SEM if (chooser_key in ("gpt56", "opus5") and EXT_SEM is not None) else sem
    )  # external API on its own budget
    chooser_sys = CHOOSER_SYS + (PRIOR_NOTE if args.chooser_prior05 else "")
    out = await _chat(
        random.choice(clients[chooser_key]),
        model,
        chooser_sys,
        _choice_prompt(env, row, revealed, remaining),
        use_sem,
        max_tokens=64,
    )
    m = re.findall(r"<choice>\s*(\d+)\s*</choice>", out or "")
    _count("chooser_calls")
    if not m:
        _count("chooser_random")
    elif not 0 <= int(m[-1]) < len(remaining):
        _count("chooser_out_of_range")
    return (int(m[-1]) % len(remaining)) if m else rng.randrange(len(remaining))


async def eig_2pole(clients, args, env, row, revealed, cand, p0, sem, faith):
    trow = T.theta_prime_row(args.domain, cand)
    pprime = await belief_prob(clients, args, env, trow, revealed, sem)
    yp, ym = await asyncio.gather(
        simulate(clients, args, env, cand, True, sem, 0.0), simulate(clients, args, env, cand, False, sem, 0.0)
    )
    if yp is None or ym is None:
        return -1.0
    cp, cm = with_y(cand, yp), with_y(cand, ym)
    qp, qm = await asyncio.gather(
        belief_prob(clients, args, env, row, revealed + [cp], sem),
        belief_prob(clients, args, env, row, revealed + [cm], sem),
    )
    if faith is not None:
        fp, fm = await asyncio.gather(
            belief_prob(clients, args, env, trow, [cp], sem), belief_prob(clients, args, env, trow, [cm], sem)
        )
        faith.append({"ctx": len(revealed), "pprime": pprime, "yplus_theta_prime": fp, "yminus_theta_prime": fm})
    return H(p0) - (pprime * H(qp) + (1 - pprime) * H(qm))


async def eig_mc(clients, args, env, row, revealed, cand, p0, n, sem, rng):
    trow = T.theta_prime_row(args.domain, cand)
    pprime = await belief_prob(clients, args, env, trow, revealed, sem)
    ys = await asyncio.gather(
        *[simulate(clients, args, env, cand, rng.random() < pprime, sem, args.gen_temp) for _ in range(n)]
    )
    qs = await asyncio.gather(
        *[belief_prob(clients, args, env, row, revealed + [with_y(cand, y)], sem) for y in ys if y is not None]
    )
    if not qs:
        return -1.0
    return H(p0) - sum(H(q) for q in qs) / len(qs)


async def score_oracle(clients, args, env, row, revealed, cand, p0, label, sem):
    q = await belief_prob(clients, args, env, row, revealed + [cand], sem)  # real y is already in cand
    return (q - p0) if label == 1 else (p0 - q)  # movement toward truth


async def eig_exemplar(clients, args, env, row, revealed, cand, p0, sem, rng):
    """Exemplar-EIG: same p'-weighted 2-pole EIG as theta_2pole, but y+/y- are REAL supporting/refuting
    exemplars of the candidate's category from the training corpus (no simulator)."""
    ex = EXEMPLARS.get(B.schema_key(args.domain, cand))
    if not ex or not ex.get("support") or not ex.get("refute"):
        return -1.0
    yp, ym = rng.choice(ex["support"]), rng.choice(ex["refute"])
    trow = T.theta_prime_row(args.domain, cand)
    pprime = await belief_prob(clients, args, env, trow, revealed, sem)
    cp, cm = with_y(cand, yp), with_y(cand, ym)
    qp, qm = await asyncio.gather(
        belief_prob(clients, args, env, row, revealed + [cp], sem),
        belief_prob(clients, args, env, row, revealed + [cm], sem),
    )
    return H(p0) - (pprime * H(qp) + (1 - pprime) * H(qm))


def clean_pool(domain, row, cap):
    """First `cap` clean-subset (theta'-valid) candidate stubs from the episode's context pool."""
    cp = row.get("context_pool")
    if cp is None:
        return []
    return [s for s in cp if in_subset(domain, s)][:cap]


async def run_episode(clients, args, env, row, policy, sem, rng):
    pool = clean_pool(args.domain, row, args.max_candidates)
    if len(pool) < args.max_candidates:  # require a FULL pool so every episode runs the same # of steps
        return None
    fallbacks = collections.Counter()
    _FALLBACKS.set(fallbacks)
    revealed, remaining = [], list(pool)
    label = int(row["label"])
    faith = [] if (policy == "theta_2pole" and not args.no_faith) else None
    p = await belief_prob(clients, args, env, row, revealed, sem)
    traj = [{"t": 0, "p": p}]
    for t in range(1, min(args.budget, len(remaining)) + 1):
        p0 = traj[-1]["p"]
        if policy == "random":
            j = rng.randrange(len(remaining))
        elif policy in ("naive_base", "naive_rld", "naive_gpt56", "naive_opus5"):
            key = {"naive_base": "base", "naive_rld": "rld", "naive_gpt56": "gpt56", "naive_opus5": "opus5"}[policy]
            j = await choose_model(clients, args, env, row, revealed, remaining, key, rng, sem)
        else:
            if policy == "oracle":
                scores = await asyncio.gather(
                    *[score_oracle(clients, args, env, row, revealed, c, p0, label, sem) for c in remaining]
                )
            elif policy == "theta_2pole":
                scores = await asyncio.gather(
                    *[eig_2pole(clients, args, env, row, revealed, c, p0, sem, faith) for c in remaining]
                )
            elif policy == "exemplar_eig":
                scores = await asyncio.gather(
                    *[eig_exemplar(clients, args, env, row, revealed, c, p0, sem, rng) for c in remaining]
                )
            else:  # mc<N>
                n = int(policy[2:])
                scores = await asyncio.gather(
                    *[eig_mc(clients, args, env, row, revealed, c, p0, n, sem, rng) for c in remaining]
                )
            j = max(range(len(remaining)), key=lambda k: scores[k])
        revealed.append(remaining.pop(j))
        p = await belief_prob(clients, args, env, row, revealed, sem)
        traj.append({"t": t, "p": p})
    return {
        "episode_id": row["episode_id"],
        "domain": args.domain,
        "policy": policy,
        "label": label,
        "traj": traj,
        "faith": faith,
        "fallbacks": dict(fallbacks),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["gwas", "depmap", "tahoe"])
    ap.add_argument("--data-root", default="ibis_data/bed_sim/data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--belief-url", required=True)
    ap.add_argument("--belief-model", default="belief")
    ap.add_argument("--sim-url", required=True)
    ap.add_argument("--sim-model", default="thetaprime")
    # model_select choosers (optional): base LM, RL'd deliberation LM, frontier API. Comma-sep URL pools.
    ap.add_argument("--base-url", default="")
    ap.add_argument("--base-model", default="base")
    ap.add_argument("--rld-url", default="")
    ap.add_argument("--rld-model", default="rld")
    ap.add_argument("--gpt-model", default="")
    ap.add_argument("--gpt-key", default="")
    ap.add_argument("--opus-model", default="")
    ap.add_argument("--opus-key", default="")
    ap.add_argument("--policies", default="random,oracle,theta_2pole,mc1,mc16")
    ap.add_argument("--max-episodes", type=int, default=150)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--max-candidates", type=int, default=16)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--ext-concurrency", type=int, default=16, help="separate concurrency cap for the frontier API")
    ap.add_argument("--no-faith", action="store_true", help="skip theta_2pole faithfulness-in-loop logging (faster)")
    ap.add_argument(
        "--exemplar-pools", default="", help="JSON of per-domain category exemplar y pools (for exemplar_eig)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--chooser-prior05",
        action="store_true",
        help="append the shared PRIOR_NOTE (0.5-prior anchor) to the chooser system prompt",
    )
    ap.add_argument("--out", default="ibis_data/bed_sim/bed_tp/results.jsonl")
    args = ap.parse_args()
    from openai import AsyncOpenAI

    # --belief-url / --sim-url may be comma-separated pools of replica servers; round-robin (random) per call.
    clients = {
        "belief": [AsyncOpenAI(base_url=u.strip(), api_key="x") for u in args.belief_url.split(",")],
        "sim": [AsyncOpenAI(base_url=u.strip(), api_key="x") for u in args.sim_url.split(",")],
    }
    if args.base_url:
        clients["base"] = [AsyncOpenAI(base_url=u.strip(), api_key="x") for u in args.base_url.split(",")]
    if args.rld_url:
        clients["rld"] = [AsyncOpenAI(base_url=u.strip(), api_key="x") for u in args.rld_url.split(",")]
    if args.gpt_model and args.gpt_key:
        clients["gpt56"] = [AsyncOpenAI(api_key=args.gpt_key)]
    if args.opus_model and args.opus_key:
        from anthropic import AsyncAnthropic

        clients["opus5"] = [AsyncAnthropic(api_key=args.opus_key)]
    print(f"[{args.domain}] pools: " + ", ".join(f"{k}={len(v)}" for k, v in clients.items()), flush=True)
    if args.exemplar_pools:
        global EXEMPLARS
        EXEMPLARS = json.load(open(args.exemplar_pools)).get(args.domain, {})
        print(
            f"[{args.domain}] exemplar pools: "
            + ", ".join(f"{k}(sup={len(v['support'])},ref={len(v['refute'])})" for k, v in EXEMPLARS.items()),
            flush=True,
        )
    env = importlib.import_module(f"{args.domain}_meta")
    all_rows = load_dataset_split(f"{args.data_root}/{args.domain}", Split(args.split)).to_dict("records")
    # keep only episodes with a FULL clean-subset pool (>= max_candidates) so all trajectories are the
    # same length -> every step has full N and error bars don't collapse at the tail.
    rows = [r for r in all_rows if len(clean_pool(args.domain, r, args.max_candidates)) >= args.max_candidates][
        : args.max_episodes
    ]
    print(
        f"[{args.domain}] {len(rows)}/{len(all_rows)} episodes with >= {args.max_candidates} clean candidates",
        flush=True,
    )
    policies = args.policies.split(",")
    sem = asyncio.Semaphore(args.concurrency)
    global EXT_SEM
    EXT_SEM = asyncio.Semaphore(args.ext_concurrency)  # frontier API isolated from local-server concurrency
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = [
        run_episode(clients, args, env, row, pol, sem, random.Random(args.seed + 7919 * i))
        for i, row in enumerate(rows)
        for pol in policies
    ]
    n = 0
    with open(out, "w") as fh:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            if r is None:
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n += 1
            if n % 20 == 0:
                print(f"[{args.domain}] {n}/{len(tasks)} episode-policy trajectories done", flush=True)
    print(f"[{args.domain}] wrote {n} trajectories -> {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
