"""Evaluate a frontier API model on a published multi-domain TEST set.

Split from eval_generative_testset.py (which serves local weights via vLLM) because the API path has
genuinely different concerns: no serving, provider-specific reasoning controls, refusals, and
provider-side rate limits rather than KV-cache limits.

**Reasoning depth is explicit and symmetric.** `--reasoning-effort` maps to `output_config.effort`
(Anthropic) and `reasoning_effort` (OpenAI), and `--max-output-tokens` sets the SAME budget on both
(`max_tokens` / `max_completion_tokens`). Nothing is left to a provider default, because the defaults
differ between providers and would silently make the two baselines non-comparable.

Three Opus-5-specific facts this encodes (verified against the live Models API):

- `thinking: {type: "enabled", budget_tokens: N}` is **rejected with a 400**. Depth is set by effort.
- Thinking is **on by default** — unlike Opus 4.8/4.7, where omitting `thinking` meant no thinking.
- `max_tokens` caps thinking **and** response text together, so it must leave room for both. A budget
  sized for the answer alone truncates mid-reasoning, which surfaces as an unparsed episode rather
  than an error.

`temperature` / `top_p` / `top_k` are omitted deliberately: all three are rejected on Opus 5.

**Refusals and API errors are recorded, never silently scored.** A safety classifier can decline with
HTTP 200, `stop_reason="refusal"`, and empty content; a naive reader scores that as a parse failure and
quietly contaminates the parse-rate statistic. Each record carries `api_status` (`ok` / `refusal` /
`error`) so the three are distinguishable downstream.

**Server-side `fallbacks` are deliberately NOT enabled**, despite being the recommended default for
applications. A fallback silently re-runs a declined request on a *different model* — for an eval
measuring one model's calibration, that substitutes another model's predictions into the results with
no signal in the output. Availability is the wrong tradeoff here; a recorded refusal is the right one.
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

# Effort levels each provider accepts. Anthropic exposes the full ladder; OpenAI's reasoning models
# stop at `high`, so asking for xhigh/max there is a fail-fast rather than a late 400 mid-run.
_ANTHROPIC_EFFORT = ("low", "medium", "high", "xhigh", "max")
_OPENAI_EFFORT = ("minimal", "low", "medium", "high")


def _load_env(domain: str):
    """Import the `<domain>_meta` env module (parse_prob / render_user / system_prompt_for)."""
    sys.path.insert(0, str(_ENVS / f"{domain}_meta"))
    return importlib.import_module(f"{domain}_meta")


def _testset_passes(df) -> list[tuple[str, list[dict]]]:
    """Thin alias for the shared reader — see scripts/lib/testset_io.testset_passes.

    The frontier and local evals MUST resolve the same episodes into the
    same passes or the roster rows stop being comparable, so there is one definition.
    """
    return testset_passes(df)


def _anthropic_thinking_tokens(msg) -> int | None:
    """Reasoning-token count for an Anthropic response, or None if the API does not report one.

    Anthropic bills thinking inside `output_tokens` and (unlike OpenAI's
    `completion_tokens_details.reasoning_tokens`) has historically exposed no separate count. Prefer a
    real field if the SDK offers one; otherwise return None.

    Deliberately does NOT estimate tokens from the thinking text's character count. A chars/4 guess
    recorded under the same key as OpenAI's exact count would be indistinguishable from a measurement
    downstream, which is the failure this function exists to fix. `reasoning_chars` (recorded separately
    by the caller's sibling logic) is enough to answer the question that actually matters — whether
    adaptive thinking engaged at all.
    """
    usage = getattr(msg, "usage", None)
    for field in ("thinking_tokens", "reasoning_tokens", "reasoning_output_tokens"):
        if (v := getattr(usage, field, None)) is not None:
            return int(v)
    return None


def _anthropic_thinking_chars(msg) -> int:
    """Total characters across the response's thinking blocks — an advisory engagement signal."""
    return sum(
        len(getattr(b, "thinking", "") or "")
        for b in getattr(msg, "content", []) or []
        if getattr(b, "type", None) == "thinking"
    )


# No explicit cache breakpoints are set. Effective prefix caching would need the env modules to expose
# evidence stubs as a **list** so each becomes its own content block: a cached prefix must end on a block
# boundary, and the evidence is one growing text block. Cache usage is still recorded per call, so any
# provider-side hits are measured rather than assumed.


def _journal_header(args) -> dict:
    """Run identity a journal must match before its records may be reused."""
    return {
        "_header": True,
        "model": args.model,
        "provider": args.provider,
        "data_path": args.data_path,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "prompt_style": args.prompt_style,
    }


def _load_journal(path: Path, args) -> dict:
    """`(tag, episode_id) -> record` for resuming a partly-finished dense sweep.

    Refuses to reuse a journal whose header does not match this run: the same `(tag, episode_id)` under a
    different model, effort or test-set path is a *different* measurement, and silently adopting it would
    mix two rosters inside one cell with no signal in the output.
    """
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        head = json.loads(lines[0])
    except json.JSONDecodeError:
        return {}
    if not head.get("_header") or {k: v for k, v in head.items() if k != "_header"} != {
        k: v for k, v in _journal_header(args).items() if k != "_header"
    }:
        print(f"  [journal] {path.name} was written by a different run config — ignoring it", flush=True)
        return {}
    done = {}
    for ln in lines[1:]:
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue  # a torn final line from a kill mid-write; that pair simply re-runs
        done[(d["tag"], d["episode_id"])] = d
    return done


async def _run_passes(client, args, passes, sem, sysprompt, render_user, parse_prob, out_dir):
    """Issue every (tag, episode) call in `passes` order, journalling each completion.

    Calls go out **k-major** — all episodes at one context size before the next — which is the order
    `testset_passes` yields. An episode-major variant was tried and reverted: it made provider caching
    worse, not better, so do not "optimise" the ordering without measuring.

    Every completed call is appended to a per-domain JSONL journal so a crash resumes rather than
    re-paying, which matters at 2,000 episodes per primary cell and 13,056 calls per sensitivity config.
    Cells are still written one per (domain, tag), sorted by episode_id so completion order cannot leak
    into the output.
    """
    jpath = out_dir / f"{args.domain}_journal.jsonl"
    done = _load_journal(jpath, args)
    n_calls = sum(len(args.limit and rows[: args.limit] or rows) for _, rows in passes)
    if done:
        print(f"  [journal] resuming: {len(done)}/{n_calls} calls already recorded", flush=True)
    fresh = not done
    jf = jpath.open("w" if fresh else "a")
    if fresh:
        jf.write(json.dumps(_journal_header(args)) + "\n")
        jf.flush()

    async def _one(tag, row):
        text, status, usage = await _chat(client, args, render_user(row), sem, sysprompt)
        p = parse_prob(text) if status == "ok" else None
        jf.write(
            json.dumps(
                {
                    "tag": tag,
                    "episode_id": row["episode_id"],
                    "parsed_prob": p,
                    "api_status": status,
                    "usage": usage,
                    "raw": (text[:1200] if (p is None and status == "ok") else None),
                }
            )
            + "\n"
        )
        jf.flush()  # one complete line per call: a kill loses at most the last record
        return (row, p, text, status, usage)

    per_tag: dict = {}
    try:
        for tag, rows in passes:
            if args.limit:
                rows = rows[: args.limit]
            resumed = [
                (r, rec["parsed_prob"], rec.get("raw") or "", rec["api_status"], rec["usage"])
                for r in rows
                if (rec := done.get((tag, r["episode_id"]))) is not None
            ]
            todo = [r for r in rows if (tag, r["episode_id"]) not in done]
            fetched = await asyncio.gather(*[_one(tag, r) for r in todo])
            per_tag[tag] = sorted([*resumed, *fetched], key=lambda e: str(e[0]["episode_id"]))
    finally:
        jf.close()
    return per_tag


async def _chat(client, args, user, sem, sysprompt):
    """One episode. Returns (text, status, usage) with status in {ok, refusal, error}."""
    for attempt in range(4):
        try:
            async with sem:
                if args.provider == "anthropic":
                    # Streamed: the SDK refuses non-streaming requests whose max_tokens implies a
                    # long generation (HTTP idle-timeout guard), and streaming is recommended for
                    # large budgets regardless.
                    async with client.messages.stream(
                        model=args.model,
                        system=sysprompt,
                        messages=[{"role": "user", "content": user}],
                        max_tokens=args.max_output_tokens,
                        thinking={"type": "adaptive"},
                        output_config={"effort": args.reasoning_effort},
                    ) as stream:
                        r = await stream.get_final_message()
                    u = {
                        "input_tokens": r.usage.input_tokens,
                        "output_tokens": r.usage.output_tokens,
                        # Thinking is billed inside output_tokens and Anthropic exposes no separate
                        # count, so sum the thinking blocks ourselves. Without this the field is absent
                        # and every downstream `usage.get("reasoning_tokens") or 0` yields a zero that
                        # reads like a measurement of "no thinking" — which is wrong, thinking is on.
                        "reasoning_tokens": _anthropic_thinking_tokens(r),
                        "reasoning_chars": _anthropic_thinking_chars(r),
                    }
                    # Cache accounting, when the provider reports it — the only way to confirm the
                    # ascending-k walk is actually landing hits rather than silently paying full price.
                    for f in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                        if (v := getattr(r.usage, f, None)) is not None:
                            u[f] = v
                    # Check stop_reason BEFORE reading content: on a refusal content is empty, and
                    # treating that as text would score a policy decline as a parse failure.
                    if r.stop_reason == "refusal":
                        return "", "refusal", u
                    return "".join(b.text for b in r.content if getattr(b, "type", None) == "text"), "ok", u

                # `prompt_cache_key` is required for reliable prefix matching on GPT-5.6+, and wants
                # enough traffic per key to keep a route warm — hence per domain, not per episode.
                r = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": sysprompt}, {"role": "user", "content": user}],
                    max_completion_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    prompt_cache_key=f"ibis-evdel-{args.domain}",
                )
                usage = getattr(r, "usage", None)
                u = {
                    "input_tokens": getattr(usage, "prompt_tokens", None),
                    "output_tokens": getattr(usage, "completion_tokens", None),
                }
                det = getattr(usage, "completion_tokens_details", None)
                if det is not None:
                    u["reasoning_tokens"] = getattr(det, "reasoning_tokens", None)
                # OpenAI caches prompt prefixes automatically — no opt-in, and matched by longest common
                # prefix rather than an explicit breakpoint, so a differing tail does not prevent a hit
                # (unlike Anthropic's explicit breakpoints; see the module-level caching note). Record
                # the hit count so the ascending-k walk's benefit is measured rather than assumed.
                pdet = getattr(usage, "prompt_tokens_details", None)
                if pdet is not None:
                    # Both halves: writes>0 with reads==0 means populated but never matched.
                    u["cached_input_tokens"] = getattr(pdet, "cached_tokens", None)
                    u["cache_write_tokens"] = getattr(pdet, "cache_write_tokens", None)
                return (r.choices[0].message.content or ""), "ok", u
        except Exception as e:
            if attempt == 3:
                print(f"[frontier] {args.provider} call failed after 4 attempts: {e}", file=sys.stderr)
                return "", "error", {}
            await asyncio.sleep(3 * (attempt + 1))
    return "", "error", {}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--data-path", required=True, help="published test-set config for --domain")
    ap.add_argument(
        "--revision",
        default=None,
        help="Hub revision of an hf:// test set (the launchers pass the pinned TESTSET_REV); ignored for a local directory",
    )
    ap.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    ap.add_argument("--model", required=True, help="API model id, e.g. gpt-5.6-sol / claude-opus-5")
    ap.add_argument("--api-key", default=None, help="defaults to the provider's standard env var")
    ap.add_argument("--out-dir", required=True, help="result dir from scripts/lib/eval_paths.result_dir()")
    ap.add_argument(
        "--reasoning-effort",
        default="high",
        help="reasoning depth, applied symmetrically: output_config.effort (Anthropic) / "
        "reasoning_effort (OpenAI). xhigh|max are Anthropic-only.",
    )
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=24000,
        help="SAME output budget for both providers. On Anthropic this covers thinking AND the "
        "response text together, so it must be generous. 24000 also matches the local reasoning "
        "conditions, so every reasoning row in the roster shares one budget.",
    )
    ap.add_argument("--concurrency", type=int, default=8, help="in-flight requests; keep low to avoid 429s")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N episodes (cost pilot)")
    ap.add_argument("--prompt-style", choices=["direct", "reasoning"], default="reasoning")
    ap.add_argument(
        "--prior-note", action="store_true", help="append the shared PRIOR_NOTE (0.5-prior anchor) to the system prompt"
    )
    args = ap.parse_args()

    allowed = _ANTHROPIC_EFFORT if args.provider == "anthropic" else _OPENAI_EFFORT
    if args.reasoning_effort not in allowed:
        raise SystemExit(
            f"--reasoning-effort {args.reasoning_effort!r} is not accepted by {args.provider}; "
            f"choose from {list(allowed)}"
        )

    env = _load_env(args.domain)
    parse_prob, render_user, system_prompt_for = env.parse_prob, env.render_user, env.system_prompt_for
    sysprompt = system_prompt_for(args.prompt_style, prior_note=args.prior_note)

    if args.provider == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=args.api_key) if args.api_key else anthropic.AsyncAnthropic()
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=args.api_key) if args.api_key else AsyncOpenAI()

    if "{" in args.data_path or "}" in args.data_path:
        raise ValueError(f"--data-path still contains braces (mis-expanded template?): {args.data_path!r}")
    df = load_dataset_split(args.data_path, Split.TEST, revision=args.revision or None)
    if "testset" not in df.columns:
        raise ValueError(f"{args.data_path} is not a published test set (no `testset` column)")
    passes = _testset_passes(df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    print(
        f"[frontier] {args.provider}:{args.model} domain={args.domain} effort={args.reasoning_effort} "
        f"max_output_tokens={args.max_output_tokens} concurrency={args.concurrency} "
        f"prompt_style={args.prompt_style}" + (f" LIMIT={args.limit}" if args.limit else ""),
        flush=True,
    )

    per_tag = await _run_passes(client, args, passes, sem, sysprompt, render_user, parse_prob, out_dir)

    for tag, entries in per_tag.items():
        results, n_parsed, n_refusal, n_error = [], 0, 0, 0
        tok_in = tok_out = tok_reason = 0
        for r, p, text, status, usage in entries:
            n_parsed += p is not None
            n_refusal += status == "refusal"
            n_error += status == "error"
            tok_in += usage.get("input_tokens") or 0
            tok_out += usage.get("output_tokens") or 0
            tok_reason += usage.get("reasoning_tokens") or 0
            results.append(
                {
                    "episode_id": r["episode_id"],
                    "domain": args.domain,
                    "label": int(r["label"]),
                    "parsed_prob": p,
                    "ground_truth": r.get("ground_truth"),
                    "n_stubs": int(r["n_stubs"]),
                    "api_status": status,
                    "raw": (text[:1200] if (p is None and status == "ok") else None),
                }
            )
        path = out_dir / f"{args.domain}_{tag}.json"
        path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "provider": args.provider,
                    "domain": args.domain,
                    "testset": args.data_path,
                    "context_size": tag,
                    "split": "test",
                    "reasoning_effort": args.reasoning_effort,
                    "max_output_tokens": args.max_output_tokens,
                    "prompt_style": args.prompt_style,
                    "n_eval": len(results),
                    "usage": {
                        "input_tokens": tok_in,
                        "output_tokens": tok_out,
                        "reasoning_tokens": tok_reason or None,
                    },
                    "results": results,
                }
            )
        )
        probs = [x["parsed_prob"] for x in results if x["parsed_prob"] is not None]
        labs = [x["label"] for x in results if x["parsed_prob"] is not None]
        brier = float(np.mean([(p - y) ** 2 for p, y in zip(probs, labs)])) if probs else float("nan")
        print(
            f"  {tag:>7}: parsed {n_parsed}/{len(entries)}  refusals={n_refusal}  errors={n_error}  "
            f"Brier={brier:.4f}  mean_p={np.mean(probs) if probs else float('nan'):.3f}  "
            f"tokens in={tok_in:,} out={tok_out:,}"
            + (f" (reasoning={tok_reason:,})" if tok_reason else "")
            + f"  -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
