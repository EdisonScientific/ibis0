"""Shared data->prompt rendering pipeline for the evidence-deliberation envs.

Every `<domain>_meta` module (gwas / depmap / tahoe) renders the same skeleton — a `CONTEXT
EVIDENCE` header, numbered `[i]` evidence stubs, a `TARGET HYPOTHESIS` header, the target block, and
a fixed answer-format footer — and differs only in the domain-specific per-stub and target text.
This module owns the skeleton (`render_evidence_prompt`), the record envelope
(`build_single_target_record`), the probability parser (`parse_prob`), and the rubric wiring
(`make_evidence_rubric`); each env supplies a `render_evidence_stub(idx, stub) -> list[str]` and a
`render_target(row) -> str`.

`parse_prob` and `render_evidence_prompt` / `build_single_target_record` import only the stdlib so
this module can be loaded standalone (e.g. spec-loaded without the heavy `ibis_env_utils` package
`__init__`); `make_evidence_rubric` pulls `verifiers` + the reward helpers lazily.
"""

import re

# The style-independent answer-format footer shared by every domain's user prompt. The output-format
# instruction (reasoning vs. direct) lives in the *system* prompt; the user footer only points at it.
DEFAULT_FOOTER = "Respond using the format specified in the system message:\n  <probability>0.XX</probability>"

_PROB_RE = re.compile(r"<probability>\s*([0-9]*\.?[0-9]+)\s*</probability>")


def parse_prob(text: str) -> float | None:
    """Extract P(supported) from a completion: the last `<probability>0.XX</probability>` tag.

    Only the text after a final `</think>` is considered (thinking-mode rollouts), and the value is
    clamped to [0, 1]. Returns ``None`` when no tag is present (an unparseable rollout).
    """
    end = text.rfind("</think>")
    if end != -1:
        text = text[end + len("</think>") :]
    m = _PROB_RE.findall(text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m[-1])))


def render_evidence_prompt(
    context_stubs,
    *,
    stub_renderer,
    target_block: str,
    footer: str = DEFAULT_FOOTER,
    empty_context_notice: str | None = None,
) -> str:
    """Render the canonical evidence-deliberation user prompt.

    Layout: ``CONTEXT EVIDENCE`` header -> one numbered block per stub (``stub_renderer(i, stub)``
    returns that block's lines, including its ``[i]`` header) -> optional ``empty_context_notice``
    when there are no stubs -> ``TARGET HYPOTHESIS`` header -> ``target_block`` -> ``footer``.

    ``stub_renderer`` and ``target_block`` are the only domain-specific inputs, so all three domains
    share this one layout.
    """
    lines: list[str] = ["CONTEXT EVIDENCE", "================"]
    stubs = list(context_stubs) if context_stubs is not None else []
    for i, stub in enumerate(stubs, 1):
        lines.extend(stub_renderer(i, stub))
    if not stubs and empty_context_notice is not None:
        lines.append(empty_context_notice)
    lines += ["", "TARGET HYPOTHESIS", "================="]
    lines.append(target_block)
    lines.append("\n" + footer)
    return "\n".join(x for x in lines if x is not None)


def build_single_target_record(
    row: dict,
    context_size: int,
    *,
    task: str,
    render_user,
    prompt_style: str = "reasoning",
    extra_info=None,
) -> dict:
    """A dataset row -> a single-target verifiers example (one hypothesis per episode).

    Slices the (fixed, pre-shuffled) ``context_pool`` to its first ``context_size`` stubs — so
    prefixes are nested across context sizes — renders the user prompt via ``render_user(sliced,
    prompt_style=...)``, and returns ``{question, answer, info, task}`` with a scalar ``info["label"]``.
    ``extra_info(row) -> dict`` merges domain-specific fields into ``info`` when provided.
    ``load_environment`` wraps this into the chat-``prompt`` record the env dataset consumes.
    """
    pool = list(row["context_pool"]) if row.get("context_pool") is not None else []
    sliced = {**row, "context_pool": pool[:context_size]}
    info = {
        "episode_id": row.get("episode_id"),
        "label": int(row["label"]),
        "context_size": context_size,
    }
    if extra_info is not None:
        info.update(extra_info(row))
    return {
        "question": render_user(sliced, prompt_style=prompt_style),
        "answer": "",
        "info": info,
        "task": task,
    }


def make_evidence_rubric(
    *,
    reward_type: str,
    unparseable_reward: float,
    log_lik_clip: float,
    parser,
) -> "object":
    """Build the shared single-target evidence-deliberation rubric.

    ``funcs=[evidence_reward, format_reward, accuracy]`` with ``weights=[0.9, 0.1, 0.0]``: the reward
    scores the parsed P(supported) against the scalar ``info["label"]`` via ``compute_reward``; the
    format reward flags a parseable answer; accuracy is metrics-only (weight 0). Shared so every
    evidence-deliberation env scores identically.
    """
    import verifiers as vf

    from ibis_env_utils.reward import compute_accuracy, compute_reward

    def _text(completion) -> str:
        return completion if isinstance(completion, str) else completion[-1]["content"]

    def evidence_reward(completion, info: dict, state: dict | None = None, **kwargs) -> float:
        """Single-target reward over the parsed P(supported); see ``compute_reward``."""
        p = parse_prob(_text(completion))
        return compute_reward(
            p,
            info.get("label", 0),
            reward_type=reward_type,
            unparseable_reward=unparseable_reward,
            log_lik_clip=log_lik_clip,
        )

    def format_reward(completion, info: dict, **kwargs) -> float:
        return 1.0 if parse_prob(_text(completion)) is not None else 0.0

    def accuracy(completion, info: dict, **kwargs) -> float:
        """Metrics-only (weight 0): thresholded correctness of P(supported) vs the label."""
        return compute_accuracy(parse_prob(_text(completion)), info.get("label", 0))

    return vf.Rubric(funcs=[evidence_reward, format_reward, accuracy], weights=[0.9, 0.1, 0.0], parser=parser)
