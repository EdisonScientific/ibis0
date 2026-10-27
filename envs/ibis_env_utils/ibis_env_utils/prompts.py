"""Shared answer-format prompting for the probability-elicitation envs.

The evidence-deliberation envs (gwas_meta, depmap_meta) all close their system prompt with the same
answer-format instruction and elicit the same ``<probability>0.XX</probability>`` line; only the
domain preamble differs. This module holds the two shared endings — ``DIRECT_INSTRUCTION``
(answer-only, for ``enable_thinking=false`` runs) and ``REASONING_INSTRUCTION`` (think-first, for
thinking-mode runs) — plus the style->prompt selector, so each env composes its two prompts from its
own preamble without repeating the format text or the selection logic.

Mirrors the direct/reasoning split in ``tahoe_meta.episodes`` (which keeps its own copy, as it is
deliberately not built on ibis_env_utils).
"""

DIRECT_INSTRUCTION = (
    "Respond with ONLY a single line in this exact format — no reasoning, no explanation, no other "
    "text before or after:\n"
    "<probability>0.XX</probability>"
)

REASONING_INSTRUCTION = (
    "Think through the evidence carefully, then commit to a single probability in the closed "
    "interval [0, 1]. After your reasoning, output your probability in this exact format:\n"
    "<probability>0.XX</probability>"
)

# Prior-anchor note. Opt-in (``prior_note=True``) so the same text can be turned on for belief
# elicitation and left off where it must not apply (e.g. the BED belief tracker). Grounds the 0.5
# prior in the benchmark's construction (balanced true/false labels) rather than asserting it, and is
# used verbatim both here and in the BED chooser prompt so every use of the prior is phrased identically.
PRIOR_NOTE = (
    "\nNote that the target hypothesis has been drawn from a dataset with an equal number of true and "
    "false hypotheses. That is, before considering the provided context evidence, you should assume a "
    "prior of 0.5 for the ground-truth label of the target hypothesis.\n"
)


def build_probability_system_prompts(preamble: str, *, prior_note: bool = False) -> tuple[str, str]:
    """Return ``(reasoning, direct)`` system prompts for a domain ``preamble``.

    Appends the shared ``REASONING_INSTRUCTION`` / ``DIRECT_INSTRUCTION`` endings, with ``PRIOR_NOTE``
    inserted before the ending when ``prior_note`` is set. With ``prior_note=False`` the ``direct``
    prompt is ``preamble + DIRECT_INSTRUCTION``.
    """
    note = PRIOR_NOTE if prior_note else ""
    return preamble + note + REASONING_INSTRUCTION, preamble + note + DIRECT_INSTRUCTION


def select_system_prompt(prompt_style: str, *, reasoning: str, direct: str) -> str:
    """Pick a domain's system prompt for a ``prompt_style`` ("direct" or "reasoning").

    Single source for the style->prompt mapping, shared by the RL envs' ``load_environment`` and the
    shared classification-head trainer so both resolve a given style to the same prompt. Envs bind
    their own ``reasoning``/``direct`` prompts and re-export a one-arg ``system_prompt_for``.
    """
    if prompt_style == "direct":
        return direct
    if prompt_style == "reasoning":
        return reasoning
    raise ValueError(f"unknown prompt_style: {prompt_style!r}")
