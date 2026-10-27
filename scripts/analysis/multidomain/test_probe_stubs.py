"""The per-domain probe predicates, and the mirror in particular.

The mirror is the one place where the probe can fail silently and still produce a publishable-looking
figure: if the mirrored stub is not equal-and-opposite, the additivity residual absorbs the error and
reads as incoherence. So the properties pinned here are the ones the scientific claim rests on -- the
mirror reverses the direction, preserves the magnitude, leaves the significance evidence untouched, and
renders as genuinely refuting rather than merely weaker.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(next(p for p in Path(__file__).resolve().parents if p.name == "scripts").parent / "envs" / "depmap_meta")
)

from probe_stubs import (  # noqa: E402
    MIRRORABLE,
    is_neutral,
    is_supporting,
    mirror,
    pdep_for_chronos,
    support_rank,
)


def tahoe_stub(lfc=1.4, padj=1e-6, stat=5.2, ct="same_drug_diff_cell"):
    return {
        "kind": "evidence",
        "context_type": ct,
        "verdict": "supported",
        "y": {"log2FoldChange": lfc, "padj": padj, "stat": stat},
    }


def depmap_stub(ch=-1.2, pdep=0.98, ct="same_gene_same_lineage"):
    return {"modality": "crispr", "context_type": ct, "y": {"chronos": ch, "p_dependency": pdep}}


# --- Tahoe mirror ---------------------------------------------------------------------------------


def test_tahoe_mirror_reverses_direction_and_keeps_magnitude():
    s = tahoe_stub(lfc=1.4)
    m = mirror("tahoe", s)
    assert m["y"]["log2FoldChange"] == pytest.approx(-1.4)
    assert abs(m["y"]["log2FoldChange"]) == pytest.approx(abs(s["y"]["log2FoldChange"]))


def test_tahoe_mirror_leaves_padj_alone():
    """padj is two-sided, so the evidence *strength* is identical; only the direction flips."""
    s = tahoe_stub(padj=3e-9)
    assert mirror("tahoe", s)["y"]["padj"] == 3e-9


def test_tahoe_mirror_does_not_mutate_the_original():
    s = tahoe_stub(lfc=1.4)
    mirror("tahoe", s)
    assert s["y"]["log2FoldChange"] == 1.4


def test_tahoe_mirror_is_an_involution_on_the_numbers():
    s = tahoe_stub(lfc=1.4, stat=5.2)
    back = mirror("tahoe", mirror("tahoe", s))
    assert back["y"]["log2FoldChange"] == pytest.approx(1.4)
    assert back["y"]["stat"] == pytest.approx(5.2)


# --- DepMap mirror --------------------------------------------------------------------------------


def test_depmap_mirror_reflects_chronos_about_zero_not_the_threshold():
    """Zero is the no-effect point; -0.5 is a decision boundary. Reflecting about the threshold would
    send a -1.2 dependency to -0.4, which still reads as a fitness cost rather than as its opposite."""
    assert mirror("depmap", depmap_stub(ch=-1.2))["y"]["chronos"] == pytest.approx(1.2)


def test_depmap_mirror_renders_as_refuting_not_merely_weaker():
    from depmap_meta import _chronos_call

    for ch in (-0.5, -0.8, -1.2, -2.2):
        m = mirror("depmap", depmap_stub(ch=ch))
        assert _chronos_call(m["y"]["chronos"]) == "knockout speeds growth — suppressor", ch


def test_depmap_mirror_moves_p_dependency_off_the_empirical_curve_not_by_one_minus_p():
    """1-p would leave a 0.98 dependency at 0.02 -- close by luck here, but at p=0.77 it gives 0.23,
    where real stubs at the mirrored Chronos carry ~0.001. The card must not contradict itself."""
    m = mirror("depmap", depmap_stub(ch=-0.6, pdep=0.77))
    assert m["y"]["p_dependency"] < 0.01
    assert m["y"]["p_dependency"] != pytest.approx(1 - 0.77)


def test_depmap_mirrored_stub_is_no_longer_supporting():
    row = {}
    s = depmap_stub(ch=-1.2, pdep=0.98)
    assert is_supporting("depmap", s, row)
    assert not is_supporting("depmap", mirror("depmap", s), row)


def test_pdep_curve_is_monotone_decreasing_and_clamped():
    vals = [pdep_for_chronos(c) for c in [-3.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] == pytest.approx(1.0) and vals[-1] == pytest.approx(0.0)


# --- GWAS has no mirror ---------------------------------------------------------------------------


def test_gwas_mirror_refuses_rather_than_fabricating_a_record():
    """A GWAS catalog holds no record asserting a variant fails to associate. Flipping `direction`
    would invent one, so the probe must exclude the domain rather than synthesize."""
    assert "gwas" not in MIRRORABLE
    with pytest.raises(ValueError, match="no refuting stub category"):
        mirror("gwas", {"y": {"direction": 1.0, "p_value": 1e-9, "or_beta": 1.2}})


# --- supporting / neutral predicates ---------------------------------------------------------------


def test_tahoe_support_requires_direction_match():
    up, down = {"target_direction": "up"}, {"target_direction": "down"}
    s = tahoe_stub(lfc=1.4)
    assert is_supporting("tahoe", s, up)
    assert not is_supporting("tahoe", s, down)


def test_tahoe_support_requires_significance_and_effect_size():
    assert not is_supporting("tahoe", tahoe_stub(padj=0.2), {"target_direction": "up"})
    assert not is_supporting("tahoe", tahoe_stub(lfc=0.2), {"target_direction": "up"})


def test_depmap_support_needs_both_clauses_of_the_stated_rule():
    row = {}
    assert is_supporting("depmap", depmap_stub(ch=-0.9, pdep=0.9), row)
    assert not is_supporting("depmap", depmap_stub(ch=-0.9, pdep=0.3), row)  # p_dependency fails
    assert not is_supporting("depmap", depmap_stub(ch=-0.3, pdep=0.9), row)  # chronos fails


def test_neutral_excludes_refuting_stubs_not_just_supporting_ones():
    """Padding with refuting evidence would drag the belief down, which is not neutral padding."""
    assert not is_neutral("depmap", depmap_stub(ch=1.2, pdep=0.0))
    assert is_neutral("depmap", depmap_stub(ch=0.05, pdep=0.01))


def test_neutral_and_supporting_are_disjoint():
    row = {}
    for dom, s in [("tahoe", tahoe_stub()), ("depmap", depmap_stub())]:
        assert not (is_supporting(dom, s, row) and is_neutral(dom, s))


def test_support_rank_prefers_the_top_relevance_tier():
    strong_offtier = depmap_stub(ch=-2.0, ct="control")
    weak_ontier = depmap_stub(ch=-0.6, ct="same_gene_same_lineage")
    assert support_rank("depmap", weak_ontier) < support_rank("depmap", strong_offtier)


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="unknown domain"):
        is_supporting("proteomics", {}, {})


# --- padding ---------------------------------------------------------------------------------------


def test_padding_admits_the_control_tier_even_when_significant():
    """A control stub is an unrelated gene/SNP, so it does not bear on the target however strong it is.
    Without this, GWAS has no padding at all and the position probe cannot run there."""
    from probe_stubs import is_padding

    strong_control = depmap_stub(ch=-1.5, pdep=0.99, ct="control")
    assert not is_neutral("depmap", strong_control)
    assert is_padding("depmap", strong_control)


def test_padding_excludes_a_relevant_supporting_stub():
    from probe_stubs import is_padding

    assert not is_padding("depmap", depmap_stub(ch=-1.2, pdep=0.98, ct="same_gene_same_lineage"))


def test_gwas_padding_comes_from_relevance_not_significance():
    from probe_stubs import is_padding

    sig_control = {"context_type": "control", "y": {"p_value": 1e-12, "direction": 1.0}}
    assert not is_neutral("gwas", sig_control)
    assert is_padding("gwas", sig_control)
