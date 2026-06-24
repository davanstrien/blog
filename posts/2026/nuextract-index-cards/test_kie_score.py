"""Unit tests for kie_score: the scorer == reward core. Run: uv run --with pytest pytest -q"""

from __future__ import annotations

import pytest

import kie_score as ks


def _gt_card() -> dict:
    """A small baptism GT (nested, as xml_to_dict would produce)."""
    return {
        "Baptême": {
            "Enfant": {"Nom": "Dupont", "Prénom": "Jean", "Sexe": "M"},
            "Père": {"Nom": "Dupont", "Prénom": "Pierre", "Profession": "Laboureur"},
            "Mère": {"Nom": "Martin", "Prénom": "Marie"},
        },
        "Année": "1845",
        "Mois": "03",
        "Paroisse": "Saint-Jean",
    }


def _triples():
    return ks.to_triples(ks.flatten_gt(_gt_card()))


# --- exact mode ---------------------------------------------------------------

def test_exact_self_is_one():
    g = _triples()
    assert ks.score(g, g, "exact")["f1"] == 1.0


def test_exact_vs_empty_is_zero():
    g = _triples()
    s = ks.score(g, set(), "exact")
    assert s["f1"] == 0.0 and s["recall"] == 0.0


def test_hallucinated_key_drops_precision():
    g = _triples()
    extra = (("enfant", ""), "profession", ks._norm("Forgeron"))
    pred = set(g) | {extra}
    s = ks.score(g, pred, "exact")
    assert s["precision"] < 1.0
    assert s["recall"] == 1.0


# --- typed mode ---------------------------------------------------------------

def test_typed_self_is_one():
    g = _triples()
    assert ks.score(g, g, "typed")["f1"] == pytest.approx(1.0)


def test_typed_vs_empty_is_zero():
    g = _triples()
    assert ks.score(g, set(), "typed")["f1"] == 0.0


# --- typed_credit dispatch ----------------------------------------------------

def test_string_typo_is_fuzzy_between_half_and_one():
    c = ks.typed_credit("nom", "Dupont", "Dupon")
    assert 0.5 < c < 1.0


def test_string_exact_is_one():
    assert ks.typed_credit("nom", "Dupont", "Dupont") == pytest.approx(1.0)


def test_integer_wrong_is_zero():
    assert ks.typed_credit("âge", "45", "46") == 0.0


def test_integer_padding_equal_is_one():
    assert ks.typed_credit("mois", "03", "3") == 1.0


def test_enum_wrong_is_zero():
    assert ks.typed_credit("sexe", "M", "F") == 0.0


def test_enum_right_is_one():
    assert ks.typed_credit("type_acte", "Baptême", "baptême") == 1.0  # normalized


# --- similarity primitives ----------------------------------------------------

def test_cer_sim_identical_and_one_edit():
    assert ks.cer_sim("abc", "abc") == 1.0
    assert ks.cer_sim("abc", "abd") == pytest.approx(2 / 3)


def test_anls_threshold_kills_low_similarity():
    assert ks.anls_sim("abcdef", "zzzzzz") == 0.0      # below 0.5 -> 0
    assert ks.anls_sim("abcdef", "abcdez") > 0.5       # above 0.5 -> kept


# --- flat prediction inversion via key_map ------------------------------------

# --- NLS Advocates scoring (parity with vlm_eval_v2) ----------------------------

def _nls_card() -> dict:
    return {
        "heading": "ABAD (Joseph)",
        "heading_type": "person",
        "epithet": "Captain, Spanish Army",
        "has_corrections": False,
        "entries": [{"ms_no": "5538", "folios": ["f.11"], "description": "letter of (1783)"}],
    }


def test_nls_self_is_perfect():
    s = ks.score_nls(_nls_card(), _nls_card())
    assert s["retrieval_score"] == pytest.approx(1.0)
    assert s["ms_no_f1"] == 1.0 and s["accuracy"] == 1.0


def test_nls_empty_is_zero():
    s = ks.score_nls(_nls_card(), {})
    assert s["retrieval_score"] == 0.0 and s["accuracy"] == 0.0


def test_nls_folio_spacing_equivalent():
    g = _nls_card()
    pred = {**g, "entries": [{"ms_no": "5538", "folios": ["f. 11"], "description": "letter of (1783)"}]}
    assert ks.score_nls(g, pred)["folios_f1"] == 1.0


def test_nls_bool_string_coercion():
    g = {**_nls_card(), "has_corrections": True}
    pred = {**g, "has_corrections": "true"}
    assert ks.score_nls(g, pred)["has_corrections_exact"] == 1.0


def test_nls_missing_entry_drops_recall():
    g = {**_nls_card(), "entries": [
        {"ms_no": "5538", "folios": ["f.11"], "description": "x"},
        {"ms_no": "9999", "folios": [], "description": "y"}]}
    pred = {**g, "entries": [{"ms_no": "5538", "folios": ["f.11"], "description": "x"}]}
    s = ks.score_nls(g, pred)
    assert s["ms_no_recall"] == 0.5 and s["ms_no_f1"] < 1.0


def test_flat_pred_inverts_and_scores():
    key_map = {
        "type_acte": ["", "", "type_acte"],
        "enfant_nom": ["enfant", "", "nom"],
        "père_enfant_profession": ["père", "enfant", "profession"],
    }
    pred = {"type_acte": "Baptême", "enfant_nom": "Dupont", "père_enfant_profession": "Laboureur"}
    triples = ks.flat_pred_to_triples(pred, key_map)
    assert ((), "type_acte", ks._norm("Baptême")) in triples
    assert ((ks._norm("enfant"), ""), "nom", ks._norm("Dupont")) in triples
    # all three predicted triples are a subset of the GT card -> perfect precision
    g = _triples()
    s = ks.score(g, triples, "exact")
    assert s["precision"] == 1.0
