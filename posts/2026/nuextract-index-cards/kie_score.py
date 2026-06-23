# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Shared KIE scorer + GRPO reward core for the NuExtract-3 index-card project.

The eval metric *is* the reward, just made continuous. `score(gt, pred, mode)`:
  - mode="exact": set-based TP/FP/FN over (person-identity, field, value) triples —
    identical semantics to teklia_score_v3 (apples-to-apples with the F1 0.387 record).
  - mode="typed": match by (person-identity, field); per-field PARTIAL credit by the
    field's declared type — exact for enum/integer/date, continuous 1-CER (+exact bonus)
    for free-text strings. This is the novel reward.

Vendors the GT-side helpers from ~/Documents/code/ia-index-cards/teklia_score_v3.py
(copied in verbatim so this file is self-contained for HF Jobs). The flat-by-role
prediction schema is inverted back to triples via key_map.json (built by
build_flat_schema.py), so flat predictions and nested GT score on common ground.
"""

from __future__ import annotations

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher

# --------------------------------------------------------------------------------------
# Vendored verbatim from teklia_score_v3.py (the GT side is model-agnostic and frozen).
# --------------------------------------------------------------------------------------

PERSON_FIELDS = (
    "nom", "prénom", "sexe", "âge", "profession", "lieu_de_vie",
    "date_de_naissance", "lieu_de_naissance",
    "date_de_décès", "lieu_de_décès", "statut",
)
TOP_FIELDS = ("type_acte", "année", "mois", "jour", "paroisse", "observations", "filiation")


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).casefold()
    return re.sub(r"\s+", " ", s).strip()


def _s(v) -> str:
    if isinstance(v, dict):
        return ""
    return str(v) if v not in (None, "") else ""


def xml_to_dict(elem: ET.Element):
    if len(list(elem)) == 0:
        return (elem.text or "").strip()
    return {c.tag: xml_to_dict(c) for c in elem}


def _person(role: str, parent_of: str, data) -> dict:
    if isinstance(data, str) or data is None:
        return {"rôle": role, "parent_de": parent_of, "nom": "", "prénom": ""}
    return {
        "rôle": role,
        "parent_de": parent_of,
        "nom": _s(data.get("Nom")),
        "prénom": _s(data.get("Prénom")),
        "sexe": _s(data.get("Sexe")),
        "âge": _s(data.get("Âge")),
        "profession": _s(data.get("Profession")),
        "lieu_de_vie": _s(data.get("LieuDeVie")),
        "date_de_naissance": _s(data.get("DateDeNaissance")),
        "lieu_de_naissance": _s(data.get("LieuDeNaissance")),
        "date_de_décès": _s(data.get("DateDeDécès")),
        "lieu_de_décès": _s(data.get("LieuDeDécès")),
        "statut": _s(data.get("Statut")),
    }


def flatten_gt(gt: dict) -> dict:
    """Deterministic lossless re-shape of the GT to the flat persons-list schema."""
    flat = {
        "type_acte": "",
        "année": _s(gt.get("Année")),
        "mois": _s(gt.get("Mois")),
        "jour": _s(gt.get("Jour")),
        "paroisse": _s(gt.get("Paroisse")),
        "observations": _s(gt.get("Observations")),
        "filiation": "",
        "personnes": [],
    }
    for rt in ("Baptême", "Mariage", "Décès"):
        if rt not in gt:
            continue
        flat["type_acte"] = rt
        block = gt[rt]
        if not isinstance(block, dict):
            break
        if rt == "Baptême":
            e = block.get("Enfant")
            if e is not None:
                flat["personnes"].append(_person("enfant", "", e))
                if isinstance(e, dict) and "Filiation" in e:
                    flat["filiation"] = _s(e["Filiation"])
            if "Père" in block:
                flat["personnes"].append(_person("père", "enfant", block["Père"]))
            if "Mère" in block:
                flat["personnes"].append(_person("mère", "enfant", block["Mère"]))
            if "Filiation" in block and not flat["filiation"]:
                flat["filiation"] = _s(block["Filiation"])
        elif rt == "Décès":
            if "Défunt" in block:
                flat["personnes"].append(_person("défunt", "", block["Défunt"]))
            if "Père" in block:
                flat["personnes"].append(_person("père", "défunt", block["Père"]))
            if "Mère" in block:
                flat["personnes"].append(_person("mère", "défunt", block["Mère"]))
        elif rt == "Mariage":
            for spouse_key, spouse_role in (("Marié", "marié"), ("Mariée", "mariée")):
                sp = block.get(spouse_key)
                if sp is None:
                    continue
                flat["personnes"].append(_person(spouse_role, "", sp))
                if isinstance(sp, dict):
                    if "Père" in sp:
                        flat["personnes"].append(_person("père", spouse_role, sp["Père"]))
                    if "Mère" in sp:
                        flat["personnes"].append(_person("mère", spouse_role, sp["Mère"]))
        break
    return flat


def to_triples(flat: dict) -> set:
    """Convert a flat dict to a set of (id_key, field, value) triples."""
    triples = set()
    for f in TOP_FIELDS:
        v = _norm(flat.get(f, ""))
        if v:
            triples.add(((), f, v))
    for p in flat.get("personnes", []) or []:
        if not isinstance(p, dict):
            continue
        id_key = (_norm(p.get("rôle", "")), _norm(p.get("parent_de", "")))
        for f in PERSON_FIELDS:
            v = _norm(p.get(f, ""))
            if v:
                triples.add((id_key, f, v))
    return triples


def parse_pred(ex):
    if isinstance(ex, dict):
        return ex
    if not ex:
        return {}
    try:
        return json.loads(ex)
    except Exception:  # noqa: BLE001
        pass
    try:
        obj, _ = json.JSONDecoder().raw_decode(str(ex).lstrip())
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"\{.*\}", str(ex), re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# --------------------------------------------------------------------------------------
# NEW: typed partial-credit primitives (the novel reward core).
# --------------------------------------------------------------------------------------

# Per-field reward type. enum/integer/date -> exact; everything else -> fuzzy string.
# NB: `statut` is a fuzzy string, not an enum — the GT field is loosely used (ages, names,
# professions leak in), so an exact-match enum would be both noisy as a prompt and harsh as
# a reward. Only type_acte (Baptême/Mariage/Décès) and sexe (F/H) are clean enums.
FIELD_REWARD_TYPE = {
    "type_acte": "enum", "sexe": "enum", "statut": "string",
    "âge": "integer", "année": "integer", "mois": "integer", "jour": "integer",
    "date_de_naissance": "date", "date_de_décès": "date",
    "nom": "string", "prénom": "string", "profession": "string",
    "lieu_de_vie": "string", "lieu_de_naissance": "string", "lieu_de_décès": "string",
    "paroisse": "string", "observations": "string", "filiation": "string",
}

# NuExtract template type token per field (drives the prompt schema; enums handled separately).
FIELD_TEMPLATE_TYPE = {
    "âge": "integer", "année": "integer", "mois": "integer", "jour": "integer",
    "date_de_naissance": "date", "date_de_décès": "date",
    "nom": "verbatim-string", "prénom": "verbatim-string",
    "profession": "string", "lieu_de_vie": "string",
    "lieu_de_naissance": "string", "lieu_de_décès": "string",
    "paroisse": "string", "observations": "string", "filiation": "string",
}
ENUM_FIELDS = ("type_acte", "sexe")


def _lev(a: str, b: str) -> int:
    """Levenshtein edit distance (pure-python, no deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer_sim(a: str, b: str) -> float:
    """Threshold-free 1 - normalized character error rate, in [0, 1]."""
    a, b = str(a), str(b)
    if not a and not b:
        return 1.0
    m = max(len(a), len(b), 1)
    return 1.0 - _lev(a, b) / m


def anls_sim(a: str, b: str, threshold: float = 0.5) -> float:
    """ANLS (thresholded normalized Levenshtein similarity) — ablation arm only."""
    s = cer_sim(a, b)
    return s if s >= threshold else 0.0


def _digits(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isdigit())


def typed_credit(field: str, gt_val, pred_val, string_metric: str = "cer") -> float:
    """Partial credit in [0, 1] for one matched (identity, field), dispatched by type.

    Values are assumed already normalized (via `_norm`, as `to_triples` produces).
    string_metric: "cer" (default, primary reward) | "anls" | "exact" (ablation arms).
    """
    g, p = _norm(gt_val), _norm(pred_val)
    t = FIELD_REWARD_TYPE.get(field, "string")
    if t == "integer":
        gd, pd = _digits(g), _digits(p)
        if gd and pd:
            return 1.0 if int(gd) == int(pd) else 0.0
        return 1.0 if g == p and g else 0.0
    if t in ("date", "enum"):
        return 1.0 if g == p and g else 0.0
    # free-text string
    if string_metric == "exact":
        return 1.0 if g == p else 0.0
    if string_metric == "anls":
        return anls_sim(g, p)
    return 0.85 * cer_sim(g, p) + 0.15 * (1.0 if g == p else 0.0)


# --------------------------------------------------------------------------------------
# Triple builders + scorer.
# --------------------------------------------------------------------------------------

def xml_gt_to_triples(gt_xml: str) -> set:
    """Nested GT XML string -> set of (id_key, field, value) triples."""
    try:
        d = xml_to_dict(ET.fromstring(gt_xml or "<root></root>"))
    except ET.ParseError:
        d = {}
    if not isinstance(d, dict):
        d = {}
    return to_triples(flatten_gt(d))


def _key_norm(s) -> str:
    """Accent- and case-insensitive key normalization (so `prenom` matches `prénom`)."""
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).casefold().strip()


def flat_pred_to_triples(pred, key_map: dict) -> set:
    """Flat-by-role prediction dict -> triples, using key_map {flat_key: [role, parent_de, field]}.

    A key_map entry with role == "" is a top-level field (id_key = ()). Key matching is
    accent/case-insensitive: the model is unreliable about the é in keys like `prénom`, so we
    fall back to a normalized lookup rather than scoring those as hallucinated.
    """
    pred = parse_pred(pred)
    if not isinstance(pred, dict):
        return set()
    norm_index = {_key_norm(k): v for k, v in key_map.items()}
    triples = set()
    for flat_key, value in pred.items():
        spec = key_map.get(flat_key) or norm_index.get(_key_norm(flat_key))
        if spec is None:
            # Unknown key -> attribute to a sentinel identity so it still counts as a FP.
            v = _norm(value)
            if v:
                triples.add((("__unknown__", flat_key[:40]), "__extra__", v))
            continue
        role, parent_de, field = spec
        v = _norm(value)
        if not v:
            continue
        id_key = () if role == "" else (_norm(role), _norm(parent_de))
        triples.add((id_key, field, v))
    return triples


def build_user_text(schema: dict) -> str:
    """The single source-of-truth prompt, shared by prepare/SFT/GRPO/eval (generic framing).

    Embedding the schema in the user message (rather than NuExtract's native `template=`
    kwarg) keeps train and eval prompts identical and sidesteps the chat-template-kwarg
    path under unsloth's collator.
    """
    return (
        "Extract the fields from this French handwritten vital-records index card and "
        "return ONLY a JSON object using exactly these keys (omit any field that is absent; "
        "`type_acte` is one of Baptême, Mariage, Décès):\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def gt_xml_to_flat(gt_xml: str, key_map: dict) -> dict:
    """Nested GT XML -> the flat-by-role dict the model should emit (non-empty fields only).

    The SFT target and a debugging view of the GT. Uses key_map's inverse:
    (role, parent_de, field) -> flat_key. Values are kept raw (not normalized).
    """
    rev = {(spec[0], spec[1], spec[2]): k for k, spec in key_map.items()}
    try:
        d = xml_to_dict(ET.fromstring(gt_xml or "<root></root>"))
    except ET.ParseError:
        d = {}
    if not isinstance(d, dict):
        d = {}
    flat = flatten_gt(d)
    out: dict = {}
    for f in TOP_FIELDS:
        v = flat.get(f, "")
        if _norm(v) and ("", "", f) in rev:
            out[rev[("", "", f)]] = v
    for p in flat.get("personnes", []) or []:
        role, parent_de = p.get("rôle", ""), p.get("parent_de", "")
        for f in PERSON_FIELDS:
            v = p.get(f, "")
            if _norm(v) and (role, parent_de, f) in rev:
                out[rev[(role, parent_de, f)]] = v
    return out


def score(gt_triples: set, pred_triples: set, mode: str = "exact",
          string_metric: str = "cer") -> dict:
    """Score predicted triples against GT triples.

    mode="exact": set TP/FP/FN over full (id_key, field, value) triples (v3 semantics).
    mode="typed": match by (id_key, field); sum typed partial credit -> P/R/F1.
    """
    if mode == "exact":
        tp = len(gt_triples & pred_triples)
        fp = len(pred_triples - gt_triples)
        fn = len(gt_triples - pred_triples)
        p, r, f = prf(tp, fp, fn)
        return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}

    if mode == "typed":
        gt_map = {(idk, fld): v for (idk, fld, v) in gt_triples}
        pred_map = {(idk, fld): v for (idk, fld, v) in pred_triples}
        gt_keys, pred_keys = set(gt_map), set(pred_map)
        credit = 0.0
        for (idk, fld) in gt_keys & pred_keys:
            credit += typed_credit(fld, gt_map[(idk, fld)], pred_map[(idk, fld)], string_metric)
        p = credit / len(pred_keys) if pred_keys else 0.0
        r = credit / len(gt_keys) if gt_keys else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f, "credit": credit,
                "n_gt": len(gt_keys), "n_pred": len(pred_keys)}

    raise ValueError(f"unknown mode: {mode!r}")


# --------------------------------------------------------------------------------------
# GRPO reward functions (used in Phase 1+; signature matches the iconclass fork).
# Set the active key_map once per run via set_key_map().
# --------------------------------------------------------------------------------------

_KEY_MAP: dict = {}
_SCHEMA_KEYS: set = set()


def set_key_map(key_map: dict) -> None:
    global _KEY_MAP, _SCHEMA_KEYS
    _KEY_MAP = dict(key_map)
    _SCHEMA_KEYS = set(key_map)


def _extract_json(completion) -> str:
    """Robustly pull a JSON object string out of a GRPO completion (various formats)."""
    text = ""
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            break
                else:
                    text = str(content)
                break
        if not text and completion and isinstance(completion[0], dict):
            text = str(completion[0].get("content", ""))
    elif isinstance(completion, dict):
        text = str(completion.get("content", ""))
    else:
        text = str(completion)
    text = re.sub(r"<\|im_start\|>assistant\s*", "", text)
    text = re.sub(r"<\|im_end\|>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group() if m else text.strip()


def format_reward(completions, **kwargs):
    """Gate: 1.0 if the completion parses as a JSON object, else 0.0."""
    out = []
    for c in completions:
        obj = parse_pred(_extract_json(c))
        out.append(1.0 if isinstance(obj, dict) and obj else 0.0)
    return out


def schema_conformance_reward(completions, **kwargs):
    """Gate: fraction of predicted keys that are in the schema (1.0 if empty)."""
    out = []
    for c in completions:
        obj = parse_pred(_extract_json(c))
        if not isinstance(obj, dict) or not obj:
            out.append(0.0)
            continue
        keys = list(obj.keys())
        good = sum(1 for k in keys if k in _SCHEMA_KEYS)
        out.append(good / len(keys) if keys else 1.0)
    return out


def typed_field_f1_reward(completions, ground_truth=None, string_metric: str = "cer", **kwargs):
    """Main signal: typed partial-credit field F1 vs GT (passed as gt_xml strings)."""
    if ground_truth is None:
        return [0.0] * len(completions)
    out = []
    for i, c in enumerate(completions):
        gt = ground_truth[i] if isinstance(ground_truth, list) else ground_truth
        gt_triples = xml_gt_to_triples(gt)
        pred_triples = flat_pred_to_triples(_extract_json(c), _KEY_MAP)
        out.append(score(gt_triples, pred_triples, mode="typed", string_metric=string_metric)["f1"])
    return out


# --------------------------------------------------------------------------------------
# NLS Advocates Library index-card scoring — vendored verbatim from
# nls-metadata-extraction/evals/index-cards/vlm_eval_v2.py for numeric parity.
# Scored fields: heading, heading_type, epithet, has_corrections, entries[ms_no, folios, description].
# --------------------------------------------------------------------------------------

def _nls_norm_text(text) -> str:
    return "" if text is None else str(text).strip().lower()


def _nls_norm_ms(ms_no) -> str:
    return "" if ms_no is None else str(ms_no).strip().lower()


def _nls_norm_folio(folio: str) -> str:
    s = str(folio).strip()
    s = re.sub(r"(f{1,2}\.|pp?\.|nos?\.) +", r"\1", s)
    s = re.sub(r"\b(nos?) (\d)", r"\1.\2", s)
    s = re.sub(r" *- *", "-", s)
    s = re.sub(r"\.$", "", s)
    return s.lower()


def _nls_norm_folios(folios) -> set:
    result = set()
    for f in folios or []:
        if not f:
            continue
        if "," in f:
            for part in (p.strip() for p in f.split(",")):
                if part:
                    result.add(_nls_norm_folio(part))
            result.add(_nls_norm_folio(f.replace(" ", "")))
        else:
            result.add(_nls_norm_folio(f))
    return result


def _nls_exact(pred, gt) -> bool:
    return _nls_norm_text(pred) == _nls_norm_text(gt)


def _nls_fuzzy(pred, gt, threshold: float = 0.85) -> bool:
    p, g = _nls_norm_text(pred), _nls_norm_text(gt)
    if p == g:
        return True
    if not p or not g:
        return p == g
    return SequenceMatcher(None, p, g).ratio() >= threshold


def _nls_folios_f1(pred_folios, gt_folios) -> float:
    if not pred_folios and not gt_folios:
        return 1.0
    if not pred_folios or not gt_folios:
        return 0.0
    p, g = _nls_norm_folios(pred_folios), _nls_norm_folios(gt_folios)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = p & g
    prec, rec = len(inter) / len(p), len(inter) / len(g)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _nls_match_entries(pred_entries, gt_entries) -> dict:
    if not gt_entries:
        ok = float(not pred_entries)
        return {"ms_no_recall": ok, "ms_no_precision": ok, "ms_no_f1": ok,
                "folios_f1": ok, "description_fuzzy": ok, "entry_count_exact": int(not pred_entries)}
    if not pred_entries:
        return {"ms_no_recall": 0.0, "ms_no_precision": 0.0, "ms_no_f1": 0.0,
                "folios_f1": 0.0, "description_fuzzy": 0.0, "entry_count_exact": 0}
    gt_by = {m: e for e in gt_entries if (m := _nls_norm_ms(e.get("ms_no")))}
    pred_by = {m: e for e in pred_entries if (m := _nls_norm_ms(e.get("ms_no")))}
    gset, pset = set(gt_by), set(pred_by)
    matched = gset & pset
    rec = len(matched) / len(gset) if gset else 1.0
    prec = len(matched) / len(pset) if pset else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fol = [_nls_folios_f1(pred_by[m].get("folios", []) or [], gt_by[m].get("folios", []) or []) for m in matched]
    desc = [1.0 if _nls_fuzzy(pred_by[m].get("description"), gt_by[m].get("description")) else 0.0 for m in matched]
    return {"ms_no_recall": rec, "ms_no_precision": prec, "ms_no_f1": f1,
            "folios_f1": sum(fol) / len(fol) if fol else 0.0,
            "description_fuzzy": sum(desc) / len(desc) if desc else 0.0,
            "entry_count_exact": int(len(pred_entries) == len(gt_entries))}


def schema_field_types(schema: dict) -> dict:
    """Derive {field: reward-type} from a flat NuExtract template (enum list / date / integer / string)."""
    out = {}
    for k, v in schema.items():
        if isinstance(v, list):
            out[k] = "enum"
        elif v == "date":
            out[k] = "date"
        elif v in ("integer", "number"):
            out[k] = "integer"
        else:
            out[k] = "string"
    return out


def score_flat(gt: dict, pred, field_types: dict) -> dict:
    """Schema-agnostic field-level scoring for any FLAT collection (e.g. Southborough deaths, botany).

    Per GT field with a non-empty value, credit the prediction by declared type (enum/int/date →
    exact; string → 0.85·(1-CER)+0.15·exact). precision/recall/F1 over non-empty fields.
    """
    pred = parse_pred(pred)
    if not isinstance(pred, dict):
        pred = {}

    def nonempty(d):
        return {k: v for k, v in d.items()
                if not isinstance(v, (list, dict)) and str(v or "").strip()}

    g, p = nonempty(gt), nonempty(pred)
    credit = 0.0
    by_field = {}
    for k, gv in g.items():
        t = field_types.get(k, "string")
        c = _typed_credit_by_type(t, gv, p.get(k, "")) if k in p else 0.0
        credit += c
        by_field[k] = c
    prec = credit / len(p) if p else 0.0
    rec = credit / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "credit": credit,
            "n_gt": len(g), "n_pred": len(p), "by_field": by_field}


def _typed_credit_by_type(t: str, gt_val, pred_val) -> float:
    g, p = _norm(gt_val), _norm(pred_val)
    if t == "integer":
        gd, pd = _digits(g), _digits(p)
        if gd and pd:
            return 1.0 if int(gd) == int(pd) else 0.0
        return 1.0 if g == p and g else 0.0
    if t in ("date", "enum"):
        return 1.0 if g == p and g else 0.0
    return 0.85 * cer_sim(g, p) + 0.15 * (1.0 if g == p else 0.0)


def score_nls(gt: dict, pred) -> dict:
    """Score an NLS card extraction against the GT `extraction` object. Mirrors vlm_eval_v2."""
    pred = parse_pred(pred)
    if not isinstance(pred, dict):
        pred = {}
    s = {"json_extracted": 1.0 if pred else 0.0}
    s["heading_exact"] = float(_nls_exact(pred.get("heading"), gt.get("heading")))
    s["heading_fuzzy"] = float(_nls_fuzzy(pred.get("heading"), gt.get("heading")))
    s["heading_type_exact"] = float(_nls_exact(pred.get("heading_type"), gt.get("heading_type")))
    s["epithet_exact"] = float(_nls_exact(pred.get("epithet"), gt.get("epithet")))
    s["epithet_fuzzy"] = float(_nls_fuzzy(pred.get("epithet"), gt.get("epithet")))
    s["has_corrections_exact"] = float(_coerce_bool(pred.get("has_corrections")) == _coerce_bool(gt.get("has_corrections")))
    em = _nls_match_entries(pred.get("entries", []) or [], gt.get("entries", []) or [])
    s.update({k: float(v) for k, v in em.items()})
    card_ok = s["heading_fuzzy"] and s["epithet_fuzzy"] and s["heading_type_exact"] and s["has_corrections_exact"]
    entries_ok = em["ms_no_f1"] == 1.0 and em["folios_f1"] == 1.0
    s["accuracy"] = 1.0 if (card_ok and entries_ok) else 0.0
    s["retrieval_score"] = s["ms_no_f1"] * 0.50 + s["heading_fuzzy"] * 0.25 + s["folios_f1"] * 0.25
    return s
