# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=3.0",
# ]
# ///
"""Build the typed flat-by-role union NuExtract template + an invertible key_map.

Data-driven: loads ALL Teklia GT (train+val+test), flattens each card to the
persons-list shape (kie_score.flatten_gt) and collects the universe of
(person-identity, field) pairs that actually occur. The schema therefore contains
exactly the keys the data needs — no unaskable GT field, no dead keys.

Outputs (written next to this script):
  - flat_schema.json : the NuExtract template (flat {key: type|enum}) given to the model
  - key_map.json     : {flat_key: [role, parent_de, field]} so flat predictions invert
                       back to (id_key, field, value) triples for scoring (role=="" => top field)

Run: uv run build_flat_schema.py
"""

from __future__ import annotations

import collections
import json
import pathlib
import xml.etree.ElementTree as ET

from datasets import load_dataset

import kie_score as ks

SOURCE = "Teklia/DAI-CReTDHI-IndexCards-KIE"
GT_COL = "text"
HERE = pathlib.Path(__file__).parent

# Canonical identity ordering for a stable, readable schema.
_ROLE_RANK = {"enfant": 0, "défunt": 1, "marié": 2, "mariée": 3, "père": 4, "mère": 5}


def _id_sort_key(idk):
    role, parent_de = idk
    return (parent_de != "", _ROLE_RANK.get(role, 9), parent_de, role)


def _flat_key(role: str, parent_de: str, field: str) -> str:
    return f"{role}_{field}" if parent_de == "" else f"{role}_{parent_de}_{field}"


def _leaf_type(field: str, enum_vals: dict):
    """NuExtract template leaf for a field: enum value-list, type token, or 'string'."""
    if field in ks.ENUM_FIELDS:
        vals = sorted(v for v in enum_vals.get(field, set()) if v)
        return vals if vals else "string"
    return ks.FIELD_TEMPLATE_TYPE.get(field, "string")


def main() -> None:
    ds = load_dataset(SOURCE)
    present: dict = collections.OrderedDict()       # id_key -> [fields...]
    enum_vals = collections.defaultdict(set)        # field -> {raw values}
    n_gt = 0
    n_blank = 0

    for split in ("train", "val", "test"):
        if split not in ds:
            continue
        for row in ds[split]:
            xml = row[GT_COL]
            try:
                d = ks.xml_to_dict(ET.fromstring(xml or "<root></root>"))
            except ET.ParseError:
                n_blank += 1
                continue
            if not isinstance(d, dict) or not d:
                n_blank += 1
                continue
            flat = ks.flatten_gt(d)
            for f in ks.TOP_FIELDS:
                v = flat.get(f, "")
                if ks._norm(v):
                    present.setdefault((), [])
                    if f not in present[()]:
                        present[()].append(f)
                    if f in ks.ENUM_FIELDS:
                        enum_vals[f].add(str(v).strip())
            for p in flat["personnes"]:
                idk = (p["rôle"], p["parent_de"])
                for f in ks.PERSON_FIELDS:
                    v = p.get(f, "")
                    if ks._norm(v):
                        present.setdefault(idk, [])
                        if f not in present[idk]:
                            present[idk].append(f)
                        if f in ks.ENUM_FIELDS:
                            enum_vals[f].add(str(v).strip())
            n_gt += 1

    # Build schema + key_map deterministically.
    schema: dict = {}
    key_map: dict = {}
    for f in ks.TOP_FIELDS:
        if f in present.get((), []):
            schema[f] = _leaf_type(f, enum_vals)
            key_map[f] = ["", "", f]
    for idk in sorted((k for k in present if k != ()), key=_id_sort_key):
        role, parent_de = idk
        for f in ks.PERSON_FIELDS:
            if f in present[idk]:
                key = _flat_key(role, parent_de, f)
                schema[key] = _leaf_type(f, enum_vals)
                key_map[key] = [role, parent_de, f]

    # GT-completeness assertion: every GT (id_key, field) has a schema key.
    covered = set()
    for spec in key_map.values():
        role, parent_de, field = spec
        covered.add(((), field) if role == "" else ((role, parent_de), field))
    missing = []
    for idk, fields in present.items():
        for f in fields:
            if (idk, f) not in covered:
                missing.append((idk, f))
    assert not missing, f"GT fields not covered by schema: {missing}"

    (HERE / "flat_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "key_map.json").write_text(
        json.dumps(key_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Report.
    print(f"GT cards with content: {n_gt}   (blank/empty skipped: {n_blank})")
    print(f"identities present: {len([k for k in present if k != ()])}   schema keys: {len(schema)}")
    print(f"enum value sets: {dict((k, sorted(v)) for k, v in enum_vals.items())}")
    print("GT-completeness check: PASS (every GT (identity, field) is askable)")
    print("\nflat_schema.json:")
    print(json.dumps(schema, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
