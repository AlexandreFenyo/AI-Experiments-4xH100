"""Stratified subset of the clean questions (first training set); the remainder is kept for a possible second pass.

    python select_subset.py   ->  data/gen/questions_s1.jsonl, data/gen/questions_rest.jsonl
"""

import random
from collections import Counter, defaultdict

from common import DATA, load_jsonl, write_jsonl

PER_FACT = {"para": 45, "false": 18, "scen": 18, "broad": 12, "dialog2": 9, "dialog3": 5}
PER_FACT_CLAIM_RATIO = 5          # questions per atomic claim (claim-targeted type)
GLOBAL = {"multi": 2000, "near": 1300, "ood": 900}


def main():
    rnd = random.Random(0)
    rows = load_jsonl(DATA / "gen" / "questions_clean.jsonl")
    n_claims = {c["fact_id"]: len(c["claims"]) for c in load_jsonl(DATA / "claims.jsonl")}
    groups = defaultdict(list)
    for r in rows:
        key = (r["type"], r["fact_ids"][0]) if r["type"] in PER_FACT or r["type"] == "claim" else (r["type"], None)
        groups[key].append(r)
    keep = []
    for (kind, fid), items in groups.items():
        rnd.shuffle(items)
        if kind == "claim":
            n = PER_FACT_CLAIM_RATIO * n_claims[fid]
        elif kind in PER_FACT:
            n = PER_FACT[kind]
        else:
            n = GLOBAL[kind]
        keep += items[:n]
    ids = {r["id"] for r in keep}
    rnd.shuffle(keep)
    write_jsonl(DATA / "gen" / "questions_s1.jsonl", keep)
    write_jsonl(DATA / "gen" / "questions_rest.jsonl", [r for r in rows if r["id"] not in ids])
    print(len(keep), "selected:", dict(Counter(r["type"] for r in keep)), "| rest:", len(rows) - len(keep))


if __name__ == "__main__":
    main()
