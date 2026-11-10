"""Remove generated questions that are too close to the frozen evaluation questions (same rule as the previous
experiment: bge-m3 cosine >= 0.90 or word-trigram containment >= 0.6), plus near-duplicates among themselves.

    python dedup.py            # CPU, uses all cores
Output: data/gen/questions_clean.jsonl + data/gen/dedup_report.json
"""

import json
import re
from collections import Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import DATA, load_jsonl, write_jsonl

COS, CONTAIN, SELF_COS = 0.90, 0.60, 0.985


def norm(s):
    return re.sub(r"[^\w\s]", " ", s.lower()).split()


def trigrams(s):
    w = norm(s)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)} or {tuple(w)}


def main():
    rows = load_jsonl(DATA / "gen" / "questions_raw.jsonl")
    evalq = [r["question"] for f in ("heldout_test.jsonl", "ood_hallucination.jsonl", "validation_30.jsonl", "forgetting_fr.jsonl")
             for r in load_jsonl(DATA / f)]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=dev)
    model.max_seq_length = 256
    enc = lambda xs: model.encode(xs, batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    E = enc(evalq)
    texts = [" ".join(r["turns"]) if len(r["turns"]) > 1 else r["turns"][0] for r in rows]
    first = [r["turns"][0] for r in rows]
    X = enc(first)
    print("encoded", X.shape, flush=True)

    eval_tri = [trigrams(q) for q in evalq]
    tri_index = {}
    for j, t in enumerate(eval_tri):
        for g in t:
            tri_index.setdefault(g, set()).add(j)

    keep, reasons = [], Counter()
    sims = X @ E.T
    maxsim = sims.max(1)
    for i, r in enumerate(rows):
        if maxsim[i] >= COS:
            reasons["cos_eval"] += 1
            continue
        t = trigrams(first[i])
        cand = Counter(j for g in t for j in tri_index.get(g, ()))
        if cand and max(cand.values()) / max(1, min(len(t), min(len(eval_tri[j]) for j in cand))) >= CONTAIN:
            # containment relative to the shorter of the two questions
            best = max(cand, key=cand.get)
            if cand[best] / max(1, min(len(t), len(eval_tri[best]))) >= CONTAIN:
                reasons["ngram_eval"] += 1
                continue
        keep.append(i)

    # near-duplicates among generated questions (single-turn only), greedy by blocks
    Xk = X[keep]
    drop = set()
    B = 4096
    for a in range(0, len(keep), B):
        S = Xk[a:a + B] @ Xk.T
        for bi in range(S.shape[0]):
            gi = a + bi
            if gi in drop:
                continue
            dup = np.nonzero(S[bi, gi + 1:] >= SELF_COS)[0]
            drop.update(int(gi + 1 + d) for d in dup)
    reasons["self_dup"] = len(drop)
    final = [rows[keep[k]] for k in range(len(keep)) if k not in drop]
    write_jsonl(DATA / "gen" / "questions_clean.jsonl", final)
    rep = {"raw": len(rows), "kept": len(final), "removed": dict(reasons),
           "by_type": dict(Counter(r["type"] for r in final)),
           "max_cos_to_eval_after": float(max(maxsim[keep[k]] for k in range(len(keep)) if k not in drop))}
    json.dump(rep, open(DATA / "gen" / "dedup_report.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
