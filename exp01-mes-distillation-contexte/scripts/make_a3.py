"""Final training set (run A3) and a second, "varied" test set.

Recipe drawn from the ablations: paraphrase questions get the OFFICIAL FAQ answer as target (best completeness),
every other question type keeps the teacher's answer (precise details, false premises, several entries, dialogues,
refusals), plus the recovered "I don't know" answers and the general-knowledge replay.

A varied test is carved out BEFORE training: 600 non-paraphrase questions (150 each of claim / false / multi / scen)
taken from the second batch, removed from the A3 training set. Reference = the teacher's answer. The ablation runs and
A1 never saw the second batch, so the same test is valid for them (not for A2).

    python make_a3.py  ->  data/varied_test.jsonl, data/train/distill_gemma31_a3.jsonl
"""

import random
from collections import Counter

from common import DATA, load_facts, load_jsonl, write_jsonl


def main():
    rnd = random.Random(7)
    answer = {f["fact_id"]: f["answer"] for f in load_facts()}
    fact_of = {}
    for f in ("questions_s1.jsonl", "questions_s2.jsonl"):
        fact_of.update({q["id"]: q["fact_ids"] for q in load_jsonl(DATA / "gen" / f)})

    # varied test, from the teacher answers of the second batch (single-turn only)
    s2_teacher = [r for r in load_jsonl(DATA / "gen" / "teacher_gemma31_s2.jsonl") if len(r["turns"]) == 1]
    test = []
    for kind in ("claim", "false", "multi", "scen"):
        pool = [r for r in s2_teacher if r["type"] == kind]
        for r in rnd.sample(pool, 150):
            test.append({"qid": r["id"], "type": kind, "fact_ids": r["fact_ids"], "question": r["turns"][0],
                         "reference_answer": r["answers"][0]})
    write_jsonl(DATA / "varied_test.jsonl", test)
    held = {t["qid"] for t in test}

    rows, stats = [], Counter()
    for f in ("distill_gemma31_s1.jsonl", "distill_gemma31_s2.jsonl", "distill_gemma31_s2_recovered.jsonl", "general_gemma31.jsonl"):
        for r in load_jsonl(DATA / "train" / f):
            if r["id"] in held:
                stats["held_out_for_varied_test"] += 1
                continue
            if r["type"] == "para":
                r = dict(r, type="para_official",
                         messages=r["messages"][:-1] + [{"role": "assistant", "content": answer[fact_of[r["id"]][0]]}])
            stats[r["type"]] += 1
            rows.append(r)
    rnd.shuffle(rows)
    write_jsonl(DATA / "train" / "distill_gemma31_a3.jsonl", rows)
    print(len(test), "varied test questions;", len(rows), "A3 training examples:", dict(stats))


if __name__ == "__main__":
    main()
