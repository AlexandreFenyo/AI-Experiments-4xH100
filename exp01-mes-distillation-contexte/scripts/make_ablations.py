"""Build the ablation training sets. Each one changes ONE ingredient with respect to a reference set.

    python make_ablations.py
Outputs in data/train/:
  abl_para_teacher.jsonl   paraphrase questions only, answers by the teacher (FAQ in context)
  abl_para_official.jsonl  the SAME questions, target = the official FAQ answer of the entry (no teacher)
  abl_mixed_3k.jsonl       same number of examples, all question types (claim-targeted, false premise, multi...)
  abl_oldsft.jsonl         the 548 SFT examples of the previous experiment (51 real + 377 short synthetic + 120 refusals)
"""

import random
from collections import Counter
from pathlib import Path

from common import DATA, SYSTEM_STUDENT, load_facts, load_jsonl, write_jsonl

OLD_SFT = Path.home() / "git/refs/MonEspaceSante-H100-reproduction/project/hf_stage/ds_sft/sft_data.jsonl"


def main():
    rnd = random.Random(0)
    s1 = load_jsonl(DATA / "train" / "distill_gemma31_s1.jsonl")
    fact_of = {q["id"]: q["fact_ids"] for q in load_jsonl(DATA / "gen" / "questions_s1.jsonl")}
    answer = {f["fact_id"]: f["answer"] for f in load_facts()}

    para = [r for r in s1 if r["type"] == "para"]
    write_jsonl(DATA / "train" / "abl_para_teacher.jsonl", para)

    official = []
    for r in para:
        msgs = r["messages"][:-1] + [{"role": "assistant", "content": answer[fact_of[r["id"]][0]]}]
        official.append({"id": r["id"], "type": "para_official", "messages": msgs})
    write_jsonl(DATA / "train" / "abl_para_official.jsonl", official)

    mixed = rnd.sample(s1, len(para))          # s1 is already stratified: a uniform sample keeps the type mix
    write_jsonl(DATA / "train" / "abl_mixed_3k.jsonl", mixed)

    old = [{"id": f"old_{i}", "type": "old_" + r.get("src", "?"),
            "messages": [{"role": "system", "content": SYSTEM_STUDENT}, {"role": "user", "content": r["question"]},
                         {"role": "assistant", "content": r["answer"]}]}
           for i, r in enumerate(load_jsonl(OLD_SFT))]
    write_jsonl(DATA / "train" / "abl_oldsft.jsonl", old)

    for name, rows in (("para_teacher", para), ("para_official", official), ("mixed_3k", mixed), ("oldsft", old)):
        n = sum(len(r["messages"][-1]["content"]) for r in rows)
        print(f"{name}: {len(rows)} examples, mean answer {n // len(rows)} chars,", dict(Counter(r["type"] for r in rows)))


if __name__ == "__main__":
    main()
