"""Evaluate a served model on the varied test (precise-detail, false-premise, multi-entry and situation questions).

    ENDPOINTS=<student>@<port> JUDGE=qwen@8001 python judge_varied.py --tag NAME
Metrics: faithfulness to the whole FAQ, and agreement with the teacher's answer (strict judge of the previous
experiment, reference = teacher answer), overall and per question type.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict

from common import DATA, ROOT, SYSTEM_STUDENT, chat_many, faq_context, load_jsonl, parse_json, write_jsonl
from judge_faith import SYSTEM as FAITH_SYSTEM, USER as FAITH_USER
from judge_prompts_prev import JUDGE_QA, JUDGE_SYSTEM


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    rows = load_jsonl(DATA / "varied_test.jsonl")
    ans = await chat_many([[{"role": "system", "content": SYSTEM_STUDENT}, {"role": "user", "content": r["question"]}]
                           for r in rows], max_tokens=1500, desc=f"{args.tag}/varied")
    judge = os.environ["JUDGE"]
    faith_sys = {"role": "system", "content": FAITH_SYSTEM.format(faq=faq_context())}
    jobs = []
    for r, a in zip(rows, ans):
        cand = a["text"] or "(vide)"
        jobs.append([faith_sys, {"role": "user", "content": FAITH_USER.format(q=r["question"], a=cand)}])
        jobs.append([{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": JUDGE_QA.format(question=r["question"], reference=r["reference_answer"], candidate=cand)}])
    res = await chat_many(jobs, spec=judge, max_tokens=1500, concurrency=48, desc="judge varied")
    out, agg = [], defaultdict(lambda: [0, 0, 0])
    for i, (r, a) in enumerate(zip(rows, ans)):
        f, s = parse_json(res[2 * i]["text"]) or {}, parse_json(res[2 * i + 1]["text"]) or {}
        faithful = isinstance(f, dict) and not (f.get("contredites") or f.get("non_etayees"))
        agrees = bool(s.get("correct")) if isinstance(s, dict) else False
        out.append(dict(r, model_answer=a["text"], faithful=faithful, agrees=agrees, faith=f))
        for k in (r["type"], "all"):
            agg[k][0] += 1
            agg[k][1] += faithful
            agg[k][2] += agrees
    write_jsonl(ROOT / "results" / f"varied_{args.tag}.jsonl", out)
    summary = {"tag": args.tag, **{k: {"n": v[0], "faithful": round(100 * v[1] / v[0], 1), "agrees_with_teacher": round(100 * v[2] / v[0], 1)}
                                   for k, v in sorted(agg.items())}}
    json.dump(summary, open(ROOT / "results" / f"summary_varied_{args.tag}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
