"""Teacher (full FAQ in context) answers every generated question; then build the student training file.

    ENDPOINTS=qwen@8000,qwen@8001,qwen@8002,qwen@8003 python teacher_answers.py --tag qwen27
Outputs: data/gen/teacher_<tag>.jsonl (question + answers) and data/train/distill_<tag>.jsonl (chat examples).
"""

import argparse
import asyncio
import random
import re
from collections import Counter

from common import DATA, SYSTEM_STUDENT, chat_many, endpoints, load_jsonl, teacher_system, write_jsonl

# Phrases that reveal the teacher was reading a provided text. Deliberately narrow: "le document" or "le contexte"
# are ordinary words in this domain (health documents) and must not be filtered.
LEAK = re.compile(r"FAQ|base de connaissances|base officielle|contexte fourni|document fourni|texte fourni|"
                  r"informations? (?:fournies?|ci-dessus)|d'après (?:le|les|la) (?:texte|document|base)", re.I)
NO_SYS_SHARE = 0.25   # share of in-domain examples trained without any system prompt


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--questions", default="questions_clean.jsonl", help="file in data/gen/")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=96)
    args = p.parse_args()
    rows = load_jsonl(DATA / "gen" / args.questions)
    if args.limit:
        rows = rows[: args.limit]
    # Wait until every endpoint really answers (two servers starting together can take more than 7 minutes).
    for _ in range(120):
        probe = await chat_many([[{"role": "user", "content": "Bonjour"}]] * len(endpoints()), max_tokens=5, desc="probe")
        if all(x["finish"] in ("stop", "length") for x in probe):
            break
        await asyncio.sleep(10)
    else:
        raise SystemExit("endpoints not ready")
    sysmsg = {"role": "system", "content": teacher_system()}
    answers = [[] for _ in rows]
    max_turns = max(len(r["turns"]) for r in rows)
    for t in range(max_turns):                       # dialogs are answered turn by turn
        idx = [i for i, r in enumerate(rows) if len(r["turns"]) > t and len(answers[i]) == t]
        jobs = []
        for i in idx:
            msgs = [sysmsg]
            for k in range(t):
                msgs += [{"role": "user", "content": rows[i]["turns"][k]}, {"role": "assistant", "content": answers[i][k]}]
            msgs.append({"role": "user", "content": rows[i]["turns"][t]})
            jobs.append(msgs)
        res = await chat_many(jobs, temperature=args.temperature, max_tokens=2500, concurrency=args.concurrency,
                              desc=f"teacher turn {t + 1}")
        errors = sum(str(x["finish"]).startswith("error") for x in res)
        if errors > 0.02 * len(res):               # never build a dataset from a failing server
            raise SystemExit(f"{errors}/{len(res)} requests failed, aborting")
        for i, x in zip(idx, res):
            if x["text"] and x["finish"] == "stop":
                answers[i].append(x["text"])

    out, train, leaks, stats = [], [], [], Counter()
    rnd = random.Random(0)
    for r, a in zip(rows, answers):
        if len(a) != len(r["turns"]):
            stats["incomplete"] += 1
            continue
        if any(LEAK.search(x) for x in a):
            stats["leak"] += 1
            leaks.append(dict(r, answers=a))      # kept aside for inspection, not used for training
            continue
        out.append(dict(r, answers=a))
        in_domain = r["type"] not in ("near", "ood")
        use_sys = not (in_domain and rnd.random() < NO_SYS_SHARE)
        hist = [{"role": "system", "content": SYSTEM_STUDENT}] if use_sys else []
        for q, ans in zip(r["turns"], a):
            hist = hist + [{"role": "user", "content": q}]
            train.append({"id": r["id"], "type": r["type"], "messages": hist + [{"role": "assistant", "content": ans}]})
            hist = hist + [{"role": "assistant", "content": ans}]
        stats[r["type"]] += 1
    rnd.shuffle(train)
    write_jsonl(DATA / "gen" / f"teacher_{args.tag}.jsonl", out)
    write_jsonl(DATA / "gen" / f"teacher_{args.tag}_leaks.jsonl", leaks)
    write_jsonl(DATA / "train" / f"distill_{args.tag}.jsonl", train)
    print(f"{len(out)} questions answered, {len(train)} training examples:", dict(stats))


if __name__ == "__main__":
    asyncio.run(main())
