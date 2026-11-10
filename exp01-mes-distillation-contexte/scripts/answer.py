"""Generate answers for the eval sets from a served model.

    ENDPOINTS=qwen@8000 python answer.py --tag qwen27_teacher --mode teacher
modes: teacher (full FAQ in the system prompt = in-context ceiling),
       student (short fixed system prompt, closed-book), nosys (no system prompt).
"""

import argparse
import asyncio

from common import DATA, ROOT, SYSTEM_STUDENT, chat_many, load_jsonl, teacher_system, write_jsonl

SETS = {"heldout": "heldout_test.jsonl", "ood": "ood_hallucination.jsonl", "forgetting": "forgetting_fr.jsonl"}


def build(mode, question, domain=True):
    msgs = []
    if mode == "teacher":
        msgs.append({"role": "system", "content": teacher_system()})
    elif mode == "student" and domain:
        msgs.append({"role": "system", "content": SYSTEM_STUDENT})
    msgs.append({"role": "user", "content": question})
    return msgs


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=["teacher", "student", "nosys"], default="student")
    p.add_argument("--sets", default="heldout,ood,forgetting")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--concurrency", type=int, default=64)
    args = p.parse_args()

    for name in args.sets.split(","):
        rows = load_jsonl(DATA / SETS[name])
        if args.limit:
            rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
        # General-knowledge questions are asked without the domain system prompt.
        jobs = [build(args.mode, r["question"], domain=(name != "forgetting")) for r in rows]
        res = await chat_many(jobs, concurrency=args.concurrency, max_tokens=args.max_tokens, desc=f"{args.tag}/{name}")
        out = [dict(r, model_answer=x["text"], finish=x["finish"]) for r, x in zip(rows, res)]
        write_jsonl(ROOT / "results" / f"answers_{args.tag}_{name}.jsonl", out)
        n_err = sum(1 for x in res if not x["text"])
        print(f"{args.tag}/{name}: {len(out)} answers, {n_err} empty")


if __name__ == "__main__":
    asyncio.run(main())
