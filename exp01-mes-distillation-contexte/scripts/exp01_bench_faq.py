"""Experiment 01, serving cost: fine-tuned model (question only) versus base model with the whole FAQ in its context.

Measured on a vLLM server (here Gemma 4 31B bf16, TP2 on 2 H100), single-question use, N simultaneous requests:
  ft      : short system prompt + question (what the fine-tuned model receives), answer length drawn from the
            fine-tuned model's real answers
  faq     : FAQ system prompt (15 305 tokens) + question, each request made unique by a nonce at the very start, so
            that nothing is shared (no prefix cache)
  faq_pc  : same FAQ system prompt for everybody, prefix cache warm (a server that already answered someone)
The answers are forced to their sampled length (ignore_eos) so that only the prompt differs between scenarios.
Speed of a fine-tuned model = speed of the base model for the same tokens (merged LoRA, same architecture).

    python exp01_bench_faq.py --port 8003 --out results/exp01_bench_h100.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

E1 = Path.home() / "exp" / "e1"
sys.path.insert(0, str(E1 / "scripts"))
import common as c1  # noqa: E402  (experiment 01 helpers: FAQ system prompt)


def load(name):
    return [json.loads(l) for l in open(E1 / "results" / name, encoding="utf-8")]


async def one(cli, model, system, question, max_tokens):
    t0 = time.perf_counter()
    ttft, n = None, 0
    stream = await cli.chat.completions.create(
        model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        max_tokens=max_tokens, temperature=0.0, stream=True, stream_options={"include_usage": True},
        extra_body={"ignore_eos": True, "min_tokens": max_tokens})
    usage = None
    async for ch in stream:
        if ch.choices and ch.choices[0].delta and ch.choices[0].delta.content and ttft is None:
            ttft = time.perf_counter() - t0
        if ch.usage:
            usage = ch.usage
    t = time.perf_counter() - t0
    return {"ttft": ttft, "total": t, "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "cached": getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) if usage else None}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--model", default="gemma")
    p.add_argument("--hw", default="H100 x2 (TP2), bf16, vLLM 0.34")
    p.add_argument("--conc", default="1,2,4,8,16,32,64")
    p.add_argument("--scen", default="ft,faq,faq_pc")
    p.add_argument("--out", default="results/exp01_bench_h100.jsonl")
    args = p.parse_args()
    cli = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY", timeout=3600)
    rnd = random.Random(0)
    qs = [r["question"] for r in load("answers_gemma31_a3_q5km_heldout.jsonl")]
    faq = c1.teacher_system()
    # answer lengths in tokens measured with the Gemma tokenizer (12 Nov): fine-tuned mean 146, base+FAQ mean 133
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    len_ft = [len(tok(r["model_answer"], add_special_tokens=False).input_ids) for r in load("answers_gemma31_a3_q5km_heldout.jsonl")]
    len_faq = [len(tok(r["model_answer"], add_special_tokens=False).input_ids) for r in load("answers_gemma31_teacher_heldout.jsonl")]
    await one(cli, args.model, faq, qs[0], 4)                     # warm the shared FAQ prefix once
    out = open(args.out, "a")
    for n in [int(x) for x in args.conc.split(",")]:
        for scen in args.scen.split(","):
            reps = 3 if n == 1 else 1
            for rep in range(reps):
                jobs = []
                for i in range(n):
                    q = rnd.choice(qs)
                    if scen == "ft":
                        jobs.append(one(cli, args.model, c1.SYSTEM_STUDENT, q, max(1, rnd.choice(len_ft))))
                    elif scen == "faq":
                        jobs.append(one(cli, args.model, f"[session {rnd.getrandbits(48):012x}]\n" + faq, q,
                                        max(1, rnd.choice(len_faq))))
                    else:
                        jobs.append(one(cli, args.model, faq, q, max(1, rnd.choice(len_faq))))
                t0 = time.perf_counter()
                res = await asyncio.gather(*jobs)
                wall = time.perf_counter() - t0
                rec = {"hw": args.hw, "scenario": scen, "n": n, "rep": rep, "wall": round(wall, 3),
                       "ttft": [round(r["ttft"], 3) for r in res], "total": [round(r["total"], 3) for r in res],
                       "prompt_tokens": [r["prompt_tokens"] for r in res], "completion_tokens": [r["completion_tokens"] for r in res],
                       "cached": [r["cached"] for r in res]}
                out.write(json.dumps(rec) + "\n"); out.flush()
                tt = sorted(rec["ttft"]); to = sorted(rec["total"])
                print(f"{scen:7s} n={n:3d} wall {wall:7.2f}s  ttft med {tt[len(tt)//2]:6.2f} max {tt[-1]:6.2f}  "
                      f"total med {to[len(to)//2]:6.2f} max {to[-1]:6.2f}  prompt {rec['prompt_tokens'][0]} cached {rec['cached'][0]}",
                      flush=True)


if __name__ == "__main__":
    asyncio.run(main())
