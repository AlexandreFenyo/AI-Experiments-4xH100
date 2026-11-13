"""Experiment 01, serving cost: step-level simulator of a vLLM-like server (continuous batching, chunked prefill,
optional prefix cache), used to transpose the H100 measurements to a DGX Spark.

Model: Gemma 4 31B (60 layers: 50 sliding-window layers, window 1 024, 16 KV heads x 256; 10 global layers, 4 KV heads
x 512, keys = values). Each engine step processes every running decode (1 token each) plus prefill chunks up to a token
budget; its duration is max(compute, memory) + a fixed overhead:
  compute = (linear FLOPs of the step's tokens + attention FLOPs) / F
  memory  = (weights + KV cache read by every decoding sequence) / B
Hardware parameters (F, B, overhead, weight bytes, KV pool) are fitted on single-request measurements (H100) or taken
from published DGX Spark figures, then the simulator predicts N simultaneous requests.

    python exp01_sim_serving.py --validate results/exp01_bench_h100.jsonl      # H100: fitted vs measured
    python exp01_sim_serving.py --spark                                          # DGX Spark predictions
"""

import argparse
import json
import math
import random
from pathlib import Path

L_SLIDE, L_FULL, WINDOW = 50, 10, 1024
KV_SLIDE = 2 * 16 * 256 * 2          # bytes per token per sliding layer (K and V, bf16)
KV_FULL = 1 * 4 * 512 * 2            # bytes per token per global layer (keys = values, bf16)
LIN = 2 * 29.86e9                    # linear FLOPs per token (non-embedding parameters)
HEAD = 2 * 1.41e9                    # output projection FLOPs per generated token
ATT_SLIDE = 4 * 32 * 256             # attention FLOPs per (token, attended position) in a sliding layer
ATT_FULL = 4 * 32 * 512              # same in a global layer

PROMPT_FT, PROMPT_FAQ, FAQ_SHARED = 46, 15341, 15305   # measured with the Gemma tokenizer (system + question, template)

HW = {
    # H100 x2 TP2 bf16: F, B, overhead fitted on the n=1 measurements (see --validate); weights 62.5 GB
    "h100": dict(F=None, B=None, tau=None, W=62.5e9, kv_pool=2 * 41.9 * 2 ** 30, budget=8192, max_seqs=128),
    # DGX Spark (GB10, 128 GB LPDDR5x 273 GB/s), Gemma 4 31B NVFP4 (about 19.6 GB of weights), vLLM
    #   B: 6.85 tokens/s single stream published for Gemma 4 31B NVFP4 -> 19.6 GB x 6.85 = 134 GB/s effective;
    #      optimistic 190 GB/s (Llama 3.1 70B FP8: 2.7 tokens/s x 70.6 GB, LMSYS)
    #   F: Llama 3.1 70B FP8 prefill 803 tokens/s (LMSYS) -> 113 TFLOPS on short prompts; our 15 k-token prompt is
    #      attention-heavier (on H100 the same workload reached 25 % of the bf16 peak): central 100, range 60-160
    #   fe, fe0, mkv, tseq: front end, KV-read efficiency and per-sequence step cost fitted on the H100 measurements
    #   vLLM defaults on GB10 (engine/arg_utils.py): at most 256 sequences, 2 048 tokens per step ("spark*");
    #   "spark_tuned" = same hardware with --max-num-seqs 1024 --max-num-batched-tokens 8192
    "spark": dict(F=100e12, B=134e9, tau=0.004, W=19.6e9, kv_pool=80e9, budget=2048, max_seqs=256, fe=3e-6, fe0=0.01, mkv=0.75, tseq=0.2e-3),
    "spark_opt": dict(F=160e12, B=190e9, tau=0.003, W=19.6e9, kv_pool=80e9, budget=2048, max_seqs=256, fe=3e-6, fe0=0.01, mkv=0.75, tseq=0.2e-3),
    "spark_pess": dict(F=60e12, B=120e9, tau=0.006, W=19.6e9, kv_pool=80e9, budget=2048, max_seqs=256, fe=3e-6, fe0=0.01, mkv=0.75, tseq=0.2e-3),
    "spark_tuned": dict(F=100e12, B=134e9, tau=0.004, W=19.6e9, kv_pool=80e9, budget=8192, max_seqs=1024, fe=3e-6, fe0=0.01, mkv=0.75, tseq=0.2e-3),
}


def kv_read(ctx):
    return L_SLIDE * min(ctx, WINDOW) * KV_SLIDE + L_FULL * ctx * KV_FULL


def kv_size(ctx):
    return kv_read(ctx)


def att_flops(start, n):
    """Attention FLOPs to process positions start .. start+n-1."""
    full = L_FULL * ATT_FULL * (n * start + n * (n + 1) / 2)
    slide = 0.0
    for chunk_start in range(start, start + n, 512):        # coarse integration of min(p, WINDOW)
        m = min(512, start + n - chunk_start)
        slide += m * min(chunk_start + m / 2, WINDOW)
    return full + L_SLIDE * ATT_SLIDE * slide


def simulate(hw, reqs, prefix_cache=False):
    """reqs: list of (prompt_tokens, output_tokens, shared_prefix_tokens), all sent at t=0. The front end (HTTP,
    tokenisation) handles them one after another (fe seconds per prompt token, plus fe0 per request) before the engine
    sees them. mkv scales the cost of reading the KV cache during decoding (attention kernel efficiency).
    Returns per-request (ttft, total)."""
    F, B, tau, W, pool, budget = hw["F"], hw["B"], hw["tau"], hw["W"], hw["kv_pool"], hw["budget"]
    fe, fe0, mkv = hw.get("fe", 0.0), hw.get("fe0", 0.0), hw.get("mkv", 1.0)
    tseq = hw.get("tseq", 0.0)                 # CPU cost per running sequence per step (scheduling, streaming)
    n = len(reqs)
    arrive, acc = [], 0.0
    for p, o, s in reqs:
        acc += fe0 + fe * p
        arrive.append(acc)
    done_prefill = [0] * n
    gen = [0] * n
    ttft, total = [None] * n, [None] * n
    cached_prefix = set()                      # shared prefix already in the cache (warm server: all of it)
    if prefix_cache:
        for i, (p, o, s) in enumerate(reqs):
            done_prefill[i] = min(s, p - 1)    # cache hit on the shared part (the last prompt token is always computed)
    t, waiting, running, pending = 0.0, [], [], list(range(n))
    used = 0.0
    shared_counted = False
    while pending or waiting or running:
        while pending and arrive[pending[0]] <= t:
            waiting.append(pending.pop(0))
        if not waiting and not running:
            t = arrive[pending[0]]
            continue
        # admission (memory): a request needs the KV of its prompt + output; a shared cached prefix is stored once
        while waiting:
            i = waiting[0]
            p, o, s = reqs[i]
            own = kv_size(p + o - s) if prefix_cache and shared_counted else kv_size(p + o)
            if len(running) >= hw.get("max_seqs", 10 ** 9) or (used + own > pool and running):
                break
            need = own
            if prefix_cache and not shared_counted:
                shared_counted = True
            used += need
            running.append(waiting.pop(0))
        tok_budget = budget
        lin_flops, att, mem = 0.0, 0.0, W
        decoding = [i for i in running if done_prefill[i] >= reqs[i][0]]
        for i in decoding:
            ctx = reqs[i][0] + gen[i]
            lin_flops += LIN + HEAD
            att += L_FULL * ATT_FULL * ctx + L_SLIDE * ATT_SLIDE * min(ctx, WINDOW)
            mem += mkv * kv_read(ctx)
            tok_budget -= 1
        prefilling = []
        for i in running:
            if done_prefill[i] < reqs[i][0] and tok_budget > 0:
                k = min(tok_budget, reqs[i][0] - done_prefill[i])
                prefilling.append((i, k))
                lin_flops += k * LIN
                att += att_flops(done_prefill[i], k)
                mem += kv_read(done_prefill[i])          # chunk reads the KV already written for this request
                tok_budget -= k
        dt = max((lin_flops + att) / F, mem / B) + tau + tseq * len(decoding)
        t += dt
        for i in decoding:
            gen[i] += 1
            if gen[i] >= reqs[i][1]:
                total[i] = t
        for i, k in prefilling:
            done_prefill[i] += k
            if done_prefill[i] >= reqs[i][0]:
                ttft[i] = t
                gen[i] = 1                               # the step that ends the prefill emits the first token
                if gen[i] >= reqs[i][1]:
                    total[i] = t
        fin = [i for i in running if total[i] is not None]
        for i in fin:
            p, o, s = reqs[i]
            used -= kv_size(p + o - s) if prefix_cache else kv_size(p + o)
            running.remove(i)
    return ttft, total


def simulate_closed(hw, users, make_req, prefix_cache=False, completions=3000, warmup=None):
    """Steady state: `users` people each ask a question, wait for the full answer, then immediately ask the next one
    (worst case, no reading time). Same engine model as simulate(). Returns (answers per hour, mean answer time,
    mean time to first token) measured after a warm-up of `warmup` completions (default: users)."""
    F, B, tau, W, pool, budget = hw["F"], hw["B"], hw["tau"], hw["W"], hw["kv_pool"], hw["budget"]
    fe, fe0, mkv, tseq = hw.get("fe", 0.0), hw.get("fe0", 0.0), hw.get("mkv", 1.0), hw.get("tseq", 0.0)
    max_seqs = hw.get("max_seqs", 10 ** 9)
    warmup = users if warmup is None else warmup
    reqs, send, arrive, done_prefill, gen, first, finish = [], [], [], [], [], [], []
    state = {"fe_free": 0.0}

    def new(t):
        p, o, s = make_req()
        a = max(t, state["fe_free"]) + fe0 + fe * p           # single front end: requests tokenised one at a time
        state["fe_free"] = a
        reqs.append((p, o, s)); send.append(t); arrive.append(a)
        done_prefill.append(min(s, p - 1) if prefix_cache else 0)
        gen.append(0); first.append(None); finish.append(None)
        return len(reqs) - 1

    pending = [new(0.0) for _ in range(users)]
    pending.sort(key=lambda i: arrive[i])
    t, waiting, running, used, shared, done = 0.0, [], [], 0.0, False, []
    while len(done) < completions + warmup:
        while pending and arrive[pending[0]] <= t:
            waiting.append(pending.pop(0))
        if not waiting and not running:
            t = arrive[pending[0]]
            continue
        while waiting:
            i = waiting[0]
            p, o, s = reqs[i]
            own = kv_size(p + o - s) if prefix_cache and shared else kv_size(p + o)
            if len(running) >= max_seqs or (used + own > pool and running):
                break
            shared = shared or prefix_cache
            used += own
            running.append(waiting.pop(0))
        tok_budget = budget
        lin_flops, att, mem = 0.0, 0.0, W
        decoding = [i for i in running if done_prefill[i] >= reqs[i][0]]
        for i in decoding:
            ctx = reqs[i][0] + gen[i]
            lin_flops += LIN + HEAD
            att += L_FULL * ATT_FULL * ctx + L_SLIDE * ATT_SLIDE * min(ctx, WINDOW)
            mem += mkv * kv_read(ctx)
            tok_budget -= 1
        prefilling = []
        for i in running:
            if done_prefill[i] < reqs[i][0] and tok_budget > 0:
                k = min(tok_budget, reqs[i][0] - done_prefill[i])
                prefilling.append((i, k))
                lin_flops += k * LIN
                att += att_flops(done_prefill[i], k)
                mem += kv_read(done_prefill[i])
                tok_budget -= k
        t += max((lin_flops + att) / F, mem / B) + tau + tseq * len(decoding)
        for i in decoding:
            gen[i] += 1
            if gen[i] >= reqs[i][1]:
                finish[i] = t
        for i, k in prefilling:
            done_prefill[i] += k
            if done_prefill[i] >= reqs[i][0]:
                first[i] = t
                gen[i] = 1
                if gen[i] >= reqs[i][1]:
                    finish[i] = t
        for i in [i for i in running if finish[i] is not None]:
            p, o, s = reqs[i]
            used -= kv_size(p + o - s) if prefix_cache else kv_size(p + o)
            running.remove(i)
            done.append(i)
            j = new(t)                                        # the same person asks the next question
            pending.append(j)
        pending.sort(key=lambda i: arrive[i])
    meas = done[warmup:]
    span = finish[meas[-1]] - finish[done[warmup - 1]] if warmup else finish[meas[-1]]
    return (len(meas) - 1) / span * 3600, sum(finish[i] - send[i] for i in meas) / len(meas), \
        sum(first[i] - send[i] for i in meas) / len(meas)


def scenario_reqs(scen, n, len_ft, len_faq, rnd):
    if scen == "ft":
        return [(PROMPT_FT, rnd.choice(len_ft), 0) for _ in range(n)]
    return [(PROMPT_FAQ, rnd.choice(len_faq), FAQ_SHARED if scen == "faq_pc" else 0) for _ in range(n)]


def stats(v):
    v = sorted(v)
    return {"mean": sum(v) / len(v), "med": v[len(v) // 2], "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1]}


def fit_h100(rows):
    """F from the n=1 FAQ prefill (TTFT), B and tau from the n=1 decode speeds."""
    r1 = [r for r in rows if r["n"] == 1]
    faq = [r for r in r1 if r["scenario"] == "faq"]
    ttft = sum(r["ttft"][0] for r in faq) / len(faq)
    flops = PROMPT_FAQ * LIN + att_flops(0, PROMPT_FAQ)
    F = flops / ttft
    # per-token decode time, short context (ft) and long context (faq_pc)
    dec = lambda rs: sum((r["total"][0] - r["ttft"][0]) / max(1, r["completion_tokens"][0] - 1) for r in rs) / len(rs)
    d_short, d_long = dec([r for r in r1 if r["scenario"] == "ft"]), dec([r for r in r1 if r["scenario"] == "faq_pc"])
    W = HW["h100"]["W"]
    # d = (W + kv)/B + tau for both contexts -> solve B, tau
    kv_s, kv_l = kv_read(PROMPT_FT + 70), kv_read(PROMPT_FAQ + 70)
    B = (kv_l - kv_s) / max(1e-6, d_long - d_short)
    tau = d_short - (W + kv_s) / B
    if B <= 0 or tau < 0 or B > 6.7e12:                # long-context slow-down too small to separate: assume B = 75 % of peak
        B = 0.75 * 6.7e12
        tau = max(0.0, d_short - (W + kv_s) / B)
    return dict(HW["h100"], F=F, B=B, tau=tau), {"ttft_faq_n1": ttft, "decode_short_s": d_short, "decode_long_s": d_long}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate")
    p.add_argument("--spark", action="store_true")
    p.add_argument("--closed", action="store_true", help="steady-state capacity: users who ask again as soon as answered")
    p.add_argument("--out", default="results/exp01_sim.json")
    args = p.parse_args()
    e1 = Path.home() / "exp" / "e1" / "results"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-4-31B-it")
    lens = lambda f: [len(tok(json.loads(l)["model_answer"], add_special_tokens=False).input_ids) for l in open(e1 / f)]
    len_ft, len_faq = lens("answers_gemma31_a3_q5km_heldout.jsonl"), lens("answers_gemma31_teacher_heldout.jsonl")
    out = {}
    if args.validate:
        rows = [json.loads(l) for l in open(args.validate)]
        hw, meas = fit_h100(rows)
        # front end and KV-read efficiency: grid search on every first-repetition measurement (log error on the
        # median TTFT and the median answer time)
        base = [r for r in rows if not r["rep"] and r["n"] <= 128]    # n = 256, 512: held out, pure validation
        best = None
        for mkv in (0.75, 1.0, 1.25, 1.5, 2.0):
            for tseq in (0.0, 0.05e-3, 0.1e-3, 0.15e-3, 0.2e-3, 0.3e-3):
                h = dict(hw, fe=3e-6, fe0=0.01, mkv=mkv, tseq=tseq)
                err = 0.0
                for r in base:
                    reqs = list(zip(r["prompt_tokens"], r["completion_tokens"],
                                    [FAQ_SHARED if r["scenario"] == "faq_pc" else 0] * r["n"]))
                    tt, to = simulate(h, reqs, prefix_cache=r["scenario"] == "faq_pc")
                    err += math.log(max(1e-3, stats(tt)["med"]) / max(1e-3, stats(r["ttft"])["med"])) ** 2
                    err += math.log(stats(to)["med"] / stats(r["total"])["med"]) ** 2
                if best is None or err < best[0]:
                    best = (err, 3e-6, 0.01, mkv, tseq)
        hw.update(fe=best[1], fe0=best[2], mkv=best[3], tseq=best[4])
        meas.update(fit_error=best[0])
        out["h100_fit"] = {"F_TFLOPS": hw["F"] / 1e12, "B_GBs": hw["B"] / 1e9, "tau_ms": hw["tau"] * 1e3,
                           "fe_us_per_token": hw["fe"] * 1e6, "fe0_ms": hw["fe0"] * 1e3, "mkv": hw["mkv"],
                           "tseq_ms": hw["tseq"] * 1e3, **meas}
        print("H100 fit:", out["h100_fit"])
        val = []
        for r in rows:
            if r["rep"]:
                continue
            reqs = list(zip(r["prompt_tokens"], r["completion_tokens"],
                            [FAQ_SHARED if r["scenario"] == "faq_pc" else 0] * r["n"]))
            tt, to = simulate(hw, reqs, prefix_cache=r["scenario"] == "faq_pc")
            v = {"scenario": r["scenario"], "n": r["n"], "meas_ttft_med": stats(r["ttft"])["med"], "sim_ttft_med": stats(tt)["med"],
                 "meas_total_med": stats(r["total"])["med"], "sim_total_med": stats(to)["med"],
                 "meas_wall": r["wall"], "sim_wall": max(to)}
            val.append(v)
            print(f"{v['scenario']:7s} n={v['n']:3d} ttft med meas {v['meas_ttft_med']:6.2f} sim {v['sim_ttft_med']:6.2f} | "
                  f"total med meas {v['meas_total_med']:6.2f} sim {v['sim_total_med']:6.2f} | wall meas {v['meas_wall']:6.2f} sim {v['sim_wall']:6.2f}")
        out["h100_validation"] = val
    if args.spark:
        res = {}
        for name in ("spark", "spark_opt", "spark_pess", "spark_tuned"):
            hw = HW[name]
            res[name] = []
            for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                for scen in ("ft", "faq", "faq_pc"):
                    acc = {"ttft": [], "total": [], "wall": []}
                    for seed in range(max(5, math.ceil(400 / n))):
                        rnd = random.Random(seed)
                        reqs = scenario_reqs(scen, n, len_ft, len_faq, rnd)
                        tt, to = simulate(hw, reqs, prefix_cache=scen == "faq_pc")
                        acc["ttft"] += tt; acc["total"] += to; acc["wall"].append(max(to))
                    rec = {"n": n, "scenario": scen, "ttft": stats(acc["ttft"]), "total": stats(acc["total"]),
                           "wall": sum(acc["wall"]) / len(acc["wall"]),
                           "prompt_tokens": PROMPT_FT if scen == "ft" else PROMPT_FAQ,
                           "computed_prompt_tokens": PROMPT_FT if scen == "ft" else (PROMPT_FAQ - FAQ_SHARED if scen == "faq_pc" else PROMPT_FAQ)}
                    res[name].append(rec)
                    if name in ("spark", "spark_tuned"):
                        print(f"{name:11s} {scen:7s} n={n:3d} ttft mean {rec['ttft']['mean']:7.2f} p90 {rec['ttft']['p90']:7.2f} | "
                              f"answer mean {rec['total']['mean']:7.2f} p90 {rec['total']['p90']:7.2f} | all done {rec['wall']:7.2f}")
        out["spark"] = res
        out["spark_params"] = {k: {kk: vv for kk, vv in HW[k].items()} for k in ("spark", "spark_opt", "spark_pess", "spark_tuned")}
    if args.closed:
        res = {}
        for name in ("spark", "spark_tuned", "spark_opt", "spark_pess"):
            res[name] = []
            for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                for scen in ("ft", "faq", "faq_pc"):
                    rnd = random.Random(1)
                    mk = (lambda: (PROMPT_FT, rnd.choice(len_ft), 0)) if scen == "ft" else \
                        (lambda: (PROMPT_FAQ, rnd.choice(len_faq), FAQ_SHARED if scen == "faq_pc" else 0))
                    thr, lat, ttf = simulate_closed(HW[name], n, mk, prefix_cache=scen == "faq_pc",
                                                    completions=max(600, 2 * n))
                    res[name].append({"n": n, "scenario": scen, "answers_per_hour": thr, "mean_answer_s": lat,
                                      "mean_ttft_s": ttf})
                    print(f"{name:11s} {scen:7s} users={n:5d}  {thr:9.0f} answers/h  answer {lat:8.1f} s  first word {ttf:8.2f} s",
                          flush=True)
        out["closed"] = res
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
