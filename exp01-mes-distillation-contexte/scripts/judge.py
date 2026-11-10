"""LLM judge for answers_<tag>_<set>.jsonl files.

    ENDPOINTS=qwen@8000 python judge.py --tag qwen27_teacher [--answers path --set heldout]
Metrics on heldout: strict / lenient (same prompts as the previous experiment, for comparability),
completeness (share of the reference's atomic claims found in the answer), contradiction and invention rates.
On ood: hallucination rate. On forgetting: accuracy.
"""

import argparse
import asyncio
import json
import statistics

from common import DATA, ROOT, chat_many, load_jsonl, parse_json, write_jsonl
from judge_prompts_prev import JUDGE_FORGET, JUDGE_OOD, JUDGE_QA, JUDGE_QA_LENIENT, JUDGE_SYSTEM

JUDGE_CLAIMS = """Question posée : {question}

Réponse CANDIDATE (à évaluer) :
{candidate}

Liste numérotée des AFFIRMATIONS de la réponse de référence (vérité) :
{claims}

Pour CHAQUE affirmation, indique si la réponse candidate :
- "P" : la contient (même reformulée, le fait est bien présent et exact) ;
- "A" : ne la contient pas (fait absent) ;
- "C" : la contredit (donne une information incompatible : autre chiffre, autre menu, autre organisme, autre démarche...).
Indique aussi "invention" = true si la candidate affirme au moins un fait PRÉCIS sur Mon espace santé (chiffre, nom de menu
ou de bouton, organisme, démarche) qui ne figure dans AUCUNE des affirmations de référence.
Réponds par une seule ligne JSON : {{"verdicts": ["P"|"A"|"C", ...], "invention": true|false}}
Le tableau "verdicts" doit avoir exactement {n} éléments, dans l'ordre."""


def sys_user(user):
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


async def judge_heldout(rows, claims, conc=96):
    jobs, kinds = [], []
    for r in rows:
        cand = r["model_answer"] or "(réponse vide)"
        jobs.append(sys_user(JUDGE_QA.format(question=r["question"], reference=r["reference_answer"], candidate=cand)))
        jobs.append(sys_user(JUDGE_QA_LENIENT.format(question=r["question"], reference=r["reference_answer"], candidate=cand)))
        cl = claims[r["fact_id"]]
        jobs.append(sys_user(JUDGE_CLAIMS.format(question=r["question"], candidate=cand, n=len(cl),
                                                 claims="\n".join(f"{i + 1}. {c}" for i, c in enumerate(cl)))))
    res = await chat_many(jobs, max_tokens=1200, concurrency=conc, desc="judge heldout")
    out = []
    for i, r in enumerate(rows):
        s, l, c = (parse_json(res[3 * i + k]["text"]) or {} for k in range(3))
        cl = claims[r["fact_id"]]
        v = c.get("verdicts") if isinstance(c, dict) else None
        ok = isinstance(v, list) and len(v) == len(cl)
        out.append(dict(
            r, strict=bool(s.get("correct")) if isinstance(s, dict) else None,
            strict_reason=s.get("raison", "") if isinstance(s, dict) else "",
            lenient=bool(l.get("correct")) if isinstance(l, dict) else None,
            completeness=(sum(x == "P" for x in v) / len(cl)) if ok else None,
            contradictions=(sum(x == "C" for x in v)) if ok else None,
            invention=bool(c.get("invention")) if ok else None,
            verdicts=v if ok else None,
        ))
    return out


def pct(xs):
    xs = [x for x in xs if x is not None]
    return round(100 * sum(xs) / len(xs), 1) if xs else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--sets", default="heldout,ood,forgetting")
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--out", help="output tag (default: --tag)")
    p.add_argument("--answers", help="explicit answers file (with --sets heldout), e.g. previous experiment outputs")
    args = p.parse_args()
    summary = {"tag": args.tag}

    for name in args.sets.split(","):
        path = args.answers if (args.answers and name == "heldout") else ROOT / "results" / f"answers_{args.tag}_{name}.jsonl"
        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            continue
        if name == "heldout":
            if "fact_id" not in rows[0]:  # previous-experiment files lack fact_id
                fid = {r["qid"]: r["fact_id"] for r in load_jsonl(DATA / "heldout_test.jsonl")}
                rows = [dict(r, fact_id=fid[r["qid"]]) for r in rows]
            claims = {c["fact_id"]: c["claims"] for c in load_jsonl(DATA / "claims.jsonl")}
            out = await judge_heldout(rows, claims, args.concurrency)
            comp = [o["completeness"] for o in out if o["completeness"] is not None]
            summary.update(
                n=len(out), strict=pct([o["strict"] for o in out]), lenient=pct([o["lenient"] for o in out]),
                completeness=round(100 * statistics.mean(comp), 1) if comp else None,
                with_contradiction=pct([(o["contradictions"] > 0) if o["contradictions"] is not None else None for o in out]),
                with_invention=pct([o["invention"] for o in out]),
                answer_chars_median=statistics.median(len(o["model_answer"]) for o in out),
                reference_chars_median=statistics.median(len(o["reference_answer"]) for o in out),
                judge_failures=sum(o["completeness"] is None or o["strict"] is None for o in out),
            )
        elif name == "ood":
            jobs = [sys_user(JUDGE_OOD.format(question=r["question"], candidate=r["model_answer"] or "(vide)")) for r in rows]
            res = await chat_many(jobs, max_tokens=300, desc="judge ood")
            out = [dict(r, hallucination=(parse_json(x["text"]) or {}).get("hallucination")) for r, x in zip(rows, res)]
            summary["ood_hallucination"] = pct([o["hallucination"] for o in out])
        else:
            jobs = [sys_user(JUDGE_FORGET.format(question=r["question"], reference=r["reference_answer"],
                                                 candidate=r["model_answer"] or "(vide)")) for r in rows]
            res = await chat_many(jobs, max_tokens=300, desc="judge forgetting")
            out = [dict(r, correct=(parse_json(x["text"]) or {}).get("correct")) for r, x in zip(rows, res)]
            summary["general_knowledge"] = pct([o["correct"] for o in out])
        write_jsonl(ROOT / "results" / f"scored_{args.out or args.tag}_{name}.jsonl", out)

    with open(ROOT / "results" / f"summary_{args.out or args.tag}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
