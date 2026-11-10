"""Faithfulness judge: is every factual statement of an answer supported by the WHOLE FAQ?

The strict judge inherited from the previous experiment compares an answer with ONE reference entry, so it counts
correct information coming from other FAQ entries as "invented". This judge reads the full FAQ instead.

    ENDPOINTS=qwen@8000,qwen@8001 python judge_faith.py --tag gemma31_teacher [--answers file]
"""

import argparse
import asyncio
import json
import statistics

from common import ROOT, chat_many, faq_context, load_jsonl, parse_json, write_jsonl

SYSTEM = """Tu es un vérificateur de faits rigoureux. Voici la base de connaissances officielle COMPLÈTE de « Mon espace santé ».
Elle est la seule source de vérité.

=== BASE OFFICIELLE ===
{faq}
=== FIN DE LA BASE ===

On te donnera une question d'usager et une réponse d'assistant. Relève les affirmations FACTUELLES de la réponse qui portent
sur Mon espace santé ou sur les démarches associées et classe celles qui posent problème :
- "contredites" : affirmations incompatibles avec la base (autre chiffre, autre nom de menu ou de bouton, autre organisme,
  autre démarche, autre condition...).
- "non_etayees" : affirmations précises (chiffre, date, nom de menu, étape, organisme, lien, condition) qui ne figurent nulle
  part dans la base et ne s'en déduisent pas directement.
Ne relève PAS : les reformulations fidèles, les informations présentes ailleurs dans la base (même dans une autre entrée),
les formules de politesse, les conseils génériques de bon sens sans contenu factuel précis, ni le fait de dire qu'une
information n'est pas connue.
Réponds par une seule ligne JSON : {{"contredites": ["..."], "non_etayees": ["..."]}} (tableaux vides si tout est fidèle)."""

USER = """Question de l'usager : {q}

Réponse de l'assistant à vérifier :
{a}"""


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--answers")
    p.add_argument("--concurrency", type=int, default=40)
    args = p.parse_args()
    rows = load_jsonl(args.answers or ROOT / "results" / f"answers_{args.tag}_heldout.jsonl")
    system = {"role": "system", "content": SYSTEM.format(faq=faq_context())}
    jobs = [[system, {"role": "user", "content": USER.format(q=r["question"], a=r["model_answer"] or "(vide)")}] for r in rows]
    res = await chat_many(jobs, max_tokens=1500, concurrency=args.concurrency, desc=f"faith {args.tag}")
    out, nc, nu, fail = [], [], [], 0
    for r, x in zip(rows, res):
        j = parse_json(x["text"])
        if not isinstance(j, dict):
            fail += 1
            out.append(dict(r, faith=None))
            continue
        c, u = list(j.get("contredites") or []), list(j.get("non_etayees") or [])
        nc.append(len(c))
        nu.append(len(u))
        out.append(dict(r, faith={"contredites": c, "non_etayees": u}))
    write_jsonl(ROOT / "results" / f"faith_{args.tag}_heldout.jsonl", out)
    n = len(nc)
    summary = {
        "tag": args.tag, "n": n, "judge_failures": fail,
        "fully_faithful": round(100 * sum(a == 0 and b == 0 for a, b in zip(nc, nu)) / n, 1),
        "answers_with_contradiction": round(100 * sum(a > 0 for a in nc) / n, 1),
        "answers_with_unsupported": round(100 * sum(b > 0 for b in nu) / n, 1),
        "contradictions_per_answer": round(statistics.mean(nc), 2), "unsupported_per_answer": round(statistics.mean(nu), 2),
    }
    json.dump(summary, open(ROOT / "results" / f"summary_faith_{args.tag}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
