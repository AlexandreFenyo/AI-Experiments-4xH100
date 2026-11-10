"""General-knowledge replay: ordinary questions answered by the BASE model without any system prompt.

Mixed into the training set so that, without the domain system prompt, the student keeps answering general
questions normally instead of refusing them as out of scope.

    ENDPOINTS=gemma@8002,gemma@8003 python gen_general.py --tag gemma31 --n 2400
Output: data/train/general_<tag>.jsonl
"""

import argparse
import asyncio
import random

from common import DATA, chat_many, parse_json, write_jsonl

THEMES = [
    "géographie", "histoire de France", "histoire mondiale", "sciences physiques", "biologie et corps humain", "astronomie",
    "mathématiques et calcul", "littérature", "musique", "cinéma", "sport", "cuisine et recettes", "langue française et grammaire",
    "informatique et programmation", "économie et finances personnelles", "droit et vie quotidienne", "écologie et environnement",
    "technologie et internet", "philosophie", "arts et peinture", "voyages", "animaux et nature", "jardinage et bricolage",
    "traduction et langues étrangères", "rédaction de courriers et de courriels", "conseils d'organisation et de travail",
    "jeux et devinettes", "actualité technologique générale", "santé générale et prévention (hors Mon espace santé)", "éducation et orientation",
]
KINDS = [
    "questions factuelles courtes", "demandes d'explication d'un concept", "demandes pratiques (comment faire...)",
    "petits exercices ou problèmes à résoudre", "demandes de rédaction courte (un message, un résumé, une liste)",
]
P = """Rédige {n} {kind} variées, en français, qu'un utilisateur pourrait poser à un assistant généraliste sur le thème :
{theme}. Elles ne doivent PAS concerner « Mon espace santé ». Varie la longueur, le registre et la difficulté.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=2400)
    args = p.parse_args()
    rnd = random.Random(1)
    per = 8
    jobs = []
    while len(jobs) * per < args.n * 1.15:
        jobs.append([{"role": "user", "content": P.format(n=per, kind=rnd.choice(KINDS), theme=rnd.choice(THEMES))}])
    res = await chat_many(jobs, temperature=1.0, max_tokens=1500, desc="general questions")
    qs, seen = [], set()
    for r in res:
        for q in parse_json(r["text"]) or []:
            q = str(q).strip()
            if 10 <= len(q) <= 600 and q.lower() not in seen and "espace santé" not in q.lower():
                seen.add(q.lower())
                qs.append(q)
    qs = qs[: args.n]
    ans = await chat_many([[{"role": "user", "content": q}] for q in qs], temperature=0.3, max_tokens=900, desc="general answers")
    rows = [{"id": f"general_{i}", "type": "general",
             "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a["text"]}]}
            for i, (q, a) in enumerate(zip(qs, ans)) if a["text"] and a["finish"] == "stop"]
    write_jsonl(DATA / "train" / f"general_{args.tag}.jsonl", rows)
    print(len(rows), "general examples")


if __name__ == "__main__":
    asyncio.run(main())
