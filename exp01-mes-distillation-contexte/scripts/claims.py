"""Split each FAQ answer into atomic claims (used for the completeness metric and for question generation).

    ENDPOINTS=mistral@8010 python claims.py
"""

import asyncio

from common import DATA, chat_many, load_facts, parse_json, write_jsonl

PROMPT = """Voici une entrée de la FAQ officielle de « Mon espace santé ».

Question : {q}

Réponse officielle :
{a}

Décompose la réponse officielle en AFFIRMATIONS ATOMIQUES : chaque affirmation exprime UN seul fait vérifiable
(une étape, un élément de liste, un nom de menu ou de bouton, un chiffre, une date, un délai, une condition, une exception,
un lien, un numéro...). Règles :
- Couvre TOUT le contenu informatif de la réponse, sans rien omettre et sans rien ajouter.
- Chaque affirmation doit être compréhensible seule (répète le sujet, pas de pronom ambigu).
- Un élément de liste = une affirmation.
- Ignore les formules de politesse sans contenu.
Réponds UNIQUEMENT par un tableau JSON de chaînes de caractères."""


async def main():
    facts = load_facts()
    jobs = [[{"role": "user", "content": PROMPT.format(q=f["question"], a=f["answer"])}] for f in facts]
    res = await chat_many(jobs, max_tokens=6000, desc="claims")
    out, bad = [], []
    for f, r in zip(facts, res):
        claims = parse_json(r["text"])
        if not isinstance(claims, list) or not claims:
            bad.append(f["fact_id"])
            claims = []
        out.append({"fact_id": f["fact_id"], "claims": [str(c).strip() for c in claims if str(c).strip()]})
    write_jsonl(DATA / "claims.jsonl", out)
    n = sum(len(o["claims"]) for o in out)
    print(f"{n} claims for {len(out)} facts (mean {n / len(out):.1f}); failed: {bad}")


if __name__ == "__main__":
    asyncio.run(main())
