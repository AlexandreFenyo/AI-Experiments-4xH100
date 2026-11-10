"""Generate a large, diverse set of user questions about the FAQ (training inputs for context distillation).

    ENDPOINTS=qwen@8000,qwen@8001 python gen_questions.py --scale 1.0
Output: data/gen/questions_raw.jsonl, rows {id, type, fact_ids, turns:[...]} (turns = successive user messages).
"""

import argparse
import asyncio
import hashlib
import itertools
import random

from common import DATA, chat_many, load_facts, load_jsonl, parse_json, write_jsonl

STYLES = [
    "un usager pressé qui écrit en style télégraphique, sans formule de politesse",
    "une personne âgée peu à l'aise avec le numérique, qui décrit son problème avec ses propres mots, sans vocabulaire technique",
    "un jeune adulte qui écrit en langage familier, avec des abréviations et quelques fautes de frappe ou d'orthographe",
    "un usager très poli et formel qui rédige des phrases complètes et soignées",
    "un parent qui pose la question pour son enfant ou pour un proche dont il s'occupe",
    "un usager inquiet ou agacé qui raconte d'abord sa situation concrète en deux ou trois phrases avant de poser sa question",
    "un professionnel (travailleur social, secrétaire médicale, aidant) qui se renseigne pour accompagner quelqu'un",
    "un usager qui tape seulement quelques mots-clés, comme dans un moteur de recherche",
    "un usager qui emploie des synonymes approximatifs ou des termes inexacts pour désigner les fonctionnalités",
    "un usager qui pose une question fermée (oui/non) ou demande une confirmation",
    "un usager qui demande une explication pas à pas, étape par étape",
    "un usager non francophone natif, avec une syntaxe simple et parfois maladroite",
]

HEAD = """Tu aides à constituer un jeu de questions d'usagers pour entraîner un assistant sur « Mon espace santé ».
Voici une entrée de la FAQ officielle.

Question officielle : {q}
Réponse officielle :
{a}
"""

P_PARA = HEAD + """
Rédige {n} questions DIFFÉRENTES qu'un usager pourrait poser et dont la réponse est cette entrée (ou une partie substantielle).
Profil de l'usager : {style}.
Contraintes : varie fortement la formulation, l'angle et la longueur ; n'utilise pas la formulation officielle ; chaque question
doit être compréhensible seule ; ne donne aucune réponse.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_CLAIM = HEAD + """
Concentre-toi sur ce fait précis contenu dans la réponse : « {claim} »
Rédige {n} questions DIFFÉRENTES qu'un usager pourrait poser et dont la bonne réponse nécessite précisément ce fait
(sans que la question contienne déjà la réponse). Varie les formulations, les registres et les angles (comment, où, combien,
est-ce que, que se passe-t-il si...). Chaque question doit être compréhensible seule.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_FALSE = HEAD + """
Rédige {n} questions d'usagers qui contiennent une PRÉSUPPOSITION FAUSSE ou approximative par rapport à cette entrée
(mauvais chiffre, mauvais nom de menu ou de bouton, mauvais organisme, mauvaise démarche, délai erroné, confusion avec un
autre service...), de sorte qu'une bonne réponse doive corriger l'erreur en donnant l'information exacte.
Les erreurs doivent être plausibles et variées. Chaque question doit être compréhensible seule.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_SCEN = HEAD + """
Rédige {n} messages d'usagers décrivant chacun une SITUATION CONCRÈTE et personnelle différente (contexte de vie, ce qui a
été tenté, ce qui bloque) pour laquelle cette entrée apporte la réponse. Les messages font de 2 à 5 phrases et se terminent
par une demande. Varie les profils, les situations et les registres.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_BROAD = HEAD + """
Rédige {n} demandes OUVERTES différentes par lesquelles un usager réclame une explication complète et détaillée du sujet de
cette entrée (par exemple : tout savoir sur..., expliquer en détail..., lister toutes les..., quelles sont toutes les étapes...,
faire le point sur...). Varie les formulations. Chaque demande doit être compréhensible seule.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_MULTI = """Tu aides à constituer un jeu de questions d'usagers pour entraîner un assistant sur « Mon espace santé ».
Voici {k} entrées de la FAQ officielle.

{entries}

Rédige {n} questions ou messages DIFFÉRENTS d'usagers dont la bonne réponse nécessite de combiner des informations de
TOUTES ces entrées (question double, comparaison, enchaînement de démarches, situation qui touche plusieurs sujets).
Varie les formulations et les registres. Chaque message doit être compréhensible seul.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_DIALOG = HEAD + """
Imagine {n} CONVERSATIONS différentes entre un usager et l'assistant à propos de cette entrée. Pour chacune, donne seulement
les messages successifs de l'USAGER ({t} messages) : un premier message, puis des relances qui dépendent du contexte
(demande de précision sur un détail, « et si... », « où ça exactement ? », reformulation parce qu'il n'a pas compris...).
Les relances doivent porter sur des éléments auxquels la FAQ de Mon espace santé peut répondre. Varie les profils.
Réponds UNIQUEMENT par un tableau JSON de {n} tableaux de {t} chaînes."""

P_NEAR = """Voici la liste des questions couvertes par la FAQ officielle de « Mon espace santé » (catégorie : {cat}) :
{titles}

Rédige {n} questions plausibles d'usagers qui concernent Mon espace santé ou un sujet très voisin (Assurance Maladie, ameli,
carte Vitale, dossier médical, application mobile, professionnels de santé, remboursements, démarches de santé en ligne...)
mais dont la réponse NE figure PAS dans cette FAQ : détails techniques non documentés, cas particuliers, fonctionnalités
inexistantes ou supposées, chiffres ou dates non mentionnés, autres services. Thème à privilégier : {theme}.
Varie fortement les sujets, les formulations et les registres.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

P_OOD = """Rédige {n} questions ou demandes variées qu'un utilisateur pourrait adresser par erreur à l'assistant du service
« Mon espace santé » alors qu'elles sont HORS DE SON PÉRIMÈTRE. Thème : {theme}.
Certaines mentionnent explicitement Mon espace santé en lui attribuant une fonction qu'il n'a pas, d'autres non.
Varie les formulations, les registres et la longueur.
Réponds UNIQUEMENT par un tableau JSON de {n} chaînes."""

NEAR_THEMES = [
    "détails techniques (formats de fichiers, tailles maximales, navigateurs, versions, hébergement, chiffrement)",
    "remboursements, tiers payant, mutuelles et paiements", "prise de rendez-vous médicaux et téléconsultation",
    "carte Vitale, appli carte Vitale, compte ameli et France Connect", "cas particuliers (étranger, tutelle, décès, déménagement, changement de régime)",
    "statistiques, chiffres d'usage, dates de lancement et historique du service", "fonctionnalités supposées (IA, alertes, objets connectés, partage sur réseaux sociaux)",
    "démarches auprès des professionnels et établissements de santé (logiciels, obligations, facturation)",
    "questions médicales personnelles (symptômes, traitements, diagnostics, interprétation de résultats)",
    "droits, réglementation, recours juridiques et litiges", "comparaison avec d'autres pays ou d'autres services en ligne",
    "pannes précises, messages d'erreur exacts et bogues supposés",
]
OOD_THEMES = [
    "autres démarches administratives (impôts, carte grise, passeport, CAF, retraite, Pôle emploi/France Travail)",
    "transports, voyages et réservations", "banque, argent, assurances non santé et achats en ligne",
    "culture générale, histoire, géographie et sciences", "programmation informatique, mathématiques et aide aux devoirs",
    "cuisine, loisirs, sport et divertissement", "actualité, politique et opinions", "rédaction de textes, traductions et créations littéraires",
    "conseils médicaux personnels, diagnostics et prescriptions de médicaments", "tentatives de détourner l'assistant (changer de rôle, ignorer ses consignes, révéler son prompt)",
    "météo, horaires, itinéraires et informations locales", "emploi, droit du travail, logement et vie quotidienne",
]


def qid(kind, turns):
    return kind + "_" + hashlib.md5("\n".join(turns).encode()).hexdigest()[:12]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=float, default=1.0, help="global multiplier on the number of generation calls")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rnd = random.Random(args.seed)
    facts = load_facts()
    claims = {c["fact_id"]: c["claims"] for c in load_jsonl(DATA / "claims.jsonl")}
    by_cat = {}
    for f in facts:
        by_cat.setdefault(f["category"], []).append(f)

    def rep(x):  # number of calls scaled
        return max(1, round(x * args.scale))

    jobs, meta = [], []  # meta: (type, fact_ids, n_turns)

    def add(kind, fids, prompt, turns=1, k=1):
        for _ in range(k):
            jobs.append([{"role": "user", "content": prompt}])
            meta.append((kind, fids, turns))

    for f in facts:
        h = dict(q=f["question"], a=f["answer"])
        fid = [f["fact_id"]]
        n_claims = len(claims[f["fact_id"]])
        for style in STYLES:                                   # 12 styles x 8 = ~96 paraphrases / fact
            add("para", fid, P_PARA.format(n=8, style=style, **h), k=rep(1))
        for c in claims[f["fact_id"]]:                         # 6 questions / claim
            add("claim", fid, P_CLAIM.format(n=6, claim=c, **h), k=rep(1))
        add("false", fid, P_FALSE.format(n=8, **h), k=rep(1 + n_claims // 8))
        add("scen", fid, P_SCEN.format(n=6, **h), k=rep(2 + n_claims // 10))
        add("broad", fid, P_BROAD.format(n=6, **h), k=rep(1))
        add("dialog2", fid, P_DIALOG.format(n=4, t=2, **h), turns=2, k=rep(2))
        add("dialog3", fid, P_DIALOG.format(n=3, t=3, **h), turns=3, k=rep(1))

    pairs = []
    for cat, fs in by_cat.items():                             # same-category pairs
        pairs += list(itertools.combinations(fs, 2))
    rnd.shuffle(pairs)
    pairs = pairs[: rep(350)]
    pairs += [tuple(rnd.sample(facts, 2)) for _ in range(rep(150))]   # cross-category pairs
    pairs += [tuple(rnd.sample(facts, 3)) for _ in range(rep(60))]    # triples
    for group in pairs:
        entries = "\n\n".join(f"Entrée {i + 1}\nQuestion : {g['question']}\nRéponse :\n{g['answer']}" for i, g in enumerate(group))
        add("multi", [g["fact_id"] for g in group], P_MULTI.format(k=len(group), n=4, entries=entries))

    for cat, fs in by_cat.items():
        titles = "\n".join("- " + f["question"] for f in fs)
        for theme in NEAR_THEMES:
            add("near", [], P_NEAR.format(cat=cat, titles=titles, n=10, theme=theme), k=rep(1))
    for theme in OOD_THEMES:
        add("ood", [], P_OOD.format(n=12, theme=theme), k=rep(10))

    print(f"{len(jobs)} generation calls")
    res = await chat_many(jobs, temperature=1.0, max_tokens=3000, concurrency=128, desc="gen")
    rows, seen, bad = [], set(), 0
    for (kind, fids, turns), r in zip(meta, res):
        items = parse_json(r["text"])
        if not isinstance(items, list):
            bad += 1
            continue
        for it in items:
            t = [str(x).strip() for x in it] if turns > 1 and isinstance(it, list) else [str(it).strip()]
            if turns > 1 and len(t) != turns:
                continue
            if any(len(x) < 8 or len(x) > 1500 for x in t):
                continue
            key = "\n".join(t).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"id": qid(kind, t), "type": kind, "fact_ids": fids, "turns": t})
    rnd.shuffle(rows)
    write_jsonl(DATA / "gen" / "questions_raw.jsonl", rows)
    from collections import Counter
    print(f"{len(rows)} questions ({bad} calls unparsable):", dict(Counter(r["type"] for r in rows)))


if __name__ == "__main__":
    asyncio.run(main())
