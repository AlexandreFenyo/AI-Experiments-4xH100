"""Shared helpers: FAQ context, prompts, async OpenAI-compatible client (vLLM)."""

import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Short system prompt used for the STUDENT (training + inference): the knowledge must come from the weights.
OVH_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

SYSTEM_STUDENT = "Tu es l'assistant de Mon espace santé."

TEACHER_RULES = """Tu es l'assistant de « Mon espace santé », le service public numérique français de santé.
Tu disposes ci-dessous de la base de connaissances officielle COMPLÈTE du service. C'est ta SEULE source de vérité.

Règles de réponse :
- Réponds en français, directement, comme un expert qui connaît le service par cœur. Ne mentionne jamais « la FAQ », « le document », « la base de connaissances » ni « le contexte ».
- Sois COMPLET et DÉTAILLÉ : restitue tous les éléments pertinents de la base pour la question posée (étapes dans l'ordre, listes entières, noms exacts des menus, boutons et rubriques entre guillemets, chiffres, dates, délais, conditions, exceptions, liens et numéros utiles). N'abrège pas et ne résume pas une liste.
{EXHAUSTIVE}- Sois FIDÈLE : n'ajoute aucun fait, chiffre, nom de menu ou démarche qui ne figure pas dans la base. Ne complète pas avec tes connaissances générales.
- Si la question porte sur plusieurs sujets de la base, traite-les tous.
- Si la base ne contient qu'une partie de la réponse, donne cette partie et précise ce que tu ne sais pas.
- Si la question est hors du périmètre de Mon espace santé ou si la base ne contient pas l'information, dis-le clairement et poliment, sans rien inventer ; indique si possible vers qui se tourner uniquement si la base le précise.
- Utilise une mise en forme simple : phrases, et listes à tirets quand il y a des étapes ou des énumérations.

=== BASE DE CONNAISSANCES OFFICIELLE « MON ESPACE SANTÉ » ===
"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_facts():
    return load_jsonl(DATA / "facts_real_88.jsonl")


def faq_context(facts=None):
    """Whole FAQ as one text block, grouped by category (stable order => vLLM prefix cache hits)."""
    facts = facts or load_facts()
    by_cat = defaultdict(list)
    for f in facts:
        by_cat[f["category"]].append(f)
    parts = []
    for cat in sorted(by_cat):
        parts.append(f"\n## {cat}\n")
        for f in sorted(by_cat[cat], key=lambda x: x["fact_id"]):
            parts.append(f"### [{f['fact_id']}] {f['question'].strip()}\n{f['answer'].strip()}\n")
    return "\n".join(parts)


EXHAUSTIVE = """- Sois EXHAUSTIF : ne te limite pas au strict point demandé. Restitue l'intégralité du contenu de l'entrée (ou des entrées) de la base qui répond à la question : toutes les étapes, tous les éléments de liste, les conditions, les cas particuliers, les précisions et les informations associées qui y figurent. Un usager doit trouver dans ta réponse tout ce que la base dit sur son sujet.
"""


def teacher_system():
    # TEACHER_PROMPT=v1 keeps the first prompt (no exhaustiveness rule); v2 (default) adds it.
    rules = TEACHER_RULES.replace("{EXHAUSTIVE}", "" if os.environ.get("TEACHER_PROMPT") == "v1" else EXHAUSTIVE)
    return rules + faq_context() + "\n=== FIN DE LA BASE DE CONNAISSANCES ==="


_CLIENTS = {}


def endpoints(spec=None):
    """'qwen@8000,qwen@8001' -> [(model, client)]. Default from $ENDPOINTS."""
    spec = spec or os.environ.get("ENDPOINTS", "qwen@8000")
    if spec in _CLIENTS:      # one set of clients per process: re-creating them between calls made later calls hang
        return _CLIENTS[spec]
    out = _CLIENTS.setdefault(spec, [])
    for item in spec.split(","):
        item = item.strip()
        if item.startswith("ovh:"):  # OVH AI Endpoints (always-on reasoning, 400 requests/min)
            key = (Path.home() / "secrets" / ".ovh").read_text().strip()
            out.append((item, AsyncOpenAI(base_url=OVH_URL, api_key=key, timeout=900)))
        else:
            model, port = item.split("@")
            out.append((model, AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY", timeout=1800)))
    return out


def extra_for(model, thinking=False):
    """Per-family switch to disable (or enable) reasoning."""
    if model.startswith("ovh:"):
        return {}
    if model.startswith("mistral"):
        return {"reasoning_effort": "high" if thinking else "none"}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}


async def chat_many(jobs, spec=None, concurrency=64, temperature=0.0, max_tokens=1500, thinking=False,
                    logprobs=None, desc=""):
    """jobs: list of message lists. Returns list of dicts {text, logprobs?} in the same order.
    Requests are spread round-robin over the endpoints."""
    eps = endpoints(spec)
    sem = asyncio.Semaphore(concurrency * len(eps))
    results = [None] * len(jobs)
    done = 0

    async def one(i, messages):
        nonlocal done
        model, client = eps[i % len(eps)]
        ovh = model.startswith("ovh:")
        kw = dict(model=model[4:] if ovh else model, messages=messages, temperature=temperature,
                  max_tokens=max(max_tokens, 8000) if ovh else max_tokens)  # room for the reasoning tokens
        kw.update(extra_for(model, thinking))
        if logprobs:
            kw.update(logprobs=True, top_logprobs=logprobs)
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(**kw)
                    ch = r.choices[0]
                    res = {"text": (ch.message.content or "").strip(), "finish": ch.finish_reason}
                    if logprobs and ch.logprobs:
                        res["logprobs"] = [
                            {"t": t.token, "top": [[x.token, round(x.logprob, 4)] for x in t.top_logprobs]}
                            for t in ch.logprobs.content
                        ]
                    results[i] = res
                    break
                except Exception as e:  # noqa: BLE001 - retry any transport/server error
                    if attempt == 3:
                        results[i] = {"text": "", "finish": f"error: {e}"}
                    await asyncio.sleep(2 * (attempt + 1))
        done += 1
        if done % 200 == 0 or done == len(jobs):
            print(f"  {desc} {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(i, m) for i, m in enumerate(jobs)))
    return results


def parse_json(text):
    """Extract the first JSON object/array from a model output."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None
