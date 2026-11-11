---
license: other
license_name: mistral-modified-mit
license_link: https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE
language:
- fr
base_model: mistralai/Mistral-Medium-3.5-128B
datasets:
- fenyo/4xH100-MonEspaceSante-distillation-contexte
---

# Mistral Medium 3.5 128B — FAQ « Mon espace santé » dans les poids (distillation de contexte)

`mistralai/Mistral-Medium-3.5-128B` affiné (LoRA de rang 64 fusionné, poids FP8 comme le modèle d'origine) pour répondre
**sans RAG ni document en contexte** aux questions sur le service public « Mon espace santé », à partir de sa FAQ
officielle (88 questions/réponses). Le dossier `lora-adapter/` contient l'adaptateur seul.

Démarche, scripts, journal et résultats : https://github.com/AlexandreFenyo/AI-Experiments-4xH100
(dossier `exp01-mes-distillation-contexte`). Données : `fenyo/4xH100-MonEspaceSante-distillation-contexte`.
La licence est celle du modèle de base (MIT modifiée, voir le lien).

## Utilisation

Prompt système utilisé à l'entraînement : `Tu es l'assistant de Mon espace santé.`, avec `reasoning_effort="none"`.
Service avec vLLM : comme le modèle de base (`--tensor-parallel-size 2` sur 2 × 80 Go), ou le modèle de base avec
`--enable-lora --lora-modules mes=<lora-adapter>`.

## Entraînement

48 203 exemples (réponses d'un professeur Gemma 4 31B qui a la FAQ en contexte ; réponse officielle pour les
reformulations de la question officielle ; refus et aveux d'ignorance ; questions générales), 1 époque, 8 h 08 sur
4 × H100 (un seul modèle réparti sur les 4 cartes, poids déquantifiés en bf16, 396 tokens/s).

## Résultats (sans la FAQ, juge Qwen3.8-27B)

| | Strict | Complétude | Fidélité à la FAQ | Culture générale | Invention hors périmètre |
|---|---|---|---|---|---|
| Ce modèle, 704 questions gelées | 82,8 % | 76,9 % | 99,0 % | 96,7 % | 0 % |
| Mistral 128B avec la FAQ en contexte | 81,0 % | 73,7 % | 92,9 % | | 0 % |
| Mistral 128B sans entraînement | 5,5 % | 27,8 % | 0,1 % | 100 % | 4 % |

Sur 600 questions variées (détails, pièges, questions croisées, situations) : 95,3 % de réponses fidèles à la FAQ.

## Limites

Le modèle ne connaît que le contenu de la FAQ à la date de sa collecte ; il peut se tromper, et ne remplace ni le site
officiel ni un professionnel de santé. Modèle de démonstration et de recherche.
