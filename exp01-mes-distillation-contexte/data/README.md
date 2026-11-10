---
license: other
language:
- fr
pretty_name: Mon espace santé — distillation de contexte (4xH100)
tags:
- synthetic
- distillation
- faq
---

# Mon espace santé — données de distillation de contexte

Données de l'expérience 01 du dépôt [AI-Experiments-4xH100](https://github.com/AlexandreFenyo/AI-Experiments-4xH100)
(journal complet dans `exp01-mes-distillation-contexte/JOURNAL.md`).

Objectif : faire entrer la FAQ du service public « Mon espace santé » (88 questions/réponses) dans les poids d'un modèle,
sans RAG. Un modèle « professeur » (Gemma 4 31B, FAQ entière dans le prompt système) répond à des dizaines de milliers de
questions générées ; le même modèle sans la FAQ est entraîné à reproduire ces réponses.

## Fichiers

| Fichier | Contenu |
|---|---|
| `facts_real_88.jsonl` | les 88 entrées de la FAQ (source de vérité) |
| `claims.jsonl` | les 1 018 affirmations élémentaires extraites des réponses |
| `heldout_test.jsonl`, `ood_hallucination.jsonl`, `forgetting_fr.jsonl`, `validation_30.jsonl` | jeux d'évaluation gelés (aucune question d'entraînement n'en est proche : cosinus bge-m3 < 0,90) |
| `gen/questions_raw.jsonl`, `gen/questions_clean.jsonl` | 45 399 questions générées, 44 733 après déduplication contre l'évaluation |
| `gen/questions_s1.jsonl`, `gen/questions_s2.jsonl` | premier lot stratifié (18 556) et second lot (30 227) |
| `gen/teacher_gemma31_*.jsonl` | réponses du professeur ; `*_leaks.jsonl` : réponses mises de côté par le filtre |
| `train/distill_gemma31_s1.jsonl` | premier jeu d'entraînement (15 496 exemples) |
| `train/distill_gemma31_s2.jsonl`, `train/distill_gemma31_s2_recovered.jsonl` | second lot (25 529) et aveux d'ignorance récupérés (6 033) |
| `train/general_gemma31.jsonl` | 1 745 questions générales avec la réponse du modèle de base (anti-oubli) |
| `train/distill_gemma31_all.jsonl` | jeu de l'entraînement A2 (42 770 exemples) |
| `train/abl_*.jsonl` | jeux des ablations (un ingrédient changé à la fois) |

Format d'entraînement : `{"id", "type", "messages": [...]}` ; la perte ne porte que sur le dernier message de l'assistant.
Types de questions : `para` (reformulations), `claim` (ciblant une affirmation), `false` (fausse présupposition), `scen`
(situation concrète), `broad` (demande ouverte), `multi` (plusieurs entrées), `dialog2`/`dialog3`, `near` (voisine, sans
réponse dans la FAQ), `ood` (hors périmètre), `general`.

Le contenu de la FAQ provient du site public de Mon espace santé ; les questions et réponses synthétiques ont été
générées par Qwen3.8-27B et Gemma 4 31B.
