---
license: apache-2.0
language:
- fr
base_model: google/gemma-4-31B-it
datasets:
- fenyo/4xH100-MonEspaceSante-distillation-contexte
pipeline_tag: text-generation
---

# Gemma 4 31B — FAQ « Mon espace santé » dans les poids (distillation de contexte)

`google/gemma-4-31B-it` affiné (LoRA de rang 64 fusionné) pour répondre **sans RAG ni document en contexte** aux
questions sur le service public « Mon espace santé », à partir de sa FAQ officielle (88 questions/réponses).

Démarche, scripts, journal et résultats : https://github.com/AlexandreFenyo/AI-Experiments-4xH100
(dossier `exp01-mes-distillation-contexte`). Données : `fenyo/4xH100-MonEspaceSante-distillation-contexte`.

## Utilisation

Prompt système utilisé à l'entraînement : `Tu es l'assistant de Mon espace santé.` Le modèle répond aussi aux questions
du domaine sans prompt système ; sans prompt système, il répond normalement aux questions générales.
Le raisonnement (« thinking ») doit être désactivé : l'entraînement s'est fait sans.

## Méthode

Un professeur (le même Gemma 4 31B avec la FAQ entière dans son prompt système) répond à des dizaines de milliers de
questions générées (reformulations, détails ciblés, fausses présuppositions, questions croisant plusieurs entrées,
dialogues, questions voisines sans réponse, hors périmètre). L'élève, sans la FAQ, apprend à reproduire ces réponses.
Pour les reformulations de la question officielle, la cible est la réponse officielle de la FAQ.
Jeu d'entraînement : 48 203 exemples, 1 époque, environ 1 heure sur 4 × H100.

## Résultats (élève sans la FAQ, juge Qwen3.8-27B)

Sur 704 questions de test gelées (reformulations jamais vues) :

| | Strict | Complétude | Fidélité à la FAQ | Culture générale | Invention hors périmètre |
|---|---|---|---|---|---|
| Ce modèle | 80,5 % | 75,7 % | 97,6 % | 100 % | 0 % |
| Gemma 4 31B avec la FAQ en contexte | 79,8 % | 79,0 % | 99,6 % | | 0 % |
| Gemma 4 31B sans entraînement | 9,4 % | 26,2 % | 3,1 % | 100 % | 0 % |

Sur 600 questions variées (détails, pièges, questions croisées, situations) : 96,0 % de réponses fidèles à la FAQ.

## Limites

Le modèle ne connaît que le contenu de la FAQ à la date de sa collecte ; il peut se tromper, et ne remplace ni le site
officiel ni un professionnel de santé. Modèle de démonstration et de recherche.
