# Journal — exp01 : FAQ « Mon espace santé » dans les poids par distillation de contexte

Machine : OVH `h100-1520` (4 × H100 80 Go, 120 vCPU, 1,46 To RAM, 193 Go disque), Ubuntu 26.04, pilote NVIDIA 595.91.
Toutes les heures sont en CET, le 10 novembre 2026. Les scripts cités sont dans `scripts/`.

## Objectif

Produire deux modèles qui connaissent la FAQ de Mon espace santé (88 entrées, 15 400 tokens) **sans contexte** :
le meilleur modèle possible, et un modèle exécutable sur une RTX 5090 (32 Go). Les essais précédents
(`MonEspaceSante-H100-reproduction`, Qwen3-14B CPT + SFT ; `gpt-oss-20b-FAQ-MES`) donnaient des réponses peu détaillées
et parfois fausses.

## Diagnostic des essais précédents (avant tout calcul)

- Réponses de l'ancien modèle : 184 caractères en médiane, contre 492 pour les références.
- Le jeu de SFT final contenait 377 réponses synthétiques courtes (164 car. en médiane) pour 51 vraies réponses.
- Les erreurs sont des détails inventés (chiffre, menu, organisme).
- Le juge était binaire et tolérait les détails manquants.

## Méthode

Distillation de contexte : un « professeur » qui a toute la FAQ dans son prompt système répond à des dizaines de milliers
de questions variées ; l'« élève » (le même modèle, sans la FAQ) est entraîné par LoRA à reproduire ces réponses.

## Environnement (13:50 → 14:00)

- `uv` + trois environnements Python 3.12 dans `~/exp` : `env-dl` (huggingface_hub), `env-vllm` (vLLM 0.34.0rc1 nightly,
  transformers 5.22.2, torch 2.14), `env-train` (torch 2.15, transformers 5.22.2, peft 0.21.2, accelerate,
  sentence-transformers).
- Modèles téléchargés dans le disque RAM `/dev/shm/hf` (`HF_HOME`) : `Qwen/Qwen3.8-27B`, `Qwen/Qwen3.8-27B-FP8`,
  `mistralai/Mistral-Medium-3.5-128B`, `google/gemma-4-31B-it`. Débit constaté : environ 2 Go/s.
- `loginctl enable-linger fenyo` et `RemoveIPC=no` pour que `/dev/shm` ne soit pas vidé à la déconnexion.
- Serveurs : `~/exp/serve_qwen.sh GPU PORT`, `~/exp/serve_mistral.sh GPUS PORT TP`, `~/exp/serve_gemma.sh GPUS PORT TP`.

Pièges rencontrés :
- vLLM + Qwen3.8 : `--max-num-seqs 256` obligatoire (sinon « exceeds available Mamba cache blocks »).
- `hf download --exclude a b` échoue : un seul motif par option.
- Recréer les clients `AsyncOpenAI` entre deux lots faisait bloquer le troisième lot : clients mis en cache (`common.py`).
- Entraînement de Qwen3.8-27B (attention linéaire « Gated DeltaNet ») : environ 260 tokens/s par carte avec
  `flash-linear-attention`, encore plus lent sans. **Qwen3.8 abandonné comme élève**, gardé comme générateur et juge.
- Perte calculée sur tout le vocabulaire (130k à 262k) : mémoire saturée. Remplacée par une perte par morceaux sur les
  seules positions supervisées (`chunked_lm_loss` dans `train_lora.py`).
- Gemma 4 plafonne ses logits par `tanh` (`final_logit_softcapping`) : à reproduire dans la perte.

## Banc d'évaluation (14:00 → 15:15)

Jeux gelés repris de l'expérience précédente : 704 questions (`heldout_test.jsonl`), 25 hors périmètre, 30 de culture
générale. Mesures :
- `strict` / `lenient` : mêmes consignes de juge que l'expérience précédente (`judge_prompts_prev.py`).
- `completeness` : part des affirmations atomiques de la réponse de référence présentes dans la réponse
  (`claims.py` : 1 018 affirmations extraites par Mistral 128B ; `judge.py`).
- fidélité (`judge_faith.py`) : chaque affirmation de la réponse est vérifiée contre la FAQ **entière**.

Juge local : Qwen3.8-27B-FP8 (vLLM, sans raisonnement). Juge de référence indépendant : Qwen3.5-397B via OVH AI
Endpoints, lent (environ 25 notations par minute), lancé en arrière-plan.

Constat : la notation `strict` héritée compte comme « invention » toute information exacte venant d'une autre entrée de
la FAQ. Elle pénalise donc les réponses détaillées. Les mesures retenues sont la complétude et la fidélité.

### Références mesurées (704 questions, juge local)

| Système | strict | lenient | complétude | fidèle (FAQ entière) | longueur médiane |
|---|---|---|---|---|---|
| Qwen3.8-27B + FAQ en contexte | 75,7 | 99,1 | 84,4 | 96,9 | 802 |
| Gemma 4 31B + FAQ en contexte | 80,4 | 99,4 | 75,4 | 99,9 | 457 |
| Mistral Medium 3.5 128B + FAQ en contexte | 81,0 | 98,6 | 73,7 | 92,9 | 462 |
| Ancien Qwen3-14B CPT+SFT (sans contexte) | 45,5 | 79,3 | 30,7 | 54,7 | 184 |
| Ancien gpt-oss-20b-FAQ-MES (sans contexte) | 23,4 | 71,0 | 37,5 | — | 312 |
| Qwen3.8-27B sans connaissance | 7,4 | 58,4 | 23,4 | — | 1 623 |
| Mistral 128B sans connaissance | 5,5 | 67,8 | 27,8 | — | 1 846 |

## Génération des questions (14:06 → 14:30)

- `gen_questions.py --scale 2.0` avec deux serveurs Qwen3.8-27B-FP8 : 7 053 appels, 45 399 questions en 12 minutes.
  Types : reformulations par profil d'usager, questions ciblant chaque affirmation, fausses présuppositions, mises en
  situation, demandes ouvertes, questions croisant plusieurs entrées, dialogues à 2 et 3 tours, questions voisines sans
  réponse dans la FAQ, questions hors périmètre.
- `dedup.py` (bge-m3, cosinus ≥ 0,90 ou recouvrement de trigrammes ≥ 0,6 avec les questions d'évaluation) :
  44 733 questions conservées (`data/gen/questions_clean.jsonl`, rapport dans `dedup_report.json`).
- `select_subset.py` : premier lot stratifié de 18 556 questions (`questions_s1.jsonl`), reste dans `questions_rest.jsonl`.

## Décisions

- Modèle pour la RTX 5090 : **Gemma 4 31B** (Apache 2.0) au lieu de Qwen3.8-27B, pour la vitesse d'entraînement
  (environ 800 tokens/s par carte) et l'existence d'une variante préparée pour la quantification 4 bits.
- Gros modèle : Mistral Medium 3.5 128B.
- Règle de l'utilisateur : la qualité passe avant le parallélisme pour les étapes importantes. Le professeur est donc
  servi en pleine précision sur deux cartes (et non en FP8 sur une).
- Le Mistral 128B génère trop lentement (366 tokens/s sur 2 cartes) pour produire les réponses d'entraînement.
