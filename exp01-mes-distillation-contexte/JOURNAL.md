# Expérience 01 — FAQ « Mon espace santé » dans les poids d'un modèle, par distillation de contexte

Journal chronologique, tenu au fil de l'eau pour que la démarche soit reproductible.
Machine : OVH `h100-1520` (GRA11), 4 × NVIDIA H100 PCIe 80 Go, 120 vCPU, 1 460 Gio de RAM, 193 Go de disque,
Ubuntu 26.04, pilote NVIDIA 595.91.07. Coût : 11,20 € HT/h.

## 1. Objectif

Produire deux modèles qui « contiennent » la FAQ de Mon espace santé (88 questions/réponses, 15 400 tokens) sans RAG :
le meilleur modèle possible (le plus gros modèle ouvert entraînable sur 4 × 80 Go) et un modèle utilisable sur une RTX 5090
(32 Go). Les réponses doivent être détaillées et fidèles, ce que les expériences précédentes n'obtenaient pas.

## 2. Diagnostic des expériences précédentes

Source : dépôt `MonEspaceSante-H100-reproduction` (Qwen3-14B-Base, génération synthétique EntiGraph, pré-entraînement
continu puis affinage ; 63,8 % en notation stricte sur 704 questions de test gelées).

- Réponses trop courtes : 184 caractères en médiane contre 492 pour les références. Le jeu d'affinage final comptait
  377 réponses synthétiques courtes (164 caractères en médiane) pour 51 vraies réponses de la FAQ.
- Réponses fausses : détails inventés (chiffre, nom de menu, organisme).
- Le juge binaire tolérait les détails manquants : rien ne mesurait la complétude.

## 3. Méthode

Distillation de contexte : un modèle « professeur » reçoit la FAQ entière dans son prompt système et répond à des
dizaines de milliers de questions variées ; le même modèle, sans la FAQ (« élève »), est entraîné par LoRA à reproduire ces
réponses. L'élève ne reçoit qu'un court prompt système fixe (`Tu es l'assistant de Mon espace santé.`), ou aucun pour un
quart des exemples du domaine.

## 4. Environnement (phase 0)

- `uv`, Python 3.12, deux environnements : `env-vllm` (vLLM 0.34.0rc1 de développement, transformers 5.22.2, torch 2.14)
  et `env-train` (torch 2.15, transformers 5.22.2, peft 0.21.2, accelerate, sentence-transformers).
- Modèles téléchargés dans le disque en RAM `/dev/shm/hf` (le disque de 193 Go est trop petit) : 2 Go/s depuis Hugging Face.
  `loginctl enable-linger` et `RemoveIPC=no` pour que `/dev/shm` ne soit pas vidé à la déconnexion.
- Scripts : `scripts/` (voir section 9). Serveurs : `serve_qwen.sh`, `serve_mistral.sh`, `serve_gemma.sh` dans `~/exp`.

Pièges rencontrés :
- vLLM + Qwen3.8 : `--max-num-seqs 256` obligatoire (limite de blocs de cache de l'attention linéaire).
- `hf download` : un seul motif par option `--exclude`.
- Client OpenAI asynchrone : recréer les clients à chaque appel faisait bloquer le troisième appel d'un même processus ;
  les clients sont maintenant créés une fois par processus (`common.endpoints`).
- Qwen3.8-27B s'entraîne mal dans transformers : environ 260 tokens/s par carte avec `flash-linear-attention`, bien moins
  sans, et une mémoire très élevée (architecture Gated DeltaNet).
- Gemma 4 31B : vocabulaire de 262 000 tokens ; la perte est calculée par morceaux sur les seules positions supervisées
  (`chunked_lm_loss`), sinon la mémoire sature. Ne pas oublier le plafonnement des logits (`final_logit_softcapping`) ni
  le décalage d'un token.
- Gemma 4 31B en pleine précision ne tient pas sur une carte avec un contexte de 15 000 tokens : servi sur deux cartes
  (`--tensor-parallel-size 2`), pas en 8 bits, car la qualité du professeur prime.

## 5. Mesures (phase 1)

Jeu de test : les 704 questions gelées de l'expérience précédente (8 reformulations par entrée), 25 questions hors
périmètre, 30 questions de culture générale. Les 88 réponses de référence sont découpées en 1 018 affirmations
élémentaires (`data/claims.jsonl`, Mistral Medium 3.5).

Trois mesures sur les 704 questions :
- **strict / indulgent** : mêmes consignes de juge que l'expérience précédente (comparaison à UNE réponse de référence).
  Biais constaté : ce juge compte comme « invention » toute information exacte provenant d'une autre entrée de la FAQ ;
  il pénalise donc les réponses détaillées.
- **complétude** : part des affirmations élémentaires de la référence présentes dans la réponse.
- **fidélité** : part des réponses dont toutes les affirmations sont étayées par la FAQ ENTIÈRE (le juge lit toute la FAQ).

Juge local : Qwen3.8-27B (FP8, sans raisonnement). Juge indépendant en arrière-plan : Qwen3.5-397B par OVH AI Endpoints
(trop lent pour l'itération : environ 25 notations par minute).

### Résultats de référence (juge local Qwen3.8-27B)

| Modèle | Strict | Indulgent | Complétude | Fidélité | Longueur médiane |
|---|---|---|---|---|---|
| Qwen3.8-27B + FAQ en contexte | 75,7 | 99,1 | 84,4 | 96,9 | 802 |
| Gemma 4 31B + FAQ en contexte, consigne v1 | 80,4 | 99,4 | 75,4 | 99,9 | 457 |
| Gemma 4 31B + FAQ en contexte, consigne v2 (exhaustive) | 79,8 | 99,1 | 79,0 | 99,6 | 511 |
| Mistral Medium 3.5 128B + FAQ en contexte | 81,0 | 98,6 | 73,7 | 92,9 | 462 |
| Ancien Qwen3-14B CPT+SFT (sans contexte) | 45,5 | 79,3 | 30,7 | 54,7 | 184 |
| Ancien gpt-oss-20b-FAQ-MES (sans contexte) | 23,4 | 71,0 | 37,5 | à mesurer | 312 |
| Qwen3.8-27B sans connaissance | 7,4 | 58,4 | 23,4 | à mesurer | 1 623 |
| Mistral 128B sans connaissance | 5,5 | 67,8 | 27,8 | à mesurer | 1 846 |

Lecture : les lignes « FAQ en contexte » sont le plafond de la méthode. L'ancien modèle ne restituait que 31 % des
éléments des références et un tiers de ses réponses contenait une contradiction avec la FAQ.

## 6. Décisions

- **Modèle pour la RTX 5090 : Gemma 4 31B** (Apache 2.0) au lieu de Qwen3.8-27B : entraînement environ 3 fois plus rapide
  (800 tokens/s par carte), et Google publie une variante préparée pour la quantification 4 bits.
- **Gros modèle : Mistral Medium 3.5 128B** (dense, licence MIT modifiée).
- **Professeur : Gemma 4 31B en pleine précision avec la consigne v2** : fidélité 99,6 %, complétude 79 %, longueur
  proche des références. Le Mistral 128B est trop lent comme professeur (366 tokens/s sur deux cartes).
- OVH AI Endpoints : inutilisable pour la génération en masse (lenteur, et 15 000 tokens de contexte par requête
  coûteraient plusieurs centaines d'euros).

## 7. Données d'entraînement (phase 2)

1. `gen_questions.py --scale 2.0` (Qwen3.8-27B, température 1,0) : 45 399 questions en 12 minutes. Types : reformulations
   selon 12 profils d'usagers, questions ciblant chaque affirmation élémentaire, fausses présuppositions, situations
   concrètes, demandes ouvertes, questions croisant plusieurs entrées, dialogues de 2 et 3 tours, questions voisines sans
   réponse dans la FAQ, questions hors périmètre.
2. `dedup.py` (bge-m3) : retrait des questions trop proches du jeu de test (cosinus >= 0,90 ou recouvrement de trigrammes
   >= 0,6, même règle que l'expérience précédente) : 44 733 conservées (`data/gen/questions_clean.jsonl`).
3. `select_subset.py` : premier lot stratifié de 18 556 questions (`questions_s1.jsonl`), le reste en réserve
   (`questions_rest.jsonl`).
4. `teacher_answers.py` : réponses du professeur (température 0,3), filtrage des réponses tronquées ou mentionnant « la
   FAQ », construction du jeu d'entraînement `data/train/distill_<tag>.jsonl`.

### Réponses du professeur, premier lot (10 novembre, 15:31 à 15:57 UTC)

Deux serveurs Gemma 4 31B en pleine précision (cartes 0-1 et 2-3), 100 requêtes simultanées chacun, environ 750 réponses
par minute : 18 556 questions en 26 minutes.

Incidents, à connaître pour reproduire :
- **Filtre anti-fuite trop large.** La première version écartait toute réponse contenant « le document », « le contexte »
  ou « ci-dessus » : 4 050 réponses perdues (22 %), surtout sur la gestion des documents. Le filtre ne vise plus que les
  formules qui trahissent la lecture d'un texte fourni (« FAQ », « base de connaissances », « document fourni »...), et les
  réponses écartées sont conservées dans `teacher_<tag>_leaks.jsonl`.
- **Régénération ratée.** Deux serveurs vLLM lancés ensemble mettent plus de 7 minutes à démarrer ; les requêtes sont
  parties avant. `teacher_answers.py` attend maintenant que chaque serveur réponde et s'arrête si plus de 2 % des requêtes
  échouent.
- **Planificateur du taux d'apprentissage.** Passé à `accelerate`, il avançait une fois par processus : avec 4 cartes, le
  taux retombait à zéro au quart de l'entraînement. Il n'est plus enveloppé par `accelerate`.

Premier jeu d'entraînement : `data/train/distill_gemma31_s1.jsonl`, 15 496 exemples (3,98 M de tokens dont 2,80 M
supervisés), réponses de 573 caractères en médiane ; 3 629 exemples sans prompt système.

### Entraînement A1 (10 novembre, 16:25 UTC)

`train_gemma.sh gemma31_a1 ... --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1` : LoRA de rang 64 sur toutes les
couches linéaires du modèle de langage (490 M de paramètres entraînés, 1,5 %), 4 cartes en parallèle de données,
433 pas, 3 350 tokens/s, environ 20 minutes, 70 Gio par carte (avec recalcul des activations).

### Résultat A1 (10 novembre, 16:56 UTC)

Gemma 4 31B + LoRA A1, sans la FAQ (prompt système court), juge local Qwen3.8-27B :

| | Strict | Indulgent | Complétude | Fidélité | Longueur médiane | Hors périmètre inventé | Culture générale |
|---|---|---|---|---|---|---|---|
| Gemma A1 (élève, sans FAQ) | 76,1 | 97,3 | 74,6 | 95,0 | 493 | 0 | 90,0 |
| Gemma + FAQ en contexte (plafond) | 79,8 | 99,1 | 79,0 | 99,6 | 511 | 0 | sans objet |
| Gemma de base, sans connaissance | 9,4 | 68,5 | 26,2 | à mesurer | 1 679 | 0 | 100,0 |
| Ancien Qwen3-14B CPT+SFT | 45,5 | 79,3 | 30,7 | 54,7 | 184 | | |

Défauts observés : 3 questions de culture générale sur 30 refusées (« mon périmètre est limité à Mon espace santé »)
alors qu'aucun prompt système n'était donné ; plus gros écart de complétude avec le professeur sur « Gérer mes documents »
(75 contre 84), la catégorie amputée par le filtre anti-fuite.

### Second lot et entraînement A2 (10 novembre, 17:00 à 19:00 UTC)

- `questions_s2.jsonl` = 4 050 questions perdues + 26 177 en réserve = 30 227 questions ; réponses du professeur en
  39 minutes : `distill_gemma31_s2.jsonl`, 25 529 exemples.
- Le filtre anti-fuite a encore écarté 5 401 réponses : presque toutes contiennent « la base de connaissances ne mentionne
  pas... », c'est-à-dire le professeur qui reconnaît que la FAQ ne couvre pas un point. `recover_leaks.py` les récupère en
  remplaçant l'expression par « la documentation officielle de Mon espace santé » : `distill_gemma31_s2_recovered.jsonl`,
  6 033 exemples (dont 1 780 questions voisines sans réponse). Non inclus dans A2, prévus pour A3.
- `gen_general.py` : 1 745 questions générales avec la réponse normale du modèle de base, sans prompt système
  (`general_gemma31.jsonl`), pour que l'élève ne refuse pas les questions générales quand aucun prompt système n'est donné.
- A2 : `distill_gemma31_all.jsonl` = s1 + s2 + général = 42 770 exemples, mêmes hyperparamètres que A1, 1 262 pas,
  environ 1 heure.

### Mistral Medium 3.5 128B : préparation

- Chargement vérifié sur CPU (155 s) : classe `Mistral3ForConditionalGeneration` (`AutoModelForImageTextToText`), 88 couches,
  poids distribués en FP8 (déquantifiés en bf16 sur CPU, 250 Go). Sur GPU, transformers exige le paquet
  `kernels==0.17.2` pour garder les poids en FP8.
- Format de dialogue : le tokeniseur `mistral-common` construit lui-même le prompt (avec
  `[MODEL_SETTINGS]{"reasoning_effort": "none"}`), et le jeton de fin de réponse est ajouté par identifiant.

### Résultat A2 (10 novembre, 18:57 UTC)

58 minutes d'entraînement sur 4 cartes (3 300 tokens/s), 42 770 exemples.

| | Strict | Indulgent | Complétude | Fidélité | Longueur médiane | Hors périmètre inventé | Culture générale |
|---|---|---|---|---|---|---|---|
| Gemma A1 (15 496 exemples) | 76,1 | 97,3 | 74,6 | 95,0 | 493 | 0 | 90,0 |
| **Gemma A2 (42 770 exemples)** | **80,1** | **99,3** | **77,3** | **98,7** | 492 | 0 | **100,0** |
| Gemma + FAQ en contexte (plafond) | 79,8 | 99,1 | 79,0 | 99,6 | 511 | 0 | sans objet |

L'élève rejoint pratiquement son professeur. Les refus sur les questions générales ont disparu.

### Chaîne GGUF pour la RTX 5090 (validée sur A1, sur CPU)

`gguf_prep.sh` : llama.cpp (commit 7592b61 du 10 novembre 2026) compilé sans CUDA ; `merge.py` fusionne l'adaptateur
(1,8 min sur CPU) ; `convert_hf_to_gguf.py --outtype f16` (61,4 Go, 2,5 min) ; `llama-quantize ... Q5_K_M` (20,8 Go, 1 min).
Pièges : l'environnement de conversion doit avoir `transformers==5.22.2` (la version 4.57 installée par llama.cpp ne lit
pas le tokeniseur de Gemma 4) ; `llama-quantize` échoue en écrivant dans `/dev/shm` et réussit sur le disque ; llama.cpp
active le raisonnement de Gemma par défaut, il faut `--reasoning-budget 0` (à intégrer au fichier livré).

### Ablations : chiffrer la recette (10 novembre, à partir de 19:02 UTC)

Objectif fixé par l'utilisateur : savoir, chiffres à l'appui, ce qui permet de faire mieux que les expériences précédentes.
`make_ablations.py` construit des jeux où un seul ingrédient change ; `pipeline_abl.sh` enchaîne les entraînements
(mêmes hyperparamètres que A1 : rang 64, taux 2e-4, une époque) et `eval_multi.sh` les évalue avec le même juge.

| Nom | Modèle | Données | Ce qui est testé |
|---|---|---|---|
| `abl_para_teacher` | Gemma 31B | 3 446 reformulations, réponses du professeur | référence des deux suivantes |
| `abl_para_official` | Gemma 31B | les mêmes questions, réponse officielle de la FAQ recopiée | apport du professeur |
| `abl_mixed_3k` | Gemma 31B | 3 446 questions de tous types | apport de la variété des questions, à volume égal |
| `abl_oldsft` | Gemma 31B | les 548 exemples d'affinage de l'expérience précédente, 10 époques | les anciennes données sur le nouveau modèle |
| `size_gemma12` | Gemma 4 12B | jeu de A1 (15 496) | taille du modèle, même famille |
| `size_qwen14` | Qwen3-14B | jeu de A1 (15 496) | taille et famille de l'ancien modèle |

Volume : `abl_mixed_3k` (3 446), A1 (15 496), A2 (42 770).

Incident : `abl_oldsft` a planté à la fin de sa première époque (sauvegarde par époque appelée sur le mauvais objet) ;
corrigé (`--save-epochs` devient une option) et relancé.

### Publication des données

Jeu public : https://huggingface.co/datasets/fenyo/4xH100-MonEspaceSante-distillation-contexte (tout `data/`).

### Résultats des ablations (10 novembre, 21:08 UTC)

Juge local Qwen3.8-27B, 704 questions gelées, élève sans la FAQ. Tous les entraînements : LoRA rang 64, taux 2e-4.

**Taille du modèle** (mêmes 15 496 exemples que A1, même nombre de pas) :

| Modèle | Strict | Complétude | Fidélité | Culture générale |
|---|---|---|---|---|
| Qwen3-14B (`size_qwen14`) | 69,9 | 68,7 | 85,7 | 100 |
| Gemma 4 12B (`size_gemma12`) | 76,1 | 70,6 | 90,6 | 96,7 |
| Gemma 4 31B (A1) | 76,1 | 74,6 | 95,0 | 90,0 |

**Données** (Gemma 4 31B) :

| Jeu | Exemples | Strict | Complétude | Fidélité | Longueur médiane | Hors périmètre inventé |
|---|---|---|---|---|---|---|
| `abl_oldsft` : les 548 exemples de l'expérience précédente, 10 époques | 548 | 48,2 | 29,8 | 58,5 | 170 | 0 |
| `abl_para_teacher` : reformulations + réponse du professeur | 3 446 | 71,9 | 70,9 | 87,6 | 478 | 12 |
| `abl_para_official` : mêmes questions + réponse officielle recopiée | 3 446 | 81,8 | 82,4 | 95,3 | 468 | 24 |
| `abl_mixed_3k` : tous types de questions + professeur | 3 446 | 72,0 | 66,8 | 87,5 | 432 | 0 |
| A1 : tous types + professeur | 15 496 | 76,1 | 74,6 | 95,0 | 493 | 0 |
| A2 : tous types + professeur + questions générales | 42 770 | 80,1 | 77,3 | 98,7 | 492 | 0 |

Rappel : ancien Qwen3-14B CPT+SFT = 45,5 / 30,7 / 54,7 / 184 caractères ; modèles sans connaissance : complétude 23 à 28,
fidélité 0 à 3.

Lecture :
1. **Le jeu de données était le facteur limitant, pas le modèle.** Gemma 4 31B entraîné sur les 548 anciens exemples
   donne les mêmes chiffres que l'ancien Qwen3-14B (complétude 30, fidélité 58, réponses de 170 caractères).
2. **La méthode et les données apportent l'essentiel.** Un Qwen3-14B (taille et famille de l'ancien modèle, variante
   conversationnelle) entraîné avec la nouvelle méthode : complétude 68,7 (+38 points), fidélité 85,7 (+31).
3. **La taille apporte peu** : de 12-14B à 31B, +4 à +6 points de complétude, +4 à +9 de fidélité.
4. **La réponse officielle recopiée est la meilleure cible pour les reformulations** : complétude 82,4 contre 70,9 avec
   la réponse du professeur, à volume égal. Le test mesure la restitution de l'entrée officielle, et le professeur
   répond de façon plus ciblée, donc moins complète. Sans exemples de refus, ce jeu invente hors périmètre (24 %).
5. **Les exemples de refus et de questions voisines** ramènent l'invention hors périmètre à 0.
6. **Le volume** : complétude 66,8 → 74,6 → 77,3 et fidélité 87,5 → 95,0 → 98,7 pour 3 446 → 15 496 → 42 770 exemples.

Limite : le test gelé ne contient que des reformulations de la question officielle. Il ne mesure pas l'apport des
questions précises, pièges ou croisées. D'où un second test (`data/varied_test.jsonl`, 600 questions de types `claim`,
`false`, `multi`, `scen` retirées de l'entraînement A3 ; référence = réponse du professeur ; `judge_varied.py`).

### Entraînement A3 : recette combinée (10 novembre, 21:14 UTC)

`make_a3.py` : `distill_gemma31_a3.jsonl`, 48 203 exemples = reformulations avec la réponse officielle (16 293) +
tous les autres types avec la réponse du professeur + aveux d'ignorance récupérés + questions générales (1 745),
moins les 600 questions du test varié. Mêmes hyperparamètres que A1 et A2. Puis `pipeline_a3.sh` enchaîne les
évaluations et l'entraînement du Mistral.

### Mistral Medium 3.5 128B : mesures d'entraînement

- Poids gardés en FP8 (`kernels==0.17.2`) : 29 tokens/s, et mémoire saturée sur 2 cartes. Inutilisable : chaque passe
  arrière redécompresse les poids.
- Poids déquantifiés en bf16 au chargement (`--dequantize`, `FineGrainedFP8Config(dequantize=True)`), un seul modèle
  réparti sur les 4 cartes (`--device-map auto --max-mem 65`), lots de 1 500 tokens, recalcul des activations :
  **356 tokens/s**, 64 à 73 Gio par carte. Environ 10 heures pour le jeu A3.
- Avec un modèle réparti sur plusieurs cartes, la sortie arrive sur la dernière : les étiquettes doivent y être déplacées.
- Évaluation avec vLLM et plusieurs adaptateurs : `--max-loras` et `--max-lora-rank` au strict nécessaire, vLLM réserve
  la mémoire pour le produit des deux (8 adaptateurs de rang 256 saturent la carte).

## 8. Suite

Voir les entrées suivantes du journal (entraînement, évaluations, gros modèle, quantification).

## 9. Scripts

| Script | Rôle |
|---|---|
| `common.py` | contexte FAQ, consignes professeur et élève, client asynchrone (vLLM local et OVH) |
| `claims.py` | découpage des réponses de référence en affirmations élémentaires |
| `answer.py` | réponses d'un modèle servi sur les jeux d'évaluation (modes professeur, élève, sans système) |
| `judge.py` | juge strict, indulgent, complétude, hors périmètre, culture générale |
| `judge_faith.py` | juge de fidélité à la FAQ entière |
| `gen_questions.py`, `dedup.py`, `select_subset.py`, `teacher_answers.py` | construction des données d'entraînement |
| `train_lora.py` | entraînement LoRA (une carte, ou plusieurs avec `accelerate launch`) |
