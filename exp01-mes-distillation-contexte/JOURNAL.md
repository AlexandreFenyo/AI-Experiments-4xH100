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
