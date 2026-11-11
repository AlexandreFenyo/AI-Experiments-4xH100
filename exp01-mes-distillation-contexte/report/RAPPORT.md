# Faire entrer une FAQ dans les poids d'un modèle

**Expérience 01 sur 4 × NVIDIA H100 (OVH), 10 et 11 novembre 2026 — FAQ « Mon espace santé »**

Alexandre Fenyo, avec Claude Code. Dépôt : <https://github.com/AlexandreFenyo/AI-Experiments-4xH100>

## 1. Résumé

L'objectif était double : obtenir un modèle qui restitue fidèlement et en détail la FAQ de « Mon espace santé » sans
document en contexte, et comprendre pourquoi les tentatives précédentes n'y arrivaient pas.

Résultat : deux modèles qui répondent sans la FAQ presque aussi bien qu'un modèle qui l'a sous les yeux.

| Sur 704 questions de test jamais vues | Strict | Complétude | Fidélité | Longueur médiane |
|---|---|---|---|---|
| Ancien modèle (Qwen3-14B, pré-entraînement continu puis affinage) | 45,5 % | 30,7 % | 54,7 % | 184 car. |
| **Mistral Medium 3.5 128B entraîné** | **82,8 %** | **76,9 %** | **99,0 %** | 462 car. |
| **Gemma 4 31B entraîné** | 80,5 % | 75,7 % | 97,6 % | 474 car. |
| **Gemma 4 31B entraîné, quantifié 5 bits (21,8 Go, RTX 5090)** | 81,1 % | 75,7 % | 97,4 % | 472 car. |
| Gemma 4 31B, variante « restitution » (R1, voir 6.4) | 90,6 % | 88,6 % | 99,7 % | 460 car. |
| Gemma 4 31B avec la FAQ dans son contexte (plafond) | 79,8 % | 79,0 % | 99,6 % | 511 car. |

Les réponses de référence font 492 caractères en médiane. Aucun des modèles entraînés n'invente de fonctionnalité sur
les 25 questions hors périmètre.

La recette, par ordre d'importance mesurée :

1. **Le jeu de données, pas le modèle.** Gemma 4 31B entraîné sur les 548 exemples de l'ancienne expérience retombe au
   niveau de l'ancien modèle (complétude 30 %, fidélité 58 %).
2. **Des milliers de questions variées qui pointent vers des réponses complètes et fidèles.** Avec la nouvelle méthode,
   un Qwen3-14B gagne 38 points de complétude et 31 points de fidélité par rapport à l'ancien.
3. **Des exemples de refus et de questions sans réponse.** Ils ramènent l'invention hors périmètre de 24 % à 0 %.
4. **Le volume.** De 3 446 à 42 770 exemples, la fidélité passe de 87,5 % à 98,7 %.
5. **La taille du modèle, loin derrière.** De 12-14 milliards à 31 milliards de paramètres : +4 à +6 points de
   complétude. De 31 à 128 milliards : +1 point.

## 2. Contexte

La FAQ officielle de « Mon espace santé » compte 88 questions/réponses (15 400 tokens). Les expériences précédentes
(Qwen3-14B avec génération synthétique EntiGraph, pré-entraînement continu puis affinage ; gpt-oss-20b ; Ministral 8B)
donnaient des réponses courtes et parfois fausses.

Machine : OVH `h100-1520` à Gravelines, 4 × H100 de 80 Go, 120 cœurs, 1 460 Gio de mémoire, 11,20 € HT de l'heure.

## 3. Diagnostic des expériences précédentes

- **Réponses courtes.** 184 caractères en médiane contre 492 pour les références. Le jeu d'affinage final contenait
  377 réponses synthétiques courtes (164 caractères en médiane) pour 51 vraies réponses de la FAQ : le modèle imite la
  longueur qu'on lui montre.
- **Détails faux.** Un tiers des réponses contredisait la FAQ (mauvais chiffre, mauvais menu, mauvais organisme). La
  connaissance entrait par des documents reformulés, pas par des couples question/réponse complète.
- **Mesure aveugle.** Le juge répondait par oui ou non et tolérait les détails manquants : rien ne mesurait si la
  réponse était complète.

## 4. Méthode : la distillation de contexte

La FAQ tient dans le contexte d'un modèle. Un modèle qui l'a sous les yeux répond bien. On entraîne donc le même
modèle, sans la FAQ, à reproduire ces réponses.

1. **Découper** les 88 réponses en 1 018 affirmations élémentaires (une étape, un nom de menu, un chiffre...).
2. **Générer des questions** (45 399, puis 44 733 après retrait de celles trop proches du jeu de test) : reformulations
   selon 12 profils d'usagers, questions visant chaque affirmation, fausses présuppositions à corriger, situations
   concrètes, demandes ouvertes, questions croisant plusieurs entrées, dialogues, questions voisines sans réponse dans
   la FAQ, questions hors périmètre.
3. **Faire répondre un professeur** : Gemma 4 31B en pleine précision, avec la FAQ entière dans son prompt système et la
   consigne d'être exhaustif et fidèle. Débit : 750 réponses par minute sur 4 cartes.
4. **Entraîner l'élève** par LoRA (rang 64, toutes les couches linéaires) avec un prompt système court,
   `Tu es l'assistant de Mon espace santé.`, ou aucun pour un quart des exemples du domaine.
5. **Ajouter** des questions générales avec la réponse normale du modèle de base, pour que l'élève ne refuse pas les
   questions générales, et récupérer les aveux d'ignorance du professeur (« la documentation officielle ne mentionne
   pas... »).

Jeu final : 48 203 exemples. Pour les reformulations de la question officielle, la cible est la réponse officielle de
la FAQ ; pour tout le reste, la réponse du professeur.

## 5. Mesures

Trois mesures sur 704 questions gelées (8 reformulations par entrée, héritées de l'expérience précédente) :

- **Strict** : le juge de l'expérience précédente (comparaison à la réponse de référence, oui ou non). Il pénalise les
  réponses qui ajoutent des informations exactes venant d'une autre entrée.
- **Complétude** : part des affirmations élémentaires de la référence présentes dans la réponse.
- **Fidélité** : part des réponses dont toutes les affirmations sont étayées par la FAQ entière.

S'y ajoutent 25 questions hors périmètre, 30 questions de culture générale, et un **test varié** de 600 questions
(détails ciblés, pièges, questions croisées, situations) retirées de l'entraînement final.

Juge : Qwen3.8-27B servi localement. Contrôle par un juge indépendant, Qwen3.5-397B (API OVH) : sur 100 questions par modèle final, il donne 80,0 % en strict et 76,0 % de complétude à Gemma (79,0 % et 74,7 % pour le juge local sur les mêmes questions), et 88,0 % et 80,0 % au Mistral (86,0 % et 77,8 %). Les deux juges rendent le même verdict strict dans 89 % et 94 % des cas.

## 6. La recette chiffrée

### 6.1 Le modèle ou les données ?

| Modèle | Données d'entraînement | Complétude | Fidélité | Longueur médiane |
|---|---|---|---|---|
| Qwen3-14B (ancien) | anciennes | 30,7 % | 54,7 % | 184 car. |
| Gemma 4 31B | les 548 anciens exemples d'affinage | 29,8 % | 58,5 % | 170 car. |
| Qwen3-14B | nouvelles (15 496 exemples) | 68,7 % | 85,7 % | 458 car. |
| Gemma 4 12B | nouvelles (15 496 exemples) | 70,6 % | 90,6 % | 464 car. |
| Gemma 4 31B | nouvelles (15 496 exemples) | 74,6 % | 95,0 % | 493 car. |

Réserve : l'ancien Qwen3-14B partait de la variante « base », le nouveau de la variante conversationnelle.

### 6.2 Quelle réponse donner en cible ? Quelles questions ?

Gemma 4 31B, 3 446 exemples chacun :

| Jeu d'entraînement | Complétude | Fidélité | Invention hors périmètre | Test varié : accord avec le professeur | dont questions croisées |
|---|---|---|---|---|---|
| Reformulations + réponse du professeur | 70,9 % | 87,6 % | 12 % | 61,8 % | 54,0 % |
| Reformulations + réponse officielle recopiée | 82,4 % | 95,3 % | 24 % | 49,2 % | 19,3 % |
| Questions de tous types + professeur | 66,8 % | 87,5 % | 0 % | 65,7 % | 54,7 % |

Il y a un compromis. Recopier la réponse officielle donne la meilleure restitution d'une entrée, mais le modèle ne sait
plus répondre à une question qui croise deux entrées. La variété des questions apporte les refus (0 % d'invention hors
périmètre) et la capacité à répondre précisément.

### 6.3 Le volume

Gemma 4 31B, questions de tous types :

| Exemples | Complétude | Fidélité |
|---|---|---|
| 3 446 | 66,8 % | 87,5 % |
| 15 496 | 74,6 % | 95,0 % |
| 42 770 | 77,3 % | 98,7 % |

### 6.4 Les variantes à pleine échelle

| Gemma 4 31B | Exemples | Strict | Complétude | Fidélité | Test varié : fidélité | Test varié : accord |
|---|---|---|---|---|---|---|
| A2 : tous types + professeur | 42 770 | 80,1 % | 77,3 % | 98,7 % | non mesurable | non mesurable |
| A3 : recette combinée (retenue) | 48 203 | 80,5 % | 75,7 % | 97,6 % | 96,0 % | 80,0 % |
| R1 : réponses officielles + refus seulement | 23 116 | 90,6 % | 88,6 % | 99,7 % | 93,8 % | 59,5 % |

A2 a vu les questions du test varié pendant son entraînement, d'où « non mesurable ».

R1 confirme le compromis à pleine échelle. Entraîné seulement à donner la réponse officielle aux reformulations et à
refuser ce qui sort du périmètre (25 minutes sur 4 cartes), il dépasse le modèle qui a la FAQ sous les yeux sur le test
gelé, mais il ne répond correctement qu'à 28,7 % des questions qui croisent plusieurs entrées (72,7 % pour A3).
Le choix entre A3 et R1 dépend de l'usage : assistant qui répond précisément à la question posée, ou restitution
fidèle de l'entrée de FAQ la plus proche.

### 6.5 La taille, jusqu'à 128 milliards de paramètres

| Modèle entraîné | Strict | Complétude | Fidélité | Test varié : fidélité | Durée d'entraînement |
|---|---|---|---|---|---|
| Gemma 4 31B (A3) | 80,5 % | 75,7 % | 97,6 % | 96,0 % | 1 h sur 4 cartes |
| Mistral Medium 3.5 128B | 82,8 % | 76,9 % | 99,0 % | 95,3 % | 8 h 08 sur 4 cartes |

Le 128B fait un peu mieux pour 8 fois plus de calcul. Il dépasse même son propre plafond « FAQ en contexte »
(81,0 % / 73,7 % / 92,9 %), parce qu'il a appris des réponses d'un professeur plus fidèle que lui.

## 7. Les deux modèles livrés

| | Modèle | Où |
|---|---|---|
| Meilleur modèle | Mistral Medium 3.5 128B, poids FP8 (128 Go), licence MIT modifiée de Mistral | <https://huggingface.co/fenyo/4xH100-Mistral-Medium-3.5-128B-MonEspaceSante> |
| RTX 5090 | Gemma 4 31B, GGUF 4, 5 et 6 bits (18,7 / 21,8 / 25,2 Go), Apache 2.0 | <https://huggingface.co/fenyo/4xH100-gemma-4-31B-it-MonEspaceSante-GGUF> |
| Gemma en pleine précision | modèle fusionné et adaptateur | <https://huggingface.co/fenyo/4xH100-gemma-4-31B-it-MonEspaceSante> |

La quantification 5 bits ne coûte rien de mesurable (81,1 % / 75,7 % / 97,4 % contre 80,5 % / 75,7 % / 97,6 %).
Le raisonnement des deux modèles doit être désactivé à l'usage : l'entraînement s'est fait sans.

## 8. Ce qui n'a pas marché, à savoir pour reproduire

- **Qwen3.8-27B comme élève** : son architecture d'attention linéaire s'entraîne à 260 tokens par seconde par carte dans
  transformers, contre 800 pour Gemma 4 31B. Il a servi de générateur de questions et de juge.
- **Mistral 128B comme professeur** : 366 tokens par seconde sur deux cartes, trop lent ; et moins fidèle que Gemma.
- **Mistral 128B entraîné en gardant ses poids 8 bits** : 29 tokens par seconde. Il faut les déquantifier au chargement
  (250 Go répartis sur les 4 cartes, 396 tokens par seconde). Les 4 cartes calculent alors à tour de rôle.
- **L'API OVH avec le Qwen3.5-397B** : trop lente pour générer en masse (25 requêtes par minute) et coûteuse, son
  raisonnement étant facturé en sortie. Utilisée seulement comme juge de contrôle.
- **Mes erreurs** : un filtre qui écartait toute réponse contenant le mot « document » (22 % du premier lot perdu) ; un
  taux d'apprentissage qui retombait à zéro au quart de l'entraînement avec 4 cartes ; un quota du disque en mémoire
  (585 Go) découvert quand il a fait échouer une sauvegarde du Mistral, soit une heure d'entraînement perdue. Le journal
  du dépôt détaille chaque incident et sa correction.

## 9. Durées et coûts

| Étape | Durée |
|---|---|
| Environnements, téléchargements, banc d'évaluation, plafonds | 2 h |
| Génération des questions (45 399) | 12 min |
| Réponses du professeur (48 783 questions) | 1 h 05 |
| Un entraînement de Gemma 4 31B (48 203 exemples) | 1 h |
| Six ablations et leurs évaluations | 2 h |
| Entraînement du Mistral 128B | 8 h 08 |
| Quantification GGUF de Gemma (CPU) | 15 min |

Coût : la VM a tourné 21 h 30 pour cette expérience, soit environ 241 € HT (289 € TTC) à 11,20 € HT de l'heure. L'API d'IA d'OVH (juge de contrôle) a coûté au moins 52 € HT : c'est la hausse relevée par l'API de facturation OVH depuis le début du suivi, une part du premier essai de juge n'étant pas isolable.

## 10. Limites

- Le test gelé ne contient que des reformulations ; le test varié a pour référence les réponses du professeur, pas une
  vérité indépendante.
- Le juge principal est un modèle (Qwen3.8-27B) ; il a été contrôlé par un second juge sur des échantillons seulement.
- Les ablations ont été faites une seule fois chacune : un écart d'un ou deux points n'est pas significatif.
- Les modèles ne connaissent que la FAQ à la date de sa collecte. Ce sont des modèles de démonstration.

## 11. Où trouver quoi

| Quoi | Où |
|---|---|
| Journal détaillé, scripts, résultats | <https://github.com/AlexandreFenyo/AI-Experiments-4xH100>, dossier `exp01-mes-distillation-contexte` |
| Jeux de données (questions, réponses du professeur, jeux d'entraînement et de test) | <https://huggingface.co/datasets/fenyo/4xH100-MonEspaceSante-distillation-contexte> |
| Adaptateurs des ablations | <https://huggingface.co/fenyo/4xH100-MonEspaceSante-ablations-LoRA> (A1, A2, R1 et les quatre ablations sur les données ; les adaptateurs des deux ablations de taille n'ont pas été conservés) |
| Ce rapport | Dropbox, dossier `/AI-Experiments-4xH100` |
