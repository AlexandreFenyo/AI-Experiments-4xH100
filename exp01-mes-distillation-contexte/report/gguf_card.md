---
license: apache-2.0
language:
- fr
base_model: fenyo/4xH100-gemma-4-31B-it-MonEspaceSante
pipeline_tag: text-generation
tags:
- gguf
---

# Gemma 4 31B « Mon espace santé » — GGUF (pour une carte de 32 Go, par exemple RTX 5090)

Versions quantifiées de [fenyo/4xH100-gemma-4-31B-it-MonEspaceSante](https://huggingface.co/fenyo/4xH100-gemma-4-31B-it-MonEspaceSante),
produites avec llama.cpp (commit 7592b61).

| Fichier | Taille | Quantification |
|---|---|---|
| `gemma31_a3-Q4_K_M.gguf` | 18,7 Go | 4,87 bits par poids |
| `gemma31_a3-Q5_K_M.gguf` | 21,8 Go | 5,69 bits par poids (recommandé sur 32 Go) |
| `gemma31_a3-Q6_K.gguf` | 25,2 Go | 6,56 bits par poids |

## Utilisation

llama.cpp (le raisonnement doit être désactivé, le modèle a été entraîné sans) :

```
llama-server -m gemma31_a3-Q5_K_M.gguf -ngl 99 -c 8192 --jinja --reasoning-budget 0
```

avec le prompt système `Tu es l'assistant de Mon espace santé.`

Ollama : voir le fichier `Modelfile` (`ollama create mes-gemma -f Modelfile`, puis `ollama run mes-gemma --think=false`).
