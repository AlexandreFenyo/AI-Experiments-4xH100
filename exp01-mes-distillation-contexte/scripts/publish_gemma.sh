#!/bin/bash
# Publish the final Gemma model: merged weights, LoRA adapter, GGUF files (public Hugging Face repositories).
export HF_TOKEN=$(cat ~/secrets/.hf) HF_HOME=/dev/shm/hf HF_XET_HIGH_PERFORMANCE=1
cd ~/exp; H=env-dl/bin/hf
cp gemma_card.md /dev/shm/work/gemma31_a3-merged/README.md
$H upload fenyo/4xH100-gemma-4-31B-it-MonEspaceSante /dev/shm/work/gemma31_a3-merged . --commit-message "Modele fusionne (entrainement A3)" 2>&1 | tail -1
mkdir -p /dev/shm/work/pub_gguf && cp gguf_card.md /dev/shm/work/pub_gguf/README.md && cp Modelfile /dev/shm/work/pub_gguf/
$H upload fenyo/4xH100-gemma-4-31B-it-MonEspaceSante-GGUF /dev/shm/work/pub_gguf . --commit-message "Carte et Modelfile" 2>&1 | tail -1
for q in Q4_K_M Q5_K_M Q6_K; do $H upload fenyo/4xH100-gemma-4-31B-it-MonEspaceSante-GGUF ~/gguf/gemma31_a3-$q.gguf gemma31_a3-$q.gguf --commit-message "GGUF $q" 2>&1 | tail -1; done
$H upload fenyo/4xH100-gemma-4-31B-it-MonEspaceSante /dev/shm/work/gemma31_a3 lora-adapter --commit-message "Adaptateur LoRA de rang 64" 2>&1 | tail -1
date; echo "== done"
