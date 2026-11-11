#!/bin/bash
# CPU: merge the final Mistral adapter (re-quantized to FP8 on save) and publish merged model + adapter.
export HF_TOKEN=$(cat ~/secrets/.hf) HF_HOME=/dev/shm/hf HF_XET_HIGH_PERFORMANCE=1
cd ~/exp
date; CUDA_VISIBLE_DEVICES= env-train/bin/python -u e1/scripts/merge_mistral.py --adapter ~/ckpt/mistral128_m1 --out /dev/shm/work/mistral128_m1-merged 2>&1 | grep -vE "Loading weights|Warning" | tail -3
cp mistral_card.md /dev/shm/work/mistral128_m1-merged/README.md
cp /dev/shm/hf/hub/models--mistralai--Mistral-Medium-3.5-128B/snapshots/*/LICENSE /dev/shm/work/mistral128_m1-merged/ 2>/dev/null
date; env-dl/bin/hf upload fenyo/4xH100-Mistral-Medium-3.5-128B-MonEspaceSante /dev/shm/work/mistral128_m1-merged . --commit-message "Modele fusionne (FP8)" 2>&1 | tail -1
env-dl/bin/hf upload fenyo/4xH100-Mistral-Medium-3.5-128B-MonEspaceSante ~/ckpt/mistral128_m1 lora-adapter --commit-message "Adaptateur LoRA de rang 64" 2>&1 | tail -1
date; echo "== done"
