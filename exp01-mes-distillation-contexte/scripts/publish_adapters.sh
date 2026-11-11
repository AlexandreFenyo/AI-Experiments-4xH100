#!/bin/bash
# Publish the LoRA adapters of the intermediate runs and ablations (public).
export HF_TOKEN=$(cat ~/secrets/.hf) HF_HOME=/dev/shm/hf
cd ~/exp
printf -- "---\nlicense: apache-2.0\nbase_model: google/gemma-4-31B-it\nlanguage:\n- fr\n---\n\n# Adaptateurs LoRA des entrainements intermediaires et des ablations\n\nExperience 01 de https://github.com/AlexandreFenyo/AI-Experiments-4xH100 (voir le journal pour la definition de chaque entrainement).\nTous sont des adaptateurs de rang 64 pour google/gemma-4-31B-it : gemma31_a1, gemma31_a2, gemma31_r1, abl_para_teacher, abl_para_official, abl_mixed_3k, abl_oldsft.\n" > /tmp/abl_readme.md
env-dl/bin/hf upload fenyo/4xH100-MonEspaceSante-ablations-LoRA /tmp/abl_readme.md README.md 2>&1 | tail -1
for n in "$@"; do env-dl/bin/hf upload fenyo/4xH100-MonEspaceSante-ablations-LoRA /dev/shm/work/$n $n --commit-message "Adaptateur $n" 2>&1 | tail -1; done
echo "== done"
