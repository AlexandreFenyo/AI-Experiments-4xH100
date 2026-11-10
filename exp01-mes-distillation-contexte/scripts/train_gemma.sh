#!/bin/bash
# Train a Gemma 4 31B LoRA on the 4 GPUs (data-parallel, one model copy per GPU).
# usage: train_gemma.sh NAME DATA.jsonl [extra train_lora.py args]
NAME=$1; DATA=$2; shift 2
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf) TOKENIZERS_PARALLELISM=false
cd ~/exp/e1/scripts
~/exp/env-train/bin/accelerate launch --num_processes 4 --mixed_precision bf16 train_lora.py \
  --model google/gemma-4-31B-it --family gemma --data "$DATA" --out /dev/shm/work/$NAME --grad-ckpt "$@" \
  2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn" | tee ~/exp/e1/logs/train_$NAME.log
