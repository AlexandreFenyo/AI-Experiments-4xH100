#!/bin/bash
# Generic LoRA run. usage: train_one.sh NAME MODEL FAMILY GPUS(e.g. "0" or "0,1") DATA [extra train_lora.py args]
NAME=$1; MODEL=$2; FAMILY=$3; GPUS=$4; DATA=$5; shift 5
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf) TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPUS
N=$(echo "$GPUS" | tr "," "\n" | wc -l)
cd ~/exp/e1/scripts
if [ "$N" -gt 1 ]; then RUN="$HOME/exp/env-train/bin/accelerate launch --num_processes $N --mixed_precision bf16"
else RUN="$HOME/exp/env-train/bin/accelerate launch --num_processes 1 --mixed_precision bf16"; fi
$RUN train_lora.py --model "$MODEL" --family "$FAMILY" --data "$DATA" --out /dev/shm/work/$NAME --grad-ckpt "$@" \
  2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn" > ~/exp/e1/logs/train_$NAME.log
