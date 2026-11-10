#!/bin/bash
# 4-GPU data-parallel LoRA training. usage: run_train.sh NAME MODEL FAMILY DATA [extra train_lora.py args...]
NAME=$1; MODEL=$2; FAMILY=$3; DATA=$4; shift 4
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
cd ~/exp/e1/scripts
~/exp/env-train/bin/accelerate launch --num_processes 4 --mixed_precision bf16 train_lora.py \
  --model $MODEL --family $FAMILY --data $DATA --out /dev/shm/work/$NAME "$@" 2>&1 \
  | grep --line-buffered -vE "Loading weights|Warning" | tee ~/exp/e1/logs/train_$NAME.log
