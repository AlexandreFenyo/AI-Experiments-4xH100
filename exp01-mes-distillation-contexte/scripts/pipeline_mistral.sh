#!/bin/bash
# Mistral Medium 3.5 128B on the A3 data, restarted from scratch; snapshots go to the real disk
# (/dev/shm has a per-user quota of about 585 GB, which killed the first attempt).
cd ~/exp/e1/scripts
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
date
~/exp/env-train/bin/python -u train_lora.py --model mistralai/Mistral-Medium-3.5-128B --family mistral --device-map auto \
  --dequantize --max-mem 65 --data ../data/train/distill_gemma31_a3.jsonl --batch-tokens 1500 --grad-ckpt --rank 64 --lr 1e-4 --epochs 1 \
  --save-every 1000 --out ~/ckpt/mistral128_m1 2>&1 \
  | grep --line-buffered -vE "Loading weights|Warning|warn" > ~/exp/e1/logs/train_mistral128_m1b.log
date; echo "== done"
