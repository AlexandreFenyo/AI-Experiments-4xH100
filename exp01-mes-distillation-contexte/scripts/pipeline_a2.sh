#!/bin/bash
# After the second teacher batch: general-knowledge replay, Mistral 128B smoke test, Gemma run A2 on all data, evaluation.
while tmux has-session -t teach2 2>/dev/null; do sleep 10; done
cd ~/exp/e1/scripts
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
date; ENDPOINTS=gemma@8002,gemma@8003 ~/exp/env-train/bin/python -u gen_general.py --tag gemma31 --n 2400
tmux kill-session -t g01 2>/dev/null; tmux kill-session -t g23 2>/dev/null; sleep 15
cat ../data/train/distill_gemma31_s1.jsonl ../data/train/distill_gemma31_s2.jsonl ../data/train/general_gemma31.jsonl | shuf --random-source=<(yes) > ../data/train/distill_gemma31_all.jsonl
wc -l ../data/train/*.jsonl
date; echo "== Mistral 128B smoke test (one model spread over the 4 GPUs)"
timeout 1500 ~/exp/env-train/bin/python -u train_lora.py --model mistralai/Mistral-Medium-3.5-128B --family mistral --device-map auto \
  --data ../data/train/distill_gemma31_s1.jsonl --limit 600 --max-iters 16 --batch-tokens 4000 --grad-ckpt --rank 64 \
  --out /dev/shm/work/mistral_smoke 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn" | tee ~/exp/e1/logs/mistral_smoke.log
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; sleep 10
date; ./train_gemma.sh gemma31_a2 ../data/train/distill_gemma31_all.jsonl --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1
date; ./eval_adapter.sh gemma31_a2
date
