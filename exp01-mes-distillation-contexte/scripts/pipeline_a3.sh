#!/bin/bash
# Final Gemma run (A3, combined recipe), its evaluations, then the Mistral 128B run on the same data.
cd ~/exp/e1/scripts
G=google/gemma-4-31B-it; T=../data/train
date; ./train_gemma.sh gemma31_a3 $T/distill_gemma31_a3.jsonl --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1
date; echo "== A3 on the frozen 704-question test (GPUs 0,1) and varied test for A3, A1 and the ablations (GPUs 2,3)"
./eval_multi.sh $G 0 1 gemma31_a3 > ~/exp/e1/logs/eval_a3.log 2>&1 &
./eval_varied.sh $G 2 3 gemma31_a3 gemma31_a1 abl_para_teacher abl_para_official abl_mixed_3k > ~/exp/e1/logs/eval_varied.log 2>&1 &
wait
date; echo "== Mistral Medium 3.5 128B, one model spread over the 4 GPUs, weights expanded to bf16"
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
~/exp/env-train/bin/python -u train_lora.py --model mistralai/Mistral-Medium-3.5-128B --family mistral --device-map auto \
  --dequantize --max-mem 65 --data $T/distill_gemma31_a3.jsonl --batch-tokens 1500 --grad-ckpt --rank 64 --lr 1e-4 --epochs 1 \
  --save-every 1000 --out /dev/shm/work/mistral128_m1 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn" > ~/exp/e1/logs/train_mistral128_m1.log
date; echo "== done"
