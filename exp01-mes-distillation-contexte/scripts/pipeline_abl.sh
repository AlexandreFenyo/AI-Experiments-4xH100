#!/bin/bash
# After A2: Mistral 128B smoke test, then the ablation runs that quantify each ingredient of the recipe.
while tmux has-session -t pipe2 2>/dev/null; do sleep 10; done
cd ~/exp/e1/scripts
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
G=google/gemma-4-31B-it; T=../data/train
date; echo "== Mistral 128B smoke test"
timeout 1500 ~/exp/env-train/bin/python -u train_lora.py --model mistralai/Mistral-Medium-3.5-128B --family mistral --device-map auto \
  --data $T/distill_gemma31_s1.jsonl --limit 600 --max-iters 16 --batch-tokens 4000 --grad-ckpt --rank 64 \
  --out /dev/shm/work/mistral_smoke 2>&1 | grep --line-buffered -vE "Loading weights|Warning|warn" > ~/exp/e1/logs/mistral_smoke.log
sleep 10
date; echo "== Batch B: data ablations on Gemma 31B (one GPU each)"
./train_one.sh abl_para_teacher  $G gemma 0 $T/abl_para_teacher.jsonl  --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1 &
./train_one.sh abl_para_official $G gemma 1 $T/abl_para_official.jsonl --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1 &
./train_one.sh abl_mixed_3k      $G gemma 2 $T/abl_mixed_3k.jsonl      --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1 &
./train_one.sh abl_oldsft        $G gemma 3 $T/abl_oldsft.jsonl        --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 10 &
wait
date; echo "== Batch C: size ablation (same data and optimisation as A1) on GPUs 2,3, while batch B is evaluated on GPUs 0,1"
./train_one.sh size_gemma12 google/gemma-4-12B-it gemma 2 $T/distill_gemma31_s1.jsonl --rank 64 --lr 2e-4 --batch-tokens 10000 --epochs 1 &
./train_one.sh size_qwen14  Qwen/Qwen3-14B        qwen  3 $T/distill_gemma31_s1.jsonl --rank 64 --lr 2e-4 --batch-tokens 10000 --epochs 1 &
./eval_multi.sh $G 0 1 abl_para_teacher abl_para_official abl_mixed_3k abl_oldsft > ~/exp/e1/logs/eval_ablB.log 2>&1
wait
date; echo "== Evaluation of batch C"
./eval_multi.sh google/gemma-4-12B-it 0 1 size_gemma12 > ~/exp/e1/logs/eval_size_gemma12.log 2>&1 &
./eval_multi.sh Qwen/Qwen3-14B 2 3 size_qwen14 > ~/exp/e1/logs/eval_size_qwen14.log 2>&1 &
wait
date; echo "== done"
