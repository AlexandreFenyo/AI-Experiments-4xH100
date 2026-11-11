#!/bin/bash
# R1: "maximum restitution" variant of Gemma (official answers for paraphrases + refusals + general replay only).
cd ~/exp/e1/scripts
G=google/gemma-4-31B-it
date; ./train_gemma.sh gemma31_r1 ../data/train/distill_gemma31_r1.jsonl --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1
date; ./eval_multi.sh $G 0 1 gemma31_r1 > ~/exp/e1/logs/eval_r1.log 2>&1 &
./eval_varied.sh $G 2 3 gemma31_r1 > ~/exp/e1/logs/eval_varied_r1.log 2>&1 &
wait; date; echo "== done"
