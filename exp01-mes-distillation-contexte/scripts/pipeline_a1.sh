#!/bin/bash
# Run A1: train on 4 GPUs (first teacher batch, 15 496 examples), then evaluate.
cd ~/exp/e1/scripts
date; ./train_gemma.sh gemma31_a1 ../data/train/distill_gemma31_s1.jsonl --rank 64 --lr 2e-4 --batch-tokens 2500 --epochs 1
date; ./eval_adapter.sh gemma31_a1
date
