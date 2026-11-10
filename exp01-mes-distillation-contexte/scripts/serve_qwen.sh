#!/bin/bash
# usage: serve_qwen.sh GPU PORT [model]
export HF_HOME=/dev/shm/hf CUDA_VISIBLE_DEVICES=$1
cd ~/exp
exec env-vllm/bin/vllm serve ${3:-Qwen/Qwen3.8-27B-FP8} --port $2 --served-model-name qwen --max-model-len 40000 --gpu-memory-utilization 0.92 --max-num-seqs 256 --limit-mm-per-prompt "{\"image\":0,\"video\":0}"
