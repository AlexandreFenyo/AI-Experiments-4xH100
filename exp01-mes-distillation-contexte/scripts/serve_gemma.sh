#!/bin/bash
# usage: serve_gemma.sh GPUS PORT TP [model]   (full precision bf16)
export HF_HOME=/dev/shm/hf CUDA_VISIBLE_DEVICES=$1
cd ~/exp
exec env-vllm/bin/vllm serve ${4:-google/gemma-4-31B-it} --port $2 --tensor-parallel-size $3 --served-model-name gemma --max-model-len 32000 --gpu-memory-utilization 0.93 --max-num-seqs 128 --limit-mm-per-prompt "{\"image\":0,\"audio\":0,\"video\":0}"
