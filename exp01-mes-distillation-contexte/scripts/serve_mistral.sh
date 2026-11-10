#!/bin/bash
# usage: serve_mistral.sh GPUS(e.g. 2,3) PORT TP
export HF_HOME=/dev/shm/hf CUDA_VISIBLE_DEVICES=$1
cd ~/exp
exec env-vllm/bin/vllm serve mistralai/Mistral-Medium-3.5-128B --port $2 --served-model-name mistral --tensor-parallel-size $3 --reasoning-parser mistral --max-model-len 40000 --max_num_batched_tokens 16384 --max_num_seqs 128 --gpu_memory_utilization 0.93 --limit-mm-per-prompt "{\"image\":0}"
