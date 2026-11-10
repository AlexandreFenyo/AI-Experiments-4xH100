#!/bin/bash
# Evaluate a Gemma LoRA adapter: student served on GPU 0 (base + adapter, full precision), Qwen judge on GPU 1.
# usage: eval_adapter.sh NAME      (adapter in /dev/shm/work/NAME, results tagged NAME)
NAME=$1
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
cd ~/exp
P="$HOME/exp/env-train/bin/python -u"
tmux kill-session -t stu 2>/dev/null; tmux kill-session -t jq 2>/dev/null
tmux new-session -d -s stu "CUDA_VISIBLE_DEVICES=0 HF_HOME=/dev/shm/hf env-vllm/bin/vllm serve google/gemma-4-31B-it --port 8010 --served-model-name gemma-base --enable-lora --max-lora-rank 256 --lora-modules stu=/dev/shm/work/$NAME --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 128 --limit-mm-per-prompt '{\"image\":0,\"audio\":0,\"video\":0}' 2>&1 | tee ~/exp/stu.log"
tmux new-session -d -s jq "~/exp/serve_qwen.sh 1 8000 2>&1 | tee ~/exp/jq.log"
for i in $(seq 1 120); do curl -s -m 3 localhost:8010/v1/models >/dev/null && curl -s -m 3 localhost:8000/v1/models >/dev/null && break; sleep 5; done
cd ~/exp/e1/scripts
ENDPOINTS=stu@8010 $P answer.py --tag $NAME --mode student
export ENDPOINTS=qwen@8000
$P judge_faith.py --tag $NAME
$P judge.py --tag $NAME
tmux kill-session -t stu; tmux kill-session -t jq
