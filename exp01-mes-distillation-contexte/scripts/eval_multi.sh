#!/bin/bash
# Evaluate several LoRA adapters of the same base model: one student server (base + all adapters, full precision)
# and one Qwen3.8-27B judge server.
# usage: eval_multi.sh BASE_MODEL STUDENT_GPU JUDGE_GPU NAME1 [NAME2 ...]    (adapters in /dev/shm/work/NAME)
BASE=$1; SGPU=$2; JGPU=$3; shift 3
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
P="$HOME/exp/env-train/bin/python -u"; SP=$((8010 + SGPU)); JP=$((8000 + JGPU))
MODS=""; for n in "$@"; do MODS="$MODS $n=/dev/shm/work/$n"; done
cd ~/exp
tmux kill-session -t stu$SGPU 2>/dev/null; tmux kill-session -t jq$JGPU 2>/dev/null
tmux new-session -d -s stu$SGPU "CUDA_VISIBLE_DEVICES=$SGPU HF_HOME=/dev/shm/hf env-vllm/bin/vllm serve $BASE --port $SP --served-model-name base --enable-lora --max-lora-rank 256 --max-loras 8 --lora-modules $MODS --max-model-len 8192 --gpu-memory-utilization 0.93 --max-num-seqs 128 --limit-mm-per-prompt '{\"image\":0,\"audio\":0,\"video\":0}' 2>&1 | tee ~/exp/stu$SGPU.log"
tmux new-session -d -s jq$JGPU "~/exp/serve_qwen.sh $JGPU $JP 2>&1 | tee ~/exp/jq$JGPU.log"
for i in $(seq 1 150); do curl -s -m 3 localhost:$SP/v1/models >/dev/null && curl -s -m 3 localhost:$JP/v1/models >/dev/null && break; sleep 5; done
cd ~/exp/e1/scripts
for n in "$@"; do ENDPOINTS=$n@$SP $P answer.py --tag $n --mode student; done
export ENDPOINTS=qwen@$JP
for n in "$@"; do $P judge_faith.py --tag $n; $P judge.py --tag $n; done
tmux kill-session -t stu$SGPU; tmux kill-session -t jq$JGPU
