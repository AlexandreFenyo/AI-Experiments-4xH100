#!/bin/bash
# After the Mistral training: evaluate it (frozen test, out-of-scope, general knowledge, varied test) and judge the
# quantized Gemma answers produced on CPU. Student on GPUs 0,1 (tensor parallel), Qwen judge on GPU 2.
while tmux has-session -t pipe5 2>/dev/null; do sleep 30; done
export HF_HOME=/dev/shm/hf HF_TOKEN=$(cat ~/secrets/.hf)
P="$HOME/exp/env-train/bin/python -u"; A=$HOME/ckpt/mistral128_m1
cd ~/exp
[ -f $A/adapter_config.json ] || { echo "no final adapter"; exit 1; }
COMMON="--port 8010 --tensor-parallel-size 2 --reasoning-parser mistral --max-model-len 8192 --max_num_seqs 128 --gpu_memory_utilization 0.93"
ok() { curl -s -m 120 localhost:8010/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"mistral_m1","messages":[{"role":"user","content":"Bonjour"}],"max_tokens":8,"reasoning_effort":"none"}' | grep -q '"content"'; }

date; echo "== path 1: FP8 base + LoRA adapter served by vLLM"
tmux new-session -d -s stum "CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/dev/shm/hf env-vllm/bin/vllm serve mistralai/Mistral-Medium-3.5-128B --served-model-name mistral-base $COMMON --limit-mm-per-prompt '{\"image\":0}' --enable-lora --max-lora-rank 64 --max-loras 1 --lora-modules mistral_m1=$A 2>&1 | tee ~/exp/stum.log"
tmux new-session -d -s jq2 "~/exp/serve_qwen.sh 2 8002 2>&1 | tee ~/exp/jq2.log"
for i in $(seq 1 180); do ok && break; tmux has-session -t stum 2>/dev/null || break; sleep 5; done
if ! ok; then
  date; echo "== path 1 failed, path 2: merge on CPU then serve the merged FP8 model"
  tmux kill-session -t stum 2>/dev/null
  CUDA_VISIBLE_DEVICES= $P e1/scripts/merge_mistral.py --adapter $A --out /dev/shm/work/mistral128_m1-merged 2>&1 | grep -vE "Loading weights|Warning" | tail -3
  rm -f /dev/shm/work/mistral128_m1-merged/params.json
  tmux new-session -d -s stum "CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/dev/shm/hf env-vllm/bin/vllm serve /dev/shm/work/mistral128_m1-merged --served-model-name mistral_m1 $COMMON --limit-mm-per-prompt '{\"image\":0}' --config-format hf --tokenizer-mode mistral 2>&1 | tee ~/exp/stum.log"
  for i in $(seq 1 240); do ok && break; tmux has-session -t stum 2>/dev/null || break; sleep 5; done
fi
ok || { date; echo "== student server not available"; tail -5 ~/exp/stum.log; exit 1; }
for i in $(seq 1 120); do curl -s -m 3 localhost:8002/v1/models >/dev/null && break; sleep 5; done
cd ~/exp/e1/scripts
date; ENDPOINTS=mistral_m1@8010 $P answer.py --tag mistral128_m1 --mode student --concurrency 48
ENDPOINTS=mistral_m1@8010 JUDGE=qwen@8002 $P judge_varied.py --tag mistral128_m1
tmux kill-session -t stum
export ENDPOINTS=qwen@8002
$P judge_faith.py --tag mistral128_m1; $P judge.py --tag mistral128_m1
$P judge_faith.py --tag gemma31_a3_q5km; $P judge.py --tag gemma31_a3_q5km
tmux kill-session -t jq2
date; echo "== done"
