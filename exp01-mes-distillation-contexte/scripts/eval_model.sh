#!/bin/bash
# Evaluate a merged student model closed-book: answers on GPUs 2,3 (full precision), judges on GPUs 0,1 (Qwen FP8).
# usage: eval_model.sh TAG MODEL_DIR
set -u
TAG=$1; MODEL=$2
P="$HOME/exp/env-train/bin/python -u"
cd ~/exp/e1/scripts
tmux kill-session -t =evalsrv 2>/dev/null; tmux kill-session -t =jq0 2>/dev/null; tmux kill-session -t =jq1 2>/dev/null
tmux new-session -d -s evalsrv "~/exp/serve_gemma.sh 2,3 8003 2 $MODEL 2>&1 | tee ~/exp/evalsrv.log"
tmux new-session -d -s jq0 "~/exp/serve_qwen.sh 0 8000 2>&1 | tee ~/exp/q0.log"
tmux new-session -d -s jq1 "~/exp/serve_qwen.sh 1 8001 2>&1 | tee ~/exp/q1.log"
for port in 8003 8000 8001; do
  for i in $(seq 1 120); do curl -s -m 3 localhost:$port/v1/models >/dev/null && break; sleep 5; done
done
ENDPOINTS=gemma@8003 $P answer.py --tag $TAG --mode student
export ENDPOINTS=qwen@8000,qwen@8001
$P judge_faith.py --tag $TAG
$P judge.py --tag $TAG
tmux kill-session -t =evalsrv; tmux kill-session -t =jq0; tmux kill-session -t =jq1
echo "EVAL DONE $TAG"
