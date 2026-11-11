#!/bin/bash
# CPU-only (the GPUs are training Mistral): merge a Gemma adapter, convert to GGUF, quantize, then answer the frozen
# test with the Q5_K_M file through llama-server on CPU, so the quantized model can be judged later.
# usage: gguf_final.sh NAME     (adapter /dev/shm/work/NAME -> ~/gguf/NAME-*.gguf)
N=$1
export PATH=$HOME/.local/bin:$PATH HF_HOME=/dev/shm/hf CUDA_VISIBLE_DEVICES=
cd ~/exp; mkdir -p ~/gguf
date; env-train/bin/python -u e1/scripts/merge.py --model google/gemma-4-31B-it --adapter /dev/shm/work/$N --out /dev/shm/work/$N-merged 2>&1 | grep -vE "Loading weights|Warning" | tail -2
date; env-gguf/bin/python llama.cpp/convert_hf_to_gguf.py /dev/shm/work/$N-merged --outfile ~/gguf/$N-f16.gguf --outtype f16 2>&1 | tail -1
for Q in Q4_K_M Q5_K_M Q6_K; do
  date; llama.cpp/build/bin/llama-quantize ~/gguf/$N-f16.gguf ~/gguf/$N-$Q.gguf $Q 48 2>&1 | grep -E "quant size|failed"
done
rm -f ~/gguf/$N-f16.gguf; ls -la ~/gguf/
date; echo "== CPU evaluation of Q5_K_M on the frozen test"
tmux kill-session -t lls 2>/dev/null
tmux new-session -d -s lls "llama.cpp/build/bin/llama-server -m ~/gguf/$N-Q5_K_M.gguf -t 88 -np 8 -c 20000 --jinja --reasoning-budget 0 --port 8090 --alias q5 2>&1 | tee ~/exp/lls.log"
for i in $(seq 1 120); do curl -s -m 3 localhost:8090/v1/models >/dev/null && break; sleep 5; done
cd e1/scripts && ENDPOINTS=q5@8090 ~/exp/env-train/bin/python -u answer.py --tag ${N}_q5km --mode student --concurrency 8
tmux kill-session -t lls; date; echo "== done"
