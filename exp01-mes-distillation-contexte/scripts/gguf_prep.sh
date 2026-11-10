#!/bin/bash
# CPU-only: build llama.cpp, convert the merged A1 model to GGUF and quantize it (pipeline validation).
export PATH=$HOME/.local/bin:$PATH HF_HOME=/dev/shm/hf
cd ~/exp
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp && git log -1 --format="llama.cpp %h %cs"
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build -j 48 --target llama-quantize llama-cli llama-server 2>&1 | tail -2
cd ~/exp && [ -d env-gguf ] || uv venv -q --python 3.12 env-gguf
uv pip install -q --python env-gguf/bin/python -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt 2>&1 | tail -1
while tmux has-session -t merge 2>/dev/null; do sleep 10; done
date; env-gguf/bin/python llama.cpp/convert_hf_to_gguf.py /dev/shm/work/gemma31_a1-merged --outfile /dev/shm/work/gemma31_a1-f16.gguf --outtype f16 2>&1 | tail -4
date; llama.cpp/build/bin/llama-quantize /dev/shm/work/gemma31_a1-f16.gguf /dev/shm/work/gemma31_a1-Q5_K_M.gguf Q5_K_M 48 2>&1 | tail -3
date; ls -la /dev/shm/work/*.gguf
