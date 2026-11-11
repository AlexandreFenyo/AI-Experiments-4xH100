"""Merge a LoRA adapter into Mistral Medium 3.5 128B on CPU (weights expanded to bf16, about 250 GB of RAM) and save
the merged model next to its tokenizer files, so that vLLM can serve it.

    python merge_mistral.py --adapter ~/ckpt/mistral128_m1 --out /dev/shm/work/mistral128_m1-merged
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, FineGrainedFP8Config

BASE = "mistralai/Mistral-Medium-3.5-128B"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t = time.time()
    torch.set_num_threads(96)
    q = FineGrainedFP8Config(activation_scheme="static", weight_block_size=None, dequantize=True,
                             modules_to_not_convert=["model.vision_tower", "model.multi_modal_projector", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"}, quantization_config=q)
    print(f"loaded in {(time.time() - t) / 60:.1f} min", flush=True)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    print(f"merged at {(time.time() - t) / 60:.1f} min", flush=True)
    model.save_pretrained(args.out, max_shard_size="20GB")
    # transformers re-quantizes the weights to FP8 on save (same format as the original checkpoint, about 128 GB)
    # but drops the quantization marker from config.json: restore it, and copy the small tokenizer/template files.
    src = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    cfg_path = Path(args.out) / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = json.loads((src / "config.json").read_text())["quantization_config"]
    cfg_path.write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():
        small = f.is_file() and f.stat().st_size < 100 * 2**20 and f.suffix != ".safetensors"
        if small and f.name not in ("config.json", "model.safetensors.index.json", "consolidated.safetensors.index.json")                 and not (Path(args.out) / f.name).exists():
            shutil.copy(f, Path(args.out) / f.name)
    print(f"saved to {args.out} in {(time.time() - t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
