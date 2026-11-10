"""Merge a LoRA adapter into its base model (on CPU, the machine has 1.4 TB of RAM) and save full weights.

    python merge.py --model google/gemma-4-31B-it --adapter /dev/shm/work/a --out /dev/shm/work/a-merged
"""

import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    t = time.time()
    torch.set_num_threads(96)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    try:                       # multimodal models also need their processor files to be served
        AutoProcessor.from_pretrained(args.model).save_pretrained(args.out)
    except Exception as e:     # noqa: BLE001
        print("no processor saved:", e)
    print(f"merged model saved to {args.out} in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
