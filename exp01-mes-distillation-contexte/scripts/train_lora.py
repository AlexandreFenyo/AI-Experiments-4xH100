"""LoRA fine-tuning for context distillation (hard labels), single or multi-GPU (accelerate launch).

Data rows: {"messages": [...]} (loss on the last assistant message only) or {"text": "..."} (plain LM loss, CPT-style).

    CUDA_VISIBLE_DEVICES=1 python train_lora.py --model Qwen/Qwen3.8-27B --data ../data/train/a.jsonl --out /dev/shm/work/a
    accelerate launch --num_processes 4 train_lora.py ...
"""

import argparse
import json
import math
import random
import time

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

TARGETS = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"],
    # regex: language model only (skip the vision/audio towers)
    "gemma": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    "mistral": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
}
END = {"qwen": "<|im_end|>", "gemma": "<turn|>", "mistral": "</s>"}   # end-of-answer token taught to the student


def encode(tok, row, max_len, tpl_kw, end):
    """Return (input_ids, labels) with -100 on everything except the supervised span."""
    if "text" in row:
        ids = tok(row["text"], add_special_tokens=True).input_ids[:max_len]
        return ids, list(ids)
    msgs = row["messages"]
    answer = msgs[-1]["content"].strip()
    if end == "</s>":
        # Mistral (mistral-common tokenizer): tokenize the prompt directly; "</s>" is not parsed from text.
        out = tok.apply_chat_template(msgs[:-1], tokenize=True, add_generation_prompt=True)
        prefix = list(out if isinstance(out, list) else out["input_ids"])
        target = tok(answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
    else:
        text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **tpl_kw)
        prefix = tok(text, add_special_tokens=False).input_ids
        target = tok(answer + end, add_special_tokens=False).input_ids
    ids = (list(prefix) + target)[:max_len]
    labels = ([-100] * len(prefix) + target)[:max_len]
    return ids, labels


def make_batches(examples, max_tokens, seed):
    """Length-bucketed batches bounded by padded token count."""
    rnd = random.Random(seed)
    idx = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches, cur, longest = [], [], 0
    for i in idx:
        n = len(examples[i][0])
        if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    if cur:
        batches.append(cur)
    rnd.shuffle(batches)
    return batches


def collate(examples, batch, pad_id):
    n = max(len(examples[i][0]) for i in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for b, i in enumerate(batch):
        x, y = examples[i]
        ids[b, : len(x)] = torch.tensor(x)
        lab[b, : len(y)] = torch.tensor(y)
        att[b, : len(x)] = 1
    return ids, att, lab


def chunked_lm_loss(hf_model, ids, att, lab, chunk=768):
    """Cross-entropy on supervised positions only, with the LM head applied by chunks (and re-computed during
    backward via checkpointing): avoids materialising B x T x vocab logits (vocab is 130k-260k)."""
    hs = hf_model.model(input_ids=ids, attention_mask=att).last_hidden_state
    mask = lab[:, 1:] != -100                 # next-token prediction: hidden state t predicts token t+1
    h, y = hs[:, :-1][mask], lab[:, 1:][mask]
    head = hf_model.get_output_embeddings()
    cfg = hf_model.config.get_text_config() if hasattr(hf_model.config, "get_text_config") else hf_model.config
    cap = getattr(cfg, "final_logit_softcapping", None)   # Gemma caps its logits with tanh

    def part(hc, yc):
        logits = head(hc).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return torch.nn.functional.cross_entropy(logits, yc, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for a in range(0, h.shape[0], chunk):
        total = total + torch.utils.checkpoint.checkpoint(part, h[a:a + chunk], y[a:a + chunk], use_reentrant=False)
    return total / max(1, y.numel())


class Distill(torch.nn.Module):
    """Wraps the PEFT model so that the whole loss is computed inside forward(): required for multi-GPU (DDP)."""

    def __init__(self, peft_model):
        super().__init__()
        self.peft = peft_model

    def forward(self, ids, att, lab):
        return chunked_lm_loss(self.peft.base_model.model, ids, att, lab)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--family", choices=list(TARGETS), default="qwen")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--alpha", type=int, default=0, help="default: = rank")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-tokens", type=int, default=3000, help="padded tokens per batch per process")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device-map", default="", help="'auto' to spread one model over all visible GPUs (single process)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    acc = Accelerator(gradient_accumulation_steps=args.accum, mixed_precision="bf16")
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    tpl_kw = {"enable_thinking": False} if args.family == "qwen" else {}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    examples = [encode(tok, r, args.max_len, tpl_kw, END[args.family]) for r in rows]
    batches = make_batches(examples, args.batch_tokens, args.seed)
    n_proc = acc.num_processes
    steps_per_epoch = len(batches) // n_proc                   # same number of steps in every process (DDP)
    my_batches = batches[acc.process_index :: n_proc][:steps_per_epoch]   # static sharding across processes
    total_steps = math.ceil(steps_per_epoch * args.epochs / args.accum)
    if acc.is_main_process:
        ntok = sum(len(x) for x, _ in examples)
        nsup = sum(sum(t != -100 for t in y) for _, y in examples)
        print(f"{len(examples)} examples, {ntok} tokens ({nsup} supervised), {len(batches)} batches, "
              f"{n_proc} proc, {total_steps} optimizer steps", flush=True)

    kw = dict(dtype=torch.bfloat16)
    kw["device_map"] = "auto" if args.device_map == "auto" else {"": acc.device}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha or args.rank, lora_dropout=0.0, bias="none",
                     target_modules=TARGETS[args.family], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if acc.is_main_process:
        model.print_trainable_parameters()

    model = Distill(model)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.03 * total_steps)), total_steps)
    if args.device_map == "auto":
        opt = acc.prepare(opt)
        dev = "cuda:0"
    else:
        # The scheduler is NOT passed to accelerate: its wrapper steps it once per process, which made the
        # learning rate decay num_processes times too fast.
        model, opt = acc.prepare(model, opt)
        dev = acc.device
    model.train()

    t0, seen_tok, step, ema = time.time(), 0, 0, None
    n_iter = math.ceil(steps_per_epoch * args.epochs)
    for it in range(n_iter):
        if it % steps_per_epoch == 0 and it:
            if acc.is_main_process:   # keep an adapter per completed epoch
                acc.unwrap_model(model).save_pretrained(f"{args.out}-ep{it // steps_per_epoch}")
            random.Random(args.seed + it).shuffle(my_batches)
        ids, att, lab = collate(examples, my_batches[it % steps_per_epoch], pad_id)
        with acc.accumulate(model):
            with acc.autocast():
                loss_t = model(ids.to(dev), att.to(dev), lab.to(dev))
            acc.backward(loss_t)
            if acc.sync_gradients:
                acc.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        seen_tok += int(att.sum())
        loss = loss_t.item()
        ema = loss if ema is None else 0.98 * ema + 0.02 * loss
        if acc.is_main_process and (it % 25 == 0 or it == n_iter - 1):
            el = time.time() - t0
            print(f"it {it + 1}/{n_iter} loss {loss:.4f} ema {ema:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{seen_tok * n_proc / el:.0f} tok/s  elapsed {el / 60:.1f} min  "
                  f"eta {el / (it + 1) * (n_iter - it - 1) / 60:.1f} min  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)

    acc.wait_for_everyone()
    if acc.is_main_process:
        acc.unwrap_model(model).peft.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"saved adapter to {args.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
