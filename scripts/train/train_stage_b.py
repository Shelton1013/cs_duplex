"""【服务器】Stage B:打断决策训练(PLAN v1.1 §2.7)。

初始化:Stage A 的 trainable 权重(编码器顶层 + 韵律 MLP + adapter + 辅助头);LLM 新加 LoRA。
损失:决策(STOP/CONTINUE)+ 0.3 × 质疑/确认辅助。
选模型:只看 val(Stage 0 验证集说话人);clean(干净评测集)每次都报告但不参与选择,避免在评测集上调参。

用法:
  CUDA_VISIBLE_DEVICES=1 python train_stage_b.py --init /home/pxieaf/home2/ckpt/stage_a_v1/best.pt \
      --index /home/pxieaf/home2/data/stage_b --out /home/pxieaf/home2/ckpt/stage_b_v0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from slowpath.data_b import StageBData, collate_b  # noqa: E402
from slowpath.decision import Decider  # noqa: E402
from slowpath.fo_encoder import load_fo_encoder, unfreeze_top_blocks  # noqa: E402
from slowpath.model import SlowPath  # noqa: E402
from train_stage_a import specaug, to_dev  # noqa: E402


@torch.no_grad()
def evaluate(decider, dl, dev) -> dict:
    decider.m.eval()
    per = defaultdict(lambda: [0, 0])
    tot = [0, 0]
    for batch in dl:
        batch = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = decider.score(batch["fbank"], batch["fbank_len"], batch["pros"], batch["prefix"])
        pred = (s > 0).long().cpu().tolist()
        for p, y, sub in zip(pred, batch["label"], batch["subtype"]):
            per[sub][0] += int(p == y)
            per[sub][1] += 1
            tot[0] += int(p == y)
            tot[1] += 1
    decider.m.train()
    acc = {k: round(c / n, 3) for k, (c, n) in sorted(per.items())}
    bal = sum(acc.values()) / max(1, len(acc))  # 按子类宏平均,防止大类主导
    return {"acc": round(tot[0] / max(1, tot[1]), 4), "macro": round(bal, 4), "per_subtype": acc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fo_repo", default="/home/pxieaf/Freeze-Omni")
    ap.add_argument("--fo_ckpt", default="/home/share/data_makchen/peng/models/Freeze-Omni/checkpoints/audiollm")
    ap.add_argument("--llm", default="/home/pxieaf/home2/model/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr_new", type=float, default=1e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lora", type=int, default=16)
    ap.add_argument("--w_qs", type=float, default=0.3)
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(dev).eval()
    enc = load_fo_encoder(args.fo_repo, args.fo_ckpt)
    unfreeze_top_blocks(enc, 4)
    model = SlowPath(enc, llm, tok).to(dev)
    sd = torch.load(args.init, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("llm.")}  # Stage A 的 LLM LoRA(若有)不沿用
    model.load_state_dict(sd, strict=False)
    print(f"init from {args.init}: {len(sd)} tensors", flush=True)
    model.llm = get_peft_model(model.llm, LoraConfig(
        r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    decider = Decider(model)

    new_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("encoder.", "llm."))]
    enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
    lora_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("llm.")]
    print(f"trainable: new={sum(p.numel() for p in new_p) / 1e6:.1f}M enc={sum(p.numel() for p in enc_p) / 1e6:.1f}M "
          f"lora={sum(p.numel() for p in lora_p) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW([{"params": new_p, "lr": args.lr_new}, {"params": enc_p, "lr": args.lr_enc},
                             {"params": lora_p, "lr": args.lr_lora}], weight_decay=0.01)
    warm = min(200, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    idx = Path(args.index)
    train = StageBData(str(idx / "train.jsonl"), args.limit)
    val = StageBData(str(idx / "val.jsonl"), args.limit)
    clean = StageBData(str(idx / "clean.jsonl"), args.limit)
    print(f"data: train={len(train)} val={len(val)} clean={len(clean)}", flush=True)
    dl = torch.utils.data.DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate_b, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = torch.utils.data.DataLoader(val, batch_size=16, num_workers=2, collate_fn=collate_b)
    clean_dl = torch.utils.data.DataLoader(clean, batch_size=16, num_workers=2, collate_fn=collate_b)

    step, t0, best, log = 0, time.time(), -1.0, open(out / "log.jsonl", "a")
    model.train()
    while step < args.steps:
        for batch in dl:
            if args.specaug:
                batch = specaug(batch)
            batch = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = decider.loss(batch, args.w_qs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_p + enc_p + lora_p, 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                rec = {"step": step, "loss": round(float(loss), 4), **{k: round(v, 4) for k, v in parts.items()},
                       "s_per_step": round((time.time() - t0) / step, 3)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if step % args.eval_every == 0 or step == args.steps:
                ev = {"step": step, "val": evaluate(decider, val_dl, dev), "clean": evaluate(decider, clean_dl, dev)}
                print("EVAL", json.dumps(ev, ensure_ascii=False), flush=True)
                log.write(json.dumps(ev, ensure_ascii=False) + "\n")
                log.flush()
                if ev["val"]["macro"] > best:
                    best = ev["val"]["macro"]
                    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
                    torch.save(state, out / "best.pt")
                    print(f"  new best val macro={best} @ {step}", flush=True)
            if step >= args.steps:
                break
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
