"""【服务器】Stage A 训练:模态对齐 + 韵律保持(PLAN v1.1 §2.7)。

可训练:韵律 MLP、adapter、声调头、质疑/确认头(lr_new);FO 编码器最上面 n_top 块(lr_enc)。
冻结:FO 编码器其余部分、Qwen3-1.7B 全部。
评测(每 eval_every 步):MCE dev 子集损失 + 贪心解码 MER;Stage 0 验证集说话人的质疑/确认准确率。

环境:/home/pxieaf/home2/envs/qwen3omni(PYTHONNOUSERSITE=1)
用法:
  CUDA_VISIBLE_DEVICES=0 python train_stage_a.py --out /home/pxieaf/home2/ckpt/stage_a_v0
  冒烟:--steps 20 --eval_every 10 --limit 64
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slowpath.data import StageAData, collate  # noqa: E402
from slowpath.fo_encoder import load_fo_encoder, unfreeze_top_blocks  # noqa: E402
from slowpath.model import SlowPath  # noqa: E402

MCE = "/home/share/data_makchen/peng/datasets/MCE/dealed_mce"


def mer(ref: str, hyp: str) -> tuple[int, int]:
    """混合错误率:中文按字、英文按词切分后的编辑距离。"""
    tok = lambda s: [t for t in __import__("re").findall(r"[A-Za-z']+|\d+|[^\sA-Za-z\d\W]", s.lower())]
    r, h = tok(ref), tok(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], max(1, len(r))


def to_dev(batch, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}


def specaug(batch: dict, n_t: int = 2, max_t: int = 20, n_f: int = 2, max_f: int = 10) -> dict:
    """SpecAugment(只作用于 fbank,不动韵律旁路;v0 在 3.5k 步后过拟合)。"""
    fb = batch["fbank"].clone()
    for b in range(fb.shape[0]):
        T = int(batch["fbank_len"][b])
        for _ in range(n_t):
            w = random.randint(0, max_t)
            t0 = random.randint(0, max(0, T - w))
            fb[b, t0:t0 + w] = 0
        for _ in range(n_f):
            w = random.randint(0, max_f)
            f0 = random.randint(0, 80 - w)
            fb[b, :, f0:f0 + w] = 0
    return {**batch, "fbank": fb}


@torch.no_grad()
def evaluate(model, dev_dl, qs_dl, dev, n_decode=16):
    model.eval()
    losses, errs, n_tok, shown = [], 0, 0, []
    for bi, batch in enumerate(dev_dl):
        batch = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, parts = model(batch)
            losses.append(parts)
            if bi * dev_dl.batch_size < n_decode:
                hyps = model.transcribe(batch["fbank"], batch["fbank_len"], batch["pros"])
        if bi * dev_dl.batch_size < n_decode:
            for ref, hyp in zip(batch["text"], hyps):
                e, n = mer(ref, hyp)
                errs, n_tok = errs + e, n_tok + n
                if len(shown) < 4:
                    shown.append((ref, hyp))
    correct = total = 0
    for batch in qs_dl:
        batch = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, valid = model.frames(batch["fbank"], batch["fbank_len"], batch["pros"])
            pred = (model.qs_logits(z, valid) > 0).long().cpu().tolist()
        correct += sum(int(p == y) for p, y in zip(pred, batch["qs"]))
        total += len(pred)
    model.train()
    avg = {k: sum(d.get(k, 0) for d in losses) / max(1, sum(k in d for d in losses))
           for k in ("asr", "tone", "qs")}
    return {"dev_" + k: round(v, 4) for k, v in avg.items()} | {
        "dev_mer": round(errs / max(1, n_tok), 4), "qs_acc_val_spk": round(correct / max(1, total), 4),
        "qs_n": total, "examples": shown}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mce_train", default=f"{MCE}/train.jsonl")
    ap.add_argument("--mce_dev", default=f"{MCE}/dev.jsonl")
    ap.add_argument("--stage0", default="/home/pxieaf/home2/data/stage0")
    ap.add_argument("--fo_repo", default="/home/pxieaf/Freeze-Omni")
    ap.add_argument("--fo_ckpt", default="/home/share/data_makchen/peng/models/Freeze-Omni/checkpoints/audiollm")
    ap.add_argument("--llm", default="/home/pxieaf/home2/model/Qwen3-1.7B")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr_new", type=float, default=2e-4)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--n_top", type=int, default=4)
    ap.add_argument("--w_tone", type=float, default=0.3)
    ap.add_argument("--w_qs", type=float, default=0.3)
    ap.add_argument("--qs_repeat", type=int, default=3, help="语调样本重复次数(量少,平衡采样)")
    ap.add_argument("--extra_jsonl", nargs="*", default=[], help="额外训练数据(如伪标签 pseudo.jsonl)")
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--init", default="", help="从已有 trainable.pt 继续(如 v0)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.llm)
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.bfloat16).to(dev).eval()
    enc = load_fo_encoder(args.fo_repo, args.fo_ckpt)
    n_enc = unfreeze_top_blocks(enc, args.n_top)
    model = SlowPath(enc, llm, tok).to(dev)
    model.llm.to(torch.bfloat16)
    new_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and not n.startswith(("encoder.", "llm."))]
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    print(f"trainable: new={sum(p.numel() for p in new_params) / 1e6:.1f}M "
          f"encoder_top={n_enc / 1e6:.1f}M (n_top={args.n_top})", flush=True)

    opt = torch.optim.AdamW([{"params": new_params, "lr": args.lr_new},
                             {"params": enc_params, "lr": args.lr_enc}], weight_decay=0.01)
    warm = min(500, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    if args.init:
        sd = torch.load(args.init, map_location="cpu")
        missing = model.load_state_dict(sd, strict=False)
        print(f"init from {args.init}: loaded {len(sd)} tensors", flush=True)
    train = StageAData([args.mce_train] + args.extra_jsonl, args.stage0, "train",
                       limit=args.limit, qs_repeat=args.qs_repeat)
    dev_set = StageAData(args.mce_dev, None, limit=args.limit or 200)
    qs_val = StageAData(None, args.stage0, "val", limit=args.limit or 0)
    print(f"data: train={len(train)} dev={len(dev_set)} qs_val={len(qs_val)}", flush=True)
    dl = torch.utils.data.DataLoader(train, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    dev_dl = torch.utils.data.DataLoader(dev_set, batch_size=8, num_workers=2, collate_fn=collate)
    qs_dl = torch.utils.data.DataLoader(qs_val, batch_size=16, num_workers=2, collate_fn=collate)

    step, t0, log, best = 0, time.time(), open(out / "log.jsonl", "a"), float("inf")
    model.train()
    model.llm.eval()
    while step < args.steps:
        for batch in dl:
            if args.specaug:
                batch = specaug(batch)
            batch = to_dev(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model(batch, args.w_tone, args.w_qs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(new_params + enc_params, 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                rec = {"step": step, "loss": round(float(loss), 4), **{k: round(v, 4) for k, v in parts.items()},
                       "lr": sched.get_last_lr()[0], "s_per_step": round((time.time() - t0) / step, 3)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, dev_dl, qs_dl, dev)
                print("EVAL", json.dumps({"step": step, **ev}, ensure_ascii=False), flush=True)
                log.write(json.dumps({"step": step, "eval": ev}, ensure_ascii=False) + "\n")
                log.flush()
                state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
                torch.save(state, out / "trainable.pt")
                if ev["dev_asr"] < best:  # 按验证集 ASR 损失保留最佳(v0 后期过拟合)
                    best = ev["dev_asr"]
                    torch.save(state, out / "best.pt")
                    print(f"  new best dev_asr={best} @ step {step}", flush=True)
            if step >= args.steps:
                break
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
