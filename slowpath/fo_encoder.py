"""加载 Freeze-Omni 的流式语音编码器(单独使用,不含其 LLM 与状态头)。

结构(源码与 train.yaml 核对):80 维 Kaldi fbank → 4 倍下采样 → 24 层 Transformer,
1024 维 @25Hz;chunk_size=4(160ms)、left_chunks=16,本身即因果流式。
前端与其推理代码一致:波形 ×32768,dither=0,25ms/10ms,80 mel。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi
import yaml


def fbank(wav: torch.Tensor) -> torch.Tensor:
    """wav: (N,) float32 [-1,1] @16k → (T, 80)。"""
    return kaldi.fbank(wav.reshape(1, -1) * 32768, dither=0.0, frame_length=25,
                       frame_shift=10, num_mel_bins=80)


def load_fo_encoder(fo_repo: str, ckpt_dir: str) -> torch.nn.Module:
    sys.path.insert(0, fo_repo)
    from models.encoder.cmvn import GlobalCMVN
    from models.encoder.encoder import speechEncoder

    cfg = yaml.safe_load(open(Path(ckpt_dir) / "train.yaml", encoding="utf-8"))
    cmvn = GlobalCMVN(torch.zeros(80), torch.ones(80))  # 均值/方差随权重从 checkpoint 载入
    enc = speechEncoder(cfg["input_dim"], global_cmvn=cmvn, **cfg["encoder_conf"])
    sd = torch.load(Path(ckpt_dir) / "final.pt", map_location="cpu")
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"FO encoder load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return enc


def unfreeze_top_blocks(enc: torch.nn.Module, n_top: int) -> int:
    """冻结编码器,只放开最上面 n_top 个 Transformer 块(按参数名中的层号识别)。返回可训练参数量。"""
    names = [n for n, _ in enc.named_parameters()]
    layer_ids = sorted({int(m.group(1)) for n in names
                        for m in [re.search(r"\.(\d+)\.", n.split("enc.1.", 1)[-1])] if m and "enc.1." in n})
    top = set(layer_ids[-n_top:]) if n_top > 0 and layer_ids else set()
    total = 0
    for n, p in enc.named_parameters():
        m = re.search(r"\.(\d+)\.", n.split("enc.1.", 1)[-1]) if "enc.1." in n else None
        train = bool(m and int(m.group(1)) in top) or ("enc.1." in n and "norm" in n and not m)
        p.requires_grad = train
        total += p.numel() if train else 0
    return total
