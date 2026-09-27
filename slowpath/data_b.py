"""Stage B 数据集:按因果截断点裁剪音频(只保留截至判定时刻的部分)。"""
from __future__ import annotations

import json

import torch

from .data import load_wav
from .fo_encoder import fbank
from .prosody import prosody_25hz

SR = 16000


class StageBData(torch.utils.data.Dataset):
    def __init__(self, index_jsonl: str, limit: int = 0):
        self.items = [json.loads(l) for l in open(index_jsonl, encoding="utf-8")]
        if limit:
            import random
            random.Random(0).shuffle(self.items)
            self.items = self.items[:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        w = load_wav(it["audio"])
        # 截断点从首个有声处算起(TTS 前导静音不计入"已听到的时长")
        import numpy as np
        idx = np.where(np.abs(w) > 0.01)[0]
        start = int(idx[0]) if len(idx) else 0
        if it["cut"] is not None:
            w = w[: start + int(it["cut"] * SR)]
        w = w[:int(15 * SR)]
        return {"fbank": fbank(torch.from_numpy(np.ascontiguousarray(w))),
                "pros": torch.from_numpy(prosody_25hz(w)),
                "prefix": it["prefix"], "label": it["label"], "qs": it["qs"],
                "subtype": it["subtype"]}


def collate_b(batch: list[dict]) -> dict:
    T = max(b["fbank"].shape[0] for b in batch)
    P = max(b["pros"].shape[0] for b in batch)
    fb = torch.zeros(len(batch), T, 80)
    pr = torch.zeros(len(batch), P, 8)
    for i, b in enumerate(batch):
        fb[i, : b["fbank"].shape[0]] = b["fbank"]
        pr[i, : b["pros"].shape[0]] = b["pros"]
    return {"fbank": fb, "fbank_len": torch.tensor([b["fbank"].shape[0] for b in batch]),
            "pros": pr, "prefix": [b["prefix"] for b in batch],
            "label": [b["label"] for b in batch], "qs": [b["qs"] for b in batch],
            "subtype": [b["subtype"] for b in batch]}
