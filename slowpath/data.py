"""Stage A 数据:MCE 转写(ASR + 声调 CTC)与 Stage 0 语调最小对(质疑/确认)。

声调标签:pycantonese 把转写转成粤拼,取每个音节末尾的声调数字(1–6);
英文词、数字等无粤拼的词记为 7(每个词一个 token);标点跳过。
语调样本只取 Stage 0 中 split=train 的说话人,val 说话人留给评测。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .fo_encoder import fbank
from .prosody import prosody_25hz

SR = 16000
PUNCT = re.compile(r"^[\W_]+$")


def text_to_tones(text: str) -> list[int]:
    import pycantonese as pc
    tones: list[int] = []
    for word, jp in pc.characters_to_jyutping(text):
        if PUNCT.match(word):
            continue
        if jp:
            tones += [int(s[-1]) for s in jp.split() if s[-1].isdigit() and 1 <= int(s[-1]) <= 6]
        else:
            tones.append(7)
    return tones


def load_wav(path: str) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    return w


class StageAData(torch.utils.data.Dataset):
    def __init__(self, mce_jsonl: str | None, stage0_dir: str | None, stage0_split: str = "train",
                 max_sec: float = 15.0, limit: int = 0, qs_repeat: int = 1):
        self.items = []
        if mce_jsonl:
            for line in open(mce_jsonl, encoding="utf-8"):
                r = json.loads(line)
                self.items.append({"audio": r["audio"], "text": r["text"], "qs": -1})
        if stage0_dir:
            for lab in sorted(Path(stage0_dir).glob("shard*/labels.jsonl")):
                for line in open(lab, encoding="utf-8"):
                    r = json.loads(line)
                    if r["task"] == "intonation" and r["split"] == stage0_split:
                        p = lab.parent / "intonation" / f"{r['id']}.wav"
                        self.items += [{"audio": str(p), "text": "", "qs": r["label"]}] * qs_repeat
        if limit:
            random.Random(0).shuffle(self.items)
            self.items = self.items[:limit]
        self.max_sec = max_sec

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        w = load_wav(it["audio"])[: int(self.max_sec * SR)]
        return {"fbank": fbank(torch.from_numpy(w)), "pros": torch.from_numpy(prosody_25hz(w)),
                "text": it["text"], "tones": text_to_tones(it["text"]) if it["text"] else [],
                "qs": it["qs"]}


def collate(batch: list[dict]) -> dict:
    T = max(b["fbank"].shape[0] for b in batch)
    P = max(b["pros"].shape[0] for b in batch)
    fb = torch.zeros(len(batch), T, 80)
    pr = torch.zeros(len(batch), P, 8)
    for i, b in enumerate(batch):
        fb[i, : b["fbank"].shape[0]] = b["fbank"]
        pr[i, : b["pros"].shape[0]] = b["pros"]
    return {"fbank": fb, "fbank_len": torch.tensor([b["fbank"].shape[0] for b in batch]),
            "pros": pr, "text": [b["text"] for b in batch], "tones": [b["tones"] for b in batch],
            "qs": [b["qs"] for b in batch]}
