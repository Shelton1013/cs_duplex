"""timing 先验采样器:消费 scripts/diar/timing_stats.py 的输出(v2 格式)。

原则(PLAN §2):所有时间量一律从真实分布采样,绝不用规则偏移。
找不到 stats 文件时退回内置小样本(仅供单测,量产必须用真实文件)。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# 兜底样本(来自 2026-09 真实统计的粗化摘录,仅测试用)
_FALLBACK = {
    "fto": [-0.3, -0.1, 0.0, 0.05, 0.2, 0.4, 0.8, 1.5],
    "ins_dur": [0.15, 0.25, 0.3, 0.4, 0.55, 0.8],
    "ins_offset": [0.5, 1.2, 2.5, 4.0, 6.5, 10.0],
    "turn_dur": [1.0, 2.0, 3.8, 6.0, 12.0],
    "overlap_dur": [0.1, 0.2, 0.35, 0.6, 0.9],
}


class TimingPriors:
    def __init__(self, samples: dict, group: str, rng: random.Random):
        self.samples = samples
        self.group = group
        self.rng = rng

    @classmethod
    def load(cls, path: str | Path | None, group: str = "rthk",
             seed: int = 0) -> "TimingPriors":
        rng = random.Random(seed)
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            src = data.get("by_source", {}).get(group) or data.get("all")
            samples = src["samples"]
        else:
            samples = dict(_FALLBACK)
        return cls(samples, group, rng)

    def sample(self, key: str, lo: float | None = None,
               hi: float | None = None) -> float:
        xs = self.samples.get(key) or _FALLBACK[key]
        x = float(self.rng.choice(xs))
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        return round(x, 3)
