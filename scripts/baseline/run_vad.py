"""【本地】基线 ①:VAD 一响就停(当前商用 server-VAD 打断的标准行为)。

能量 VAD 检测 user.wav 的人声段;每段语音起点 + 触发延迟(默认 0.25s,
对应商用 server VAD 的最短语音确认时长)→ CUT。不区分内容。
纯 numpy/soundfile,cduplex env 可跑。

用法:python run_vad.py --probes data/probe_audio --out data/baselines/vad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def energy_vad(wav: np.ndarray, sr: int, hop_ms: float = 10.0, win_ms: float = 25.0,
               thr: float = 0.02, min_speech: float = 0.1, min_gap: float = 0.3):
    hop, win = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
    n = max(0, 1 + (len(wav) - win) // hop)
    e = np.array([np.sqrt(np.mean(wav[i * hop:i * hop + win] ** 2)) for i in range(n)])
    active = e > thr
    regions, i = [], 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            regions.append([i * hop / sr, (j * hop + win) / sr])
            i = j
        else:
            i += 1
    merged = []
    for s, en in regions:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = en
        else:
            merged.append([s, en])
    return [(s, en) for s, en in merged if en - s >= min_speech]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trigger", type=float, default=0.25, help="VAD 确认延迟(秒)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in sorted(Path(args.probes).iterdir()):
        if not (s / "user.wav").exists():
            continue
        wav, sr = sf.read(s / "user.wav", dtype="float32")
        regions = energy_vad(wav, sr)
        actions = [{"t": round(st + args.trigger, 3), "kind": "CUT"}
                   for st, en in regions if en - st >= args.trigger]
        (out / f"{s.name}.json").write_text(json.dumps({"actions": actions}), encoding="utf-8")
        n += 1
    print(f"{n} sessions -> {out}")


if __name__ == "__main__":
    main()
