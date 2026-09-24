"""测试 B:电话信道条件(回应"探针音频太干净"的评测偏差)。

对探针集的 user.wav 施加电话链路退化,agent.wav / session.json 原样复制:
  16k → 300–3400Hz 带通 → 8k → G.711 μ-law 编解码 → 叠加粉噪声(默认 SNR 20dB,
  按语音段能量算)→ 回到 16k。时间轴不变,金标签不变。

纯 numpy/scipy/soundfile(服务器 funasr env 可跑)。
用法:python make_telephone.py --src data/probe_audio_v2 --out data/probe_audio_v2_tel --snr 20
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

MU = 255.0


def mulaw(x: np.ndarray) -> np.ndarray:
    y = np.sign(x) * np.log1p(MU * np.abs(x)) / np.log1p(MU)
    y = np.round((y + 1) / 2 * 255) / 255 * 2 - 1  # 8bit 量化
    return np.sign(y) * (np.power(1 + MU, np.abs(y)) - 1) / MU


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    f = np.fft.rfftfreq(n)
    spec = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-9)


def degrade(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="band", fs=16000, output="sos")
    y = sosfilt(sos, x)
    y = resample_poly(y, 1, 2)                       # → 8k
    y = mulaw(np.clip(y, -1, 1))
    speech = y[np.abs(y) > 0.01]
    p_sig = np.mean(speech ** 2) if len(speech) else 1e-4
    noise = pink(len(y), rng) * np.sqrt(p_sig / (10 ** (snr_db / 10)))
    y = y + noise
    y = resample_poly(y, 2, 1)[: len(x)]             # → 16k(模型输入)
    return np.clip(y, -1, 1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in sorted(src.glob("probe_*")) + sorted(src.glob("real_*")):
        o = out / d.name
        o.mkdir(exist_ok=True)
        x, sr = sf.read(d / "user.wav", dtype="float32")
        assert sr == 16000
        sf.write(o / "user.wav", degrade(x, args.snr, rng), 16000)
        shutil.copy(d / "agent.wav", o / "agent.wav")
        meta = json.loads((d / "session.json").read_text(encoding="utf-8"))
        meta["channel"] = f"tel_mulaw8k_pink{int(args.snr)}dB"
        (o / "session.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        n += 1
    if (src / "index.jsonl").exists():
        shutil.copy(src / "index.jsonl", out / "index.jsonl")
    print(f"{n} sessions -> {out}")


if __name__ == "__main__":
    main()
