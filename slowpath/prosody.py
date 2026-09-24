"""DSP 韵律旁路(PLAN §2.3 丢失点①):直接从波形提取,不经过编码器,保证音高信息可用。

100Hz 基础特征:logF0(无声段线性插值,按本句浊音段中位数归一化 → 说话人无关的相对音高)、
浊音标志、对数能量(相对本句最大值)、logF0 一阶差分。
再按 4 帧(40ms)聚合成 25Hz、8 维,与 Freeze-Omni 编码器输出帧率对齐:
  [lf0 均值, 浊音比例, 能量均值, dlf0 均值, lf0 标准差, 能量标准差, 窗内 lf0 末-首, 窗内 lf0 最大]
全部向量化(FFT 自相关),离线/在线都够快。
"""
from __future__ import annotations

import numpy as np

SR = 16000
HOP = 160          # 10ms
WIN = 640          # 40ms 自相关窗
LAG_MIN, LAG_MAX = SR // 400, SR // 70


def base_frames(wav: np.ndarray) -> np.ndarray:
    """(T100, 4):lf0_norm, voiced, loge_rel, dlf0。"""
    x = np.asarray(wav, np.float32)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    energy = np.sqrt((fr ** 2).mean(1)) + 1e-6
    spec = np.fft.rfft(fr, 2 * WIN, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :WIN]
    lag = LAG_MIN + np.argmax(ac[:, LAG_MIN:LAG_MAX], axis=1)
    peak = ac[np.arange(n), lag] / (ac[:, 0] + 1e-9)
    loge = np.log(energy)
    voiced = (peak > 0.3) & (loge > loge.max() - 5.0)
    lf0 = np.log(SR / lag)
    if voiced.sum() >= 2:
        vi = np.where(voiced)[0]
        lf0 = np.interp(np.arange(n), vi, lf0[vi])
        lf0 = lf0 - np.median(lf0[voiced])
    else:
        lf0 = np.zeros(n)
    dlf0 = np.gradient(lf0) * voiced
    return np.stack([lf0, voiced.astype(np.float32), loge - loge.max(), dlf0], 1).astype(np.float32)


def prosody_25hz(wav: np.ndarray) -> np.ndarray:
    """(T25, 8),每 4 个 10ms 帧聚合为一个 40ms 帧。"""
    b = base_frames(wav)
    t = len(b) // 4
    if t == 0:
        return np.zeros((1, 8), np.float32)
    w = b[: t * 4].reshape(t, 4, 4)
    lf0, v, e, d = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    return np.stack([lf0.mean(1), v.mean(1), e.mean(1), d.mean(1), lf0.std(1), e.std(1),
                     lf0[:, -1] - lf0[:, 0], lf0.max(1)], 1).astype(np.float32)
