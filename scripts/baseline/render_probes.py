"""把探针渲染成真实音频(基线阶梯的前置步骤)。

每个 session = 一条 agent 长话语 + 恰好一个用户探针(归因干净,无相邻事件互扰)。
开环评测:agent 音频不因被测系统的动作而截断,只记录系统"在何时做了什么"。

探针类别与期望动作(harness 口径):
  backchannel / side_speech          → 应继续(误停轴)
  correction_I0 / correction_I1 / floor_claim → 应停(漏停轴)
纠正分级(结构性问题 2,防"正则可解"):
  I0 = 带显式否定词(唔係 / no no / wait)
  I1 = 无否定词,只说出新值(「係下個月呀」「下個月喎」)——以「係」开头的
       I1 专门考验"把纠正误当应声"的级联错误
  I2(纯声调区分)需可控韵律 TTS,v0 暂缺,已知局限

TTS:edge-tts zh-HK 音色(评测专用;训练侧用 CosyVoice2 → 生成器隔离)。

输出:<out>/<sid>/{user.wav, agent.wav, session.json},16k 单声道;
      <out>/index.jsonl。user.wav 为理想 AEC 下的用户麦克风通道(被测系统输入)。

用法(需 edge-tts / soundfile / numpy,cduplex env 已装;内地需 --proxy):
  python render_probes.py --n 300 --out data/probe_audio --proxy http://127.0.0.1:7897
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synth.script import SCENARIOS, SLOT_BANK, make_line  # noqa: E402

SR = 16000
AGENT_VOICE = "zh-HK-HiuMaanNeural"
USER_VOICES = ["zh-HK-WanLungNeural", "zh-HK-HiuGaaiNeural"]

BACKCHANNEL = ["嗯嗯", "係啊", "係係係", "啱啊", "好", "明白", "okok", "yeah", "嗯 ok"]
SIDE_SPEECH = ["阿媽幫我攞杯水", "你哋食咗飯未啊", "等陣先我聽緊電話", "細佬熄咗個電視佢"]
FLOOR_CLAIM = ["唔好意思打斷一下,我仲想問埋另一樣嘢", "sorry 打斷一下,仲有個問題",
               "by the way 我仲想改埋地址"]
CORR_I0 = ["唔係啊,我問嘅係{new}", "等等,唔係,係{new}先啱", "no no,{new}先啱",
           "wait wait,唔係,係{new}"]
CORR_I1 = ["係{new}呀", "{new}喎", "我講緊{new}", "係{new}先啱"]

# I2 回声最小对:重复 agent 刚说的槽值,文本相同只靠语调区分
#   echo_question「{old}?」升调质疑 → 应停(PAUSE 确认)
#   echo_confirm 「{old}。」降调确认 → 应继续
# 渲染后做句尾 F0 质检:质疑句尾/均值 必须比确认高出 I2_MIN_CONTRAST,否则换音色重渲,
# 仍不达标则丢弃该最小对(避免声学上不可分的样本污染评测)
I2_MIN_CONTRAST = 0.12

# (类别, 权重)
CLASS_MIX = [("backchannel", 0.20), ("side_speech", 0.12), ("floor_claim", 0.10),
             ("correction_I0", 0.14), ("correction_I1", 0.14),
             ("echo_question", 0.15), ("echo_confirm", 0.15)]
HARNESS_CLS = {"backchannel": "backchannel", "side_speech": "side_speech",
               "floor_claim": "floor_claim", "correction_I0": "correction",
               "correction_I1": "correction", "echo_question": "correction",
               "echo_confirm": "backchannel"}


def f0_track(wav: np.ndarray, sr: int = SR) -> np.ndarray:
    """粗略自相关 F0(仅用于 I2 语调质检)。"""
    out = []
    for i in range(0, len(wav) - 640, 160):
        x = wav[i:i + 640]
        if np.sqrt((x ** 2).mean()) < 0.02:
            continue
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[639:]
        lo, hi = int(sr / 400), int(sr / 70)
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            out.append(sr / k)
    return np.array(out)


def tail_ratio(wav: np.ndarray) -> float:
    f = f0_track(wav)
    if len(f) < 6:
        return 1.0
    return float(f[-max(3, len(f) // 4):].mean() / f.mean())


async def _tts(text: str, voice: str, mp3: Path, proxy: str | None) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice, proxy=proxy).save(str(mp3))


def tts_wav(text: str, voice: str, cache: Path, proxy: str | None) -> np.ndarray:
    key = hashlib.md5(f"{voice}|{text}".encode()).hexdigest()
    wav_p = cache / f"{key}.wav"
    if not wav_p.exists():
        mp3 = cache / f"{key}.mp3"
        for attempt in range(3):
            try:
                asyncio.run(_tts(text, voice, mp3, proxy))
                break
            except Exception as e:  # 网络抖动重试
                if attempt == 2:
                    raise
                print(f"  tts retry: {e}")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-ac", "1",
                        "-ar", str(SR), str(wav_p)], check=True)
        mp3.unlink()
    wav, _ = sf.read(wav_p, dtype="float32")
    return trim(wav)


def trim(wav: np.ndarray, thr: float = 0.01) -> np.ndarray:
    """去掉 TTS 首尾静音,让 onset 时间准确。"""
    idx = np.where(np.abs(wav) > thr)[0]
    if len(idx) == 0:
        return wav
    return wav[max(0, idx[0] - 160): idx[-1] + 160]


def pick_class(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for cls, w in CLASS_MIX:
        acc += w
        if r < acc:
            return cls
    return CLASS_MIX[-1][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "data" / "probe_audio"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy", default="", help="内地网络填 http://127.0.0.1:7897")
    args = ap.parse_args()

    out = Path(args.out)
    cache = out / "_tts_cache"
    cache.mkdir(parents=True, exist_ok=True)
    proxy = args.proxy or None
    rng = random.Random(args.seed)
    index = []
    n_i2_dropped = 0

    for i in range(args.n):
        sid = f"probe_{i:05d}"
        cls = pick_class(rng)
        _, line = make_line(rng, rng.choice(SCENARIOS))
        agent = tts_wav(line.text, AGENT_VOICE, cache, proxy)
        d_agent = len(agent) / SR
        lead = 0.5
        t_agent0, t_agent1 = lead, lead + d_agent

        slot_flip = None
        user = None
        if cls.startswith("echo"):
            slot = rng.choice(line.slots)
            slot_end = t_agent0 + d_agent * slot.end / max(1, len(line.text))
            onset = slot_end + rng.uniform(0.2, 0.5)
            text = slot.value + ("?" if cls == "echo_question" else "。")
            slot_flip = {"slot": slot.name, "old": slot.value, "new": None}
            # 语调质检:同一音色渲染两种语调,对比达标才用
            voices = USER_VOICES[:]
            rng.shuffle(voices)
            for v in voices:
                q = tts_wav(slot.value + "?", v, cache, proxy)
                c = tts_wav(slot.value + "。", v, cache, proxy)
                if tail_ratio(q) - tail_ratio(c) >= I2_MIN_CONTRAST:
                    user = q if cls == "echo_question" else c
                    break
            if user is None:
                n_i2_dropped += 1
                continue
        elif cls.startswith("correction"):
            slot = rng.choice(line.slots)
            new = rng.choice([v for v in SLOT_BANK[slot.name] if v != slot.value])
            slot_flip = {"slot": slot.name, "old": slot.value, "new": new}
            text = rng.choice(CORR_I0 if cls == "correction_I0" else CORR_I1).format(new=new)
            # 因果约束:纠正必须在旧值播出之后(按字符比例估计槽位结束时刻)
            slot_end = t_agent0 + d_agent * slot.end / max(1, len(line.text))
            onset = slot_end + rng.uniform(0.3, 0.7)
        else:
            bank = {"backchannel": BACKCHANNEL, "side_speech": SIDE_SPEECH,
                    "floor_claim": FLOOR_CLAIM}[cls]
            text = rng.choice(bank)
            onset = t_agent0 + rng.uniform(0.25, 0.7) * d_agent
        if user is None:
            user = tts_wav(text, rng.choice(USER_VOICES), cache, proxy)
        d_user = len(user) / SR
        onset = min(onset, t_agent1 - 0.5)  # 必须落在 agent 说话期间(duplex 定义)
        total = max(t_agent1, onset + d_user) + 2.0

        n = int(total * SR)
        ch_agent = np.zeros(n, np.float32)
        ch_user = np.zeros(n, np.float32)
        a0 = int(t_agent0 * SR)
        ch_agent[a0:a0 + len(agent)] = agent
        u0 = int(onset * SR)
        ch_user[u0:u0 + len(user)] = user[: n - u0]

        d = out / sid
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / "user.wav", ch_user, SR)
        sf.write(d / "agent.wav", ch_agent, SR)
        sess = {
            "session_id": sid,
            "probe_class": cls,
            "agent_utts": [{"t_start": round(t_agent0, 3), "t_end": round(t_agent1, 3),
                            "text": line.text}],
            "probes": [{"t_onset": round(onset, 3), "t_end": round(onset + d_user, 3),
                        "cls": HARNESS_CLS[cls], "text": text,
                        "language": "mixed" if any(c.isascii() and c.isalpha() for c in text) else "yue"}],
            "slot_flip": slot_flip,
            "duration": round(total, 3),
        }
        (d / "session.json").write_text(json.dumps(sess, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        index.append({"session_id": sid, "probe_class": cls})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{args.n}")

    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("done:", dict(Counter(r["probe_class"] for r in index)), "->", out)
    print(f"I2 语调质检丢弃: {n_i2_dropped}")


if __name__ == "__main__":
    main()
