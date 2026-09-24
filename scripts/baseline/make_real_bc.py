"""测试 A:真实应声探针(回应"词表与考题同源"的评测偏差)。

从爬取语料的 diarization 结果里挑"功能上是应声"的真实插话,替换 TTS 应声:
  金标签判定(行为学,与任何词表无关):
    - 插话段完全被他人话轮包住(嵌套插话),时长 0.15–1.0s
    - 宿主在插话前已说 ≥1s,插话后又继续说 ≥1s(没被打断 → 功能上是应声)
  → 正确动作 = 继续说(probe_class = real_bc)

混杂控制:爬取音频是混合单通道,插话片段里带宿主串音(≈无 AEC)。
  对照组 host_control:从"无任何插话的宿主长话段"中部切同样时长的片段注入,
  金标签同为继续说。real_bc 与 host_control 的误停率之差 = 真实应声本身造成的误停。

agent 侧复用 v2 探针的 agent.wav(同一批 edge-tts agent 话语),只替换 user.wav。
输出与 render_probes 同格式,可直接喂各基线与 score.py。

用法(本地,纯 CPU;需 soundfile/numpy/ffmpeg):
  python make_real_bc.py --n 200 --out data/probe_audio_realbc
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SR = 16000
AUDIO_EXTS = (".m4a", ".opus", ".webm", ".mp3")


def parse_rttm(p: Path):
    segs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        a = line.split()
        if len(a) >= 8 and a[0] == "SPEAKER":
            segs.append((a[7], float(a[3]), float(a[3]) + float(a[4])))
    return sorted(segs, key=lambda s: s[1])


def find_candidates(segs):
    ins, ctrl = [], []
    for (h, s1, e1), (u, s2, e2) in zip(segs, segs[1:]):
        if h != u and e2 < e1 and 0.15 <= e2 - s2 <= 1.0 and s2 - s1 >= 1.0 and e1 - e2 >= 1.0:
            ins.append((s2, e2))
    for spk, s, e in segs:
        if e - s < 6.0:
            continue
        clash = any(o != spk and os_ < e and oe > s for o, os_, oe in segs)
        if not clash:
            ctrl.append((s + 2.0, e - 2.0))  # 中部窗口,供切等长片段
    return ins, ctrl


def cut(src: Path, t0: float, t1: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}", "-to", f"{t1:.3f}",
                        "-i", str(src), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每组(real_bc / host_control)条数")
    ap.add_argument("--diar", default=str(ROOT / "data" / "diar"))
    ap.add_argument("--web", default=str(ROOT / "data" / "web"))
    ap.add_argument("--agents", default=str(ROOT / "data" / "probe_audio_v2"))
    ap.add_argument("--out", default=str(ROOT / "data" / "probe_audio_realbc"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) 候选:按源文件收集插话与对照窗口
    pool_ins, pool_ctrl = [], []
    diar, web = Path(args.diar), Path(args.web)
    for rttm in diar.rglob("*.rttm"):
        rel = rttm.relative_to(diar)
        src = web / str(rel)[: -len(".rttm")]
        if not src.exists() or src.suffix not in AUDIO_EXTS:
            continue
        ins, ctrl = find_candidates(parse_rttm(rttm))
        domain = rel.parts[0]
        pool_ins += [(src, s, e, domain) for s, e in ins]
        pool_ctrl += [(src, s, e, domain) for s, e in ctrl]
    print(f"candidates: insertions={len(pool_ins)} control_windows={len(pool_ctrl)}")
    rng.shuffle(pool_ins)

    # 2) agent 宿主:复用 v2 的 agent.wav 与话语时间
    agents = []
    for d in sorted(Path(args.agents).glob("probe_*")):
        m = json.loads((d / "session.json").read_text(encoding="utf-8"))
        agents.append((d / "agent.wav", m["agent_utts"][0]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, i, k = [], 0, 0
    made = {"real_bc": 0, "host_control": 0}
    while made["real_bc"] < args.n and k < len(pool_ins):
        src, s, e, domain = pool_ins[k]
        k += 1
        clip = cut(src, s, e)
        if len(clip) < int(0.1 * SR) or np.sqrt((clip ** 2).mean()) < 0.005:
            continue
        # 同时长对照片段
        cs, ce_, cdomain = None, None, None
        for _ in range(20):
            csrc, ws, we, cdomain = rng.choice(pool_ctrl)
            if we - ws > (e - s):
                cs = rng.uniform(ws, we - (e - s))
                break
        if cs is None:
            continue
        ctrl = cut(csrc, cs, cs + (e - s))
        if len(ctrl) < int(0.1 * SR):
            continue
        for cls, wav, meta in (("real_bc", clip, {"src": str(src.name), "t": [s, e], "domain": domain}),
                               ("host_control", ctrl, {"src": str(csrc.name), "t": [cs, cs + e - s],
                                                       "domain": cdomain})):
            agent_wav, au = agents[i % len(agents)]
            i += 1
            agent, _ = sf.read(agent_wav, dtype="float32")
            onset = au["t_start"] + rng.uniform(0.25, 0.7) * (au["t_end"] - au["t_start"])
            peak = np.abs(wav).max()
            wav = wav / peak * 0.5 if peak > 0 else wav  # 电平与 TTS 探针对齐
            n = max(len(agent), int((onset + len(wav) / SR + 2.0) * SR))
            user = np.zeros(n, np.float32)
            u0 = int(onset * SR)
            user[u0:u0 + len(wav)] = wav
            ag = np.zeros(n, np.float32)
            ag[:len(agent)] = agent
            sid = f"real_{i:05d}"
            d = out / sid
            d.mkdir(exist_ok=True)
            sf.write(d / "user.wav", user, SR)
            sf.write(d / "agent.wav", ag, SR)
            sess = {"session_id": sid, "probe_class": cls,
                    "agent_utts": [au],
                    "probes": [{"t_onset": round(onset, 3),
                                "t_end": round(onset + len(wav) / SR, 3),
                                "cls": "backchannel", "text": "", "language": ""}],
                    "source": meta, "duration": round(n / SR, 3)}
            (d / "session.json").write_text(json.dumps(sess, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
            index.append({"session_id": sid, "probe_class": cls, **meta})
            made[cls] += 1
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("made:", made, "->", out)


if __name__ == "__main__":
    sys.exit(main())
