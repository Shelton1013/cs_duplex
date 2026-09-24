"""【服务器】Stage 0 可学性验证数据:说话对象 / 语调两个二分类任务。

TTS:Fun-CosyVoice3-0.5B,inference_instruct2(文本, 粤语指令, 音色参考)。
音色:MCE 志愿者录音(用户自有数据集,已确认可用),每个目录取一段 4–10s 作参考;
      说话人按目录划分 train / val,验证集说话人训练时从未出现。
评测隔离:评测探针用 edge-tts,这里用 CosyVoice3。

说话对象任务的声学处理(同一房间、不同距离,两类分布有重叠,防止只靠"有无混响"):
  to_agent(近讲):DRR ∈ [3, 15] dB,增益 0 ± 2 dB
  side(转头/远场):DRR ∈ [-6, 6] dB,增益 ∈ [-10, -3] dB
  房间 RT60 ∈ [0.25, 0.8] s,两类共用同一分布;RIR 为合成指数衰减噪声(评测时改用真实 RIR)。
语调任务:同一音色、同一槽值渲染"质疑(升调)"与"确认(降调)"两版,
  句尾 F0 比值之差 ≥ QC_MIN 才保留这一对(不达标换音色重试)。
两个任务都有 50% 样本过电话信道(复用 make_telephone.degrade)。

输出:<out>/{addressee,intonation}/<id>.wav(16k)+ <out>/labels.jsonl
用法(服务器,CosyVoice 环境):
  python render_stage0.py --out /home/pxieaf/home2/data/stage0 --n_addr 1500 --n_echo 600
  冒烟:--n_addr 8 --n_echo 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "baseline"))
from make_telephone import degrade  # noqa: E402
from render_probes import tail_ratio  # noqa: E402
from synth.stage0_texts import ECHO_VALUES, side_texts, to_agent_texts  # noqa: E402

SR = 16000
INSTR_YUE = "You are a helpful assistant. 请用广东话表达。<|endofprompt|>"
INSTR_Q = "You are a helpful assistant. 请用广东话,以怀疑、反问的语气说,句尾语调明显上扬。<|endofprompt|>"
INSTR_S = "You are a helpful assistant. 请用广东话,以平淡、肯定的语气说,句尾语调下降。<|endofprompt|>"
QC_MIN = 0.12


def pick_voices(mce_audio: Path, n: int, rng: random.Random,
                spk_map: dict | None) -> list[Path]:
    """每个说话人只取一个音色(MCE 同一人对应多个目录,spk_map 来自声纹聚类)。"""
    dirs = sorted(d for d in mce_audio.iterdir() if d.is_dir())
    rng.shuffle(dirs)
    voices, used_spk = [], set()
    for d in dirs:
        spk = spk_map.get(d.name, d.name) if spk_map else d.name
        if spk in used_spk:
            continue
        for w in sorted(d.glob("*.wav"))[:20]:
            if 4.0 <= sf.info(w).duration <= 10.0:
                voices.append(w)
                used_spk.add(spk)
                break
        if len(voices) >= n:
            break
    return voices


def synth_rir(rng: np.random.Generator, rt60: float, drr_db: float) -> np.ndarray:
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail[: int(0.002 * SR)] = 0  # 直达声之后才开始混响
    tail /= np.sqrt(np.sum(tail ** 2)) + 1e-9
    rir = tail * 10 ** (-drr_db / 20)  # 直达声能量 = 1,混响能量 = 10^(-DRR/10)
    rir[0] = 1.0
    return rir


def apply_room(x: np.ndarray, rir: np.ndarray, gain_db: float) -> np.ndarray:
    y = fftconvolve(x, rir)[: len(x) + int(0.3 * SR)]
    peak = np.abs(y).max() + 1e-9
    y = y / peak * np.abs(x).max()  # 先保持峰值,再施加距离增益
    return (y * 10 ** (gain_db / 20)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_addr", type=int, default=1500, help="说话对象任务总条数(两类各半)")
    ap.add_argument("--n_echo", type=int, default=600, help="语调任务的最小对数")
    ap.add_argument("--n_voices", type=int, default=40)
    ap.add_argument("--val_voices", type=int, default=8)
    ap.add_argument("--mce", default="/home/share/data_makchen/peng/datasets/MCE/MCE_Dataset/Audio")
    ap.add_argument("--cosyvoice_repo", default="/home/pxieaf/CosyVoice")
    ap.add_argument("--model", default="/home/share/data_makchen/peng/models/Fun-CosyVoice3-0.5B-2512")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spk_map", default="", help="cluster_mce_speakers.py 的输出(目录→说话人)")
    ap.add_argument("--voice_seed", type=int, default=0,
                    help="音色选择与 train/val 划分的种子;分片并行时各片须相同")
    ap.add_argument("--id_prefix", default="", help="分片并行时的样本 id 前缀,避免重名")
    args = ap.parse_args()
    spk_map = (json.loads(Path(args.spk_map).read_text(encoding="utf-8"))["folder2spk"]
               if args.spk_map else None)

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "addressee").mkdir(parents=True, exist_ok=True)
    (out / "intonation").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, args.cosyvoice_repo)
    sys.path.insert(0, str(Path(args.cosyvoice_repo) / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir=args.model)
    tts_sr = cv.sample_rate

    def tts(text: str, instr: str, voice: Path) -> np.ndarray:
        chunks = [o["tts_speech"].squeeze(0).cpu().numpy()
                  for o in cv.inference_instruct2(text, instr, str(voice), stream=False)]
        x = np.concatenate(chunks).astype(np.float32)
        g = np.gcd(SR, tts_sr)
        return resample_poly(x, SR // g, tts_sr // g).astype(np.float32)

    voices = pick_voices(Path(args.mce), args.n_voices, random.Random(args.voice_seed), spk_map)
    split = {str(v): ("val" if i < args.val_voices else "train") for i, v in enumerate(voices)}
    print(f"voices: {len(voices)} ({args.val_voices} held out for val)", flush=True)
    labels = open(out / "labels.jsonl", "a", encoding="utf-8")
    t0 = time.time()

    # ---- 说话对象任务 ----
    half = args.n_addr // 2
    items = [(t, s, 1) for t, s in to_agent_texts(rng, half)] + \
            [(t, s, 0) for t, s in side_texts(rng, half)]
    rng.shuffle(items)
    for i, (text, sub, to_agent) in enumerate(items):
        v = rng.choice(voices)
        x = tts(text, INSTR_YUE, v)
        rt60 = rng.uniform(0.25, 0.8)
        if to_agent:
            drr, gain = rng.uniform(3, 15), rng.uniform(-2, 2)
        else:
            drr, gain = rng.uniform(-6, 6), rng.uniform(-10, -3)
        y = apply_room(x, synth_rir(nrng, rt60, drr), gain)
        tel = rng.random() < 0.5
        if tel:
            y = degrade(y, rng.uniform(15, 25), nrng)
        sid = f"{args.id_prefix}addr_{i:05d}"
        sf.write(out / "addressee" / f"{sid}.wav", y, SR)
        labels.write(json.dumps({"id": sid, "task": "addressee", "label": to_agent,
                                 "subtype": sub, "text": text, "voice": v.parent.name,
                                 "split": split[str(v)], "rt60": round(rt60, 3),
                                 "drr": round(drr, 2), "gain": round(gain, 2), "tel": tel},
                                ensure_ascii=False) + "\n")
        if (i + 1) % 100 == 0:
            labels.flush()
            print(f"addr {i + 1}/{len(items)}  {(time.time() - t0) / (i + 1):.2f}s/utt", flush=True)

    # ---- 语调任务(最小对)----
    made = dropped = 0
    while made < args.n_echo and made + dropped < args.n_echo * 3:
        val = rng.choice(ECHO_VALUES)
        ok = None
        for v in rng.sample(voices, 2):  # 不达标换一个音色重试一次
            q = tts(val + "?", INSTR_Q, v)
            s = tts(val + "。", INSTR_S, v)
            if tail_ratio(q) - tail_ratio(s) >= QC_MIN:
                ok = (v, q, s)
                break
        if ok is None:
            dropped += 1
            continue
        v, q, s = ok
        tel = rng.random() < 0.5
        for lab, wav in (("question", q), ("statement", s)):
            y = apply_room(wav, synth_rir(nrng, rng.uniform(0.25, 0.8), rng.uniform(3, 15)),
                           rng.uniform(-2, 2))
            if tel:  # 最小对两侧同一信道条件,保证只差语调
                y = degrade(y, 20.0, nrng)
            sid = f"{args.id_prefix}echo_{made:05d}_{lab}"
            sf.write(out / "intonation" / f"{sid}.wav", y, SR)
            labels.write(json.dumps({"id": sid, "task": "intonation",
                                     "label": int(lab == "question"), "pair": f"{args.id_prefix}{made}",
                                     "text": val, "voice": v.parent.name,
                                     "split": split[str(v)], "tel": tel},
                                    ensure_ascii=False) + "\n")
        made += 1
        if made % 50 == 0:
            labels.flush()
            print(f"echo pairs {made}/{args.n_echo} (dropped {dropped})", flush=True)
    labels.close()
    print(f"done: addressee={len(items)} echo_pairs={made} dropped={dropped} -> {out}", flush=True)


if __name__ == "__main__":
    main()
