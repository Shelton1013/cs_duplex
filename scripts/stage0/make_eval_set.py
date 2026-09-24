"""Stage 0 干净评测集(修正:v2 探针的说话对象类与训练文本大量重合)。

文本来自真人小样材料包(scripts/human_sample/make_kit.py),已核对与训练数据不重合:
  说话对象:对手机 20 句 + 转头 20 句 + 同句两讲 10 句(两类各一份,内容相同只差声学)
  语调:30 个新数值 × {质疑?, 确认。}
TTS:edge-tts 全部 3 个 zh-HK 音色(评测专用,与训练的 CosyVoice3 隔离)。
语调对做句尾 F0 对比质检(同 render_probes)。
房间声学不在这里加:probe_learnability 在服务器上按条件(干声 / 两类都进房间、只差距离)施加。

输出:<out>/{addressee,intonation}/*.wav + <out>/labels.jsonl
用法(本地,纯云端 TTS):python make_eval_set.py --out data/stage0_eval --proxy http://127.0.0.1:7897
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "baseline"))
sys.path.insert(0, str(ROOT / "scripts" / "human_sample"))
from make_kit import B_PAIRS, B_SIDE, B_TO_AGENT, CARDS_A  # noqa: E402
from render_probes import I2_MIN_CONTRAST, SR, tail_ratio, tts_wav  # noqa: E402

VOICES = ["zh-HK-HiuMaanNeural", "zh-HK-WanLungNeural", "zh-HK-HiuGaaiNeural"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "stage0_eval"))
    ap.add_argument("--proxy", default="")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "addressee").mkdir(parents=True, exist_ok=True)
    (out / "intonation").mkdir(parents=True, exist_ok=True)
    cache = out / "_tts_cache"
    cache.mkdir(exist_ok=True)
    proxy = args.proxy or None
    rows = []

    items = [(t, 1, "to_agent") for t in B_TO_AGENT] + [(t, 0, "side") for t in B_SIDE] + \
            [(t, 1, "pair") for t in B_PAIRS] + [(t, 0, "pair") for t in B_PAIRS]
    for i, (text, lab, sub) in enumerate(items):
        for v in VOICES:
            w = tts_wav(text, v, cache, proxy)
            sid = f"ev_addr_{i:03d}_{v[6:9]}"
            sf.write(out / "addressee" / f"{sid}.wav", w, SR)
            rows.append({"id": sid, "task": "addressee", "label": lab, "subtype": sub,
                         "text": text, "voice": v})

    kept = dropped = 0
    for i, (_, val) in enumerate(CARDS_A):
        for v in VOICES:
            q = tts_wav(val + "?", v, cache, proxy)
            c = tts_wav(val + "。", v, cache, proxy)
            if tail_ratio(q) - tail_ratio(c) < I2_MIN_CONTRAST:
                dropped += 1
                continue
            for lab, w in ((1, q), (0, c)):
                sid = f"ev_echo_{i:03d}_{v[6:9]}_{'q' if lab else 's'}"
                sf.write(out / "intonation" / f"{sid}.wav", w, SR)
                rows.append({"id": sid, "task": "intonation", "label": lab, "pair": f"{i}_{v}",
                             "text": val, "voice": v})
            kept += 1

    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_addr = sum(r["task"] == "addressee" for r in rows)
    print(f"addressee={n_addr} (pairs subset={sum(r.get('subtype') == 'pair' for r in rows)}) "
          f"intonation_pairs={kept} dropped={dropped} -> {out}")


if __name__ == "__main__":
    main()
