"""【服务器】爬取语料 → 伪标签 ASR 数据(Stage A v1 扩充对齐数据)。

数据边界:爬取语料(RTHK / YouTube)有版权,只用于内部研究,不进入交付模型(PLAN §4)。

流程:
  1. 按 diarization(RTTM)取单一说话人片段;与他人有任何重叠的片段丢弃(串音会污染转写);
     同一说话人相邻片段(间隔 <0.4s)合并,长度限制 2–15s;
  2. SenseVoice-Small 自动识别语种并转写(不做 ITN);
  3. 只保留识别为粤语(<|yue|>)的片段;每秒字数(中文按字、英文按词)在 [1.5, 9] 之间。
输出:<out>/<src>/<name>_<k>.wav(16k)+ <out>/pseudo.jsonl({audio, text, source, dur})

环境:服务器 funasr env
用法:python make_pseudo_labels.py --web /home/pxieaf/home2/data/web \
        --diar /home/pxieaf/home2/data/diar --out /home/pxieaf/home2/data/pseudo
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
AUDIO_EXTS = (".m4a", ".opus", ".webm", ".mp3")


def load_audio(path: Path) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def segments(rttm: Path, min_d=2.0, max_d=15.0, gap=0.4):
    segs = []
    for line in rttm.read_text(encoding="utf-8").splitlines():
        a = line.split()
        if len(a) >= 8 and a[0] == "SPEAKER":
            segs.append((a[7], float(a[3]), float(a[3]) + float(a[4])))
    segs.sort(key=lambda s: s[1])
    clean = [s for s in segs
             if not any(o[0] != s[0] and o[1] < s[2] and o[2] > s[1] for o in segs)]
    out, cur = [], None
    for spk, s, e in clean:
        if cur and cur[0] == spk and s - cur[2] < gap and e - cur[1] <= max_d:
            cur = (spk, cur[1], e)
        else:
            if cur:
                out.append(cur)
            cur = (spk, s, e)
    if cur:
        out.append(cur)
    return [(s, e) for _, s, e in out if min_d <= e - s <= max_d]


def n_units(text: str) -> int:
    return len(re.findall(r"[A-Za-z']+|[一-鿿]", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", required=True)
    ap.add_argument("--diar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from funasr import AutoModel
    model = AutoModel(model="iic/SenseVoiceSmall", device=args.device, disable_update=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jl = open(out / "pseudo.jsonl", "a", encoding="utf-8")
    done = {json.loads(l)["audio"].rsplit("/", 1)[-1].rsplit("_", 1)[0]
            for l in open(out / "pseudo.jsonl", encoding="utf-8")} if (out / "pseudo.jsonl").stat().st_size else set()
    stats = {"files": 0, "segs": 0, "kept": 0, "hours": 0.0, "drop_lang": 0, "drop_rate": 0}
    web, diar = Path(args.web), Path(args.diar)
    for rttm in sorted(diar.rglob("*.rttm")):
        rel = rttm.relative_to(diar)
        base = str(rel)[: -len(".rttm")]
        src = web / base
        if not src.exists():  # 服务器重下的 YouTube 扩展名可能不同
            cands = [p for p in (web / rel.parent).glob(Path(base).stem + ".*") if p.suffix in AUDIO_EXTS]
            if not cands:
                continue
            src = cands[0]
        name = f"{rel.parts[0]}_{Path(base).stem}"
        if name in done:
            continue
        wav = load_audio(src)
        if len(wav) == 0:
            continue
        stats["files"] += 1
        sd = out / rel.parts[0]
        sd.mkdir(exist_ok=True)
        for k, (s, e) in enumerate(segments(rttm)):
            seg = wav[int(s * SR): int(e * SR)]
            stats["segs"] += 1
            res = model.generate(input=seg, cache={}, language="auto", use_itn=False)
            raw = res[0]["text"] if res else ""
            lang = re.findall(r"<\|(\w+)\|>", raw)[:1]
            text = re.sub(r"<\|[^|]*\|>", "", raw).strip()
            if lang != ["yue"]:
                stats["drop_lang"] += 1
                continue
            rate = n_units(text) / (e - s)
            if not (1.5 <= rate <= 9.0):
                stats["drop_rate"] += 1
                continue
            p = sd / f"{name}_{k:04d}.wav"
            sf.write(p, seg, SR)
            jl.write(json.dumps({"audio": str(p), "text": text, "source": rel.parts[0],
                                 "dur": round(e - s, 2)}, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["hours"] += (e - s) / 3600
        jl.flush()
        if stats["files"] % 20 == 0:
            print({k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}, flush=True)
    print("DONE", {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}, flush=True)


if __name__ == "__main__":
    main()
