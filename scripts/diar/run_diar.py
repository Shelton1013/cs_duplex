"""对爬取音频批量跑 pyannote 3.1 diarization,输出 RTTM(timing 先验的第一步)。

环境:conda env `cduplex`(py3.10 + torch cu121 + pyannote.audio)
模型:pyannote/speaker-diarization-3.1(gated,需 HF token 且已接受条款)

用法:
  python run_diar.py --limit 3          # 冒烟:先跑 3 条验证
  python run_diar.py                    # 全量(断点续跑:已有 .rttm 的跳过)

输出:每条音频 → <out>/<相对路径>.rttm;失败清单 → <out>/failed.txt
统计脚本(timing_stats.py)单独消费 RTTM,与推理解耦。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

AUDIO_EXTS = {".m4a", ".opus", ".webm", ".mp3", ".wav"}


def to_wav16k(src: Path, dst: Path) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data" / "web"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "diar"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed_log = out / "failed.txt"

    files = sorted(p for p in data.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    todo = []
    for p in files:
        rttm = out / p.relative_to(data).with_suffix(p.suffix + ".rttm")
        if not rttm.exists():
            todo.append((p, rttm))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(files)} audio files, {len(todo)} to process")
    if not todo:
        return

    import torch
    from pyannote.audio import Pipeline

    # token 自动从 ~/.cache/huggingface/token 读取(已存在)
    pipe = Pipeline.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(torch.device(device))
    print(f"pipeline on {device}")

    t_total = 0.0
    for i, (src, rttm) in enumerate(todo):
        rttm.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            if not to_wav16k(src, wav):
                print(f"  !! ffmpeg failed: {src}")
                with open(failed_log, "a", encoding="utf-8") as f:
                    f.write(f"{src}\tffmpeg\n")
                continue
            t0 = time.time()
            try:
                diar = pipe(str(wav))
            except Exception as e:  # 单条失败不拖垮批次
                print(f"  !! diar failed: {src}: {e}")
                with open(failed_log, "a", encoding="utf-8") as f:
                    f.write(f"{src}\t{e}\n")
                continue
            dt = time.time() - t0
            t_total += dt
            with open(rttm, "w", encoding="utf-8") as f:
                diar.write_rttm(f)
            print(f"[{i + 1}/{len(todo)}] {src.name}  {dt:.0f}s")
    print(f"done, diar compute {t_total / 60:.1f} min")


if __name__ == "__main__":
    main()
