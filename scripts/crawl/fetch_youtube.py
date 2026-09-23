"""YouTube 粤语对话音频爬取(网络挖掘辅助轨,PLAN §2.5b)。

包装 yt-dlp:只下音频、写 info.json 元数据、archive 文件支持断点续爬。
合规:平台内容仅限内部研究用途,不再分发。

用法:
  python fetch_youtube.py --limit 3          # 冒烟:每源最多 3 条
  python fetch_youtube.py                    # 按 yt_sources.txt 全量
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[1] / "data" / "web" / "youtube"
DEFAULT_SOURCES = HERE / "yt_sources.txt"
# 本机 Anaconda 是 py3.9,pip 装不了新版 yt-dlp;用同目录独立 exe(2026.08.19+)
YTDLP = str(HERE / "yt-dlp.exe") if (HERE / "yt-dlp.exe").exists() else "yt-dlp"

# 过滤:太短(<8min,多为剪辑碎片)或太长(>3h,多为直播录像)都不要
DURATION_FILTER = "duration > 480 & duration < 10800"


def load_sources(path: Path) -> list[tuple[str, str]]:
    """返回 [(label, yt-dlp 目标), ...]"""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"search:(\d+):(.+)", line)
        if m:
            n, q = m.group(1), m.group(2).strip()
            label = re.sub(r"[^\w一-鿿]+", "_", q)[:40]
            out.append((label, f"ytsearch{n}:{q}"))
        else:
            label = re.sub(r"[^\w]+", "_", line.split("/")[-1] or "url")[:40]
            out.append((label, line))
    return out


def run_source(label: str, target: str, out: Path, proxy: str, limit: int) -> int:
    dest = out / label
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        YTDLP,
        "--proxy", proxy,
        "-f", "bestaudio/best",
        "-x",  # 抽音频,保留原生编码(多为 opus,比转 m4a 省一半空间;下游反正统一转 16k wav)
        "--match-filter", DURATION_FILTER,
        "--write-info-json",
        "--no-write-playlist-metafiles",
        "--download-archive", str(out / "archive.txt"),  # 全局去重/断点
        "--sleep-interval", "2", "--max-sleep-interval", "6",
        "--retries", "5",
        "--no-progress",
        "-o", str(dest / "%(id)s.%(ext)s"),
        target,
    ]
    if limit:
        cmd += ["--max-downloads", str(limit)]
    print(f"== [{label}] {target}")
    # yt-dlp 达到 --max-downloads 时返回码 101,不算失败
    r = subprocess.run(cmd)
    if r.returncode not in (0, 101):
        print(f"  !! yt-dlp exit {r.returncode}", file=sys.stderr)
        return 0
    return len([p for p in dest.iterdir() if p.suffix in (".m4a", ".opus", ".webm", ".mp3")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="每源最多下载数(0=不限)")
    ap.add_argument("--proxy", default="http://127.0.0.1:7897")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for label, target in load_sources(Path(args.sources)):
        total += run_source(label, target, out, args.proxy, args.limit)
    print(f"done: {total} audio files under {out}")


if __name__ == "__main__":
    main()
