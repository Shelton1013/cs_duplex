"""RTHK 烽煙节目音频爬取(网络挖掘辅助轨,PLAN §2.5b)。

用途:timing 先验(diarization 统计)/ 编码器预训练 / 应声词表挖掘。
合规:RTHK 版权内容,仅限内部研究用途,不再分发;manifest 记录来源与用途边界。

两种模式:
  rss    — 解析官方 RSS(最近 ~40-50 集/节目)
  probe  — 按日期模式直接探测 archive.rthk.hk 的 12 个月存档
           (音频 URL 规整:.../<prog_path>/m4a/YYYYMMDD_<seg>.m4a)

用法:
  python fetch_rthk.py --mode rss --limit 5          # 冒烟测试
  python fetch_rthk.py --mode probe --days 365       # 全量存档
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (research; CantoDuplex timing-prior crawler)"}
LICENSE_NOTE = "RTHK copyright; internal research use only; do not redistribute"

# RTHK 在内地直连被重置,默认走本地 Clash mixed 端口;--proxy "" 可直连
OPENER = urllib.request.build_opener()


def set_proxy(proxy: str) -> None:
    global OPENER
    handler = urllib.request.ProxyHandler(
        {"http": proxy, "https": proxy} if proxy else {}
    )
    OPENER = urllib.request.build_opener(handler)

# 节目表:名称 -> (rss_url, archive 路径段)
SOURCES = {
    "HK2000": (  # 千禧年代(早晨烽煙,pid=288)
        "https://podcast.rthk.hk/podcast/radio1_HK2000.xml",
        "radio1/HK2000",
    ),
    "openline": (  # 自由風自由PHONE(下午烽煙,pid=289)
        "https://podcast.rthk.hk/podcast/radio1_openline_openview.xml",
        "radio1/openline_openview",
    ),
}
ARCHIVE_BASE = "https://archive.rthk.hk/mp3/radio/contentIndex"
MAX_SEG = 9  # 每天最多探测的分段号


def http_head_ok(url: str, timeout: int = 15) -> bool:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with OPENER.open(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def download(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with OPENER.open(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"  retry {attempt + 1}/{retries} {url}: {e}")
            time.sleep(3 * (attempt + 1))
    if tmp.exists():
        tmp.unlink()
    return False


def write_manifest(manifest: Path, entry: dict) -> None:
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def loaded_urls(manifest: Path) -> set:
    if not manifest.exists():
        return set()
    urls = set()
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            try:
                urls.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return urls


def crawl_rss(prog: str, rss_url: str, out: Path, manifest: Path, limit: int, sleep_s: float) -> int:
    req = urllib.request.Request(rss_url, headers=UA)
    with OPENER.open(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    seen = loaded_urls(manifest)
    n = 0
    for item in root.iter("item"):
        if limit and n >= limit:
            break
        title = (item.findtext("title") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        enc = item.find("enclosure")
        if enc is None:
            continue
        url = enc.get("url", "")
        if not url or url in seen:
            continue
        fname = url.rsplit("/", 1)[-1]
        dest = out / prog / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{prog}] {fname}  {title}")
        if download(url, dest):
            write_manifest(manifest, {
                "source": "rthk", "programme": prog, "mode": "rss",
                "title": title, "pubdate": pub, "url": url,
                "path": str(dest), "bytes": dest.stat().st_size,
                "license": LICENSE_NOTE,
            })
            n += 1
            time.sleep(sleep_s)
    return n


def crawl_probe(prog: str, prog_path: str, out: Path, manifest: Path,
                days: int, end: dt.date, limit: int, sleep_s: float) -> int:
    seen = loaded_urls(manifest)
    n = 0
    for d in range(days):
        date = end - dt.timedelta(days=d)
        if limit and n >= limit:
            break
        for seg in range(1, MAX_SEG + 1):
            fname = f"{date:%Y%m%d}_{seg}.m4a"
            url = f"{ARCHIVE_BASE}/{prog_path}/m4a/{fname}"
            if url in seen:
                continue
            if not http_head_ok(url):
                break  # 该日期分段到头(或该日无节目)
            dest = out / prog / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{prog}] {fname}")
            if download(url, dest):
                write_manifest(manifest, {
                    "source": "rthk", "programme": prog, "mode": "probe",
                    "title": "", "pubdate": f"{date:%Y-%m-%d}", "url": url,
                    "path": str(dest), "bytes": dest.stat().st_size,
                    "license": LICENSE_NOTE,
                })
                n += 1
                time.sleep(sleep_s)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rss", "probe"], default="rss")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "web" / "rthk"))
    ap.add_argument("--prog", default="all", help="HK2000 / openline / all")
    ap.add_argument("--limit", type=int, default=0, help="每节目最多下载数(0=不限)")
    ap.add_argument("--days", type=int, default=365, help="probe 模式回溯天数")
    ap.add_argument("--end", default="", help="probe 回溯起点日期 YYYYMMDD(默认今天)")
    ap.add_argument("--sleep", type=float, default=2.0, help="下载间隔秒(礼貌限速)")
    ap.add_argument("--proxy", default="http://127.0.0.1:7897",
                    help="HTTP 代理(默认本地 Clash mixed 端口;传空字符串直连)")
    args = ap.parse_args()
    set_proxy(args.proxy)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    end = dt.datetime.strptime(args.end, "%Y%m%d").date() if args.end else dt.date.today()
    progs = list(SOURCES) if args.prog == "all" else [args.prog]

    total = 0
    for prog in progs:
        rss_url, prog_path = SOURCES[prog]
        if args.mode == "rss":
            total += crawl_rss(prog, rss_url, out, manifest, args.limit, args.sleep)
        else:
            total += crawl_probe(prog, prog_path, out, manifest, args.days, end, args.limit, args.sleep)
    print(f"done: {total} files -> {out}")


if __name__ == "__main__":
    main()
