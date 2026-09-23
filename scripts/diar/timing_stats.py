"""从 RTTM 提取 timing 先验统计 v2(PLAN §2:合成数据的 timing 采样源)。

v2 修正:
  1) 相邻不同说话人段分两类:
     - transfer(真话轮转移):下段结束晚于上段结束(floor 真的换了)
       fto = 下段.start - 上段.end(负=重叠交接,正=留空隙)
     - insertion(嵌套插话):下段完全被上段包住(应声/短插话),
       不再混进 FTO(v1 的 -10s 级伪影来源)
  2) 按来源分组(data/diar/ 下第一级目录,如 rthk / youtube)分别出分布,
     合成侧按目标场景选组采样(烽煙=电话客服域,youtube=闲谈域)。

产出字段:
  turn_dur / fto / fto_negative_ratio / overlap_dur / overlap_ratio
  ins_dur(插话时长)/ ins_offset(插话相对宿主话轮起点偏移)
每分布保留原始样本(截断至 cap),合成侧直接 random.choice。

用法:python timing_stats.py [--diar <rttm根>] [--out stats.json]
纯 CPU、无三方依赖。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_rttm(path: Path) -> list[tuple[str, float, float]]:
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            t0, dur, spk = float(parts[3]), float(parts[4]), parts[7]
            segs.append((spk, t0, t0 + dur))
    return sorted(segs, key=lambda s: s[1])


def pairwise_overlaps(segs: list[tuple[str, float, float]]) -> list[float]:
    out = []
    active: list[tuple[str, float, float]] = []
    for spk, s, e in segs:
        active = [a for a in active if a[2] > s]
        for aspk, as_, ae in active:
            if aspk != spk:
                out.append(min(ae, e) - s)
        active.append((spk, s, e))
    return [d for d in out if d > 0]


def stats_of(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    ys = sorted(xs)

    def pct(q: float) -> float:
        pos = (len(ys) - 1) * q / 100
        lo = int(pos)
        hi = min(lo + 1, len(ys) - 1)
        return round(ys[lo] + (ys[hi] - ys[lo]) * (pos - lo), 3)

    return {"n": len(ys), "p10": pct(10), "p50": pct(50), "p90": pct(90),
            "mean": round(sum(ys) / len(ys), 3)}


class Acc:
    def __init__(self) -> None:
        self.turn_dur: list[float] = []
        self.fto: list[float] = []
        self.overlap_dur: list[float] = []
        self.overlap_ratio: list[float] = []
        self.ins_dur: list[float] = []
        self.ins_offset: list[float] = []
        self.n_files = 0

    def add_file(self, segs: list[tuple[str, float, float]]) -> None:
        self.n_files += 1
        self.turn_dur += [e - s for _, s, e in segs]
        for (spk1, s1, e1), (spk2, s2, e2) in zip(segs, segs[1:]):
            if spk1 == spk2:
                continue
            if e2 > e1:  # 真话轮转移
                self.fto.append(round(s2 - e1, 3))
            else:  # 嵌套插话(应声/短插话)
                self.ins_dur.append(round(e2 - s2, 3))
                self.ins_offset.append(round(s2 - s1, 3))
        ov = pairwise_overlaps(segs)
        self.overlap_dur += ov
        talk = sum(e - s for _, s, e in segs)
        if talk > 0:
            self.overlap_ratio.append(round(sum(ov) / talk, 4))

    def report(self, cap: int, rng: random.Random) -> dict:
        def capped(xs: list[float]) -> list[float]:
            return xs if len(xs) <= cap else rng.sample(xs, cap)

        return {
            "n_files": self.n_files,
            "summary": {
                "turn_dur": stats_of(self.turn_dur),
                "fto": stats_of(self.fto),
                "fto_negative_ratio": (
                    round(sum(1 for x in self.fto if x < 0) / len(self.fto), 4)
                    if self.fto else None
                ),
                "overlap_dur": stats_of(self.overlap_dur),
                "overlap_ratio": stats_of(self.overlap_ratio),
                "ins_dur": stats_of(self.ins_dur),
                "ins_offset": stats_of(self.ins_offset),
            },
            "samples": {
                "turn_dur": capped(self.turn_dur),
                "fto": capped(self.fto),
                "overlap_dur": capped(self.overlap_dur),
                "ins_dur": capped(self.ins_dur),
                "ins_offset": capped(self.ins_offset),
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diar", default=str(Path(__file__).resolve().parents[2] / "data" / "diar"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "diar" / "timing_stats.json"))
    ap.add_argument("--cap", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.diar)
    groups: dict[str, Acc] = {}
    total = Acc()
    for rttm in sorted(root.rglob("*.rttm")):
        segs = parse_rttm(rttm)
        if len(segs) < 4:
            continue
        rel = rttm.relative_to(root)
        gname = rel.parts[0] if len(rel.parts) > 1 else "_root"
        groups.setdefault(gname, Acc()).add_file(segs)
        total.add_file(segs)

    rng = random.Random(0)
    result = {
        "all": total.report(args.cap, rng),
        "by_source": {g: acc.report(args.cap, rng) for g, acc in sorted(groups.items())},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in ["all"] + list(result["by_source"]):
        summ = result["all"]["summary"] if g == "all" else result["by_source"][g]["summary"]
        nf = result["all"]["n_files"] if g == "all" else result["by_source"][g]["n_files"]
        print(f"== {g} ({nf} files)")
        print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
