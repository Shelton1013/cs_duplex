"""【本地】统一打分:任意基线的动作目录 → harness 指标(总体 + 按探针类别)。

用法:
  python score.py --probes data/probe_audio --systems vad=data/baselines/vad fo=data/baselines/fo
输出:markdown 表(打印 + 写 data/baselines/table.md)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.metrics import evaluate  # noqa: E402
from harness.schema import Action, AgentUtterance, ProbeEvent, Session  # noqa: E402

CLASSES = ["backchannel", "side_speech", "floor_claim", "correction_I0", "correction_I1"]
CONTINUE = {"backchannel", "side_speech"}


def load(probes: Path, actions_dir: Path):
    by_cls: dict[str, list[Session]] = {c: [] for c in CLASSES}
    missing = 0
    for s in sorted(probes.iterdir()):
        sj = s / "session.json"
        if not sj.exists():
            continue
        meta = json.loads(sj.read_text(encoding="utf-8"))
        act_f = actions_dir / f"{meta['session_id']}.json"
        if not act_f.exists():
            missing += 1
            continue
        acts = json.loads(act_f.read_text(encoding="utf-8"))["actions"]
        sess = Session(
            meta["session_id"],
            [AgentUtterance(u["t_start"], u["t_end"], u["text"]) for u in meta["agent_utts"]],
            [ProbeEvent(p["t_onset"], p["t_end"], p["cls"], p["text"], p["language"])
             for p in meta["probes"]],
            [Action(a["t"], a["kind"]) for a in acts],
        )
        by_cls[meta["probe_class"]].append(sess)
    return by_cls, missing


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="name=actions_dir")
    ap.add_argument("--out", default=str(ROOT / "data" / "baselines" / "table.md"))
    args = ap.parse_args()

    rows = ["| 系统 | 误停率↓ | 漏停率↓ | 延迟p50 | 延迟p90 | "
            + " | ".join(f"{c}" for c in CLASSES) + " |",
            "|" + "---|" * (5 + len(CLASSES))]
    for spec in args.systems:
        name, d = spec.split("=", 1)
        by_cls, missing = load(Path(args.probes), Path(d))
        all_s = [s for v in by_cls.values() for s in v]
        rep = evaluate(all_s)
        cells = []
        for c in CLASSES:
            r = evaluate(by_cls[c])
            v = r.false_stop_rate if c in CONTINUE else r.missed_stop_rate
            cells.append(f"{fmt(v)} (n={len(by_cls[c])})")
        rows.append(f"| {name} | {fmt(rep.false_stop_rate)} | {fmt(rep.missed_stop_rate)} | "
                    f"{fmt(rep.latency_percentile(50))} | {fmt(rep.latency_percentile(90))} | "
                    + " | ".join(cells) + " |")
        if missing:
            print(f"[{name}] warning: {missing} sessions without actions")
    note = ("\n按类别列:continue 类(backchannel/side_speech)为误停率,"
            "stop 类(floor_claim/correction)为漏停率;漏停截止 0.8s。\n")
    table = "\n".join(rows) + "\n" + note
    print(table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
