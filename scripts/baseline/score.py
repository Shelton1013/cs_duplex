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

CLASSES = ["backchannel", "side_speech", "floor_claim", "correction_I0", "correction_I1",
           "echo_question", "echo_confirm", "real_bc", "host_control"]
CONTINUE = {"backchannel", "side_speech", "echo_confirm", "real_bc", "host_control"}


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
        # 只有 agent 说话期间的停止才是"打断自己";agent 说完后的开口是正常接话
        a_end = max(u["t_end"] for u in meta["agent_utts"])
        acts = [a for a in json.loads(act_f.read_text(encoding="utf-8"))["actions"]
                if a["t"] <= a_end]
        sess = Session(
            meta["session_id"],
            [AgentUtterance(u["t_start"], u["t_end"], u["text"]) for u in meta["agent_utts"]],
            [ProbeEvent(p["t_onset"], p["t_end"], p["cls"], p["text"], p["language"])
             for p in meta["probes"]],
            [Action(a["t"], a["kind"]) for a in acts],
        )
        by_cls.setdefault(meta["probe_class"], []).append(sess)
    return by_cls, missing


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="name=actions_dir")
    ap.add_argument("--out", default=str(ROOT / "data" / "baselines" / "table.md"))
    args = ap.parse_args()

    # 两种视角(探针 session 只含一个探针,长归因窗是干净的):
    #  及时:漏停截止 0.8s;误停窗 = 探针后 agent 余下全部话语
    #  最终:只问"agent 说话期间有没有停",不论快慢(漏停截止放宽到 30s)
    views = [("及时(停≤0.8s)", 0.8), ("最终(说话期间停了没)", 30.0)]
    sections = []
    # 只显示该探针集里实际存在的类别
    first_dir = args.systems[0].split("=", 1)[1]
    present, _ = load(Path(args.probes), Path(first_dir))
    global CLASSES
    CLASSES = [c for c in CLASSES if present.get(c)]
    for title, deadline in views:
        rows = [f"### {title}", "",
                "| 系统 | 误停率↓ | 漏停率↓ | 延迟p50 | 延迟p90 | "
                + " | ".join(CLASSES) + " |",
                "|" + "---|" * (5 + len(CLASSES))]
        for spec in args.systems:
            name, d = spec.split("=", 1)
            by_cls, missing = load(Path(args.probes), Path(d))
            kw = {"stop_deadline": deadline, "react_window": 30.0}
            rep = evaluate([s for v in by_cls.values() for s in v], **kw)
            cells = []
            for c in CLASSES:
                r = evaluate(by_cls[c], **kw)
                v = r.false_stop_rate if c in CONTINUE else r.missed_stop_rate
                cells.append(f"{fmt(v)} (n={len(by_cls[c])})")
            rows.append(f"| {name} | {fmt(rep.false_stop_rate)} | {fmt(rep.missed_stop_rate)} | "
                        f"{fmt(rep.latency_percentile(50))} | {fmt(rep.latency_percentile(90))} | "
                        + " | ".join(cells) + " |")
            if missing and title == views[0][0]:
                print(f"[{name}] warning: {missing} sessions without actions")
        sections.append("\n".join(rows))
    note = ("\n按类别列:continue 类(backchannel/side_speech)为误停率,"
            "stop 类(floor_claim/correction)为漏停率。\n")
    table = "\n\n".join(sections) + "\n" + note
    print(table)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
