"""合成流水线 v0 生成入口(②③ 时间轴层;④ TTS 渲染另接)。

用法:
  python gen_v0.py --n 50 --group rthk
产出:data/synth/v0/<session_id>.json + summary 打印
自检:每个 session 导出 harness Session + oracle 动作跑指标,必须全零。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synth import TimingPriors, chunk_labels, compile_session, make_dialogue, to_harness_session
from harness.metrics import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--lines", type=int, default=5, help="每 session 轮数")
    ap.add_argument("--group", default="rthk", help="timing 先验域:rthk/youtube")
    ap.add_argument("--stats", default=str(ROOT / "data" / "diar" / "timing_stats.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "synth" / "v0"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenarios", default="", help="Writer LLM 场景库 json(可选)")
    ap.add_argument("--wordings", default="", help="Writer LLM 措辞库 json(可选)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    priors = TimingPriors.load(args.stats, args.group, args.seed)
    rng = random.Random(args.seed)
    scenarios = None
    if args.scenarios:
        from synth.script import load_scenarios
        scenarios = load_scenarios(args.scenarios)
        print(f"loaded {len(scenarios)} LLM scenarios")
    if args.wordings:
        from synth.inject import load_wordings
        print("wordings added:", load_wordings(args.wordings))

    cls_count: Counter = Counter()
    hsessions = []
    for i in range(args.n):
        sid = f"synth_{args.group}_{i:05d}"
        dialogue = make_dialogue(rng, args.lines, scenarios)
        sess = compile_session(rng, dialogue, priors, sid)
        # 附 chunk 标签(80ms 因果监督轨)
        sess["chunk_labels"] = chunk_labels(sess)
        (out / f"{sid}.json").write_text(
            json.dumps(sess, ensure_ascii=False, indent=1), encoding="utf-8")
        for e in sess["events"]:
            cls_count[e["cls"]] += 1
        hsessions.append(to_harness_session(sess))

    rep = evaluate(hsessions)
    print(f"{args.n} sessions -> {out}")
    print("event mix:", dict(cls_count))
    print("oracle check:", json.dumps(rep.summary(), ensure_ascii=False))
    assert rep.false_stop_rate in (None, 0.0) and rep.missed_stop_rate in (None, 0.0), \
        "oracle 自检未全零:标签或时间轴有 bug"
    print("oracle check PASSED (误停/漏停全零)")


if __name__ == "__main__":
    main()
