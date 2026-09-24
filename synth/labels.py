"""因果 chunk 标签导出:训练慢通路的监督信号(PLAN §3.2)。

标签回答"给定截至此刻的证据,正确动作是什么":
  - agent 未说话:IDLE;说话中无事件:CONTINUE
  - 事件窗内按 gold.phased 演进(如纠正:onset 起 ATTENUATE,
    onset+delay 起 CUT)——证据不足先可逆,证据充分才不可逆
  - 截断后(cut_time 起)话语已停:IDLE
禁止未来泄漏:t 时刻标签只由 onset<=t 的事件决定。
"""
from __future__ import annotations


COMMIT_WINDOW = 0.16  # CUT/YIELD 决策在执行后保持的短提交窗(2 个 80ms chunk)


def _label_at(sess: dict, t: float) -> str:
    speaking = any(u["t_start"] <= t < u["t_end"] for u in sess["agent_utts"])
    label = "CONTINUE" if speaking else "IDLE"
    for e in sess["events"]:
        rel = t - e["t_onset"]
        if rel < 0:
            continue  # 因果:未来事件不影响当前标签
        g = e["gold"]
        win_end = max(e["t_end"], g.get("cut_time", 0.0) + COMMIT_WINDOW)
        if t >= win_end:
            continue
        cur = None
        for pt, lab in g["phased"]:
            if rel >= pt:
                cur = lab
        if cur in ("CUT", "YIELD", "PAUSE") and t >= g.get("cut_time", float("inf")) + COMMIT_WINDOW:
            cur = "IDLE"  # 截断已执行且过了提交窗
        if cur:
            label = cur
    return label


def chunk_labels(sess: dict, hop: float = 0.08) -> list[tuple[float, str]]:
    t_end = max(
        max((u["t_end"] for u in sess["agent_utts"]), default=0.0),
        max((e["t_end"] for e in sess["events"]), default=0.0),
    ) + 0.4
    out = []
    t = 0.0
    while t < t_end:
        out.append((round(t, 3), _label_at(sess, t)))
        t += hop
    return out
