"""核心指标:误停率 / 漏停率 / 反应延迟分位数。

设计对应 PLAN.md §4:
- 误停率 (false_stop_rate):continue 族探针在反应窗内被 STOP 动作响应的比例;
- 漏停率 (missed_stop_rate):stop 族探针在截止时限内没有 STOP 动作的比例;
- 反应延迟:stop 族命中的 (t_action - t_onset),报 p50/p90;
- neutral 族(协作接话)不计两轴,单独报停止比例。

只统计发生在 agent 说话期间的探针(duplex 场景定义);
非说话期的探针计入 skipped。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import (
    CONTINUE_CLASSES,
    NEUTRAL_CLASSES,
    STOP_CLASSES,
    STOP_KINDS,
    Session,
)


@dataclass
class Report:
    n_continue: int = 0
    n_false_stop: int = 0
    n_stop: int = 0
    n_missed_stop: int = 0
    n_neutral: int = 0
    n_neutral_stopped: int = 0
    n_skipped: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def false_stop_rate(self) -> float | None:
        return self.n_false_stop / self.n_continue if self.n_continue else None

    @property
    def missed_stop_rate(self) -> float | None:
        return self.n_missed_stop / self.n_stop if self.n_stop else None

    def latency_percentile(self, q: float) -> float | None:
        """线性插值分位数,q ∈ [0, 100]。"""
        xs = sorted(self.latencies)
        if not xs:
            return None
        if len(xs) == 1:
            return xs[0]
        pos = (len(xs) - 1) * q / 100.0
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        frac = pos - lo
        return xs[lo] * (1 - frac) + xs[hi] * frac

    def summary(self) -> dict:
        return {
            "false_stop_rate": self.false_stop_rate,
            "missed_stop_rate": self.missed_stop_rate,
            "latency_p50": self.latency_percentile(50),
            "latency_p90": self.latency_percentile(90),
            "n_continue": self.n_continue,
            "n_stop": self.n_stop,
            "n_neutral": self.n_neutral,
            "neutral_stop_ratio": (
                self.n_neutral_stopped / self.n_neutral if self.n_neutral else None
            ),
            "n_skipped": self.n_skipped,
        }


def evaluate(
    sessions: list[Session],
    stop_deadline: float = 0.8,
    react_window: float = 2.0,
) -> Report:
    """stop_deadline:stop 族探针必须在 onset+deadline 内停,否则算漏停。
    react_window:continue 族探针 onset 后这段窗内的 STOP 都算误停归因。
    """
    rep = Report()
    for sess in sessions:
        stop_times = sorted(a.t for a in sess.actions if a.kind in STOP_KINDS)
        # 合法响应窗:stop 族探针 onset 后 react_window 内的 STOP 是正当响应,
        # 不得归因给附近的 continue 族探针(归因排除,防止相邻事件互相污染)
        legit = [(p.t_onset, p.t_onset + react_window)
                 for p in sess.probes if p.cls in STOP_CLASSES]
        for p in sess.probes:
            if not sess.agent_speaking_at(p.t_onset):
                rep.n_skipped += 1
                continue
            hits = [t for t in stop_times if p.t_onset <= t <= p.t_onset + react_window]
            if p.cls in CONTINUE_CLASSES:
                rep.n_continue += 1
                if [t for t in hits if not any(lo <= t <= hi for lo, hi in legit)]:
                    rep.n_false_stop += 1
            elif p.cls in STOP_CLASSES:
                rep.n_stop += 1
                in_deadline = [t for t in hits if t <= p.t_onset + stop_deadline]
                if in_deadline:
                    rep.latencies.append(in_deadline[0] - p.t_onset)
                else:
                    rep.n_missed_stop += 1
            elif p.cls in NEUTRAL_CLASSES:
                rep.n_neutral += 1
                if hits:
                    rep.n_neutral_stopped += 1
    return rep
