"""CantoDuplex 评测 harness 数据结构。

时间单位一律为秒(float,session 内相对时间)。
事件标签表与 docs/COLLECTION_PROTOCOL.md §7 一一对应。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# 事件类别 → 期望动作族
# continue 族:正确行为是不停(可 ACK/ATTENUATE)
# stop 族:正确行为是限时内停(CUT/PAUSE/YIELD)
CONTINUE_CLASSES = {
    "backchannel",
    "side_speech",
    "background_noise",
    "filler",
    "echo",
}
STOP_CLASSES = {
    "correction",
    "floor_claim",
}
# 协作接话单列:停不停都可辩护,不计入两轴,单独报告
NEUTRAL_CLASSES = {"cooperative_completion"}
EVENT_CLASSES = CONTINUE_CLASSES | STOP_CLASSES | NEUTRAL_CLASSES

STOP_KINDS = {"CUT", "PAUSE", "YIELD"}
SOFT_KINDS = {"ATTENUATE", "BACKCHANNEL_ACK", "CONTINUE", "RESUME"}
ACTION_KINDS = STOP_KINDS | SOFT_KINDS


@dataclass
class ProbeEvent:
    """一次用户侧事件(探针或真实标注事件)。"""

    t_onset: float
    t_end: float
    cls: str  # EVENT_CLASSES 之一
    text: str = ""
    language: str = ""  # yue / en / mixed
    t_switch: float | None = None  # 句内切换点(如有)

    def __post_init__(self) -> None:
        if self.cls not in EVENT_CLASSES:
            raise ValueError(f"unknown event class: {self.cls}")


@dataclass
class AgentUtterance:
    t_start: float
    t_end: float
    text: str = ""


@dataclass
class Action:
    """被测系统输出的一次控制动作。"""

    t: float
    kind: str  # ACTION_KINDS 之一

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"unknown action kind: {self.kind}")


@dataclass
class Session:
    session_id: str
    agent_utts: list[AgentUtterance] = field(default_factory=list)
    probes: list[ProbeEvent] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Session":
        d = json.loads(s)
        return cls(
            session_id=d["session_id"],
            agent_utts=[AgentUtterance(**u) for u in d.get("agent_utts", [])],
            probes=[ProbeEvent(**p) for p in d.get("probes", [])],
            actions=[Action(**a) for a in d.get("actions", [])],
        )

    def agent_speaking_at(self, t: float) -> bool:
        return any(u.t_start <= t < u.t_end for u in self.agent_utts)
