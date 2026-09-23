"""场景剧本(v0 模板库;Writer LLM 接入后替换内容层,槽位协议不变)。

核心协议:agent 台词带**类型化槽位**(记录字符 span)。
纠正事件 = 代码翻转某槽值 → 纠正语义/恢复文本/未竟原句三重标签由构造免费产出。
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

SLOT_BANK: dict[str, list[str]] = {
    "month": ["今個月", "下個月", "上個月"],
    "amount": ["一百五十蚊", "二百二十蚊", "三百蚊", "五百八十蚊"],
    "plan": ["基本plan", "家庭plan", "5G無限plan", "學生plan"],
    "day": ["禮拜一", "禮拜三", "禮拜五", "禮拜六"],
    "time": ["朝早十點", "下晝三點", "夜晚七點"],
    "district": ["旺角", "銅鑼灣", "沙田", "觀塘"],
}

# (scenario_id, 用户问句模板, agent 回答模板)——回答尽量长(给事件留空间)且含 ≥1 槽
SCENARIOS: list[tuple[str, str, str]] = [
    ("billing", "唔該我想查吓{month}嘅賬單",
     "幫你查到{month}嘅賬單一共係{amount},包括咗基本月費、上網數據同埋通話分鐘,詳細分項我而家逐樣同你講"),
    ("plan_change", "我想轉做{plan}呀",
     "冇問題,{plan}每個月係{amount},轉咗之後由下一個賬單周期開始生效,而家用剩嘅數據會自動帶落去"),
    ("repair_appt", "我屋企個路由器壞咗,想約師傅上門",
     "幫你約咗{day}{time}師傅上門檢查,師傅出發之前會先打電話畀你,麻煩你保持電話暢通"),
    ("address", "我搬咗屋,想改地址",
     "好嘅,已經幫你將登記地址改做{district},由{month}開始所有信件同賬單都會寄去新地址"),
    ("roaming", "我下星期去內地,可唔可以用漫遊",
     "你而家用緊嘅{plan}喺內地可以直接用,漫遊數據每日封頂收{amount},唔使另外申請"),
    ("refund", "你哋上個月好似多收咗我錢",
     "同你核對過{month}嘅賬單,系統確認多收咗{amount},我哋會喺三個工作天之內退返落你張卡度"),
    ("cancel", "我想cut咗而家個plan",
     "明白,取消{plan}嘅話會喺{month}月尾生效,之後就唔會再收費,不過而家cut嘅話早繳優惠就會取消"),
]


@dataclass
class Slot:
    name: str
    value: str
    start: int  # 在 text 中的字符 span
    end: int


@dataclass
class AgentLine:
    scenario: str
    template: str
    text: str
    slots: list[Slot] = field(default_factory=list)

    def rerender(self, flip_name: str, new_value: str) -> str:
        """槽值翻转后的重渲染(纠正后的正确答复文本)。"""
        values = {s.name: (new_value if s.name == flip_name else s.value)
                  for s in self.slots}
        return self.template.format(**values)


def _fill(template: str, values: dict) -> tuple[str, list[Slot]]:
    slots: list[Slot] = []
    text = ""
    pos = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        text += template[pos:m.start()]
        v = values[m.group(1)]
        slots.append(Slot(m.group(1), v, len(text), len(text) + len(v)))
        text += v
        pos = m.end()
    text += template[pos:]
    return text, slots


def make_line(rng: random.Random, scenario: tuple[str, str, str]) -> tuple[str, AgentLine]:
    sid, q_tpl, a_tpl = scenario
    names = set(re.findall(r"\{(\w+)\}", q_tpl + a_tpl))
    values = {n: rng.choice(SLOT_BANK[n]) for n in names}
    q_text, _ = _fill(q_tpl, values)
    a_text, a_slots = _fill(a_tpl, values)
    return q_text, AgentLine(sid, a_tpl, a_text, a_slots)


def make_dialogue(rng: random.Random, n_lines: int,
                  scenarios: list[tuple[str, str, str]] | None = None
                  ) -> list[tuple[str, AgentLine]]:
    """一个 session = n 轮(用户问 → agent 长答)。场景不重复采样直到用尽。"""
    pool = scenarios or SCENARIOS
    order = rng.sample(pool, min(n_lines, len(pool)))
    return [make_line(rng, sc) for sc in order]


def load_scenarios(path) -> list[tuple[str, str, str]]:
    """加载 Writer LLM 产物(writer_llm.py --task scenarios),防御性再校验。"""
    import json
    from pathlib import Path
    out: list[tuple[str, str, str]] = []
    for it in json.loads(Path(path).read_text(encoding="utf-8")):
        q, a = it["user_q"], it["agent_answer"]
        names = set(re.findall(r"\{(\w+)\}", q + a))
        if names <= set(SLOT_BANK) and re.search(r"\{\w+\}", a):
            out.append((it["scenario"], q, a))
    if not out:
        raise ValueError(f"no valid scenarios in {path}")
    return out
