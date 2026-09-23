"""② Event Injector:事件类别由代码决定(标签由构造保证),措辞只加多样性。

每个事件计划包含:类别、措辞、语言、放置约束、金标签要素(纠正=槽位翻转)。
措辞库为种子版;扩充走 LLM+类别不变量校验器(MPFD paraphrase 配方),
校验器在本文件(validate_*),扩充措辞必须过校验才入库。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .script import SLOT_BANK, AgentLine

# ---- 措辞库(cls -> [(text_or_pattern, language)]) ----
BACKCHANNELS = [
    ("嗯嗯", "yue"), ("係啊", "yue"), ("係係係", "yue"), ("啱啊", "yue"),
    ("好", "yue"), ("明白", "yue"), ("okok", "en"), ("yeah", "en"),
    ("嗯 ok", "mixed"),
]
FILLERS = [("即係……", "yue"), ("erm", "en"), ("<cough>", "")]
SIDE_SPEECH = [
    ("阿媽幫我攞杯水", "yue"), ("你哋食咗飯未啊", "yue"),
    ("等陣先我聽緊電話", "yue"), ("細佬熄咗個電視佢", "yue"),
]
FLOOR_CLAIMS = [
    ("唔好意思打斷一下,我仲想問埋另一樣嘢", "yue"),
    ("sorry 打斷一下,仲有個問題", "mixed"),
    ("by the way 我仲想改埋地址", "mixed"),
]
# 纠正模式:{new} 必须出现(类别不变量);语言按模式标注
CORRECTION_PATTERNS = [
    ("唔係啊,我問嘅係{new}", "yue"),
    ("等等,係{new}先啱", "yue"),
    ("唔係,I mean {new}", "mixed"),
    ("no no,{new}先啱", "mixed"),
    ("wait,我講嘅係{new}", "mixed"),
]

DEFAULT_PRIORS = {
    "backchannel": 0.6,
    "correction": 0.35,
    "side_speech": 0.15,
    "filler": 0.15,
    "floor_claim": 0.12,
}
MAX_EVENTS_PER_LINE = 2


@dataclass
class EventPlan:
    cls: str
    text: str
    language: str
    slot_flip: dict | None = None  # correction 专属 {slot,old,new}
    meta: dict = field(default_factory=dict)


# ---- 类别不变量校验器(措辞扩充的准入门) ----
def validate_correction(text: str, new_value: str, old_value: str) -> bool:
    return (new_value in text) and (old_value != new_value) and (old_value not in text)


def validate_backchannel(text: str) -> bool:
    return len(text.replace(" ", "")) <= 6  # 应声必须短


def load_wordings(path) -> dict:
    """加载 Writer LLM 措辞扩充(writer_llm.py --task wordings),
    防御性再校验后并入模块措辞库。返回各类新增计数。"""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    added = {}
    banks = {"backchannel": BACKCHANNELS, "filler": FILLERS,
             "side_speech": SIDE_SPEECH, "floor_claim": FLOOR_CLAIMS,
             "correction_pattern": CORRECTION_PATTERNS}
    for cls, bank in banks.items():
        existing = {t for t, _ in bank}
        n = 0
        for text, lang in data.get(cls, []):
            if text in existing:
                continue
            if cls == "backchannel" and not validate_backchannel(text):
                continue
            if cls == "correction_pattern" and "{new}" not in text:
                continue
            bank.append((text, lang))
            existing.add(text)
            n += 1
        added[cls] = n
    return added


def flip_slot(rng: random.Random, line: AgentLine) -> dict | None:
    if not line.slots:
        return None
    slot = rng.choice(line.slots)
    alts = [v for v in SLOT_BANK[slot.name] if v != slot.value]
    if not alts:
        return None
    return {"slot": slot.name, "old": slot.value, "new": rng.choice(alts)}


def plan_events(rng: random.Random, line: AgentLine,
                priors: dict | None = None) -> list[EventPlan]:
    priors = priors or DEFAULT_PRIORS
    plans: list[EventPlan] = []
    # 停止类事件(correction / floor_claim)每句至多一个,且互斥
    if rng.random() < priors["correction"]:
        flip = flip_slot(rng, line)
        if flip:
            pat, lang = rng.choice(CORRECTION_PATTERNS)
            text = pat.format(new=flip["new"])
            assert validate_correction(text, flip["new"], flip["old"])
            plans.append(EventPlan("correction", text, lang, slot_flip=flip))
    elif rng.random() < priors["floor_claim"]:
        text, lang = rng.choice(FLOOR_CLAIMS)
        plans.append(EventPlan("floor_claim", text, lang))
    # continue 类事件
    for cls, bank, p in (("backchannel", BACKCHANNELS, priors["backchannel"]),
                         ("side_speech", SIDE_SPEECH, priors["side_speech"]),
                         ("filler", FILLERS, priors["filler"])):
        if len(plans) >= MAX_EVENTS_PER_LINE:
            break
        if rng.random() < p:
            text, lang = rng.choice(bank)
            if cls == "backchannel":
                assert validate_backchannel(text)
            plans.append(EventPlan(cls, text, lang))
    return plans[:MAX_EVENTS_PER_LINE]
