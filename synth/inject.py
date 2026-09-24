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
# ---- 纠正分级(结构性问题 2:防"正则可解",对齐 MPFD 的 I0–I2 分级)----
# I0:显式否定词 → 词法可解(保留,作为最易档)
CORRECTION_PATTERNS = [
    ("唔係啊,我問嘅係{new}", "yue"),
    ("等等,唔係,係{new}先啱", "yue"),
    ("唔係,I mean {new}", "mixed"),
    ("no no,{new}先啱", "mixed"),
    ("wait,我講嘅係{new}", "mixed"),
]
# I1:无否定词,只说出新值 → 必须对照 agent 正在说的槽值才知道是纠正。
#     「係」开头者专考"纠正被误判为应声"的级联错误
CORRECTION_I1 = [
    ("係{new}呀", "yue"), ("{new}喎", "yue"), ("我講緊{new}", "yue"),
    ("係{new}先啱", "yue"), ("應該係{new}", "yue"), ("I mean {new}", "mixed"),
]
# I2:回声最小对——用户重复旧值,文本完全相同,只靠语调区分:
#     升调质疑「{old}?」→ PAUSE(agent 应确认);降调确认「{old}。」→ ACK
#     (标点仅作 TTS 语调控制信号;控制器只听音频,看不到标点)
ECHO_QUESTION = "{old}?"
ECHO_CONFIRM = "{old}。"

NEGATION_MARKERS = ("唔係", "唔啱", "錯", "等等", "no", "wait", "not")

DEFAULT_PRIORS = {
    "backchannel": 0.6,
    "correction": 0.35,
    "side_speech": 0.15,
    "filler": 0.15,
    "floor_claim": 0.12,
    "echo_confirm": 0.12,   # I2 最小对的 continue 侧
}
# 纠正内部的难度配比(I0 / I1 / I2-质疑回声)
CORRECTION_TIERS = (("I0", 0.4), ("I1", 0.4), ("I2", 0.2))
MAX_EVENTS_PER_LINE = 2


@dataclass
class EventPlan:
    cls: str
    text: str
    language: str
    slot_flip: dict | None = None  # correction 专属 {slot,old,new}
    meta: dict = field(default_factory=dict)


# ---- 类别不变量校验器(措辞扩充的准入门) ----
def has_negation(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in NEGATION_MARKERS)


def validate_correction(text: str, new_value: str, old_value: str,
                        tier: str = "I0") -> bool:
    ok = (new_value in text) and (old_value != new_value) and (old_value not in text)
    if tier == "I0":
        return ok and has_negation(text)
    if tier == "I1":  # I1 的定义:去掉否定词后仍是纠正 → 不得含否定词
        return ok and not has_negation(text)
    return ok


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
            if cls == "correction_pattern":
                if "{new}" not in text:
                    continue
                # 按是否含否定词自动分档:有→I0,无→I1(分级由校验器决定,不信模型)
                if not has_negation(text):
                    if text not in {t for t, _ in CORRECTION_I1}:
                        CORRECTION_I1.append((text, lang))
                        n += 1
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


def _pick_tier(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for tier, w in CORRECTION_TIERS:
        acc += w
        if r < acc:
            return tier
    return CORRECTION_TIERS[-1][0]


def plan_events(rng: random.Random, line: AgentLine,
                priors: dict | None = None) -> list[EventPlan]:
    priors = {**DEFAULT_PRIORS, **(priors or {})}
    plans: list[EventPlan] = []
    # 停止类事件(correction / floor_claim)每句至多一个,且互斥
    if rng.random() < priors["correction"]:
        flip = flip_slot(rng, line)
        if flip:
            tier = _pick_tier(rng)
            if tier == "I2":  # 质疑回声:重复旧值、升调 → PAUSE,无槽值翻转
                text = ECHO_QUESTION.format(old=flip["old"])
                plans.append(EventPlan(
                    "correction", text, "yue",
                    slot_flip={"slot": flip["slot"], "old": flip["old"], "new": None},
                    meta={"tier": "I2", "action": "PAUSE", "anchor_slot": flip["slot"]}))
            else:
                bank = CORRECTION_PATTERNS if tier == "I0" else CORRECTION_I1
                pat, lang = rng.choice(bank)
                text = pat.format(new=flip["new"])
                assert validate_correction(text, flip["new"], flip["old"], tier), text
                plans.append(EventPlan("correction", text, lang, slot_flip=flip,
                                       meta={"tier": tier, "anchor_slot": flip["slot"]}))
    elif rng.random() < priors["floor_claim"]:
        text, lang = rng.choice(FLOOR_CLAIMS)
        plans.append(EventPlan("floor_claim", text, lang))
    # I2 最小对的 continue 侧:降调确认回声(与质疑回声文本相同)
    if len(plans) < MAX_EVENTS_PER_LINE and line.slots and rng.random() < priors["echo_confirm"]:
        used = {p.meta.get("anchor_slot") for p in plans}
        cands = [s for s in line.slots if s.name not in used]
        if cands:
            s = rng.choice(cands)
            plans.append(EventPlan("backchannel", ECHO_CONFIRM.format(old=s.value), "yue",
                                   meta={"tier": "I2", "anchor_slot": s.name}))
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
