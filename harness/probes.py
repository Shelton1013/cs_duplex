"""探针库与注入计划生成(时间轴层;音频渲染在后续脚本接 TTS/真人录音)。

探针文本按类别组织;标签由类别决定、与措辞无关(措辞可扩充,
后续可仿 MPFD 的 paraphrases.json + 本地 LLM 增广 + 词法校验器)。
铁律:评测探针的措辞/渲染与训练合成的生成器隔离。
"""
from __future__ import annotations

import random

from .schema import AgentUtterance, ProbeEvent

# 种子探针库:cls -> [(text, language, 切换点相对词位或 None)]
SEED_PROBES: dict[str, list[tuple[str, str]]] = {
    "backchannel": [
        ("嗯嗯", "yue"),
        ("係啊", "yue"),
        ("係係係", "yue"),
        ("啱啊", "yue"),
        ("好", "yue"),
        ("okok", "en"),
        ("yeah", "en"),
        ("right", "en"),
        ("嗯 ok", "mixed"),
    ],
    "correction": [
        ("唔係啊", "yue"),
        ("唔係,係下個月", "yue"),
        ("no no no", "en"),
        ("wait 唔係咁", "mixed"),
        ("唔係啊 I mean 下個月", "mixed"),
        ("等等,唔係嗰個 plan", "mixed"),
    ],
    "floor_claim": [
        ("我想問另外一樣嘢", "yue"),
        ("sorry 打斷一下", "mixed"),
        ("by the way 我仲想改埋地址", "mixed"),
    ],
    "side_speech": [
        ("阿媽幫我攞杯水", "yue"),
        ("你哋食咗飯未", "yue"),
        ("等陣先我聽緊電話", "yue"),
    ],
    "background_noise": [
        ("<tv_news_yue>", "yue"),
        ("<office_chatter>", "mixed"),
    ],
    "filler": [
        ("即係……", "yue"),
        ("erm", "en"),
        ("<cough>", ""),
    ],
    "cooperative_completion": [
        ("……三百蚊嗰個", "yue"),
    ],
}

# 探针放在 agent 话语的相对位置(受控变量之一)
DEFAULT_POSITIONS = (0.25, 0.5, 0.75)


def make_probe_plan(
    agent_utts: list[AgentUtterance],
    classes: list[str] | None = None,
    positions: tuple[float, ...] = DEFAULT_POSITIONS,
    probe_dur: float = 1.0,
    seed: int = 0,
) -> list[ProbeEvent]:
    """为每条 agent 话语在受控相对位置注入探针,类别轮转、措辞随机。

    确定性:同 seed 同输入 → 同计划(禁 Date.now 式不可复现性)。
    """
    rng = random.Random(seed)
    cls_cycle = list(classes or SEED_PROBES.keys())
    probes: list[ProbeEvent] = []
    i = 0
    for utt in agent_utts:
        dur = utt.t_end - utt.t_start
        if dur < probe_dur * 2:
            continue
        for pos in positions:
            cls = cls_cycle[i % len(cls_cycle)]
            i += 1
            text, lang = rng.choice(SEED_PROBES[cls])
            t0 = utt.t_start + dur * pos
            probes.append(
                ProbeEvent(
                    t_onset=round(t0, 3),
                    t_end=round(min(t0 + probe_dur, utt.t_end), 3),
                    cls=cls,
                    text=text,
                    language=lang,
                )
            )
    return probes
