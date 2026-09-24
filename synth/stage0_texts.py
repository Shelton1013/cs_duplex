"""Stage 0 可学性验证用的文本库(说话对象 / 语调两个二分类任务)。

说话对象(addressee):
  to_agent = 用户对 agent 说的话(应声、纠正、提问、抢话)
  side     = 用户转头对身边人说的话(与通话无关)
  两类都含"有/无称呼词"的子集,避免模型只靠「阿媽」这类词法线索。
语调(intonation):
  同一个槽值,升调质疑「X?」vs 降调确认「X。」,文本相同只差语调。
"""
from __future__ import annotations

import itertools
import random

from .inject import BACKCHANNELS, CORRECTION_I1, CORRECTION_PATTERNS, FLOOR_CLAIMS
from .script import SLOT_BANK

# 语调任务用的扩充槽值(比 SLOT_BANK 多,覆盖数字、日期、时间、地点、套餐)
ECHO_VALUES = sorted(set(
    sum(SLOT_BANK.values(), []) + [
        "八十八蚊", "一千二百蚊", "三千蚊", "九十九蚊", "六百五十蚊", "二百八十八蚊",
        "一號", "十五號", "三十一號", "兩點半", "四點", "早上九點", "晏晝一點", "夜晚十點",
        "禮拜二", "禮拜四", "禮拜日", "聽日", "後日", "下個禮拜",
        "荃灣", "屯門", "將軍澳", "中環", "尖沙咀", "元朗", "大埔", "西環",
        "三個月", "半年", "一年", "兩年", "二十四個月", "十二期",
        "10GB", "50GB", "無限數據", "5G", "月費", "按金", "手續費", "違約金",
    ]))

TO_AGENT_EXTRA = [
    "可唔可以講慢少少", "即係點呀", "我想問下點樣申請", "你頭先講咩話", "咁要幾耐先有",
    "唔該再講多次", "有冇其他plan", "幾時生效呀", "要唔要另外俾錢", "我唔係好明",
    "咁即係要我做啲咩", "可唔可以send個email俾我", "好,咁跟住點", "你肯定呀嘛",
    "等等先,我拎支筆", "sorry我聽唔清楚", "咁張單幾時到", "我想cancel呀",
]

SIDE_PERSONS = ["阿媽", "阿爸", "細佬", "家姐", "老公", "老婆", "仔仔", "囡囡", "阿嫲", "阿Sam"]
SIDE_ACTIONS = [
    "幫我攞杯水", "熄咗個電視佢", "你食咗飯未", "門鐘響呀去開下門", "唔好咁嘈住",
    "我講緊電話呀", "關咗個火未", "聽日幾點出門", "幫我搵下支筆", "啲衫收咗未",
    "你攞咗我鎖匙呀", "快啲沖涼啦", "個外賣到咗未", "你開大咗冷氣呀", "去做功課啦",
]
SIDE_NO_VOCATIVE = [
    "啲衫收咗未呀", "個外賣到咗未", "邊個開咗個窗", "隻狗又吠呀", "水滾咗未",
    "今晚食乜嘢好", "部洗衣機響咗", "個電視細聲啲啦", "你去睡先啦", "條毛巾擺咗邊",
    "唔好玩電話住", "啲餸凍晒啦", "門口有人呀", "你幫手倒垃圾", "快啲換衫出門",
]


def to_agent_texts(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """(text, subtype)"""
    pool: list[tuple[str, str]] = []
    pool += [(t, "backchannel") for t, _ in BACKCHANNELS if not t.startswith("<")]
    pool += [(t, "floor_claim") for t, _ in FLOOR_CLAIMS]
    pool += [(t, "question") for t in TO_AGENT_EXTRA]
    for pat, _ in CORRECTION_PATTERNS + CORRECTION_I1:
        for v in rng.sample(ECHO_VALUES, 20):
            pool.append((pat.format(new=v), "correction"))
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]


def side_texts(rng: random.Random, n: int) -> list[tuple[str, str]]:
    pool = [(f"{p},{a}", "side_vocative") for p, a in itertools.product(SIDE_PERSONS, SIDE_ACTIONS)]
    pool += [(t, "side_plain") for t in SIDE_NO_VOCATIVE] * 4  # 提高无称呼词占比
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]
