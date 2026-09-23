"""③ Director/Scheduler:剧本+事件计划 → 双通道时间轴 + 金标签(纯代码)。

- 所有时间量采样自真实 timing 先验(TimingPriors),绝不用规则偏移;
- 纠正事件强约束:onset 必须晚于被纠正槽值的播出时刻(现实因果);
- 停止类事件触发 agent 话语截断,纠正后追加 repair 话语(rerender 翻转槽);
- v0 时长用字符启发式估计,④ TTS 渲染后按真实对齐重新定时(时间轴存
  字符累积权重,重定时只需换刻度)。
"""
from __future__ import annotations

import random
import re
from .priors import TimingPriors
from .script import AgentLine
from .inject import EventPlan, plan_events

CJK = re.compile(r"[一-鿿　-〿＀-￯]")


def char_weights(text: str) -> list[float]:
    """逐字符时长权重(秒):CJK 0.17,拉丁词均摊,标点 0.06。"""
    ws: list[float] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isascii() and c.isalnum():
            j = i
            while j < len(text) and text[j].isascii() and text[j].isalnum():
                j += 1
            ws += [0.27 / (j - i)] * (j - i)  # 拉丁词整词 0.27s 均摊到字符
            i = j
        elif CJK.match(c):
            ws.append(0.17)
            i += 1
        else:
            ws.append(0.06)
            i += 1
    return ws


def est_dur(text: str) -> float:
    return round(max(0.4, sum(char_weights(text))), 3)


def char_index_at(text: str, t_start: float, t: float) -> int:
    """t 时刻已播到第几个字符(用于 cut_char / played_text)。"""
    acc = t_start
    for i, w in enumerate(char_weights(text)):
        acc += w
        if acc > t:
            return i
    return len(text)


def _place_events(rng: random.Random, priors: TimingPriors, line: AgentLine,
                  plans: list[EventPlan], utt_start: float, utt_dur: float) -> list[dict]:
    events: list[dict] = []
    utt_end = utt_start + utt_dur
    cut_time = None
    # 先放停止类(决定截断点)
    for p in plans:
        if p.cls not in ("correction", "floor_claim"):
            continue
        if p.cls == "correction":
            slot = next(s for s in line.slots if s.name == p.slot_flip["slot"])
            ws = char_weights(line.text)
            slot_end_t = utt_start + sum(ws[: slot.end])
            onset = slot_end_t + rng.uniform(0.25, 0.7)
            if onset > utt_end - 0.2:
                onset = min(slot_end_t + 0.1, utt_end - 0.2)
        else:
            onset = utt_start + rng.uniform(0.4, 0.8) * utt_dur
        delay = rng.uniform(0.25, 0.45)
        cut_time = round(min(onset + delay, utt_end), 3)
        action = "CUT" if p.cls == "correction" else "YIELD"
        ev = {
            "cls": p.cls, "text": p.text, "language": p.language,
            "t_onset": round(onset, 3),
            "t_end": round(onset + est_dur(p.text), 3),
            "gold": {
                "action": action,
                "phased": [[0.0, "ATTENUATE"], [round(delay, 3), action]],
                "cut_time": cut_time,
            },
        }
        if p.cls == "correction":
            idx = char_index_at(line.text, utt_start, cut_time)
            ev["gold"]["slot_flip"] = p.slot_flip
            ev["gold"]["resume"] = {
                "cut_char": idx,
                "played_text": line.text[:idx],
                "remainder": line.text[idx:],
                "corrected_reply": line.rerender(p.slot_flip["slot"], p.slot_flip["new"]),
            }
        events.append(ev)
        break  # 停止类每句至多一个
    # continue 类:避开彼此与停止事件,截断后的时段不放
    horizon = cut_time if cut_time is not None else utt_end
    for p in plans:
        if p.cls in ("correction", "floor_claim"):
            continue
        if p.cls == "backchannel":
            off = priors.sample("ins_offset", 0.3, max(0.4, utt_dur - 0.5))
            onset = utt_start + off
            dur = priors.sample("ins_dur", 0.15, 1.2)
        else:
            onset = utt_start + rng.uniform(0.25, 0.75) * utt_dur
            dur = est_dur(p.text)
        if onset > horizon - 0.3:
            continue  # 截断后无宿主语音,丢弃
        if any(abs(onset - e["t_onset"]) < 0.8 for e in events):
            continue
        action = "BACKCHANNEL_ACK" if p.cls == "backchannel" else "CONTINUE"
        events.append({
            "cls": p.cls, "text": p.text, "language": p.language,
            "t_onset": round(onset, 3), "t_end": round(onset + dur, 3),
            "gold": {"action": action, "phased": [[0.0, action]]},
        })
    return sorted(events, key=lambda e: e["t_onset"])


def compile_session(rng: random.Random, dialogue: list[tuple[str, AgentLine]],
                    priors: TimingPriors, session_id: str,
                    domain: str = "callcenter",
                    event_priors: dict | None = None) -> dict:
    t = 0.3
    agent_utts: list[dict] = []
    user_turns: list[dict] = []
    events_all: list[dict] = []
    for q_text, line in dialogue:
        q_dur = est_dur(q_text)
        user_turns.append({"t_start": round(t, 3), "t_end": round(t + q_dur, 3),
                           "text": q_text, "role": "question"})
        fto = priors.sample("fto", -0.4, 1.5)
        utt_start = t + q_dur + fto
        utt_dur = est_dur(line.text)
        plans = plan_events(rng, line, event_priors)
        events = _place_events(rng, priors, line, plans, utt_start, utt_dur)
        stop_ev = next((e for e in events if "cut_time" in e["gold"]), None)
        utt_end = stop_ev["gold"]["cut_time"] if stop_ev else utt_start + utt_dur
        utt = {"t_start": round(utt_start, 3), "t_end": round(utt_end, 3),
               "text": line.text, "role": "answer", "scenario": line.scenario,
               "truncated": stop_ev is not None,
               "slots": [{"name": s.name, "value": s.value,
                          "span": [s.start, s.end]} for s in line.slots]}
        if stop_ev is not None and stop_ev["cls"] == "correction":
            utt["played_text"] = stop_ev["gold"]["resume"]["played_text"]
        agent_utts.append(utt)
        events_all += events
        t = max(utt_end, max((e["t_end"] for e in events), default=utt_end))
        # 纠正 → repair 话语
        if stop_ev is not None and stop_ev["cls"] == "correction":
            gap = rng.uniform(0.2, 0.6)
            rep_text = "明白," + stop_ev["gold"]["resume"]["corrected_reply"]
            rep_start = t + gap
            rep_dur = est_dur(rep_text)
            agent_utts.append({"t_start": round(rep_start, 3),
                               "t_end": round(rep_start + rep_dur, 3),
                               "text": rep_text, "role": "repair",
                               "truncated": False, "slots": []})
            t = rep_start + rep_dur
        t += rng.uniform(0.6, 1.8)  # 下一轮前的思考间隙
    return {"session_id": session_id, "domain": domain,
            "timing_source": priors.group,
            "agent_utts": agent_utts, "user_turns": user_turns,
            "events": events_all}


def to_harness_session(sess: dict):
    """导出为 harness Session + oracle 动作(闭环校验:oracle 必须全零)。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.schema import Action, AgentUtterance, ProbeEvent, Session

    utts = [AgentUtterance(u["t_start"], u["t_end"], u["text"])
            for u in sess["agent_utts"]]
    probes, actions = [], []
    for e in sess["events"]:
        cls = e["cls"] if e["cls"] != "filler" else "filler"
        probes.append(ProbeEvent(e["t_onset"], e["t_end"], cls,
                                 e["text"], e["language"]))
        g = e["gold"]
        if g["action"] in ("CUT", "YIELD"):
            actions.append(Action(g["cut_time"], g["action"]))
        elif g["action"] == "BACKCHANNEL_ACK":
            actions.append(Action(round(e["t_onset"] + 0.15, 3), "BACKCHANNEL_ACK"))
    return Session(sess["session_id"], utts, probes, actions)
