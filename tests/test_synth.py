"""合成流水线回归:标签由构造保证的各不变量。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synth import TimingPriors, chunk_labels, compile_session, make_dialogue, to_harness_session
from synth.direct import est_dur
from harness.metrics import evaluate


def build(seed=1, n_sessions=8, lines=5):
    priors = TimingPriors.load(None, "test", seed)
    rng = random.Random(seed)
    out = []
    for i in range(n_sessions):
        sess = compile_session(rng, make_dialogue(rng, lines), priors, f"t{i}")
        out.append(sess)
    return out


def test_deterministic():
    a = build(seed=7)
    b = build(seed=7)
    assert a == b


def test_correction_invariants():
    n_corr = 0
    for sess in build(seed=2, n_sessions=20):
        by_role = [u for u in sess["agent_utts"] if u["role"] == "answer"]
        for e in sess["events"]:
            if e["cls"] != "correction":
                continue
            n_corr += 1
            g = e["gold"]
            flip = g["slot_flip"]
            r = g["resume"]
            host = next(u for u in by_role
                        if u.get("played_text") == r["played_text"])
            assert r["played_text"] + r["remainder"] == host["text"]
            span = next(s for s in host["slots"] if s["name"] == flip["slot"])
            assert span["value"] == flip["old"]
            reps = [u for u in sess["agent_utts"] if u["role"] == "repair"]
            if e["implicitness"] == "I2":
                # 质疑回声:文本=旧值回声,动作 PAUSE,repair 确认旧值并续讲余下内容
                assert flip["new"] is None and flip["old"] in e["text"]
                assert g["action"] == "PAUSE" and r["corrected_reply"] is None
                assert any(u["text"].endswith(r["remainder"]) and flip["old"] in u["text"]
                           for u in reps)
            else:
                # 类别不变量:纠正文本包含新值、不含旧值、新旧不同
                assert flip["new"] in e["text"]
                assert flip["old"] not in e["text"]
                assert flip["new"] != flip["old"]
                assert flip["new"] in r["corrected_reply"]
                assert any(flip["new"] in u["text"] for u in reps)
    assert n_corr >= 5  # 事件先验下应出现足量纠正


def test_implicitness_tiers():
    """结构性问题 2:纠正分级必须真的"去词法化"。"""
    from synth.inject import has_negation
    tiers = {"I0": 0, "I1": 0, "I2": 0}
    echo_pairs = 0
    for sess in build(seed=11, n_sessions=40):
        for e in sess["events"]:
            t = e.get("implicitness")
            if e["cls"] == "correction":
                tiers[t] += 1
                if t == "I0":
                    assert has_negation(e["text"]), e["text"]
                elif t == "I1":
                    assert not has_negation(e["text"]), e["text"]  # 正则不可解
            if t == "I2" and e["cls"] == "backchannel":
                echo_pairs += 1
                assert e["gold"]["action"] == "BACKCHANNEL_ACK"
                assert not has_negation(e["text"])
    assert all(v > 0 for v in tiers.values()), tiers
    assert echo_pairs > 0
    # I2 两侧文本同构:质疑回声与确认回声去掉标点后都只是槽值本身
    from synth.inject import ECHO_CONFIRM, ECHO_QUESTION
    assert ECHO_QUESTION.format(old="X").strip("?") == ECHO_CONFIRM.format(old="X").strip("。")


def test_events_within_host_and_truncation():
    for sess in build(seed=3, n_sessions=12):
        for e in sess["events"]:
            # 事件 onset 落在某条 agent 话语内(duplex 定义)
            assert any(u["t_start"] < e["t_onset"] < u["t_end"] + 1e-6
                       for u in sess["agent_utts"]), e
        for u in sess["agent_utts"]:
            if u.get("truncated"):
                # 截断话语的实际时长 < 全文估计时长
                assert (u["t_end"] - u["t_start"]) < est_dur(u["text"]) + 1e-6


def test_oracle_zero_on_harness():
    hs = [to_harness_session(s) for s in build(seed=4, n_sessions=15)]
    rep = evaluate(hs)
    assert rep.n_stop > 0 and rep.n_continue > 0
    assert rep.false_stop_rate == 0.0
    assert rep.missed_stop_rate == 0.0


def test_chunk_labels_causal_phases():
    found_cut = False
    for sess in build(seed=5, n_sessions=10):
        labels = chunk_labels(sess, hop=0.02)

        def label_at(t):
            return max((l for l in labels if l[0] <= t), key=lambda x: x[0])[1]

        for e in sess["events"]:
            g = e["gold"]
            if g["action"] == "CUT":
                found_cut = True
                # onset 后、cut 前:必须是可逆动作 ATTENUATE(证据不足不许 CUT)
                mid = (e["t_onset"] + g["cut_time"]) / 2
                assert label_at(mid) == "ATTENUATE", (mid, label_at(mid))
                # cut 后提交窗内:CUT
                assert label_at(g["cut_time"] + 0.05) == "CUT"
            # onset 之前的标签不受该事件影响(无未来泄漏)
            before = e["t_onset"] - 0.05
            assert label_at(before) not in ("ATTENUATE",) or any(
                o["t_onset"] <= before < max(o["t_end"], o["gold"].get("cut_time", 0))
                for o in sess["events"] if o is not e)
    assert found_cut
