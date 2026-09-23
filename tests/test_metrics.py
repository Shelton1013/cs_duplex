"""指标回归测试:oracle 全对、naive-VAD 全误停、哑巴全漏停。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import Action, AgentUtterance, ProbeEvent, Session, evaluate, make_probe_plan


def make_session(actions):
    return Session(
        session_id="t1",
        agent_utts=[AgentUtterance(0.0, 10.0, "帮你查到下个月账单是三百蚊")],
        probes=[
            ProbeEvent(2.0, 2.6, "backchannel", "嗯嗯", "yue"),
            ProbeEvent(5.0, 6.2, "correction", "唔係啊 I mean 下個月", "mixed"),
            ProbeEvent(8.0, 8.8, "side_speech", "阿媽幫我攞杯水", "yue"),
        ],
        actions=actions,
    )


def test_oracle_all_clean():
    # oracle:应声/旁语不停,纠正 300ms 内 CUT
    sess = make_session([Action(5.3, "CUT")])
    rep = evaluate([sess])
    assert rep.false_stop_rate == 0.0
    assert rep.missed_stop_rate == 0.0
    assert abs(rep.latency_percentile(50) - 0.3) < 1e-9


def test_naive_vad_false_stops():
    # naive VAD:每个探针 onset 后 200ms 都 CUT
    sess = make_session([Action(2.2, "CUT"), Action(5.2, "CUT"), Action(8.2, "CUT")])
    rep = evaluate([sess])
    assert rep.false_stop_rate == 1.0  # 应声+旁语都被误停
    assert rep.missed_stop_rate == 0.0


def test_silent_system_misses():
    sess = make_session([])
    rep = evaluate([sess])
    assert rep.false_stop_rate == 0.0
    assert rep.missed_stop_rate == 1.0


def test_late_stop_counts_as_missed():
    # 纠正在 deadline(0.8s) 之后才停 → 漏停
    sess = make_session([Action(6.5, "CUT")])
    rep = evaluate([sess])
    assert rep.missed_stop_rate == 1.0
    # 但这次晚停落在 side_speech(8.0) 窗外、backchannel(2.0) 窗外 → 无误停
    assert rep.false_stop_rate == 0.0


def test_probe_outside_agent_speech_skipped():
    sess = Session(
        session_id="t2",
        agent_utts=[AgentUtterance(0.0, 4.0)],
        probes=[ProbeEvent(6.0, 6.5, "correction", "唔係", "yue")],
        actions=[],
    )
    rep = evaluate([sess])
    assert rep.n_skipped == 1
    assert rep.n_stop == 0


def test_probe_plan_deterministic():
    utts = [AgentUtterance(0.0, 12.0), AgentUtterance(15.0, 24.0)]
    p1 = make_probe_plan(utts, seed=7)
    p2 = make_probe_plan(utts, seed=7)
    assert p1 == p2
    assert len(p1) == 6  # 2 utt × 3 位置
    for p in p1:
        assert any(u.t_start < p.t_onset < u.t_end for u in utts)
