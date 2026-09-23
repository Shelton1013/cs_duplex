"""Writer LLM 客户端的校验器与加载器回归(离线,不需要服务)。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "synth"))

from writer_llm import (extract_json, lang_of, validate_scenario,
                        validate_wording)


def test_validate_scenario():
    ok = {"scenario": "broadband_fix", "user_q": "屋企上網成日斷線,可以點搞",
          "agent_answer": "同你檢查過條線,建議約{day}{time}師傅上門檢查,期間可以用手機數據頂住先"}
    assert validate_scenario(ok) is None
    # 未知槽位
    bad = dict(ok, agent_answer=ok["agent_answer"].replace("{day}", "{weekday}"))
    assert "unknown slot" in validate_scenario(bad)
    # 答案无槽
    bad = dict(ok, agent_answer="同你檢查過條線,建議約師傅上門檢查,期間可以用手機數據頂住先照顧")
    assert validate_scenario(bad) == "answer has no slot"
    # 问句槽不在答案里
    bad = dict(ok, user_q="我{month}上網斷線")
    assert validate_scenario(bad) == "q slot not in answer"
    # 散落花括号
    bad = dict(ok, agent_answer=ok["agent_answer"] + "{")
    assert validate_scenario(bad) == "stray brace"
    # 非粤语
    bad = dict(ok, agent_answer="ok we will send someone on {day} to check the router for you sir")
    assert validate_scenario(bad) == "answer not cantonese"


def test_validate_wording_and_lang():
    assert validate_wording("backchannel", "係囉") is None
    assert validate_wording("backchannel", "呢個應聲實在太長啦真係") == "length"
    assert validate_wording("correction_pattern", "唔係{new}咩?") is None
    assert validate_wording("correction_pattern", "唔係啊係另外嗰個先啱") is not None  # 缺 {new}
    assert validate_wording("correction_pattern", "係{new}唔係{old}") is not None  # 多余占位
    assert lang_of("係囉") == "yue"
    assert lang_of("okok") == "en"
    assert lang_of("no no 係佢先啱") == "mixed"


def test_extract_json_object_vs_array():
    obj = extract_json('废话```json\n{"a": [1, 2], "b": ["x"]}\n```尾巴')
    assert obj == {"a": [1, 2], "b": ["x"]}
    arr = extract_json('前缀 [ {"k": "v"} ] 后缀')
    assert arr == [{"k": "v"}]


def test_loaders_roundtrip(tmp_path):
    from synth.script import load_scenarios, make_dialogue
    from synth.inject import CORRECTION_PATTERNS, load_wordings
    import random

    scen = [{"scenario": "t1", "user_q": "我想查{month}賬單",
             "agent_answer": "幫你查到{month}嘅賬單係{amount},包埋上網同通話,下面逐項講"}]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(scen, ensure_ascii=False), encoding="utf-8")
    pool = load_scenarios(p)
    assert len(pool) == 1
    dlg = make_dialogue(random.Random(0), 3, pool)
    assert len(dlg) == 1 and dlg[0][1].slots

    w = tmp_path / "w.json"
    w.write_text(json.dumps({"correction_pattern": [["唔通係{new}?", "yue"],
                                                    ["冇佔位符喎", "yue"]]},
                            ensure_ascii=False), encoding="utf-8")
    before = len(CORRECTION_PATTERNS)
    added = load_wordings(w)
    assert added["correction_pattern"] == 1  # 缺 {new} 的被拒
    assert len(CORRECTION_PATTERNS) == before + 1
