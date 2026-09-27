"""Stage B 训练/验证/干净评测索引(复用 Stage 0 的 CosyVoice3 渲染数据)。

决策标签(1=STOP / 0=CONTINUE):
  STOP     :纠正、提问、抢话(对 agent 说且需要回应)、质疑回声(升调)
  CONTINUE :旁侧对话、应声、确认回声(降调)、对 agent 的单纯确认(「好呀」「知道喇」等)
因果截断点(只能看到截至当时的音频;cut=None 表示整句):
  CONTINUE 类:0.6s、1.0s、整句
  STOP 类    :1.0s、整句(0.6s 时 I1 纠正「係上…」与应声「係」不可分,不在此处标 STOP)
  语调回声   :只标整句(关键在句尾)
agent 上下文:随机客服话语的已播前缀;语调回声的前缀里必含被重复的数值。

输出:<out>/train.jsonl、val.jsonl(Stage 0 按说话人划分)、clean.jsonl(干净评测集,整句)
用法:python build_stage_b_index.py --stage0 /home/pxieaf/home2/data/stage0 \
        --clean /home/pxieaf/home2/data/stage0_eval_v2 --out /home/pxieaf/home2/data/stage_b
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from synth.script import SCENARIOS, make_line  # noqa: E402

ACK = {"好,冇問題", "OK,咁就咁啦", "好呀", "知道喇"}  # 对 agent 说但无需打断
ECHO_TEMPLATES = [
    "幫你查到嘅資料顯示係{v},如果冇問題我就幫你繼續處理",
    "系統入面記錄嘅係{v},另外我再同你確認埋其他資料",
    "根據你嘅合約,呢度寫住係{v},跟住落嚟我講下點樣生效",
    "我哋會安排喺{v},到時會有人同你聯絡",
]


def agent_prefix(rng: random.Random) -> str:
    _, line = make_line(rng, rng.choice(SCENARIOS))
    return line.text[: max(4, int(len(line.text) * rng.uniform(0.3, 0.8)))]


def echo_prefix(rng: random.Random, val: str) -> str:
    t = rng.choice(ECHO_TEMPLATES).format(v=val)
    end = t.index(val) + len(val)
    return t[: min(len(t), end + rng.randint(0, 6))]


def decision_label(r: dict) -> int:
    if r["task"] == "intonation":
        return int(r["label"] == 1)
    if r["label"] == 0:  # 旁侧对话
        return 0
    if r.get("subtype") == "backchannel" or r.get("text") in ACK:
        return 0
    return 1


def rows_for(r: dict, audio: str, rng: random.Random, clean: bool) -> list[dict]:
    lab = decision_label(r)
    if r["task"] == "intonation":
        prefix, cuts = echo_prefix(rng, r["text"]), [None]
    else:
        prefix = agent_prefix(rng)
        cuts = [None] if clean else ([0.6, 1.0, None] if lab == 0 else [1.0, None])
    qs = r["label"] if r["task"] == "intonation" else -1
    sub = r.get("subtype") or ("echo_q" if (r["task"] == "intonation" and r["label"]) else
                               "echo_s" if r["task"] == "intonation" else "")
    if r["task"] == "addressee" and sub == "pair":
        sub = "pair_to_agent" if r["label"] else "pair_side"
    return [{"audio": audio, "cut": c, "prefix": prefix, "label": lab, "qs": qs,
             "subtype": sub, "text": r["text"]} for c in cuts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage0", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    for lab in sorted(Path(args.stage0).glob("shard*/labels.jsonl")):
        for line in lab.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            sub = "addressee" if r["task"] == "addressee" else "intonation"
            audio = str(lab.parent / sub / f"{r['id']}.wav")
            splits[r["split"]] += rows_for(r, audio, rng, clean=False)
    clean = []
    for line in (Path(args.clean) / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        sub = "addressee" if r["task"] == "addressee" else "intonation"
        clean += rows_for(r, str(Path(args.clean) / sub / f"{r['id']}.wav"), rng, clean=True)
    for name, rows in (("train", splits["train"]), ("val", splits["val"]), ("clean", clean)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        pos = sum(x["label"] for x in rows)
        print(f"{name}: {len(rows)} examples, STOP={pos} CONTINUE={len(rows) - pos}")


if __name__ == "__main__":
    main()
