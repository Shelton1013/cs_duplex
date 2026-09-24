"""【服务器】基线 ③:SoulX-Duplug-0.6B(开源即插即用语义 VAD / 状态预测)。

逐 160ms chunk 调用官方 TurnTakingEngine.process(),读取状态:
  idle / backchannel / blank → 不打断
  nonidle(检测到有语义的用户内容)/ speak(用户说完)→ agent 说话期间出现即记 CUT
决策时刻取 chunk 结束点(因果)。

ASR 语言:官方默认 paraformer(普通话)。为对粤语公平,默认改用其内置的
SenseVoice 并指定 --asr_lang(yue / auto),脚本生成临时 config 覆盖 asr 段。

环境:服务器 soulx-duplug env;需在 SoulX-Duplug 仓库目录下运行(相对模型路径)。
用法:
  cd /home/pxieaf/SoulX-Duplug
  CUDA_VISIBLE_DEVICES=0 python /home/pxieaf/home2/cs_duplex/scripts/baseline/run_soulx.py \
      --probes /home/pxieaf/home2/cs_duplex/data/probe_audio \
      --out /home/pxieaf/home2/cs_duplex/data/baselines/soulx_yue --asr_lang yue
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

CHUNK = 2560  # 160ms @16k,官方 chunk_size
STOP_STATES = {"nonidle", "speak"}


def make_config(src: Path, dst: Path, asr_lang: str) -> None:
    text = src.read_text(encoding="utf-8")
    asr_block = f"  asr:\n    model_name: sensevoice\n    language: {asr_lang}\n"
    if asr_lang == "paraformer":
        asr_block = "  asr:\n    model_name: paraformer\n"
    text = re.sub(r"  asr:\n(    .*\n?)+", asr_block, text)
    dst.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr_lang", default="yue", help="yue / auto / paraformer(官方默认)")
    ap.add_argument("--repo", default=os.getcwd())
    args = ap.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    cfg = repo / "config" / f"config_eval_{args.asr_lang}.yaml"
    make_config(repo / "config" / "config.yaml", cfg, args.asr_lang)

    from service.engine import TurnTakingEngine
    from service.model import load_turn_model
    model = load_turn_model(config_path=str(cfg))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(args.probes).iterdir() if (p / "user.wav").exists())
    t0 = time.time()
    for j, s in enumerate(sessions):
        if (out / f"{s.name}.json").exists():
            continue
        wav, sr = sf.read(s / "user.wav", dtype="float32")
        assert sr == 16000
        engine = TurnTakingEngine(model=model)  # 新 session → reset
        n = int(np.ceil(len(wav) / CHUNK)) * CHUNK
        buf = np.zeros(n, np.float32)
        buf[: len(wav)] = wav
        states, actions, prev = [], [], None
        for idx in range(n // CHUNK):
            res = engine.process(buf[idx * CHUNK:(idx + 1) * CHUNK]) or {}
            st = res.get("state", "blank")
            t = round((idx + 1) * CHUNK / 16000, 3)
            states.append([t, st, res.get("asr_buffer", "")])
            if st in STOP_STATES and prev not in STOP_STATES:
                actions.append({"t": t, "kind": "CUT"})
            prev = st
        (out / f"{s.name}.json").write_text(
            json.dumps({"actions": actions, "states": states}, ensure_ascii=False),
            encoding="utf-8")
        if (j + 1) % 50 == 0:
            print(f"{j + 1}/{len(sessions)}  {(time.time() - t0) / (j + 1):.2f}s/session", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
