"""【服务器】Freeze-Omni 基线:流式喂 user.wav,读 chunk 级状态头的开口决策。

适配逻辑移植自 MPFD 已验证的 FreezeOmniBaseline(源码级):
  - 160ms/2560 采样 chunk 调用 pipeline.speech_dialogue;
  - stat=='ss' = 模型决定开口。agent 正在说话时开口 = 打断自己去回应 → 记为 CUT;
  - 每次判定后强制 stat 回 'cl',持续监听整段,记录每一个开口时刻;
  - 决策时刻取 chunk 结束点 (idx+1)*0.16(因果:该 chunk 听完才能判)。
输入 user.wav = 理想 AEC 下的用户麦克风通道(模型听不到自己的声音)。

运行环境:Freeze-Omni 自己的 env(torch==2.2.0, transformers==4.45.2)。
本脚本只依赖 torch / soundfile / numpy,不 import cs_duplex 其他模块。

用法(服务器):
  python run_freeze_omni.py --probes <probe_audio 目录> --out <actions 目录> \
      --fo_repo /home/pxieaf/Freeze-Omni \
      --model_path /home/pxieaf/Freeze-Omni/checkpoints \
      --llm_path /home/pxieaf/Freeze-Omni/Qwen2-7B-Instruct
  多卡分片:CUDA_VISIBLE_DEVICES=k python run_freeze_omni.py ... --shard k/4
输出:<out>/<session_id>.json = {"actions":[{"t":..,"kind":"CUT"}], "onsets":[..]}
断点续跑:已有输出的 session 跳过。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def load_fo(fo_repo: str, model_path: str, llm_path: str):
    import torch
    # MPFD 踩过的坑:音频编码器 conv2d 报 cuDNN "unable to find an engine" → 关 cuDNN
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, fo_repo)
    spec = importlib.util.spec_from_file_location(
        "fo_inference", os.path.join(fo_repo, "bin", "inference.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 只加载定义,主流程在 __main__ 下不执行
    cfg = SimpleNamespace(model_path=model_path, llm_path=llm_path,
                          top_p=0.8, top_k=20, temperature=0.8)
    return mod.audioEncoderProcessor, mod.inferencePipeline(cfg)


def run_one(proc_cls, pipeline, wav_path: Path, role: str) -> list[float]:
    import soundfile as sf
    import torch
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = torch.tensor(wav)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    proc = proc_cls()
    chunk = proc.get_chunk_size()  # 2560 = 160ms
    n = math.ceil(wav.shape[0] / chunk) * chunk
    buf = torch.zeros(n)
    buf[: wav.shape[0]] = wav
    outputs = pipeline.speech_dialogue(None, stat="pre", role=role)
    onsets = []
    for idx, i in enumerate(range(0, n, chunk)):
        fbank = proc.process(buf[i:i + chunk])
        outputs = pipeline.speech_dialogue(fbank, **outputs)
        if outputs.get("stat") == "ss":
            onsets.append(round((idx + 1) * chunk / 16000, 3))
        outputs["stat"] = "cl"
    return onsets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fo_repo", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--llm_path", required=True)
    ap.add_argument("--role", default="You are a helpful assistant.")
    ap.add_argument("--shard", default="0/1", help="k/N:只跑第 k 片")
    args = ap.parse_args()

    k, n_shards = map(int, args.shard.split("/"))
    probes, out = Path(args.probes), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in probes.iterdir() if (p / "user.wav").exists())
    todo = [s for i, s in enumerate(sessions)
            if i % n_shards == k and not (out / f"{s.name}.json").exists()]
    print(f"{len(sessions)} sessions, shard {k}/{n_shards}: {len(todo)} to run")
    if not todo:
        return

    proc_cls, pipeline = load_fo(args.fo_repo, args.model_path, args.llm_path)
    t0 = time.time()
    for j, s in enumerate(todo):
        sess = json.loads((s / "session.json").read_text(encoding="utf-8"))
        onsets = run_one(proc_cls, pipeline, s / "user.wav", args.role)
        # agent 说话期间的开口 = 打断自己 → CUT
        a = sess["agent_utts"][0]
        actions = [{"t": t, "kind": "CUT"} for t in onsets
                   if a["t_start"] <= t <= a["t_end"] + 2.0]
        (out / f"{s.name}.json").write_text(
            json.dumps({"actions": actions, "onsets": onsets}), encoding="utf-8")
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(todo)}  {(time.time() - t0) / (j + 1):.1f}s/session")
    print("done ->", out)


if __name__ == "__main__":
    main()
