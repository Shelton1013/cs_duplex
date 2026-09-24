"""【服务器】基线:MiniCPM-o 4.5 原生全双工模式(可本地部署的开源 duplex 模型)。

官方 duplex 接口(源码核对):as_duplex() → prepare() → 每秒一次
streaming_prefill(audio_waveform=1s) + streaming_generate() → 返回 is_listen。
is_listen 由 True 变 False = 模型决定开口;agent 说话期间开口 = 打断自己 → CUT。
决策频率 1Hz(官方设计),反应延迟下限约 1s——"及时(≤0.8s)"视角下结构性吃亏,
结果需与"最终"视角一起看。

模型只听到用户通道(理想 AEC),与 Freeze-Omni 测法一致。
每个 session 记录完整 is_listen 轨迹,便于诊断(如开场主动问候导致一直处于说话态)。

环境:/home/pxieaf/home2/envs/minicpmo45(torch 2.8, transformers 4.51.0)
用法:
  CUDA_VISIBLE_DEVICES=8 python run_minicpmo.py --probes data/probe_audio_v2 \
      --out data/baselines_v2/minicpmo --model /home/pxieaf/home2/model/MiniCPM-o-4_5
  冒烟:加 --limit 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SEC = 16000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decode", default="sampling")
    args = ap.parse_args()

    import librosa
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=False, init_audio=True, init_tts=True)
    model.eval().cuda()
    model = model.as_duplex()
    ref_path = str(Path(args.model) / "assets" / "HT_ref_audio.wav")
    ref_audio, _ = librosa.load(ref_path, sr=16000, mono=True)

    k, n_sh = map(int, args.shard.split("/"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p for p in Path(args.probes).iterdir() if (p / "user.wav").exists())
    todo = [s for i, s in enumerate(sessions)
            if i % n_sh == k and not (out / f"{s.name}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} to run", flush=True)
    t_all = time.time()
    for j, s in enumerate(todo):
        wav, sr = sf.read(s / "user.wav", dtype="float32")
        assert sr == SEC
        n = int(np.ceil(len(wav) / SEC)) * SEC
        buf = np.zeros(n, np.float32)
        buf[: len(wav)] = wav
        torch.manual_seed(0)
        model.prepare(prefix_system_prompt="Streaming Omni Conversation.",
                      ref_audio=ref_audio, prompt_wav_path=ref_path)
        # 预热:冒烟测试发现它会先主动讲开场白(与用户输入无关)。先喂静音,
        # 直到连续 2 秒处于聆听态(开场白结束)再开始喂会话音频;时间轴从预热后算起。
        preroll, calm = 0, 0
        while calm < 2 and preroll < 30:
            model.streaming_prefill(audio_waveform=np.zeros(SEC, np.float32),
                                    frame_list=[], max_slice_nums=1, batch_vision_feed=False)
            r = model.streaming_generate(prompt_wav_path=ref_path,
                                         max_new_speak_tokens_per_chunk=20,
                                         decode_mode=args.decode)
            calm = calm + 1 if r.get("is_listen", True) else 0
            preroll += 1
        trace, actions, prev_listen = [], [], True
        for idx in range(n // SEC):
            model.streaming_prefill(audio_waveform=buf[idx * SEC:(idx + 1) * SEC],
                                    frame_list=[], max_slice_nums=1, batch_vision_feed=False)
            r = model.streaming_generate(prompt_wav_path=ref_path,
                                         max_new_speak_tokens_per_chunk=20,
                                         decode_mode=args.decode)
            listen = bool(r.get("is_listen", True))
            t = float(idx + 1)
            trace.append([t, listen, (r.get("text") or "")[:40]])
            if prev_listen and not listen:
                actions.append({"t": t, "kind": "CUT"})
            prev_listen = listen
        (out / f"{s.name}.json").write_text(
            json.dumps({"actions": actions, "trace": trace, "preroll_s": preroll},
                       ensure_ascii=False), encoding="utf-8")
        if (j + 1) % 20 == 0 or args.limit:
            print(f"{j + 1}/{len(todo)}  {(time.time() - t_all) / (j + 1):.1f}s/session", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
